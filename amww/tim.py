"""Sony TIM textures and a virtual PS1 VRAM for resolving model textures.

TIM layout:
  u32 id (0x10), u32 flags (bits 0-1 depth: 4/8/16/24bpp, bit 3 has CLUT)
  [CLUT block]  u32 size, u16 x, y, w, h, w*h 16-bit colours
  image block   u32 size, u16 x, y, w (in 16-bit units), h, pixel data
"""

import struct
from array import array

_LUT = None


def _rgba_lut():
    """65536-entry table of 15-bit PS1 colour -> RGBA bytes.

    0x0000 is the PS1's "fully transparent" colour; everything else is opaque.
    """
    global _LUT
    if _LUT is None:
        out = bytearray(65536 * 4)
        for c in range(65536):
            r, g, b = c & 31, (c >> 5) & 31, (c >> 10) & 31
            i = c * 4
            out[i] = (r << 3) | (r >> 2)
            out[i + 1] = (g << 3) | (g >> 2)
            out[i + 2] = (b << 3) | (b >> 2)
            out[i + 3] = 0 if c == 0 else 255
        _LUT = bytes(out)
    return _LUT


class Tim:
    def __init__(self, bpp, clut_rect, clut, img_rect, pixels, size):
        self.bpp = bpp
        self.clut_rect = clut_rect  # (x, y, w, h) in VRAM or None
        self.clut = clut            # array('H') or None
        self.img_rect = img_rect    # (x, y, w16, h) in VRAM
        self.pixels = pixels        # raw image bytes
        self.size = size            # total bytes in file

    @property
    def width(self):
        w16 = self.img_rect[2]
        return {4: w16 * 4, 8: w16 * 2, 16: w16, 24: w16 * 2 // 3}[self.bpp]

    @property
    def height(self):
        return self.img_rect[3]

    @property
    def palette_count(self):
        if self.bpp > 8 or not self.clut:
            return 1
        return max(1, len(self.clut) // (16 if self.bpp == 4 else 256))

    def to_rgba(self, palette=0):
        w, h = self.width, self.height
        lut = _rgba_lut()
        px = self.pixels
        if self.bpp == 24:
            out = bytearray(w * h * 4)
            stride = self.img_rect[2] * 2
            for y in range(h):
                row = px[y * stride:y * stride + w * 3]
                o = y * w * 4
                out[o:o + w * 4] = b"".join(
                    row[x * 3:x * 3 + 3] + b"\xff" for x in range(w))
            return bytes(out)
        if self.bpp == 16:
            vals = array("H")
            vals.frombytes(px[:w * h * 2])
            if array("H", [1]).tobytes() != b"\x01\x00":
                vals.byteswap()
            return b"".join(lut[v * 4:v * 4 + 4] for v in vals)
        n = 16 if self.bpp == 4 else 256
        if self.clut:
            pal = list(self.clut[palette * n:(palette + 1) * n])
            pal += [0] * (n - len(pal))
            colors = [lut[c * 4:c * 4 + 4] for c in pal]
        else:  # no CLUT: greyscale ramp so the image is at least viewable
            step = 255 // (n - 1)
            colors = [bytes((i * step,) * 3 + (255,)) for i in range(n)]
        if self.bpp == 8:
            return b"".join(colors[b] for b in px[:w * h])
        out = []
        for b in px[:w * h // 2]:
            out.append(colors[b & 15])
            out.append(colors[b >> 4])
        return b"".join(out)


def parse_tim(buf, off=0):
    """Parse a TIM at buf[off:]. Returns Tim or None if it doesn't validate."""
    end = len(buf)
    if off + 8 > end:
        return None
    magic, flags = struct.unpack_from("<II", buf, off)
    if magic != 0x10 or flags & ~0xB:
        return None
    bpp = (4, 8, 16, 24)[flags & 3]
    p = off + 8
    clut_rect = clut = None
    if flags & 8:
        if p + 12 > end:
            return None
        bnum, cx, cy, cw, ch = struct.unpack_from("<IHHHH", buf, p)
        if not cw or not ch or cx >= 1024 or cy >= 512 or bnum != 12 + cw * ch * 2:
            return None
        if p + bnum > end:
            return None
        clut = array("H")
        clut.frombytes(bytes(buf[p + 12:p + bnum]))
        if array("H", [1]).tobytes() != b"\x01\x00":
            clut.byteswap()
        clut_rect = (cx, cy, cw, ch)
        p += bnum
    elif bpp <= 8:
        # Paletted without a CLUT is legal but almost always a false positive
        # when scanning; accept it only at the very start of a file.
        if off != 0:
            return None
    if p + 12 > end:
        return None
    bnum, dx, dy, w, h = struct.unpack_from("<IHHHH", buf, p)
    if not w or not h or w > 1024 or h > 512 or dx >= 1024 or dy >= 512:
        return None
    if bnum != 12 + w * h * 2 or p + bnum > end:
        return None
    if bpp == 24 and (w * 2) % 3:
        return None
    pixels = bytes(buf[p + 12:p + bnum])
    return Tim(bpp, clut_rect, clut, (dx, dy, w, h), pixels, p + bnum - off)


class Vram:
    """1024x512 16-bit PS1 VRAM, filled by uploading TIMs at their stored
    coordinates, so TMD texture page / CLUT references can be resolved."""

    W, H = 1024, 512

    def __init__(self):
        self.mem = array("H", bytes(self.W * self.H * 2))
        self.used = bytearray(self.W * self.H)

    def _blit(self, x, y, w, h, data):
        for row in range(h):
            yy = y + row
            if yy >= self.H:
                break
            for col in range(w):
                xx = (x + col) % self.W
                i = yy * self.W + xx
                self.mem[i] = data[row * w + col]
                self.used[i] = 1

    def load_raw(self, x, y, w, h, data):
        """Upload a raw VRAM dump (w x h halfwords) at (x, y)."""
        px = array("H")
        px.frombytes(bytes(data[:w * h * 2]))
        if array("H", [1]).tobytes() != b"\x01\x00":
            px.byteswap()
        self._blit(x, y, w, h, px)

    def load_tim(self, tim):
        if tim.bpp == 24:
            return
        px = array("H")
        px.frombytes(tim.pixels)
        if array("H", [1]).tobytes() != b"\x01\x00":
            px.byteswap()
        x, y, w, h = tim.img_rect
        self._blit(x, y, w, h, px)
        if tim.clut_rect:
            cx, cy, cw, ch = tim.clut_rect
            self._blit(cx, cy, cw, ch, tim.clut)

    def page_coverage(self, tpage, clut):
        """Fraction of the texture page + CLUT that has been uploaded."""
        depth = (tpage >> 7) & 3
        tx, ty = (tpage & 0xF) * 64, ((tpage >> 4) & 1) * 256
        wpage = {0: 64, 1: 128}.get(depth, 256)
        total = hit = 0
        for y in range(ty, ty + 256, 8):
            for x in range(tx, tx + wpage, 4):
                total += 1
                hit += self.used[y * self.W + (x % self.W)]
        if depth < 2:
            cx, cy = (clut & 0x3F) * 16, (clut >> 6) & 0x1FF
            total += 1
            hit += self.used[cy * self.W + cx] if cy < self.H else 0
        return hit / total if total else 0.0

    def texture(self, tpage, clut):
        """Decode a 256x256 RGBA texture page as the GPU would see it."""
        depth = (tpage >> 7) & 3
        tx, ty = (tpage & 0xF) * 64, ((tpage >> 4) & 1) * 256
        cx, cy = (clut & 0x3F) * 16, min((clut >> 6) & 0x1FF, self.H - 1)
        lut = _rgba_lut()
        mem, W = self.mem, self.W
        out = []
        for v in range(256):
            row = (ty + v) * W
            for u in range(256):
                if depth == 0:
                    hw = mem[row + (tx + u // 4) % W]
                    c = mem[cy * W + (cx + ((hw >> ((u & 3) * 4)) & 15)) % W]
                elif depth == 1:
                    hw = mem[row + (tx + u // 2) % W]
                    c = mem[cy * W + (cx + ((hw >> ((u & 1) * 8)) & 255)) % W]
                else:
                    c = mem[row + (tx + u) % W]
                out.append(lut[c * 4:c * 4 + 4])
        return b"".join(out)
