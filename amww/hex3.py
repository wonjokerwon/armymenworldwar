"""3DO '3HEX' textures (menu/UI art in Army Men: World War - Team Assault).

  char[4] '3HEX', u32 type (1 = 4bpp, 2 = 8bpp or 16bpp), u32 width,
  u32 height, u32 flags, u32 checksum, pixel data, [CLUT], 36-byte trailer

Type 1 (4bpp + 16-colour CLUT) is decoded. Type 2 (full-screen menu
backgrounds) uses a different pixel layout that hasn't been worked out yet.
"""

import struct

from .tim import _rgba_lut


def parse_hex3(buf, off=0):
    """Returns (width, height, rgba bytes, size) or None."""
    if bytes(buf[off:off + 4]) != b"3HEX" or off + 24 > len(buf):
        return None
    kind, w, h, _flags, _sum = struct.unpack_from("<5I", buf, off + 4)
    if not (0 < w <= 1024 and 0 < h <= 512) or kind not in (1, 2):
        return None
    lut = _rgba_lut()
    p = off + 24
    end = len(buf)
    if kind == 1:
        npx = w * h // 2
        if p + npx + 32 > end:
            return None
        clut = struct.unpack_from("<16H", buf, p + npx)
        cols = [lut[c * 4:c * 4 + 4] for c in clut]
        out = []
        for b in buf[p:p + npx]:
            out.append(cols[b & 15])
            out.append(cols[b >> 4])
        return w, h, b"".join(out), 24 + npx + 32 + 36
    return None  # type 2 (8/16bpp menu backgrounds) is not decoded yet
