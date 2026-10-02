"""Unit tests for the 3DO formats, using hand-built (non-game) data."""

import os
import struct
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from amww.hex3 import parse_hex3  # noqa: E402
from amww.mdl3 import export_obj, parse_mdl3  # noqa: E402
from amww.tim import Vram  # noqa: E402


def w(op, val=0, flags=0):
    return struct.pack("<I", (val << 8) | op | flags)


def pair(op, lo, hi, flags=0):
    return w(op, lo | (hi << 12), flags)


def build_mdl3(verts, stream, slots=()):
    vblock = b"".join(struct.pack("<hhhH", *v, 0x8000 if i == len(verts) - 1 else 0)
                      for i, v in enumerate(verts))
    body = vblock + stream + w(0x68)
    npoly = sum(1 for i in range(0, len(stream), 4) if stream[i] & 3)
    size = 24 + len(body)
    out = b"3MDL" + struct.pack("<IIIII", 1, len(verts), npoly, 0x1DF, size) + body
    for u, v, tpage, clut in slots:
        out += struct.pack("<BBHH", u, v, tpage, clut) + b"\xf3" * 12 + b"\0\0"
    return out


class Mdl3Test(unittest.TestCase):
    def test_box_quads(self):
        # unit box, bottom 0-3 at y=0, top 4-7 at y=-100 (PS1 Y is down)
        verts = [(0, 0, 0), (100, 0, 0), (0, 0, 100), (100, 0, 100),
                 (0, -100, 0), (100, -100, 0), (0, -100, 100), (100, -100, 100)]
        s = w(0x58, 0) + w(0x5C, 0xD0 << 16)
        s += w(0x20, 0) + w(0x24, 31 << 8) + w(0x28, 31 | 31 << 8) + w(0x2C, 31)
        s += pair(0x40, 0, 4) + pair(0x48, 5, 1)          # face x-z plane z=0
        s += w(0x0C, 0x808080, 1)                        # draw
        s += pair(0x44, 7, 4)                            # D=7, A=4
        s += w(0x3C, 6, 1)                               # C=6, draw top face
        s += pair(0x4C, 3, 1, 1)                         # D=3, B=1 -> draw
        data = build_mdl3(verts, s, [(32, 64, 0x0B, 0x367F)])
        m = parse_mdl3(data)
        self.assertEqual([p[0] for p in m.polys],
                         [[0, 4, 5, 1], [4, 4, 6, 7], [4, 1, 6, 3]])
        self.assertEqual(m.polys[0][1], [(0, 0), (0, 31), (31, 31), (31, 0)])
        self.assertEqual(m.slots[0].clut, 0x367F)
        self.assertEqual((m.slots[0].u, m.slots[0].v, m.slots[0].tpage), (32, 64, 0x0B))

    def test_triangle_fan_and_singles(self):
        # pyramid: apex 0, base 1-4; triangles are corners A, B, D
        verts = [(0, -100, 0), (-50, 0, -50), (50, 0, -50), (50, 0, 50), (-50, 0, 50)]
        s = w(0x5C, 0x80 << 16)
        s += pair(0x40, 1, 0) + w(0x38, 2) + w(0x00, 0x808080, 1)
        s += pair(0x44, 3, 2, 1) + pair(0x44, 4, 3, 1) + pair(0x44, 1, 4, 1)
        s += w(0x34, 3, 2) + w(0x30, 2, 1)
        m = parse_mdl3(build_mdl3(verts, s))
        self.assertEqual([p[0] for p in m.polys],
                         [[1, 0, 2], [2, 0, 3], [3, 0, 4], [4, 0, 1], [3, 0, 1], [3, 2, 1]])

    def test_untextured_and_export(self):
        verts = [(0, 0, 0), (10, 0, 0), (0, -10, 0)]
        s = w(0x5C, 0x40 << 16) + pair(0x40, 0, 1) + w(0x38, 2) + w(0x00, 0x0000FF, 1)
        m = parse_mdl3(build_mdl3(verts, s))
        self.assertEqual(m.polys[0][3], -1)
        with tempfile.TemporaryDirectory() as d:
            export_obj(m, os.path.join(d, "t.obj"), name="t")
            obj = open(os.path.join(d, "t.obj")).read()
            mtl = open(os.path.join(d, "t.mtl")).read()
        self.assertIn("v 0 10 0", obj)
        self.assertIn("f 1/1 2/2 3/3", obj)
        self.assertIn("Kd 1.000 0.000 0.000", mtl)

    def test_bad_header(self):
        self.assertIsNone(parse_mdl3(b"3MDL" + bytes(20)))
        self.assertIsNone(parse_mdl3(b"XXXX" + bytes(40)))


class Hex3Test(unittest.TestCase):
    def test_4bpp(self):
        px = bytes([0x10, 0x32] * 8)  # 4x8, indices 0,1,2,3 per row
        clut = struct.pack("<16H", 0, 31, 31 << 5, 31 << 10, *([0x8000] * 12))
        data = b"3HEX" + struct.pack("<5I", 1, 4, 8, 0, 0) + px + clut + bytes(36)
        w_, h, rgba, size = parse_hex3(data)
        self.assertEqual((w_, h, size), (4, 8, len(data)))
        self.assertEqual(rgba[0:16], bytes([0, 0, 0, 0, 255, 0, 0, 255,
                                            0, 255, 0, 255, 0, 0, 255, 255]))


class VramTest(unittest.TestCase):
    def test_raw_upload_and_texture(self):
        v = Vram()
        page = bytearray(64 * 2 * 256)
        page[0] = 0x21  # pixels (0,0)=1, (1,0)=2
        v.load_raw(512, 0, 64, 256, bytes(page))
        v.load_raw(1008, 217, 16, 1, struct.pack("<16H", 0, 31, 31 << 5, *([0] * 13)))
        tex = v.texture(0x08, (217 << 6) | 63)
        self.assertEqual(tex[0:8], bytes([255, 0, 0, 255, 0, 255, 0, 255]))


if __name__ == "__main__":
    unittest.main()
