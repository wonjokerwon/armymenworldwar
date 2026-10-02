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
                         [[0, 1, 5, 4], [4, 7, 6, 4], [4, 3, 6, 1]])
        # flag bit 0 = stored with reversed winding; corner 0 stays first
        self.assertEqual(m.polys[0][1], [(0, 31), (31, 31), (31, 0), (0, 0)])
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
                         [[1, 2, 0], [2, 3, 0], [3, 4, 0], [4, 1, 0], [3, 0, 1], [3, 1, 2]])

    def test_triangle_uv_pairing(self):
        # each corner's UV comes from its partner register:
        # B(0x30)-0x20, A(0x34)-0x24, D(0x38)-0x28
        verts = [(0, 0, 0), (10, 0, 0), (0, -10, 0)]
        s = w(0x5C, 0x80 << 16) + w(0x58, 0)
        s += w(0x24, 1 | 1 << 8) + w(0x34, 0)
        s += w(0x20, 2 | 2 << 8) + w(0x30, 1)
        s += w(0x28, 3 | 3 << 8) + w(0x38, 2, 2)  # flag 2: stored winding
        m = parse_mdl3(build_mdl3(verts, s))
        self.assertEqual(m.polys[0][0], [0, 1, 2])
        self.assertEqual(m.polys[0][1], [(1, 1), (2, 2), (3, 3)])

    def test_untextured_and_export(self):
        verts = [(0, 0, 0), (10, 0, 0), (0, -10, 0)]
        s = w(0x5C, 0x40 << 16) + pair(0x40, 0, 1) + w(0x38, 2)
        s += w(0x00, 0x0000FF) + w(0x04, 0x0000FF) + w(0x08, 0x0000FF, 1)
        m = parse_mdl3(build_mdl3(verts, s))
        self.assertEqual(m.polys[0][3], -1)
        with tempfile.TemporaryDirectory() as d:
            export_obj(m, os.path.join(d, "t.obj"), name="t")
            obj = open(os.path.join(d, "t.obj")).read()
            mtl = open(os.path.join(d, "t.mtl")).read()
        self.assertIn("v 0 10 0", obj)
        self.assertIn("f 1/1 3/2 2/3", obj)
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


def build_amdl(verts_bones, stream, parents, locals_, slots):
    """Character block: pre-header, AMDL mesh, skeleton, one slot set."""
    nv = len(verts_bones)
    vblock = b"".join(struct.pack("<hhhH", *v, bone) for v, bone in verts_bones)
    body = vblock + stream + w(0x68)
    size = 24 + len(body)
    npoly = sum(1 for i in range(0, len(stream), 4) if stream[i] & 3)
    mesh = b"AMDL" + struct.pack("<IIIII", len(parents), nv, npoly, 0x1DF, size) + body
    skel = bytes(12) + struct.pack("<I", len(parents))
    skel += bytes(p & 0xFF for p in parents).ljust((len(parents) + 3) // 4 * 4, b"\0")
    for rows, t in locals_:
        skel += struct.pack("<9h", *[v for r in rows for v in r]) + b"\0\0" + struct.pack("<3i", *t)
    slot = b"".join(struct.pack("<BBHH", u, v, tp, cl) + b"\xf3" * 12 + b"\0\0"
                    for u, v, tp, cl in slots)
    return struct.pack("<II", len(mesh) + len(skel), len(slots)) + mesh + skel + slot


class AmdlTest(unittest.TestCase):
    I = ((4096, 0, 0), (0, 4096, 0), (0, 0, 4096))

    def make(self):
        from amww.amdl import parse_amdl
        # root (rotated -90 about X, which is dropped) -> child at +100 Y
        rx = ((4096, 0, 0), (0, 0, -4096), (0, 4096, 0))
        verts = [((0, 0, 0), 0), ((10, 0, 0), 1), ((0, 10, 0), 1)]
        s = w(0x70, 0x84) + struct.pack("<II", 1, 0) + b"torso".ljust(16, b"\0")
        s += w(0x5C, 0x80 << 16) + w(0x58, 0)
        s += pair(0x40, 0, 1) + w(0x38, 2) + w(0x00, 0x808080, 1)
        data = b"pad!" + build_amdl(verts, s, [-1, 0], [(rx, (0, 0, 0)), (self.I, (0, 100, 0))],
                                    [(0, 0, 0x0A, 0x7F)])
        return parse_amdl(data, data.index(b"AMDL"))

    def test_parse_and_pose(self):
        a = self.make()
        self.assertEqual(a.parents, [-1, 0])
        self.assertEqual(a.model.groups[0][2], "torso")
        self.assertEqual([p[0] for p in a.model.polys], [[0, 2, 1]])
        self.assertEqual(a.posed_vertices(), [(0, 0, 0), (10, 100, 0), (0, 110, 0)])
        self.assertEqual(a.variants[0][0].clut, 0x7F)

    def test_export_glb_and_obj(self):
        import json
        from amww.amdl import export_amdl
        a = self.make()
        with tempfile.TemporaryDirectory() as d:
            base = os.path.join(d, "soldier")
            export_amdl(a, base, a.variants[0], lambda tp, cl: ("tex/x.png", b"\x89PNG"))
            with open(base + ".glb", "rb") as fh:
                glb = fh.read()
            obj = open(base + ".obj").read()
        magic, version, length = struct.unpack_from("<III", glb)
        self.assertEqual((magic, version, length), (0x46546C67, 2, len(glb)))
        js = json.loads(glb[20:20 + struct.unpack_from("<I", glb, 12)[0]])
        self.assertEqual(len(js["skins"][0]["joints"]), 2)
        self.assertEqual([m["name"] for m in js["meshes"]], ["torso"])
        self.assertIn("v 10 -100 -0", obj.replace("v 10 -100 0", "v 10 -100 -0"))
        self.assertIn("o torso", obj)
