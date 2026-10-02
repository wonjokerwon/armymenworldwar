"""End-to-end test on a synthetic disc: python -m unittest discover tests"""

import os
import shutil
import struct
import sys
import tempfile
import unittest
import zlib

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import make_test_disc  # noqa: E402
from amww.cli import main  # noqa: E402


def read_png(path):
    with open(path, "rb") as fh:
        data = fh.read()
    w, h = struct.unpack(">II", data[16:24])
    idat = b""
    p = 8
    while p < len(data):
        n = struct.unpack(">I", data[p:p + 4])[0]
        if data[p + 4:p + 8] == b"IDAT":
            idat += data[p + 8:p + 8 + n]
        p += 12 + n
    raw = zlib.decompress(idat)
    rows = [raw[y * (w * 4 + 1) + 1:(y + 1) * (w * 4 + 1)] for y in range(h)]
    return w, h, rows


def read_wav_samples(path):
    with open(path, "rb") as fh:
        data = fh.read()
    n = struct.unpack("<I", data[40:44])[0]
    return list(struct.unpack("<%dh" % (n // 2), data[44:44 + n]))


class PipelineTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp()
        image = make_test_disc.build(os.path.join(cls.tmp, "disc"))
        cls.out = os.path.join(cls.tmp, "out")
        main(["extract", image, "-o", cls.out])

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp)

    def path(self, *p):
        return os.path.join(self.out, *p)

    def test_filesystem(self):
        with open(self.path("files", "MODEL.TMD"), "rb") as fh:
            self.assertEqual(fh.read(), make_test_disc.tmd())
        self.assertTrue(os.path.exists(self.path("files", "LEVELS", "L1.DAT")))
        # Form 2 file is kept as raw 2352-byte sectors
        self.assertEqual(os.path.getsize(self.path("files", "MOVIE.STR")), 4 * 2352)
        unlisted = os.listdir(self.path("files", "_unlisted"))
        self.assertEqual(len(unlisted), 1)

    def test_tim_4bpp(self):
        w, h, rows = read_png(self.path("textures", "TEX.TIM@00000000_4bpp.png"))
        self.assertEqual((w, h), (16, 16))
        # pixel (0,0) uses palette index 0 = colour 0x0000 = transparent
        self.assertEqual(rows[0][3], 0)
        # pixel (1,0) uses index 1: r=2, g=29 (5-bit)
        self.assertEqual(tuple(rows[0][4:8]), ((2 << 3) | 0, (29 << 3) | (29 >> 2), 0, 255))

    def test_embedded_assets(self):
        self.assertTrue(os.path.exists(self.path("textures", "DATA.PAK@00000010_16bpp.png")))
        self.assertTrue(os.path.exists(self.path("music", "DATA.PAK@00000de4.mid")))
        with open(self.path("music", "DATA.PAK@00000de4.mid"), "rb") as fh:
            self.assertTrue(fh.read().startswith(b"MThd"))

    def test_model(self):
        d = self.path("models", "MODEL.TMD@00000000")
        with open(os.path.join(d, "MODEL.TMD@00000000.obj")) as fh:
            obj = fh.read().splitlines()
        faces = [line for line in obj if line.startswith("f ")]
        self.assertEqual(len(faces), 3)
        self.assertEqual(len(faces[0].split()), 5)  # quad
        self.assertIn("v -100 100 0", obj)  # Y flipped to Y-up
        with open(os.path.join(d, "MODEL.TMD@00000000.mtl")) as fh:
            self.assertIn("map_Kd tex_000a_7800.png", fh.read())
        # texture page decoded from VRAM matches the TIM's top-left pixels
        _, _, rows = read_png(os.path.join(d, "tex_000a_7800.png"))
        _, _, tim_rows = read_png(self.path("textures", "TEX.TIM@00000000_4bpp.png"))
        self.assertEqual(rows[5][:64], tim_rows[5][:64])

    def test_audio(self):
        expected = [n * 16 for n in make_test_disc.NIBBLES]
        vag = read_wav_samples(self.path("audio", "vag", "SOUND.VAG@00000000_testtone.wav"))
        self.assertEqual(vag[:len(expected)], expected)
        vab = read_wav_samples(self.path("audio", "vab", "DATA.PAK@00000164", "sample_000.wav"))
        self.assertEqual(vab[:len(expected)], expected)
        cdda = read_wav_samples(self.path("audio", "cdda", "track02.wav"))
        self.assertEqual(max(cdda), 7999)

    def test_report(self):
        with open(self.path("REPORT.md")) as fh:
            report = fh.read()
        self.assertIn("LEVELS/L1.DAT", report)


if __name__ == "__main__":
    unittest.main()
