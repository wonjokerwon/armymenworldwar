"""Build a small synthetic PS1 disc (MODE2/2352 + CD-DA track) for testing.

It contains hand-made TIM/TMD/VAG/VAB/SEQ assets, both as standalone files
and packed inside an archive, a Form 2 "STR" file, a subdirectory and data
outside the filesystem. No game data involved.
"""

import math
import os
import random
import shutil
import struct
import subprocess
import sys

SYNC = b"\x00" + b"\xff" * 10 + b"\x00"

# 4bpp TIM goes to VRAM x=640 (tpage 10), CLUT at (0, 480)
TPAGE = 10
CLUT = (0 // 16) | (480 << 6)


def tim_4bpp():
    clut = [0] + [((i * 2) & 31) | (((31 - i * 2) & 31) << 5) | (1 << 15) for i in range(1, 16)]
    w, h = 16, 16
    px = bytearray()
    for y in range(h):
        for x in range(0, w, 2):
            a, b = (x + y) % 16, (x + 1 + y) % 16
            px.append(a | (b << 4))
    out = struct.pack("<II", 0x10, 0x8)
    out += struct.pack("<IHHHH", 12 + 32, 0, 480, 16, 1) + struct.pack("<16H", *clut)
    out += struct.pack("<IHHHH", 12 + len(px), 640, 0, w // 4, h) + bytes(px)
    return out


def tim_16bpp():
    w, h = 8, 4
    px = struct.pack("<%dH" % (w * h), *[(x * 4) | ((y * 8) << 5) | (31 << 10) for y in range(h) for x in range(w)])
    return struct.pack("<II", 0x10, 0x2) + struct.pack("<IHHHH", 12 + len(px), 768, 256, w, h) + px


def tmd():
    verts = [(-100, -100, 0), (100, -100, 0), (-100, 100, 0), (100, 100, 0), (0, 0, -150)]
    vblock = b"".join(struct.pack("<hhhh", *v, 0) for v in verts)
    nblock = struct.pack("<hhhh", 0, 0, -4096, 0)
    prims = b""
    # textured, lit, flat quad: mode 0x2C, ilen 7
    uv = struct.pack("<BBH", 0, 0, CLUT) + struct.pack("<BBH", 15, 0, TPAGE) + \
        struct.pack("<BBH", 0, 15, 0) + struct.pack("<BBH", 15, 15, 0)
    idx = struct.pack("<5H", 0, 0, 1, 2, 3) + b"\x00\x00"
    prims += bytes((9, 7, 0, 0x2C)) + uv + idx
    # unlit flat triangle: mode 0x20 flag 1, ilen 3
    prims += bytes((4, 3, 1, 0x20)) + bytes((255, 0, 0, 0x20)) + struct.pack("<4H", 0, 1, 4, 0)
    # lit gouraud triangle: mode 0x30, ilen 4
    prims += bytes((6, 4, 0, 0x30)) + bytes((0, 255, 0, 0x30)) + struct.pack("<6H", 0, 2, 0, 3, 0, 4)
    vt = 28
    nt = vt + len(vblock)
    pt = nt + len(nblock)
    obj = struct.pack("<IIIIIIi", vt, len(verts), nt, 1, pt, 3, 0)
    return struct.pack("<III", 0x41, 0, 1) + obj + vblock + nblock + prims


def adpcm_blocks(nibbles):
    """Filter 0 / shift 8 blocks: each decoded sample is nibble * 16."""
    out = b""
    for i in range(0, len(nibbles), 28):
        chunk = nibbles[i:i + 28] + [0] * (28 - len(nibbles[i:i + 28]))
        last = i + 28 >= len(nibbles)
        body = bytes((chunk[k] & 15) | ((chunk[k + 1] & 15) << 4) for k in range(0, 28, 2))
        out += bytes((8, 1 if last else 0)) + body
    return out


NIBBLES = [round(7 * math.sin(i / 5)) for i in range(28 * 6)]


def vag():
    data = adpcm_blocks(NIBBLES)
    return (b"VAGp" + struct.pack(">IIII", 3, 0, len(data), 11025) + bytes(12) +
            b"testtone".ljust(16, b"\x00") + data)


def vab():
    body = adpcm_blocks(NIBBLES)
    ps, ts, vs = 1, 1, 1
    hdr_size = 0x20 + 0x800 + ps * 0x200 + 0x200
    hdr = b"pBAV" + struct.pack("<III", 7, 0, hdr_size + len(body)) + b"\x00\x00"
    hdr += struct.pack("<HHH", ps, ts, vs) + bytes((127, 64, 0, 0)) + bytes(4)
    hdr += bytes(0x800 + ps * 0x200)
    table = struct.pack("<256H", *([0, len(body) // 8] + [0] * 254))
    return hdr + table + body


def seq():
    events = bytes((0x00, 0x90, 60, 100, 0x60, 0x80, 60, 0, 0x00, 0xFF, 0x2F, 0x00))
    return b"pQES" + struct.pack(">IH", 1, 480) + bytes((0x07, 0xA1, 0x20)) + bytes((4, 2)) + events


def pack_archive():
    rnd = random.Random(1)
    parts = [b"PAK0" + bytes(12), tim_16bpp(), rnd.randbytes(100), tmd(),
             vab(), seq()]
    out = b""
    for p in parts:
        out += p
        out += bytes(-len(out) % 4)
    return out


def dirrec(name, lba, size, is_dir=False, xa=None):
    su = b""
    if xa is not None:
        su = struct.pack(">HHH", 0, 0, xa) + b"XA" + bytes(6)
    nlen = len(name)
    pad = b"\x00" if nlen % 2 == 0 else b""
    body = (struct.pack("<I", lba) + struct.pack(">I", lba) +
            struct.pack("<I", size) + struct.pack(">I", size) +
            bytes(7) + bytes((2 if is_dir else 0, 0, 0)) +
            struct.pack("<H", 1) + struct.pack(">H", 1) + bytes((nlen,)) + name + pad + su)
    rec = bytes((0, 0)) + body
    if len(rec) % 2:
        rec += b"\x00"
    return bytes((len(rec),)) + rec[1:]


def build_iso():
    """Returns list of (2048-byte user data, is_form2) sectors."""
    files = [
        (b"SYSTEM.CNF;1", b"BOOT = cdrom:\\SLUS_000.00;1\r\n"),
        (b"TEX.TIM;1", tim_4bpp()),
        (b"MODEL.TMD;1", tmd()),
        (b"SOUND.VAG;1", vag()),
        (b"DATA.PAK;1", pack_archive()),
        (b"MOVIE.STR;1", None),
    ]
    level = random.Random(2).randbytes(5000)
    lba = 20
    layout = []
    for name, data in files:
        n = 4 if data is None else (len(data) + 2047) // 2048
        layout.append((name, data, lba, n))
        lba += n
    level_lba = lba
    lba += (len(level) + 2047) // 2048
    hidden_lba = lba + 4
    total = hidden_lba + 20

    sectors = [(bytes(2048), False) for _ in range(total)]

    def dirsector(entries, self_lba, parent_lba):
        d = dirrec(b"\x00", self_lba, 2048, True) + dirrec(b"\x01", parent_lba, 2048, True)
        for e in entries:
            d += e
        return d.ljust(2048, b"\x00")

    root_entries = []
    for name, data, flba, n in layout:
        if data is None:
            root_entries.append(dirrec(name, flba, n * 2048, xa=0x3D55 & ~0x0800 | 0x1000 | 0x2000))
        else:
            root_entries.append(dirrec(name, flba, len(data), xa=0x0D55))
    root_entries.append(dirrec(b"LEVELS", 19, 2048, True, xa=0x8D55))
    sectors[18] = (dirsector(root_entries, 18, 18), False)
    sectors[19] = (dirsector([dirrec(b"L1.DAT;1", level_lba, len(level))], 19, 18), False)

    pvd = bytearray(2048)
    pvd[0:6] = b"\x01CD001"
    pvd[6] = 1
    pvd[8:40] = b"PLAYSTATION".ljust(32)
    pvd[40:72] = b"TESTDISC".ljust(32)
    pvd[80:88] = struct.pack("<I", total) + struct.pack(">I", total)
    pvd[128:132] = struct.pack("<H", 2048) + struct.pack(">H", 2048)
    pvd[156:156 + 34] = dirrec(b"\x00", 18, 2048, True)
    sectors[16] = (bytes(pvd), False)
    sectors[17] = (b"\xffCD001\x01".ljust(2048, b"\x00"), False)

    for name, data, flba, n in layout:
        for i in range(n):
            if data is None:
                sectors[flba + i] = (bytes((i,)) * 2324, True)
            else:
                sectors[flba + i] = (data[i * 2048:(i + 1) * 2048].ljust(2048, b"\x00"), False)
    for i in range((len(level) + 2047) // 2048):
        sectors[level_lba + i] = (level[i * 2048:(i + 1) * 2048].ljust(2048, b"\x00"), False)
    for i in range(20):
        sectors[hidden_lba + i] = (b"HIDDEN" + bytes((i,)) * 2042, False)
    return sectors


def bcd(v):
    return ((v // 10) << 4) | (v % 10)


def raw_sector(lba, data, form2):
    a = lba + 150
    hdr = SYNC + bytes((bcd(a // 4500), bcd(a // 75 % 60), bcd(a % 75), 2))
    sub = bytes((0, 0, 0x24 if form2 else 0x08, 0)) * 2
    if form2:
        return hdr + sub + data[:2324].ljust(2324, b"\x00") + bytes(4)
    return hdr + sub + data + bytes(280)


def build(out_dir, make_chd=True):
    os.makedirs(out_dir, exist_ok=True)
    binf = os.path.join(out_dir, "test.bin")
    sectors = build_iso()
    with open(binf, "wb") as fh:
        for i, (data, form2) in enumerate(sectors):
            fh.write(raw_sector(i, data, form2))
        # CD-DA track: 1 second of a 440 Hz stereo tone
        pcm = b"".join(struct.pack("<hh", s, s) for s in
                       (int(8000 * math.sin(2 * math.pi * 440 * i / 44100)) for i in range(44100)))
        pcm = pcm.ljust((len(pcm) + 2351) // 2352 * 2352, b"\x00")
        fh.write(pcm)
    t2 = len(sectors)
    cue = os.path.join(out_dir, "test.cue")
    with open(cue, "w") as fh:
        fh.write('FILE "test.bin" BINARY\n  TRACK 01 MODE2/2352\n    INDEX 01 00:00:00\n'
                 '  TRACK 02 AUDIO\n    INDEX 01 %02d:%02d:%02d\n' % (t2 // 4500, t2 // 75 % 60, t2 % 75))
    if make_chd and shutil.which("chdman"):
        chd = os.path.join(out_dir, "test.chd")
        subprocess.run(["chdman", "createcd", "-i", cue, "-o", chd, "-f"],
                       check=True, capture_output=True)
        return chd
    return cue


if __name__ == "__main__":
    print(build(sys.argv[1] if len(sys.argv) > 1 else "testdisc"))
