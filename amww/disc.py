"""Disc image access: CHD -> BIN/CUE, CUE parsing, raw sector reads and ISO9660.

PlayStation discs are CD-ROM XA. Data sectors are either Mode 2 Form 1
(2048 bytes of user data, used for normal files) or Mode 2 Form 2
(2324 bytes, used for streamed XA audio and STR video). Form 2 files can't
be represented as plain 2048-byte files, so they are dumped as raw sectors,
which is what jPSXdec / ffmpeg expect.
"""

import os
import re
import shutil
import struct
import subprocess

RAW_SECTOR = 2352
SYNC = b"\x00" + b"\xff" * 10 + b"\x00"

# XA attribute bits from the directory record system-use area.
XA_FORM1 = 0x0800
XA_FORM2 = 0x1000
XA_INTERLEAVED = 0x2000
XA_CDDA = 0x4000
XA_DIR = 0x8000

SECTOR_SIZES = {
    "MODE1/2048": 2048,
    "MODE1/2352": 2352,
    "MODE2/2336": 2336,
    "MODE2/2352": 2352,
    "AUDIO": 2352,
}


class Track:
    def __init__(self, number, mode, path):
        self.number = number
        self.mode = mode
        self.path = path
        self.sector_size = SECTOR_SIZES.get(mode, 2352)
        self.start = 0  # sector offset of INDEX 01 within the file
        self.length = 0  # sectors

    @property
    def is_audio(self):
        return self.mode == "AUDIO"

    def __repr__(self):
        return "Track(%d, %s, start=%d, length=%d)" % (
            self.number, self.mode, self.start, self.length)


def chd_to_cue(chd_path, out_dir):
    """Convert a CHD to BIN/CUE with MAME's chdman. Returns the .cue path."""
    chdman = shutil.which("chdman")
    if not chdman:
        raise RuntimeError(
            "chdman not found. Install MAME tools (Debian/Ubuntu: "
            "'apt install mame-tools', macOS: 'brew install rom-tools', "
            "Windows: chdman.exe ships with MAME) and make sure it's on PATH.")
    os.makedirs(out_dir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(chd_path))[0]
    cue = os.path.join(out_dir, stem + ".cue")
    binf = os.path.join(out_dir, stem + ".bin")
    subprocess.run([chdman, "extractcd", "-i", chd_path, "-o", cue,
                    "-ob", binf, "-f"], check=True)
    return cue


def _msf(m, s, f):
    return (int(m) * 60 + int(s)) * 75 + int(f)


def parse_cue(cue_path):
    base = os.path.dirname(os.path.abspath(cue_path))
    tracks = []
    cur_file = None
    with open(cue_path, encoding="latin-1") as fh:
        for line in fh:
            tok = line.strip()
            m = re.match(r'FILE\s+"(.*)"\s+\S+$', tok, re.I) or \
                re.match(r'FILE\s+(\S+)\s+\S+$', tok, re.I)
            if m:
                cur_file = os.path.join(base, m.group(1))
                continue
            m = re.match(r"TRACK\s+(\d+)\s+(\S+)", tok, re.I)
            if m:
                tracks.append(Track(int(m.group(1)), m.group(2).upper(), cur_file))
                continue
            m = re.match(r"INDEX\s+01\s+(\d+):(\d+):(\d+)", tok, re.I)
            if m and tracks:
                tracks[-1].start = _msf(*m.groups())
    for i, t in enumerate(tracks):
        nxt = tracks[i + 1] if i + 1 < len(tracks) else None
        if nxt is not None and nxt.path == t.path:
            t.length = nxt.start - t.start
        else:
            t.length = os.path.getsize(t.path) // t.sector_size - t.start
    return tracks


def tracks_for_image(path):
    """Return tracks for a .cue, .bin or .iso (CHDs must be converted first)."""
    ext = os.path.splitext(path)[1].lower()
    if ext == ".cue":
        return parse_cue(path)
    with open(path, "rb") as fh:
        head = fh.read(16)
    if head[:12] == SYNC:
        mode = "MODE2/2352" if head[15] == 2 else "MODE1/2352"
    else:
        mode = "MODE1/2048"
    t = Track(1, mode, path)
    t.length = os.path.getsize(path) // t.sector_size
    return [t]


class DataTrack:
    """Sector-level reader for a data track."""

    def __init__(self, track):
        self.track = track
        self.ss = track.sector_size
        self.fh = open(track.path, "rb")

    def close(self):
        self.fh.close()

    @property
    def length(self):
        return self.track.length

    def raw(self, lba):
        self.fh.seek((self.track.start + lba) * self.ss)
        return self.fh.read(self.ss)

    def user(self, lba):
        """Return (user_data, is_form2) for a sector."""
        raw = self.raw(lba)
        if self.ss == 2048:
            return raw, False
        if self.ss == 2336:
            sub = raw[0:8]
            hdr = 8
        else:
            if raw[:12] != SYNC:
                return b"\x00" * 2048, False
            if raw[15] == 1:
                return raw[16:16 + 2048], False
            sub = raw[16:24]
            hdr = 24
        if len(sub) < 8:
            return b"\x00" * 2048, False
        if sub[2] & 0x20:
            return raw[hdr:hdr + 2324], True
        return raw[hdr:hdr + 2048], False

    def is_form2(self, lba):
        if self.ss == 2048:
            return False
        self.fh.seek((self.track.start + lba) * self.ss)
        head = self.fh.read(24)
        if self.ss == 2336:
            return len(head) >= 3 and bool(head[2] & 0x20)
        return len(head) >= 24 and head[15] == 2 and bool(head[18] & 0x20)


class IsoEntry:
    def __init__(self, path, lba, size, is_dir, xa):
        self.path = path
        self.lba = lba
        self.size = size
        self.is_dir = is_dir
        self.xa = xa

    @property
    def sectors(self):
        return (self.size + 2047) // 2048


def _parse_record(rec):
    lba = struct.unpack_from("<I", rec, 2)[0]
    size = struct.unpack_from("<I", rec, 10)[0]
    flags = rec[25]
    nlen = rec[32]
    name = rec[33:33 + nlen]
    su = rec[33 + nlen + (1 - nlen % 2):]
    xa = 0
    if len(su) >= 14 and su[6:8] == b"XA":
        xa = struct.unpack_from(">H", su, 4)[0]
    return name, lba, size, bool(flags & 2), xa


def walk_iso(dt):
    """Yield IsoEntry for every file and directory on the disc."""
    pvd, _ = dt.user(16)
    if pvd[1:6] != b"CD001":
        raise ValueError("No ISO9660 primary volume descriptor at sector 16")
    _, lba, size, _, _ = _parse_record(pvd[156:156 + 34])
    seen = set()
    yield from _walk_dir(dt, lba, size, "", seen)


def _walk_dir(dt, lba, size, prefix, seen):
    if lba in seen:
        return
    seen.add(lba)
    subdirs = []
    for s in range((size + 2047) // 2048):
        data, _ = dt.user(lba + s)
        pos = 0
        while pos < 2048:
            rl = data[pos]
            if rl == 0:
                break
            rec = data[pos:pos + rl]
            pos += rl
            name, elba, esize, is_dir, xa = _parse_record(rec)
            if name in (b"\x00", b"\x01"):
                continue
            name = name.decode("latin-1").split(";")[0].rstrip(".")
            path = prefix + name
            entry = IsoEntry(path, elba, esize, is_dir, xa)
            yield entry
            if is_dir:
                subdirs.append(entry)
    for d in subdirs:
        yield from _walk_dir(dt, d.lba, d.size, d.path + "/", seen)


def extract_file(dt, entry, dest):
    """Write a file out. Returns True if dumped as raw sectors (Form 2 data)."""
    n = entry.sectors
    raw = bool(entry.xa & (XA_FORM2 | XA_INTERLEAVED)) or \
        any(dt.is_form2(entry.lba + i) for i in range(n))
    os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
    with open(dest, "wb") as out:
        if raw and dt.ss != 2048:
            for i in range(n):
                out.write(dt.raw(entry.lba + i))
        else:
            remaining = entry.size
            for i in range(n):
                data, _ = dt.user(entry.lba + i)
                chunk = data[:min(2048, remaining)]
                out.write(chunk)
                remaining -= len(chunk)
    return raw and dt.ss != 2048


def unlisted_regions(dt, entries, min_sectors=16):
    """Sector ranges on the data track not claimed by any ISO entry.

    Some PS1 games read data by raw LBA, outside the filesystem; those
    regions are worth dumping and scanning too.
    """
    used = bytearray(dt.length)
    for i in range(min(dt.length, 24)):  # system area + volume descriptors
        used[i] = 1
    for e in entries:
        for s in range(e.lba, min(dt.length, e.lba + max(1, e.sectors))):
            used[s] = 1
    regions = []
    s = 0
    while s < dt.length:
        if used[s]:
            s += 1
            continue
        e = s
        while e < dt.length and not used[e]:
            e += 1
        if e - s >= min_sectors:
            # skip all-zero padding
            nonzero = any(dt.user(i)[0].strip(b"\x00") for i in range(s, e))
            if nonzero:
                regions.append((s, e - s))
        s = e
    return regions


def write_cdda_wav(track, dest):
    from .wav import write_wav_header
    os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
    nbytes = track.length * RAW_SECTOR
    with open(track.path, "rb") as src, open(dest, "wb") as out:
        write_wav_header(out, nbytes, 44100, 2)
        src.seek(track.start * RAW_SECTOR)
        remaining = nbytes
        while remaining > 0:
            chunk = src.read(min(remaining, 1 << 20))
            if not chunk:
                break
            out.write(chunk)
            remaining -= len(chunk)


def is_packed_xa(path):
    """True for a Form 1 file that stores whole 2336-byte XA sectors
    (subheader + data) back to back, as this disc does for its .XA music
    and .STR movies. The subheader is stored twice at the start of each."""
    size = os.path.getsize(path)
    if size < 2336 * 4 or size % 2336:
        return False
    with open(path, "rb") as fh:
        head = fh.read(8)
        fh.seek(2336)
        head2 = fh.read(8)
    return head[:4] == head[4:8] and head2[:4] == head2[4:8]


def packed_xa_to_raw(src, dest):
    """Rebuild raw 2352-byte sectors (sync + header + 2336 bytes)."""
    os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
    with open(src, "rb") as fi, open(dest, "wb") as fo:
        lba = 0
        while True:
            chunk = fi.read(2336)
            if len(chunk) < 2336:
                break
            a = lba + 150
            bcd = [((v // 10) << 4) | (v % 10) for v in (a // 4500, a // 75 % 60, a % 75)]
            fo.write(SYNC + bytes(bcd) + b"\x02" + chunk)
            lba += 1
