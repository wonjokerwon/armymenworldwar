"""Signature scanning: find standard PS1 asset formats anywhere inside a file.

Games often pack TIM/TMD/VAB data inside their own archive formats. As long
as the archive isn't compressed, the embedded assets keep their headers and
can be found by magic number + structural validation.
"""

import math

from .audio import parse_vab, parse_vag, seq_to_midi
from .hex3 import parse_hex3
from .mdl3 import parse_mdl3
from .tim import parse_tim
from .tmd import parse_tmd

SIGNATURES = (
    ("tim", b"\x10\x00\x00\x00", lambda b, o: parse_tim(b, o)),
    ("tmd", b"\x41\x00\x00\x00", lambda b, o: parse_tmd(b, o)),
    ("vag", b"VAGp", parse_vag),
    ("vab", b"pBAV", parse_vab),
    ("seq", b"pQES", seq_to_midi),
    # 3DO's own formats used by Army Men: World War - Team Assault
    ("mdl3", b"3MDL", parse_mdl3),
    ("hex3", b"3HEX", parse_hex3),
)


class Found:
    def __init__(self, kind, offset, obj, size):
        self.kind = kind
        self.offset = offset
        self.obj = obj
        self.size = size


def _size_of(kind, obj):
    if kind == "seq":
        return obj[1]
    if kind == "hex3":
        return obj[3]
    return obj.size


def scan_buffer(buf):
    found = []
    for kind, magic, parser in SIGNATURES:
        pos = buf.find(magic)
        while pos >= 0:
            # TIM/TMD headers are word-aligned in practice; requiring that
            # cuts false positives in random data by 4x.
            if kind in ("tim", "tmd", "mdl3") and pos % 4:
                pos = buf.find(magic, pos + 1)
                continue
            try:
                obj = parser(buf, pos)
            except (IndexError, ValueError, OverflowError):
                obj = None
            if obj is not None:
                found.append(Found(kind, pos, obj, _size_of(kind, obj)))
            pos = buf.find(magic, pos + 1)
    found.sort(key=lambda f: f.offset)
    return found


def entropy(buf, sample=1 << 20):
    data = buf[:sample]
    if not data:
        return 0.0
    counts = [0] * 256
    for b in data:
        counts[b] += 1
    n = len(data)
    return -sum(c / n * math.log2(c / n) for c in counts if c)


def coverage(found, length):
    """Fraction of a file's bytes accounted for by recognised assets."""
    if not length:
        return 0.0
    spans = sorted((f.offset, f.offset + f.size) for f in found)
    covered = 0
    cur_s = cur_e = None
    for s, e in spans:
        if cur_e is None or s > cur_e:
            if cur_e is not None:
                covered += cur_e - cur_s
            cur_s, cur_e = s, e
        else:
            cur_e = max(cur_e, e)
    if cur_e is not None:
        covered += cur_e - cur_s
    return min(1.0, covered / length)


def identify(buf):
    """Best-effort label for a file's own header."""
    head = bytes(buf[:16])
    if head.startswith(b"PS-X EXE"):
        return "PS-X EXE (game executable)"
    if head.startswith(b"RIFF") and head[8:12] == b"CDXA":
        return "RIFF CDXA (XA audio/STR video)"
    if head.startswith(b"BOOT") or head.startswith(b"cdrom:"):
        return "SYSTEM.CNF (boot config)"
    for kind, magic, _ in SIGNATURES:
        if head.startswith(magic):
            return kind.upper()
    return None
