"""PS1 SPU audio: VAG samples, VAB sound banks and SEQ music -> WAV / MIDI.

The SPU uses 4-bit ADPCM in 16-byte blocks:
  byte 0: shift (low nibble) | filter (high nibble)
  byte 1: flags (bit 0 = end of sample / loop end)
  14 bytes: 28 nibbles, low nibble first
"""

import struct
from array import array

FILTERS = ((0, 0), (60, 0), (115, -52), (98, -55), (122, -60))


def decode_adpcm(data, stop_at_end=True):
    out = array("h")
    s1 = s2 = 0
    for i in range(0, len(data) - 15, 16):
        sf = data[i]
        flags = data[i + 1]
        shift = sf & 0xF
        if shift > 12:
            shift = 9
        f0, f1 = FILTERS[min((sf >> 4) & 0xF, 4)]
        for j in range(i + 2, i + 16):
            b = data[j]
            for nib in (b & 0xF, b >> 4):
                s = (nib << 12) & 0xFFFF
                if s & 0x8000:
                    s -= 0x10000
                s = (s >> shift) + ((s1 * f0 + s2 * f1 + 32) >> 6)
                if s > 32767:
                    s = 32767
                elif s < -32768:
                    s = -32768
                out.append(s)
                s2, s1 = s1, s
        if stop_at_end and flags & 1:
            break
    return out


class Vag:
    def __init__(self, name, rate, data, size):
        self.name = name
        self.rate = rate
        self.data = data
        self.size = size


def parse_vag(buf, off=0):
    if bytes(buf[off:off + 4]) != b"VAGp" or off + 0x30 > len(buf):
        return None
    _version, _, size, rate = struct.unpack_from(">IIII", buf, off + 4)
    if not 4000 <= rate <= 48000 or size == 0 or off + 0x30 + size > len(buf) + 16:
        return None
    name = bytes(buf[off + 0x20:off + 0x30]).split(b"\x00")[0].decode("latin-1")
    data = bytes(buf[off + 0x30:off + 0x30 + size])
    return Vag(name, rate, data, 0x30 + size)


class Vab:
    """A VAB bank (header 'pBAV'). bodies is None for a header-only .VH."""

    def __init__(self, programs, tones, sizes, header_size, bodies, size):
        self.programs = programs
        self.tones = tones
        self.sizes = sizes
        self.header_size = header_size
        self.bodies = bodies
        self.size = size

    def attach_bodies(self, vb):
        """Split a separate .VB body file using the header's size table."""
        self.bodies = _split(vb, 0, self.sizes)


def _split(buf, p, sizes):
    out = []
    for s in sizes:
        out.append(bytes(buf[p:p + s]))
        p += s
    return out


def parse_vab(buf, off=0):
    if bytes(buf[off:off + 4]) != b"pBAV" or off + 0x20 > len(buf):
        return None
    _version, _vab_id, fsize = struct.unpack_from("<III", buf, off + 4)
    ps, ts, vs = struct.unpack_from("<HHH", buf, off + 0x12)
    if not 0 < ps <= 128 or ts > ps * 16 or not 0 < vs <= 254:
        return None
    header_size = 0x20 + 0x800 + ps * 0x200 + 0x200
    table = off + header_size - 0x200
    if table + 0x200 > len(buf):
        return None
    sizes = [v * 8 for v in struct.unpack_from("<%dH" % vs, buf, table + 2)]
    body_start = off + header_size
    if body_start + sum(sizes) <= len(buf) and fsize >= header_size + sum(sizes) - 16:
        bodies = _split(buf, body_start, sizes)
        size = header_size + sum(sizes)
    else:
        bodies = None
        size = header_size
    return Vab(ps, ts, sizes, header_size, bodies, size)


def seq_to_midi(buf, off=0):
    """Convert a Sony SEQ ('pQES') to a Standard MIDI File. Returns
    (midi_bytes, consumed) or None.

    SEQ is an SMF type-0 track body with a short header:
      'pQES', u32 version, u16 ticks/quarter, u24 tempo, u16 rhythm
    """
    if bytes(buf[off:off + 4]) != b"pQES" or off + 15 > len(buf):
        return None
    resolution = struct.unpack_from(">H", buf, off + 8)[0]
    tempo = bytes(buf[off + 10:off + 13])
    if resolution == 0:
        return None
    start = off + 15
    end = bytes(buf[start:start + 4 * 1024 * 1024]).find(b"\xff\x2f\x00")
    if end < 0:
        return None
    events = bytes(buf[start:start + end + 3])
    track = b"\x00\xff\x51\x03" + tempo + events
    midi = (b"MThd" + struct.pack(">IHHH", 6, 0, 1, resolution) +
            b"MTrk" + struct.pack(">I", len(track)) + track)
    return midi, 15 + end + 3
