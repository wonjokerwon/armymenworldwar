"""Tiny WAV and PNG writers so the toolkit has no third-party dependencies."""

import struct
import zlib


def write_wav_header(fh, data_bytes, rate, channels, bits=16):
    block = channels * bits // 8
    fh.write(b"RIFF" + struct.pack("<I", 36 + data_bytes) + b"WAVE")
    fh.write(b"fmt " + struct.pack("<IHHIIHH", 16, 1, channels, rate,
                                   rate * block, block, bits))
    fh.write(b"data" + struct.pack("<I", data_bytes))


def write_wav(path, samples, rate, channels=1):
    """samples: array('h') of interleaved little-endian 16-bit PCM."""
    data = samples.tobytes()
    if struct.pack("=H", 1) != struct.pack("<H", 1):
        s = type(samples)(samples.typecode, samples)
        s.byteswap()
        data = s.tobytes()
    with open(path, "wb") as fh:
        write_wav_header(fh, len(data), rate, channels)
        fh.write(data)


def write_png(path, width, height, rgba):
    stride = width * 4
    raw = b"".join(b"\x00" + rgba[y * stride:(y + 1) * stride]
                   for y in range(height))

    def chunk(tag, body):
        return (struct.pack(">I", len(body)) + tag + body +
                struct.pack(">I", zlib.crc32(tag + body) & 0xffffffff))

    with open(path, "wb") as fh:
        fh.write(b"\x89PNG\r\n\x1a\n")
        fh.write(chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 6, 0, 0, 0)))
        fh.write(chunk(b"IDAT", zlib.compress(raw, 9)))
        fh.write(chunk(b"IEND", b""))
