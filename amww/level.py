"""Level layout for Army Men: World War - Team Assault (.DAT files).

A level .DAT starts with an 11-u32 header (the 3rd value is the number of
named object types, the last is the byte size of the name table), the name
table at 0x2C (52-byte entries: char name[8] + 11 u32, the 6th being the
byte size of the object's 3MDL model), then a chain of sections, each
  u32 id, u32 size, data
ending with id 18 (the models). Known sections:

  8   object placements, 22-byte records of i16:
        x, y, z            position: x/z in 1/64 of a terrain cell, y (down)
                           in model units (16x finer)
        x, y, z            repeated (start position)
        flags
        (scale & 0xFF00) | angle   angle in 1/256 turns about the vertical
                           axis (checked against walls lining a trench)
        type               low byte: index into the name table
        0
        scale              4096 = 1.0
      Model vertices are in units of 1/16 world unit.
  10  terrain: 256 x 128 cells of 4 bytes, row-major (row = z / 64,
      column = x / 64), followed by a 5140-byte table:
        byte 0  height; y = byte * 64 (model units, Y down)
        byte 1  ground tile (48x48 4bpp texture, 25 per texture page in
                pages 13, 14, 15, 29, 30, 31; palette row 12 + tile at x=1008)
        byte 2  flags (top two bits vary per cell; meaning not confirmed)
        byte 3  appears to be a per-cell shade value
  15  256 x 128 u16 collision / occupancy map (0x8000 = blocked)
  17  named path nodes ("PD00"...)
"""

import math
import struct

GRID_W, GRID_H, CELL = 256, 128, 64
MODEL_SCALE = 1 / 16  # model vertex units are 1/16 of a world unit
TILE = 48
TILE_PAGES = (13, 14, 15, 29, 30, 31)
TILE_CLUT_BASE = 12


class Placement:
    __slots__ = ("type", "name", "x", "y", "z", "angle", "scale", "flags")


class Level:
    def __init__(self):
        self.names = []        # (name, model byte size)
        self.sections = {}     # id -> (offset, size)
        self.placements = []
        self.cells = None      # bytes, 256*128*4


def parse_level(buf):
    if len(buf) < 0x2C:
        return None
    hdr = struct.unpack_from("<11I", buf, 0)
    count, table = hdr[2], hdr[10]
    if not (0 < count < 1000 and table == count * 52):
        return None
    lv = Level()
    for i in range(count):
        o = 0x2C + i * 52
        name = buf[o:o + 8].split(b"\0")[0].decode("latin-1")
        lv.names.append((name, struct.unpack_from("<I", buf, o + 28)[0]))
    p = 0x2C + table
    while p + 8 <= len(buf):
        sid, size = struct.unpack_from("<II", buf, p)
        lv.sections[sid] = (p + 8, size)
        if size == 0 or sid >= 18:
            break
        p += 8 + size
    if 8 in lv.sections:
        o, size = lv.sections[8]
        for k in range(size // 22):
            r = struct.unpack_from("<11h", buf, o + k * 22)
            if not any(r[:3]):
                continue
            pl = Placement()
            pl.x, pl.y, pl.z = r[0], r[1], r[2]
            pl.flags = r[6] & 0xFFFF
            pl.angle = (r[7] & 0xFF) / 256.0 * 2 * math.pi
            pl.type = r[8] & 0xFF
            pl.scale = (r[10] or 4096) / 4096.0
            pl.name = lv.names[pl.type][0] if pl.type < len(lv.names) else None
            lv.placements.append(pl)
    if 10 in lv.sections:
        o, size = lv.sections[10]
        if size >= GRID_W * GRID_H * 4:
            lv.cells = bytes(buf[o:o + GRID_W * GRID_H * 4])
    return lv


def cell(lv, i, j):
    k = (i * GRID_W + j) * 4
    return lv.cells[k], lv.cells[k + 1], lv.cells[k + 2], lv.cells[k + 3]


def tile_rgba(vram, t):
    """Decode ground tile t to 48x48 RGBA bytes (None if out of range)."""
    if not 0 <= t < len(TILE_PAGES) * 25:
        return None
    from .tim import _rgba_lut
    lut = _rgba_lut()
    tpage = TILE_PAGES[t // 25]
    tx = (tpage & 15) * 64 + (t % 25 % 5) * (TILE // 4)
    ty = ((tpage >> 4) & 1) * 256 + (t % 25 // 5) * TILE
    clut_y = TILE_CLUT_BASE + t
    W = vram.W
    pal = [vram.mem[clut_y * W + 1008 + k] for k in range(16)]
    out = []
    for y in range(TILE):
        row = (ty + y) * W
        for x in range(TILE):
            hw = vram.mem[row + tx + x // 4]
            c = pal[(hw >> ((x & 3) * 4)) & 15]
            out.append(lut[c * 4:c * 4 + 4])
    return b"".join(out)


def model_index(lv, models):
    """Map name-table index -> model offset (models in file order, matched
    to the table by byte size)."""
    out = {}
    j = 0
    for off, m in models:
        if j < len(lv.names) and lv.names[j][1] == m.size:
            out[j] = off
            j += 1
    return out


def _png_bytes(w, h, rgba):
    import io
    import zlib
    stride = w * 4
    raw = b"".join(b"\0" + rgba[y * stride:(y + 1) * stride] for y in range(h))

    def chunk(tag, body):
        return (struct.pack(">I", len(body)) + tag + body +
                struct.pack(">I", zlib.crc32(tag + body) & 0xffffffff))
    bio = io.BytesIO()
    bio.write(b"\x89PNG\r\n\x1a\n")
    bio.write(chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 6, 0, 0, 0)))
    bio.write(chunk(b"IDAT", zlib.compress(raw, 9)))
    bio.write(chunk(b"IEND", b""))
    return bio.getvalue()


def _conv(x, y, z):
    """Level space -> glTF. X/Z positions count 1/64 of a terrain cell while
    heights (placement Y, terrain bytes * 64) are in the 16x finer model
    units, so Y is divided by 16 to keep the scene in proportion. PS1 Y is
    down; glTF Y is up."""
    return (x, -y * MODEL_SCALE, -z)


def terrain_atlas(lv, vram):
    """Atlas of every ground tile the level uses: 16 tiles per row."""
    used = sorted({lv.cells[k + 1] for k in range(0, len(lv.cells), 4)})
    used = [t for t in used if t < len(TILE_PAGES) * 25]
    cols = 16
    rows = (len(used) + cols - 1) // cols
    w, h = cols * TILE, max(1, rows) * TILE
    atlas = bytearray(w * h * 4)
    where = {}
    for n, t in enumerate(used):
        px = tile_rgba(vram, t)
        cx, cy = (n % cols) * TILE, (n // cols) * TILE
        for y in range(TILE):
            o = ((cy + y) * w + cx) * 4
            atlas[o:o + TILE * 4] = px[y * TILE * 4:(y + 1) * TILE * 4]
        where[t] = (cx, cy)
    return w, h, bytes(atlas), where


def terrain_mesh(lv, where, aw, ah):
    """Quads between neighbouring cell centres, textured with the tile of
    the cell at their corner. Returns positions, uvs, triangles."""
    pos, uvs, tris = [], [], []
    hs = [lv.cells[k] * CELL for k in range(0, len(lv.cells), 4)]
    inset = 0.5
    for i in range(GRID_H - 1):
        for j in range(GRID_W - 1):
            _, t, _, _ = cell(lv, i, j)
            if t not in where:
                continue
            cx, cy = where[t]
            base = len(pos)
            for di, dj in ((0, 0), (0, 1), (1, 1), (1, 0)):
                ii, jj = i + di, j + dj
                pos.append(_conv(jj * CELL, hs[ii * GRID_W + jj], ii * CELL))
                u = cx + (inset if dj == 0 else TILE - inset)
                v = cy + (inset if di == 0 else TILE - inset)
                uvs.append((u / aw, v / ah))
            tris.append((base, base + 2, base + 1))
            tris.append((base, base + 3, base + 2))
    return pos, uvs, tris


def model_mesh(m, texture_png, mats, mat_index):
    """3MDL -> (positions, uvs, {material: triangles})."""
    pos, uvs, tris = [], [], {}
    for corners, puv, cols, slot in m.polys:
        s = m.slots[slot] if 0 <= slot < len(m.slots) else None
        key = (s.tpage, s.clut) if s else ("col",) + tuple(cols[0])
        if key not in mat_index:
            if s:
                mats.append(("tex_%04x_%04x" % key, texture_png(s.tpage, s.clut), (1, 1, 1, 1)))
            else:
                mats.append(("col_%02x%02x%02x" % tuple(cols[0]), None,
                             tuple(c / 255 for c in cols[0]) + (1,)))
            mat_index[key] = len(mats) - 1
        mi = mat_index[key]
        base = len(pos)
        for c, (u, v) in zip(corners, puv):
            x, y, z = m.vertices[c]
            pos.append((x, -y, -z))  # model units; the node scales by 1/16
            if s:
                u, v = (u + s.u) & 0xFF, (v + s.v) & 0xFF
            uvs.append(((u + 0.5) / 256, (v + 0.5) / 256))
        for k in range(1, len(corners) - 1):
            tris.setdefault(mi, []).append((base, base + k, base + k + 1))
    return pos, uvs, tris


def export_level(path, lv, vram, models, name="level"):
    """Write a .glb scene: terrain mesh + one node per placed object."""
    from .gltf import write_scene_glb
    mats, mat_index = [], {}
    page_png = {}

    def texture_png(tpage, clut):
        k = (tpage, clut)
        if k not in page_png:
            page_png[k] = _png_bytes(256, 256, vram.texture(tpage, clut))
        return page_png[k]

    meshes, nodes = [], []
    if lv.cells:
        aw, ah, atlas, where = terrain_atlas(lv, vram)
        mats.append(("terrain", _png_bytes(aw, ah, atlas), (1, 1, 1, 1)))
        pos, uvs, tris = terrain_mesh(lv, where, aw, ah)
        meshes.append(("terrain", pos, uvs, {len(mats) - 1: tris}))
        nodes.append(("terrain", 0, (0, 0, 0), 0.0, 1.0))
    idx = model_index(lv, models)
    by_off = dict(models)
    mesh_of = {}
    for pl in lv.placements:
        off = idx.get(pl.type)
        if off is None:
            continue
        if off not in mesh_of:
            pos, uvs, tris = model_mesh(by_off[off], texture_png, mats, mat_index)
            meshes.append((lv.names[pl.type][0], pos, uvs, tris))
            mesh_of[off] = len(meshes) - 1
        nodes.append((pl.name, mesh_of[off], _conv(pl.x, pl.y, pl.z), pl.angle,
                      pl.scale * MODEL_SCALE))
    write_scene_glb(path, meshes, mats, nodes)
    return len(nodes) - (1 if lv.cells else 0)


def heightmap_png(lv):
    """8-bit greyscale height image (white = high), 256 x 128."""
    hs = [lv.cells[k] for k in range(0, len(lv.cells), 4)]
    lo, hi = min(hs), max(hs)
    rng = max(1, hi - lo)
    rgba = b"".join(bytes((v, v, v, 255)) for v in
                    (255 - (h - lo) * 255 // rng for h in hs))
    return _png_bytes(GRID_W, GRID_H, rgba)


def tilemap_png(lv, vram, px=8):
    """Top-down picture of the ground: px pixels per cell."""
    step = TILE // px
    tiles = {}
    w, h = GRID_W * px, GRID_H * px
    out = bytearray(w * h * 4)
    for i in range(GRID_H):
        for j in range(GRID_W):
            t = cell(lv, i, j)[1]
            if t not in tiles:
                full = tile_rgba(vram, t)
                tiles[t] = None if full is None else [
                    b"".join(full[((y * step) * TILE + x * step) * 4:((y * step) * TILE + x * step) * 4 + 4]
                             for x in range(px)) for y in range(px)]
            rows = tiles[t]
            if rows is None:
                continue
            for y in range(px):
                o = ((i * px + y) * w + j * px) * 4
                out[o:o + px * 4] = rows[y]
    return _png_bytes(w, h, bytes(out))


def placements_csv(lv):
    lines = ["name,type,x,y,z,yaw_degrees,scale,flags"]
    for pl in lv.placements:
        lines.append("%s,%d,%d,%d,%d,%.2f,%.4f,0x%04x" % (
            pl.name or "", pl.type, pl.x, pl.y, pl.z, math.degrees(pl.angle), pl.scale, pl.flags))
    return "\n".join(lines) + "\n"
