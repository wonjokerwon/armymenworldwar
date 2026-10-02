"""3DO '3MDL' models (Army Men: World War - Team Assault, PS1).

Reverse-engineered layout:

  char[4] '3MDL'
  u32     version (1)
  u32     vertex count
  u32     polygon count
  u32     unknown (0x1DF/0x1E0)
  u32     size of the model in bytes (header + vertices + command stream)
  vertex[count]: i16 x, y, z, u16 flags (0x8000 marks the last vertex)
  command stream of u32 words, terminated by opcode 0x68

Each command word is a register write: the low byte is the register address
(low 2 bits are flags), the upper 24 bits are the value. Any write whose
flag bits are non-zero emits a polygon from the current register state.

  0x00-0x0C  vertex colours (r, g, b)
  0x20-0x2C  texture coordinates (u, v)
  0x30-0x3C  single vertex index (low 12 bits)
  0x40-0x4C  two vertex indices (low 12 bits, high 12 bits)
  0x58       texture slot select (top byte of the value)
  0x5C       render mode (top byte): 0x80 = textured, 0x10 = quads
             (else triangles)
  0x68       end of model

The model is followed by one 20-byte record per texture slot:

  u8 u_offset, u8 v_offset   placement of the sub-texture inside the page
  u16 tpage                  PS1 texture page (bit 15 unknown flag)
  u16 clut                   PS1 CLUT position
  14 bytes                   padding (0xF3...) and a zero terminator

The four polygon corners A, B, C, D (quad drawn as the loop A-B-C-D, a
triangle as A-B-D) are written by:

  0x30 -> B     0x40 -> A, B
  0x34 -> A     0x44 -> D, A
  0x38 -> D     0x48 -> C, D
  0x3C -> C     0x4C -> D, B

(Derived by hand from simple shapes and confirmed by maximising shared
edges over every model on the disc.)
"""

import struct

SINGLE = {0x30: 1, 0x34: 0, 0x38: 3, 0x3C: 2}
PAIR = {0x40: (0, 1), 0x44: (3, 0), 0x48: (2, 3), 0x4C: (3, 1)}
# texture-coordinate / colour register -> corner, per primitive type
QUAD_ATTR = {0: 0, 1: 1, 2: 2, 3: 3}
TRI_ATTR = {0: 0, 1: 1, 2: 3}


class TexSlot:
    __slots__ = ("u", "v", "tpage", "clut")

    def __init__(self, u, v, tpage, clut):
        self.u, self.v, self.tpage, self.clut = u, v, tpage, clut


class Mdl3:
    def __init__(self, vertices, polys, size, unknown, slots):
        self.vertices = vertices
        self.polys = polys  # list of (indices, uvs, colors, slot)
        self.size = size
        self.unknown = unknown
        self.slots = slots  # list of TexSlot

    @property
    def poly_count(self):
        return len(self.polys)


def parse_mdl3(buf, off=0):
    end = len(buf)
    if bytes(buf[off:off + 4]) != b"3MDL" or off + 24 > end:
        return None
    version, nv, npoly, unknown, size = struct.unpack_from("<IIIII", buf, off + 4)
    if version != 1 or not 0 < nv < 4096 or npoly > 20000 or size < 24 + nv * 8:
        return None
    if off + size > end:
        return None
    verts = [struct.unpack_from("<hhh", buf, off + 24 + i * 8) for i in range(nv)]

    idx = [0, 0, 0, 0]
    uv = [(0, 0)] * 4
    col = [(128, 128, 128)] * 4
    page = 0
    quads = False
    textured = True
    polys = []
    p = off + 24 + nv * 8
    stop = off + size
    while p + 4 <= stop:
        w = struct.unpack_from("<I", buf, p)[0]
        p += 4
        op, flags, val = w & 0xFC, w & 3, w >> 8
        if op == 0x68:
            break
        if op in PAIR:
            a, b = PAIR[op]
            idx[a], idx[b] = val & 0xFFF, val >> 12
        elif op in SINGLE:
            idx[SINGLE[op]] = val & 0xFFF
        elif op <= 0x0C:
            col[op >> 2] = (val & 0xFF, (val >> 8) & 0xFF, val >> 16)
        elif 0x20 <= op <= 0x2C:
            uv[(op - 0x20) >> 2] = (val & 0xFF, (val >> 8) & 0xFF)
        elif op == 0x58:
            page = val >> 16
        elif op == 0x5C:
            quads = bool((val >> 16) & 0x10)
            textured = bool((val >> 16) & 0x80)
        if flags:
            if quads:
                corners = list(idx)
                attrs = [QUAD_ATTR[k] for k in range(4)]
            else:
                corners = [idx[0], idx[1], idx[3]]
                attrs = [TRI_ATTR[k] for k in range(3)]
            if all(c < nv for c in corners):
                polys.append((corners, [uv[a] for a in attrs],
                              [col[a] for a in attrs], page if textured else -1))
    nslots = max([pg for _, _, _, pg in polys] + [-1]) + 1
    slots = []
    for i in range(nslots):
        q = off + size + i * 20
        if q + 20 > end:
            break
        u, v, tpage, clut = struct.unpack_from("<BBHH", buf, q)
        slots.append(TexSlot(u, v, tpage & 0x7FFF, clut))
    return Mdl3(verts, polys, size, unknown, slots)


def export_obj(model, path, texture_for=None, name="model"):
    """Write an OBJ + MTL. texture_for(tpage, clut) -> PNG filename or None.

    UVs are page-relative (slot offset + polygon UV) so each material's
    texture is the whole 256x256 texture page decoded with the slot's CLUT.
    """
    import os
    stem = os.path.splitext(os.path.basename(path))[0]
    lines = ["# Army Men: World War - Team Assault 3MDL model, exported by amww",
             "mtllib %s.mtl" % stem, "o " + name]
    # PS1 space: Y down. Flip to Y up (x, -y, -z keeps handedness).
    for x, y, z in model.vertices:
        lines.append("v %d %d %d" % (x, -y, -z))
    mats = {}
    current = None
    vt = 0
    for corners, uvs, cols, slot in model.polys:
        s = model.slots[slot] if 0 <= slot < len(model.slots) else None
        if s and texture_for:
            mat = "slot%d" % slot
        else:  # flat colour from the polygon's first vertex colour
            mat = "col_%02x%02x%02x" % cols[0]
        if mat not in mats:
            mats[mat] = texture_for(s.tpage, s.clut) if (s and texture_for) else cols[0]
        if mat != current:
            lines.append("usemtl " + mat)
            current = mat
        face = []
        for c, (u, v) in zip(corners, uvs):
            if s:
                u, v = (u + s.u) & 0xFF, (v + s.v) & 0xFF
            lines.append("vt %.6f %.6f" % ((u + 0.5) / 256, 1 - (v + 0.5) / 256))
            vt += 1
            face.append("%d/%d" % (c + 1, vt))
        lines.append("f " + " ".join(face))
    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    mtl = []
    for mat, tex in mats.items():
        if isinstance(tex, tuple):
            mtl += ["newmtl " + mat, "Kd %.3f %.3f %.3f" % tuple(c / 255 for c in tex), "illum 1"]
        else:
            mtl += ["newmtl " + mat, "Kd 1 1 1", "illum 1"]
            if tex:
                mtl += ["map_Kd " + tex, "map_d " + tex]
        mtl.append("")
    with open(os.path.join(os.path.dirname(path), stem + ".mtl"), "w") as fh:
        fh.write("\n".join(mtl))
