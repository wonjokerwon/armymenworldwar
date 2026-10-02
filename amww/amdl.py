"""3DO 'AMDL' skinned character models (the soldiers in Army Men: World War -
Team Assault). Found inside the level .DAT files, 6-8 per level.

Each character block:

  u32 skeleton_end   bytes from the AMDL magic to the end of the skeleton
  u32 slot_count     texture slots per colour variant
  'AMDL' model       same header and command stream as 3MDL (see mdl3.py),
                     except the version field holds the bone count and each
                     vertex's 4th u16 is the bone it is attached to; the
                     stream is split into named part groups (opcode 0x70)
  12 zero bytes
  u32 bone_count
  u8 parent[bone_count]          (0xFF = root), padded to 4 bytes
  bone_count x { i16 rotation[3][3] (4096 = 1.0), u16 pad, i32 translation[3] }
  slot_count x 20-byte texture slots (as in 3MDL), repeated once per colour
  variant; the variants differ only in CLUT (team colours)

Vertices are stored in their bone's local space; the bind pose is
world = parent_world * (R * v + T). The root bone's rotation only orients the
character in the game world and is dropped so the model stands upright.
"""

import struct

from .mdl3 import TexSlot, parse_mdl3


class Amdl:
    def __init__(self, model, bones, parents, locals_, variants, size):
        self.model = model          # Mdl3 (vertices in bone space)
        self.bones = bones          # per-vertex bone index
        self.parents = parents      # per-bone parent (-1 = root)
        self.locals = locals_       # per-bone (3x3 rows, translation)
        self.variants = variants    # list of slot lists (colour variants)
        self.size = size

    def world(self):
        """Bone world transforms (rows of a 3x3 matrix, translation)."""
        out = []
        for i, (r, t) in enumerate(self.locals):
            p = self.parents[i]
            if p < 0 or p >= i:
                r = ((1, 0, 0), (0, 1, 0), (0, 0, 1))  # drop root orientation
                out.append((r, t))
                continue
            pr, pt = out[p]
            out.append((_mul(pr, r), _add(_apply(pr, t), pt)))
        return out

    def posed_vertices(self):
        w = self.world()
        verts = []
        for (x, y, z), b in zip(self.model.vertices, self.bones):
            r, t = w[b] if b < len(w) else w[0]
            verts.append(_add(_apply(r, (x, y, z)), t))
        return verts


def _mul(a, b):
    return tuple(tuple(sum(a[i][k] * b[k][j] for k in range(3)) for j in range(3))
                 for i in range(3))


def _apply(r, v):
    return tuple(sum(r[i][k] * v[k] for k in range(3)) for i in range(3))


def _add(a, b):
    return tuple(a[i] + b[i] for i in range(3))


def _slot_set(buf, q, n):
    slots = []
    for i in range(n):
        if q + 20 > len(buf):
            return None
        u, v, tpage, clut = struct.unpack_from("<BBHH", buf, q + i * 20)
        if bytes(buf[q + i * 20 + 18:q + i * 20 + 20]) != b"\0\0" or clut & 0x3F != 0x3F:
            return None
        slots.append(TexSlot(u, v, tpage & 0x7FFF, clut))
    return slots


def parse_amdl(buf, off):
    """Parse a character whose 'AMDL' magic is at buf[off]."""
    if bytes(buf[off:off + 4]) != b"AMDL" or off < 8 or off + 24 > len(buf):
        return None
    skel_end, nslots = struct.unpack_from("<II", buf, off - 8)
    nbones_hdr, nv, npoly, unknown, size = struct.unpack_from("<5I", buf, off + 4)
    if not (0 < nv < 4096 and 0 < nbones_hdr < 128 and off + size + 20 <= len(buf)):
        return None
    if not (size < skel_end <= len(buf) - off and nslots < 256):
        # Some single-bone props have no pre-header: derive both values.
        skel_end = size + 16 + (nbones_hdr + 3) // 4 * 4 + 32 * nbones_hdr
        nslots = None
    hdr = bytearray(buf[off:off + size])
    hdr[0:8] = b"3MDL" + struct.pack("<I", 1)
    model = parse_mdl3(bytes(hdr), 0)
    if model is None:
        return None
    bones = [struct.unpack_from("<H", buf, off + 24 + i * 8 + 6)[0] for i in range(nv)]

    q = off + size + 12
    nb = struct.unpack_from("<I", buf, q)[0]
    q += 4
    if nb != nbones_hdr:
        return None
    parents = [-1 if p == 0xFF else p for p in buf[q:q + nb]]
    q += (nb + 3) // 4 * 4
    locals_ = []
    for _ in range(nb):
        m = struct.unpack_from("<9h", buf, q)
        t = struct.unpack_from("<3i", buf, q + 20)
        locals_.append((tuple(tuple(m[r * 3 + c] / 4096 for c in range(3)) for r in range(3)), t))
        q += 32

    if nslots is None:
        nslots = max([pg for _, _, _, pg in model.polys] + [-1]) + 1
    variants = []
    q = off + skel_end
    seen = set()
    while nslots:
        slots = _slot_set(buf, q, nslots)
        if slots is None:
            break
        key = tuple((s.u, s.v, s.tpage, s.clut) for s in slots)
        if key not in seen:
            seen.add(key)
            variants.append(slots)
        q += 20 * nslots
    model.slots = variants[0] if variants else []
    return Amdl(model, bones, parents, locals_, variants, q - off)


# Parts shown on a normal soldier; the rest are alternate weapons and
# damage/dismemberment pieces the game switches on as needed.
DEFAULT_PARTS = {"base", "torso", "arms", "legs", "sarges bk", "torso back", "torso side",
                 "helmet", "face", "m16 side", "m16 top", "m-16 end"}

# Y-down PS1 space -> Y-up (x, -y, -z)
_C = (1, -1, -1)


def _conv_r(r):
    return tuple(tuple(r[i][j] * _C[i] * _C[j] for j in range(3)) for i in range(3))


def _conv_t(t):
    return tuple(t[i] * _C[i] for i in range(3))


def _mat4(r, t):
    """Column-major 4x4 from rows + translation."""
    return (r[0][0], r[1][0], r[2][0], 0, r[0][1], r[1][1], r[2][1], 0,
            r[0][2], r[1][2], r[2][2], 0, t[0], t[1], t[2], 1)


def _inverse(r, t):
    # general 3x3 inverse (bone matrices are near-orthonormal but not exactly)
    a, b, c = r
    det = (a[0] * (b[1] * c[2] - b[2] * c[1]) - a[1] * (b[0] * c[2] - b[2] * c[0])
           + a[2] * (b[0] * c[1] - b[1] * c[0]))
    inv = ((b[1] * c[2] - b[2] * c[1], a[2] * c[1] - a[1] * c[2], a[1] * b[2] - a[2] * b[1]),
           (b[2] * c[0] - b[0] * c[2], a[0] * c[2] - a[2] * c[0], a[2] * b[0] - a[0] * b[2]),
           (b[0] * c[1] - b[1] * c[0], a[1] * c[0] - a[0] * c[1], a[0] * b[1] - a[1] * b[0]))
    inv = tuple(tuple(v / det for v in row) for row in inv)
    it = _apply(inv, t)
    return inv, (-it[0], -it[1], -it[2])


def _part_ranges(model):
    groups = list(model.groups)
    if not groups or groups[0][3] > 0:
        # polygons before the first named group (e.g. a vehicle body)
        groups.insert(0, (0, 0, "base", 0))
    bounds = [g[3] for g in groups] + [len(model.polys)]
    return [(g[2] or "part%d" % i, bounds[i], bounds[i + 1]) for i, g in enumerate(groups)]


def export_amdl(amdl, out_base, slots, texture_png, rigid=False):
    """Write out_base.glb (rigged, all parts as separate meshes),
    out_base.obj (default parts) and out_base_allparts.obj.

    texture_png(tpage, clut) -> (relative png path, png bytes)
    """
    import os
    m = amdl.model
    verts = amdl.posed_vertices()
    parts = _part_ranges(m)
    mats, mat_index = [], {}

    def material(slot):
        s = slots[slot] if 0 <= slot < len(slots) else None
        key = (s.tpage, s.clut) if s else None
        if key not in mat_index:
            if s:
                rel, png = texture_png(s.tpage, s.clut)
                mats.append(("tex_%04x_%04x" % key, png, (1, 1, 1, 1), rel))
            else:
                mats.append(("untextured", None, (0.7, 0.7, 0.7, 1), None))
            mat_index[key] = len(mats) - 1
        return mat_index[key], s

    # --- glTF: per-corner vertices so every face keeps its own UVs
    meshes = []
    for name, a, b in parts:
        pos, uvs, jids, tris = [], [], [], {}
        for corners, puv, cols, slot in m.polys[a:b]:
            mi, s = material(slot)
            base = len(pos)
            for c, (u, v) in zip(corners, puv):
                pos.append(_conv_t(verts[c]))
                if s:
                    u, v = (u + s.u) & 0xFF, (v + s.v) & 0xFF
                uvs.append(((u + 0.5) / 256, (v + 0.5) / 256))
                jids.append(0 if rigid else amdl.bones[c])
            n = len(corners)
            for k in range(1, n - 1):
                tris.setdefault(mi, []).append((base, base + k, base + k + 1))
        meshes.append((name, pos, uvs, jids, tris))
    joints = None
    if not rigid:
        world = amdl.world()
        joints = []
        for i, (r, t) in enumerate(amdl.locals):
            if amdl.parents[i] < 0:
                r = ((1, 0, 0), (0, 1, 0), (0, 0, 1))
            wr, wt = world[i]
            ir, itt = _inverse(_conv_r(wr), _conv_t(wt))
            joints.append(("bone%02d" % i, amdl.parents[i],
                           _mat4(_conv_r(r), _conv_t(t)), _mat4(ir, itt)))
    from .gltf import write_glb
    write_glb(out_base + ".glb", meshes,
              [(n, png, f) for n, png, f, _ in mats], joints)

    # --- OBJ: default soldier, and every part as its own object
    stem = os.path.basename(out_base)
    for suffix, keep in (("", lambda n: rigid or n in DEFAULT_PARTS), ("_allparts", lambda n: True)):
        lines = ["# Army Men: World War - Team Assault AMDL character, exported by amww",
                 "mtllib %s.mtl" % stem]
        for x, y, z in verts:
            lines.append("v %g %g %g" % _conv_t((x, y, z)))
        vt = 0
        for name, a, b in parts:
            if not keep(name):
                continue
            lines.append("o " + name.replace(" ", "_"))
            cur = None
            for corners, puv, cols, slot in m.polys[a:b]:
                mi, s = material(slot)
                if mi != cur:
                    lines.append("usemtl " + mats[mi][0])
                    cur = mi
                face = []
                for c, (u, v) in zip(corners, puv):
                    if s:
                        u, v = (u + s.u) & 0xFF, (v + s.v) & 0xFF
                    lines.append("vt %.6f %.6f" % ((u + 0.5) / 256, 1 - (v + 0.5) / 256))
                    vt += 1
                    face.append("%d/%d" % (c + 1, vt))
                lines.append("f " + " ".join(face))
        with open(out_base + suffix + ".obj", "w") as fh:
            fh.write("\n".join(lines) + "\n")
    mtl = []
    for name, png, f, rel in mats:
        mtl += ["newmtl " + name, "Kd %g %g %g" % f[:3], "illum 1"]
        if rel:
            mtl += ["map_Kd " + rel, "map_d " + rel]
        mtl.append("")
    with open(out_base + ".mtl", "w") as fh:
        fh.write("\n".join(mtl))
