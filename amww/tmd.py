"""Sony TMD 3D models -> Wavefront OBJ/MTL.

TMD layout:
  u32 id (0x41), u32 flags (bit 0 FIXP), u32 object count
  object table: per object u32 vert_top, n_vert, normal_top, n_normal,
                prim_top, n_prim, i32 scale (offsets relative to the table)
  vertices / normals: i16 x, y, z, pad
  primitives: u8 olen, u8 ilen, u8 flag, u8 mode, then ilen words of data

Polygon packet data is [UV block if textured] [colours] [normal/vertex
indices], with the exact contents decided by the mode/flag bits.
"""

import os
import struct

MAX_OBJECTS = 4096
MAX_VERTS = 65535


class Poly:
    __slots__ = ("verts", "uvs", "colors", "tpage", "clut", "semi")

    def __init__(self, verts, uvs, colors, tpage, clut, semi):
        self.verts = verts
        self.uvs = uvs
        self.colors = colors
        self.tpage = tpage
        self.clut = clut
        self.semi = semi


class TmdObject:
    def __init__(self, vertices, polys, scale, skipped):
        self.vertices = vertices
        self.polys = polys
        self.scale = scale
        self.skipped = skipped  # lines/sprites/undecodable packets


class Tmd:
    def __init__(self, objects, size):
        self.objects = objects
        self.size = size

    @property
    def poly_count(self):
        return sum(len(o.polys) for o in self.objects)

    @property
    def textured(self):
        return {(p.tpage, p.clut) for o in self.objects for p in o.polys
                if p.tpage is not None}


def _decode_poly(mode, flag, data, nv):
    n = 4 if mode & 0x08 else 3
    textured = bool(mode & 0x04)
    gouraud = bool(mode & 0x10)
    unlit = bool(flag & 1)
    gradation = bool(flag & 4)
    p = 0
    uvs = tpage = clut = None
    if textured:
        if len(data) < 4 * n:
            return None
        uvs = [(data[i * 4], data[i * 4 + 1]) for i in range(n)]
        clut = struct.unpack_from("<H", data, 2)[0]
        tpage = struct.unpack_from("<H", data, 6)[0]
        p = 4 * n
    if unlit:
        ncol = n if (gouraud or gradation) else 1
    elif textured:
        ncol = 0
    else:
        ncol = n if gradation else 1
    colors = []
    for i in range(ncol):
        if p + 4 > len(data):
            return None
        colors.append(tuple(data[p:p + 3]))
        p += 4
    if unlit:
        nidx = n
    elif gouraud:
        nidx = 2 * n
    else:
        nidx = 1 + n
    if p + ((nidx * 2 + 3) // 4) * 4 != len(data):
        return None
    idx = struct.unpack_from("<%dH" % nidx, data, p)
    if unlit:
        verts = list(idx)
    elif gouraud:
        verts = list(idx[1::2])
    else:
        verts = list(idx[1:])
    if any(v >= nv for v in verts):
        return None
    if ncol == 1:
        colors = colors * n
    elif ncol == 0:
        colors = None
    return Poly(verts, uvs, colors, tpage, clut, bool(mode & 0x02))


def parse_tmd(buf, off=0, strict=True):
    """Parse a TMD at buf[off:]. Returns Tmd or None.

    strict=True (used when signature-scanning inside unknown files) rejects
    the candidate if any primitive fails to decode.
    """
    end = len(buf)
    if off + 12 > end:
        return None
    magic, flags, nobj = struct.unpack_from("<III", buf, off)
    if magic != 0x41 or flags not in (0, 1) or not 1 <= nobj <= MAX_OBJECTS:
        return None
    if flags & 1:
        return None  # FIXP: offsets are RAM addresses, not file offsets
    base = off + 12
    if base + nobj * 28 > end:
        return None
    objects = []
    max_end = base + nobj * 28
    total_polys = 0
    for i in range(nobj):
        vt, nv, nt, nn, pt, npr, scale = struct.unpack_from("<IIIIIIi", buf, base + i * 28)
        if not 0 < nv <= MAX_VERTS or npr > 200000 or nn > MAX_VERTS:
            return None
        if not -16 <= scale <= 16:
            return None
        vstart = base + vt
        if vt % 4 or vstart + nv * 8 > end or base + nt + nn * 8 > end:
            return None
        verts = [struct.unpack_from("<hhh", buf, vstart + k * 8) for k in range(nv)]
        max_end = max(max_end, vstart + nv * 8, base + nt + nn * 8)
        p = base + pt
        if pt % 4:
            return None
        polys = []
        skipped = 0
        for _ in range(npr):
            if p + 4 > end:
                return None
            olen, ilen, pflag, mode = buf[p], buf[p + 1], buf[p + 2], buf[p + 3]
            data = bytes(buf[p + 4:p + 4 + ilen * 4])
            if len(data) != ilen * 4 or ilen == 0:
                return None
            p += 4 + ilen * 4
            code = mode >> 5
            if code == 1:
                poly = _decode_poly(mode, pflag, data, nv)
                if poly is None:
                    if strict:
                        return None
                    skipped += 1
                else:
                    polys.append(poly)
            elif code in (2, 3):
                skipped += 1
            elif strict:
                return None
            else:
                skipped += 1
        max_end = max(max_end, p)
        total_polys += len(polys)
        objects.append(TmdObject(verts, polys, scale, skipped))
    if strict and total_polys == 0:
        return None
    return Tmd(objects, max_end - off)


def export_obj(tmd, path, texture_for=None):
    """Write tmd to path (.obj) plus a .mtl.

    texture_for(tpage, clut) -> relative PNG path or None. Vertices are
    emitted per-corner so each face keeps its own UVs and vertex colours.
    PS1 space is X right, Y down, Z forward; output is Y up (x, -y, -z).
    """
    stem = os.path.splitext(os.path.basename(path))[0]
    mtl_name = stem + ".mtl"
    materials = {}
    lines = ["# Exported by amww from a PlayStation TMD", "mtllib " + mtl_name]
    vi = ti = 0
    for oi, obj in enumerate(tmd.objects):
        lines.append("o object_%d" % oi)
        s = 2.0 ** obj.scale
        current = None
        for poly in obj.polys:
            if poly.tpage is not None:
                mat = "tex_%04x_%04x" % (poly.tpage, poly.clut)
                if mat not in materials:
                    materials[mat] = texture_for(poly.tpage, poly.clut) if texture_for else None
            else:
                mat = "untextured"
                materials.setdefault(mat, None)
            if mat != current:
                lines.append("usemtl " + mat)
                current = mat
            # PS1 quads are ordered 0,1,2,3 = TL,TR,BL,BR -> polygon 0,1,3,2
            order = [0, 1, 3, 2] if len(poly.verts) == 4 else [0, 1, 2]
            face = []
            for k in order:
                x, y, z = obj.vertices[poly.verts[k]]
                if poly.colors:
                    r, g, b = poly.colors[k]
                    lines.append("v %g %g %g %.4f %.4f %.4f" % (
                        x * s, -y * s, -z * s, r / 255, g / 255, b / 255))
                else:
                    lines.append("v %g %g %g" % (x * s, -y * s, -z * s))
                vi += 1
                if poly.uvs:
                    u, v = poly.uvs[k]
                    lines.append("vt %.6f %.6f" % ((u + 0.5) / 256, 1 - (v + 0.5) / 256))
                    ti += 1
                    face.append("%d/%d" % (vi, ti))
                else:
                    face.append("%d" % vi)
            lines.append("f " + " ".join(face))
    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    mtl = []
    for name, tex in materials.items():
        mtl += ["newmtl " + name, "Kd 1 1 1", "Ka 0 0 0", "illum 1"]
        if tex:
            mtl += ["map_Kd " + tex, "map_d " + tex]
        mtl.append("")
    with open(os.path.join(os.path.dirname(path), mtl_name), "w") as fh:
        fh.write("\n".join(mtl))
