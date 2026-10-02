"""Tiny numpy software rasteriser for turntable-style preview PNGs of models."""

import math

import numpy as np

from .wav import write_png


def render(tris, textures, size=256, yaw=0.6, pitch=0.45):
    """tris: list of (xyz[3], uv[3] in 0..1, texture key or None).
    textures: key -> (H, W, 4) uint8 array. Returns RGBA bytes."""
    if not tris:
        return bytes(size * size * 4)
    pts = np.array([t[0] for t in tris], dtype=np.float64).reshape(-1, 3)
    centre = (pts.max(0) + pts.min(0)) / 2
    radius = max(np.linalg.norm(pts - centre, axis=1).max(), 1e-6)
    cy, sy, cp, sp = math.cos(yaw), math.sin(yaw), math.cos(pitch), math.sin(pitch)
    rot = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]]) @ \
        np.array([[1, 0, 0], [0, cp, -sp], [0, sp, cp]])
    img = np.zeros((size, size, 4), np.uint8)
    img[..., :3] = 40
    img[..., 3] = 255
    zbuf = np.full((size, size), np.inf)
    scale = size * 0.45 / radius
    light = np.array([0.3, 0.8, -0.5])
    light /= np.linalg.norm(light)
    for xyz, uv, key in tris:
        p = (np.array(xyz, dtype=np.float64) - centre) @ rot.T
        sx = p[:, 0] * scale + size / 2
        sy2 = -p[:, 1] * scale + size / 2
        z = p[:, 2]
        n = np.cross(p[1] - p[0], p[2] - p[0])
        nl = np.linalg.norm(n)
        shade = 0.55 + 0.45 * abs(n @ light) / nl if nl else 1.0
        x0, x1 = int(max(0, math.floor(sx.min()))), int(min(size - 1, math.ceil(sx.max())))
        y0, y1 = int(max(0, math.floor(sy2.min()))), int(min(size - 1, math.ceil(sy2.max())))
        if x0 > x1 or y0 > y1:
            continue
        gx, gy = np.meshgrid(np.arange(x0, x1 + 1) + 0.5, np.arange(y0, y1 + 1) + 0.5)
        d = (sy2[1] - sy2[2]) * (sx[0] - sx[2]) + (sx[2] - sx[1]) * (sy2[0] - sy2[2])
        if abs(d) < 1e-9:
            continue
        w0 = ((sy2[1] - sy2[2]) * (gx - sx[2]) + (sx[2] - sx[1]) * (gy - sy2[2])) / d
        w1 = ((sy2[2] - sy2[0]) * (gx - sx[2]) + (sx[0] - sx[2]) * (gy - sy2[2])) / d
        w2 = 1 - w0 - w1
        inside = (w0 >= 0) & (w1 >= 0) & (w2 >= 0)
        if not inside.any():
            continue
        zz = w0 * z[0] + w1 * z[1] + w2 * z[2]
        sub = zbuf[y0:y1 + 1, x0:x1 + 1]
        mask = inside & (zz < sub)
        if not mask.any():
            continue
        tex = textures.get(key) if key is not None else None
        if tex is not None:
            uv = np.array(uv)
            u = w0 * uv[0, 0] + w1 * uv[1, 0] + w2 * uv[2, 0]
            v = w0 * uv[0, 1] + w1 * uv[1, 1] + w2 * uv[2, 1]
            th, tw = tex.shape[:2]
            tx = np.clip((u * tw).astype(int), 0, tw - 1)
            ty = np.clip(((1 - v) * th).astype(int), 0, th - 1)
            col = tex[ty, tx]
            mask &= col[..., 3] > 0
            rgb = col[..., :3].astype(np.float64)
        else:
            rgb = np.full(gx.shape + (3,), 170.0)
        rgb = np.clip(rgb * shade, 0, 255).astype(np.uint8)
        region = img[y0:y1 + 1, x0:x1 + 1]
        region[..., :3][mask] = rgb[mask]
        sub[mask] = zz[mask]
    return img.tobytes()


def load_obj_tris(obj_path, load_png):
    """Parse an OBJ written by amww into render() triangles."""
    import os
    base = os.path.dirname(obj_path)
    v, vt, tris, textures = [], [], [], {}
    mats, cur = {}, None
    for line in open(obj_path):
        parts = line.split()
        if not parts:
            continue
        if parts[0] == "mtllib":
            m = None
            for ml in open(os.path.join(base, parts[1])):
                mp = ml.split()
                if mp and mp[0] == "newmtl":
                    m = mp[1]
                elif mp and mp[0] == "map_Kd" and m:
                    mats[m] = mp[1]
        elif parts[0] == "v":
            v.append([float(a) for a in parts[1:4]])
        elif parts[0] == "vt":
            vt.append([float(a) for a in parts[1:3]])
        elif parts[0] == "usemtl":
            cur = mats.get(parts[1])
            if cur and cur not in textures:
                textures[cur] = load_png(os.path.join(base, cur))
        elif parts[0] == "f":
            idx = [[int(i) - 1 if i else None for i in (c.split("/") + [""])[:2]] for c in parts[1:]]
            for k in range(1, len(idx) - 1):
                tri = [idx[0], idx[k], idx[k + 1]]
                xyz = [v[a] for a, _ in tri]
                uvs = [vt[b] if b is not None else (0, 0) for _, b in tri]
                tris.append((xyz, uvs, cur))
    return tris, textures


def save_preview(path, rgba, size):
    write_png(path, size, size, rgba)
