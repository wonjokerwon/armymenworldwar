"""Minimal binary glTF 2.0 (.glb) writer for skinned, textured meshes."""

import json
import struct


def _pad(b, fill=b"\0"):
    return b + fill * (-len(b) % 4)


def write_glb(path, meshes, materials, joints=None):
    """Write a .glb.

    meshes:    list of (name, positions, uvs, joint_ids, triangles_by_material)
               positions: [(x, y, z)], uvs: [(u, v)] (0..1, top-left origin),
               joint_ids: [int] or None, triangles_by_material: {mat: [(a, b, c)]}
    materials: list of (name, png_bytes or None, rgba factor)
    joints:    None or list of (name, parent_index or -1, local 4x4 column-major,
               inverse bind 4x4 column-major)
    """
    gl = {"asset": {"version": "2.0", "generator": "amww"},
          "scene": 0, "scenes": [{"nodes": []}], "nodes": [], "meshes": [],
          "buffers": [], "bufferViews": [], "accessors": [],
          "materials": [], "textures": [], "images": [],
          "samplers": [{"magFilter": 9728, "minFilter": 9728}]}
    blob = bytearray()

    def view(data, target=None):
        blob.extend(b"\0" * (-len(blob) % 4))
        v = {"buffer": 0, "byteOffset": len(blob), "byteLength": len(data)}
        if target:
            v["target"] = target
        blob.extend(data)
        gl["bufferViews"].append(v)
        return len(gl["bufferViews"]) - 1

    def accessor(fmt, comps, ctype, rows, target=None, minmax=False, normalized=False):
        data = b"".join(struct.pack("<" + fmt * comps, *r) for r in rows)
        a = {"bufferView": view(data, target), "componentType": ctype,
             "count": len(rows), "type": {1: "SCALAR", 2: "VEC2", 3: "VEC3",
                                          4: "VEC4", 16: "MAT4"}[comps]}
        if normalized:
            a["normalized"] = True
        if minmax:
            a["min"] = [min(r[i] for r in rows) for i in range(comps)]
            a["max"] = [max(r[i] for r in rows) for i in range(comps)]
        gl["accessors"].append(a)
        return len(gl["accessors"]) - 1

    for name, png, factor in materials:
        m = {"name": name, "doubleSided": True, "alphaMode": "MASK", "alphaCutoff": 0.5,
             "pbrMetallicRoughness": {"baseColorFactor": list(factor),
                                      "metallicFactor": 0.0, "roughnessFactor": 1.0}}
        if png:
            gl["images"].append({"bufferView": view(png), "mimeType": "image/png"})
            gl["textures"].append({"source": len(gl["images"]) - 1, "sampler": 0})
            m["pbrMetallicRoughness"]["baseColorTexture"] = {"index": len(gl["textures"]) - 1}
        gl["materials"].append(m)

    skin = None
    if joints:
        first = len(gl["nodes"])
        for i, (name, parent, local, _) in enumerate(joints):
            gl["nodes"].append({"name": name, "matrix": list(local)})
        for i, (_, parent, _, _) in enumerate(joints):
            if parent >= 0:
                gl["nodes"][first + parent].setdefault("children", []).append(first + i)
            else:
                gl["scenes"][0]["nodes"].append(first + i)
        ibm = accessor("f", 16, 5126, [j[3] for j in joints])
        gl["skins"] = [{"joints": list(range(first, first + len(joints))),
                        "inverseBindMatrices": ibm, "skeleton": first}]
        skin = 0

    for name, pos, uvs, jids, tris in meshes:
        prims = []
        for mat, faces in tris.items():
            if not faces:
                continue
            used = sorted({i for f in faces for i in f})
            remap = {o: n for n, o in enumerate(used)}
            attrs = {"POSITION": accessor("f", 3, 5126, [pos[i] for i in used], 34962, True),
                     "TEXCOORD_0": accessor("f", 2, 5126, [uvs[i] for i in used], 34962)}
            if skin is not None:
                attrs["JOINTS_0"] = accessor("H", 4, 5123, [(jids[i], 0, 0, 0) for i in used], 34962)
                attrs["WEIGHTS_0"] = accessor("f", 4, 5126, [(1.0, 0, 0, 0) for _ in used], 34962)
            idx = accessor("I", 1, 5125, [(remap[i],) for f in faces for i in f], 34963)
            prims.append({"attributes": attrs, "indices": idx, "material": mat})
        if not prims:
            continue
        gl["meshes"].append({"name": name, "primitives": prims})
        node = {"name": name, "mesh": len(gl["meshes"]) - 1}
        if skin is not None:
            node["skin"] = skin
        gl["nodes"].append(node)
        gl["scenes"][0]["nodes"].append(len(gl["nodes"]) - 1)

    for k in ("textures", "images", "materials", "accessors", "bufferViews"):
        if not gl[k]:
            del gl[k]
    if "textures" not in gl:
        del gl["samplers"]
    gl["buffers"] = [{"byteLength": len(blob)}]
    js = _pad(json.dumps(gl, separators=(",", ":")).encode(), b" ")
    bin_ = _pad(bytes(blob))
    with open(path, "wb") as fh:
        fh.write(struct.pack("<III", 0x46546C67, 2, 12 + 8 + len(js) + 8 + len(bin_)))
        fh.write(struct.pack("<II", len(js), 0x4E4F534A) + js)
        fh.write(struct.pack("<II", len(bin_), 0x004E4942) + bin_)


def write_scene_glb(path, meshes, materials, nodes):
    """Static scene: meshes are (name, positions, uvs, {material: tris}),
    nodes are (name, mesh index, translation, yaw radians, uniform scale).
    Several nodes may share one mesh (instancing)."""
    import math
    gl = {"asset": {"version": "2.0", "generator": "amww"},
          "scene": 0, "scenes": [{"nodes": []}], "nodes": [], "meshes": [],
          "buffers": [], "bufferViews": [], "accessors": [],
          "materials": [], "textures": [], "images": [],
          "samplers": [{"magFilter": 9728, "minFilter": 9728}]}
    blob = bytearray()

    def view(data, target=None):
        blob.extend(b"\0" * (-len(blob) % 4))
        v = {"buffer": 0, "byteOffset": len(blob), "byteLength": len(data)}
        if target:
            v["target"] = target
        blob.extend(data)
        gl["bufferViews"].append(v)
        return len(gl["bufferViews"]) - 1

    def accessor(fmt, comps, ctype, rows, target=None, minmax=False):
        data = struct.pack("<%d%s" % (len(rows) * comps, fmt), *[x for r in rows for x in r])
        a = {"bufferView": view(data, target), "componentType": ctype, "count": len(rows),
             "type": {1: "SCALAR", 2: "VEC2", 3: "VEC3"}[comps]}
        if minmax:
            a["min"] = [min(r[i] for r in rows) for i in range(comps)]
            a["max"] = [max(r[i] for r in rows) for i in range(comps)]
        gl["accessors"].append(a)
        return len(gl["accessors"]) - 1

    for name, png, factor in materials:
        m = {"name": name, "doubleSided": True, "alphaMode": "MASK", "alphaCutoff": 0.5,
             "pbrMetallicRoughness": {"baseColorFactor": list(factor),
                                      "metallicFactor": 0.0, "roughnessFactor": 1.0}}
        if png:
            gl["images"].append({"bufferView": view(png), "mimeType": "image/png"})
            gl["textures"].append({"source": len(gl["images"]) - 1, "sampler": 0})
            m["pbrMetallicRoughness"]["baseColorTexture"] = {"index": len(gl["textures"]) - 1}
        gl["materials"].append(m)

    mesh_ids = []
    for name, pos, uvs, tris in meshes:
        prims = []
        for mat, faces in tris.items():
            used = sorted({i for f in faces for i in f})
            remap = {o: n for n, o in enumerate(used)}
            attrs = {"POSITION": accessor("f", 3, 5126, [pos[i] for i in used], 34962, True),
                     "TEXCOORD_0": accessor("f", 2, 5126, [uvs[i] for i in used], 34962)}
            idx = accessor("I", 1, 5125, [(remap[i],) for f in faces for i in f], 34963)
            prims.append({"attributes": attrs, "indices": idx, "material": mat})
        if prims:
            gl["meshes"].append({"name": name, "primitives": prims})
            mesh_ids.append(len(gl["meshes"]) - 1)
        else:
            mesh_ids.append(None)

    for name, mesh, t, yaw, scale in nodes:
        if mesh_ids[mesh] is None:
            continue
        node = {"name": name or "object", "mesh": mesh_ids[mesh]}
        if any(t):
            node["translation"] = [float(v) for v in t]
        if yaw:
            # level yaw turns the opposite way to a glTF rotation about +Y
            node["rotation"] = [0.0, math.sin(-yaw / 2), 0.0, math.cos(-yaw / 2)]
        if scale != 1.0:
            node["scale"] = [scale] * 3
        gl["nodes"].append(node)
        gl["scenes"][0]["nodes"].append(len(gl["nodes"]) - 1)

    for k in ("textures", "images", "materials", "accessors", "bufferViews"):
        if not gl[k]:
            del gl[k]
    if "textures" not in gl:
        del gl["samplers"]
    gl["buffers"] = [{"byteLength": len(blob)}]
    js = _pad(json.dumps(gl, separators=(",", ":")).encode(), b" ")
    bin_ = _pad(bytes(blob))
    with open(path, "wb") as fh:
        fh.write(struct.pack("<III", 0x46546C67, 2, 12 + 8 + len(js) + 8 + len(bin_)))
        fh.write(struct.pack("<II", len(js), 0x4E4F534A) + js)
        fh.write(struct.pack("<II", len(bin_), 0x004E4942) + bin_)
