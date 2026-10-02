"""Command-line entry point: python -m amww extract <disc> -o <out>"""

import argparse
import json
import os
import sys
import time
from collections import Counter, defaultdict

from . import disc
from .audio import decode_adpcm
from .mdl3 import export_obj as export_mdl3
from .media import convert_stream
from .scan import coverage, entropy, identify, scan_buffer
from .tim import Vram
from .tmd import export_obj
from .wav import write_png, write_wav

MAX_PALETTES = 16


def log(msg):
    print(msg, flush=True)


class Extractor:
    def __init__(self, out, vab_rate=22050, media=True):
        self.out = out
        self.vab_rate = vab_rate
        self.media = media
        self.manifest = {"files": [], "streams": [], "cdda": [], "notes": []}
        self.counts = Counter()
        self.global_vram = Vram()
        self.pending_models = []  # (rel, found, file_vram)
        self.pending_mdl3 = defaultdict(list)  # source rel -> [(offset, Mdl3)]

    def p(self, *parts):
        path = os.path.join(self.out, *parts)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        return path

    # ---- disc stage -----------------------------------------------------
    def extract_disc(self, image):
        ext = os.path.splitext(image)[1].lower()
        if ext == ".chd":
            log("[disc] converting CHD to BIN/CUE with chdman ...")
            image = disc.chd_to_cue(image, os.path.join(self.out, "_disc"))
        tracks = disc.tracks_for_image(image)
        log("[disc] %d track(s): %s" % (len(tracks), ", ".join(
            "%d:%s" % (t.number, t.mode) for t in tracks)))
        files_root = os.path.join(self.out, "files")
        raw_files, normal_files = [], []
        for t in tracks:
            if t.is_audio:
                dest = self.p("audio", "cdda", "track%02d.wav" % t.number)
                disc.write_cdda_wav(t, dest)
                self.manifest["cdda"].append(os.path.relpath(dest, self.out))
                self.counts["cdda"] += 1
                continue
            dt = disc.DataTrack(t)
            try:
                entries = list(disc.walk_iso(dt))
                for e in entries:
                    if e.is_dir:
                        continue
                    dest = os.path.join(files_root, e.path)
                    raw = disc.extract_file(dt, e, dest)
                    if raw:
                        raw_files.append((e.path, dest))
                    elif disc.is_packed_xa(dest):
                        conv = os.path.join(self.out, "_streams", e.path)
                        disc.packed_xa_to_raw(dest, conv)
                        raw_files.append((e.path, conv))
                    else:
                        normal_files.append(e.path)
                log("[disc] extracted %d files (%d raw XA/STR)" % (
                    len(raw_files) + len(normal_files), len(raw_files)))
                for lba, n in disc.unlisted_regions(dt, entries):
                    rel = "_unlisted/lba_%06d_%d.bin" % (lba, n)
                    with open(self.p("files", rel), "wb") as fh:
                        for i in range(n):
                            fh.write(dt.user(lba + i)[0][:2048])
                    normal_files.append(rel)
                    self.manifest["notes"].append(
                        "Data outside the filesystem at LBA %d (%d sectors) dumped to files/%s"
                        % (lba, n, rel))
            finally:
                dt.close()
        return files_root, normal_files, raw_files

    # ---- asset stage ----------------------------------------------------
    def scan_file(self, root, rel):
        path = os.path.join(root, rel)
        with open(path, "rb") as fh:
            buf = fh.read()
        found = scan_buffer(buf)
        info = {
            "path": rel, "size": len(buf), "header": identify(buf),
            "magic": buf[:4].hex(), "entropy": round(entropy(buf), 2),
            "assets": dict(Counter(f.kind for f in found)),
            "coverage": round(coverage(found, len(buf)), 3),
        }
        self.manifest["files"].append(info)
        if not found:
            return
        stem = rel.replace("\\", "/")
        file_vram = Vram()
        for f in found:
            tag = "%s@%08x" % (os.path.basename(stem), f.offset)
            sub = os.path.dirname(stem)
            if f.kind == "tim":
                self.write_tim(f.obj, sub, tag)
                file_vram.load_tim(f.obj)
                self.global_vram.load_tim(f.obj)
            elif f.kind == "tmd":
                self.pending_models.append((sub, tag, f.obj, file_vram))
            elif f.kind == "vag":
                dest = self.p("audio", "vag", sub, "%s_%s.wav" % (tag, f.obj.name or "sample"))
                write_wav(dest, decode_adpcm(f.obj.data), f.obj.rate)
                self.counts["vag"] += 1
            elif f.kind == "vab":
                if f.obj.bodies is None:
                    vb = os.path.splitext(path)[0]
                    for ext in (".VB", ".vb"):
                        if os.path.exists(vb + ext) and f.offset == 0:
                            with open(vb + ext, "rb") as fh:
                                f.obj.attach_bodies(fh.read())
                            break
                self.write_vab(f.obj, sub, tag)
            elif f.kind == "mdl3":
                self.pending_mdl3[rel].append((f.offset, f.obj))
            elif f.kind == "hex3":
                w, h, rgba, _ = f.obj
                write_png(self.p("textures", sub, tag + ".png"), w, h, rgba)
                self.counts["hex3"] += 1
            elif f.kind == "seq":
                dest = self.p("music", sub, tag + ".mid")
                with open(dest, "wb") as fh:
                    fh.write(f.obj[0])
                self.counts["seq"] += 1

    def write_tim(self, tim, sub, tag):
        n = min(tim.palette_count, MAX_PALETTES)
        for pal in range(n):
            suffix = "" if n == 1 else "_pal%02d" % pal
            dest = self.p("textures", sub, "%s_%dbpp%s.png" % (tag, tim.bpp, suffix))
            write_png(dest, tim.width, tim.height, tim.to_rgba(pal))
        self.counts["tim"] += 1

    def write_vab(self, vab, sub, tag):
        if not vab.bodies:
            self.manifest["notes"].append(
                "VAB header %s/%s has no sample data (separate .VB not found)" % (sub, tag))
            return
        for i, body in enumerate(vab.bodies):
            dest = self.p("audio", "vab", sub, tag, "sample_%03d.wav" % i)
            write_wav(dest, decode_adpcm(body), self.vab_rate)
        self.counts["vab"] += 1
        self.counts["vab_samples"] += len(vab.bodies)

    def write_models(self):
        tex_cache = {}
        for sub, tag, tmd, file_vram in self.pending_models:
            model_dir = self.p("models", sub, tag, "x")[:-2]
            os.makedirs(model_dir, exist_ok=True)

            def texture_for(tpage, clut, file_vram=file_vram, model_dir=model_dir):
                vram = file_vram
                if file_vram.page_coverage(tpage, clut) < 0.25:
                    vram = self.global_vram
                if vram.page_coverage(tpage, clut) == 0:
                    return None
                name = "tex_%04x_%04x.png" % (tpage, clut)
                key = (id(vram), tpage, clut)
                if key not in tex_cache:
                    tex_cache[key] = vram.texture(tpage, clut)
                write_png(os.path.join(model_dir, name), 256, 256, tex_cache[key])
                return name

            export_obj(tmd, os.path.join(model_dir, tag + ".obj"), texture_for)
            self.counts["tmd"] += 1
            self.counts["polygons"] += tmd.poly_count

    def write_mdl3(self, root):
        for rel, models in sorted(self.pending_mdl3.items()):
            stem = os.path.splitext(rel)[0]
            vram = level_vram(os.path.join(root, stem))
            names = model_names(os.path.join(root, rel), models)
            group = os.path.basename(stem) if vram else rel.replace("/", "_")
            out_dir = self.p("models", group, "x")[:-2]
            tex_dir = os.path.join(out_dir, "tex")
            written = {}

            def texture_for(tpage, clut):
                name = "tex/%04x_%04x.png" % (tpage, clut)
                if name not in written:
                    os.makedirs(tex_dir, exist_ok=True)
                    write_png(os.path.join(out_dir, name), 256, 256, vram.texture(tpage, clut))
                    written[name] = True
                return name

            sheet = []
            for i, (off, m) in enumerate(models):
                if not m.polys:
                    continue
                label = "%03d_%s" % (i, names.get(off) or "%08x" % off)
                path = os.path.join(out_dir, label + ".obj")
                export_mdl3(m, path, texture_for if vram else None, name=label)
                self.counts["mdl3"] += 1
                self.counts["mdl3_polygons"] += m.poly_count
                if vram and m.slots:
                    self.counts["mdl3_textured"] += 1
                sheet.append(path)
            if sheet:
                write_contact_sheet(sheet, os.path.join(out_dir, "_preview.png"))
            log("[models]   %-24s %4d models%s" % (
                group, len(sheet), " (textured from %s.VR1/VR2)" % os.path.basename(stem)
                if vram else ""))

    def convert_streams(self, raw_files):
        for rel, path in raw_files:
            if not self.media:
                break
            log("[media]   %s" % rel)
            kind = "music" if rel.upper().endswith(".XA") else "video"
            out_base = self.p(kind, os.path.splitext(rel)[0])
            outputs, note = convert_stream(path, out_base)
            self.manifest["streams"].append({
                "path": rel, "outputs": [os.path.relpath(o, self.out) for o in outputs],
                "note": note})
            self.counts["stream_outputs"] += len(outputs)

    # ---- report ---------------------------------------------------------
    def write_report(self):
        with open(self.p("manifest.json"), "w") as fh:
            json.dump(self.manifest, fh, indent=1)
        files = self.manifest["files"]
        lines = ["# Extraction report", ""]
        lines += ["| Asset | Count |", "|---|---|"]
        for k in ("tim", "hex3", "mdl3", "mdl3_textured", "mdl3_polygons", "tmd",
                  "polygons", "vag", "vab", "vab_samples", "seq",
                  "cdda", "stream_outputs"):
            lines.append("| %s | %d |" % (k, self.counts[k]))
        lines += ["", "## Files with recognised assets", "",
                  "| File | Size | Assets | Coverage |", "|---|---|---|---|"]
        for f in files:
            if f["assets"]:
                lines.append("| %s | %d | %s | %.0f%% |" % (
                    f["path"], f["size"],
                    ", ".join("%s×%d" % kv for kv in sorted(f["assets"].items())),
                    f["coverage"] * 100))
        lines += ["", "## Files not (fully) decoded", "",
                  "Formats the toolkit doesn't know yet. High entropy (>7.5) usually "
                  "means compression; the first 4 bytes are the best clue to the format.",
                  "", "| File | Size | Magic | Entropy | Header | Coverage |",
                  "|---|---|---|---|---|---|"]
        for f in sorted(files, key=lambda f: -f["size"]):
            if f["coverage"] < 0.9:
                lines.append("| %s | %d | `%s` | %.2f | %s | %.0f%% |" % (
                    f["path"], f["size"], f["magic"], f["entropy"],
                    f["header"] or "", f["coverage"] * 100))
        magic = Counter(f["magic"] for f in files if not f["assets"])
        if magic:
            lines += ["", "### Most common unknown magics", ""]
            lines += ["- `%s` × %d" % kv for kv in magic.most_common(15)]
        if self.manifest["streams"]:
            lines += ["", "## XA / STR streams", ""]
            for s in self.manifest["streams"]:
                lines.append("- %s → %s%s" % (
                    s["path"], ", ".join(s["outputs"]) or "(none)",
                    " — " + s["note"] if s["note"] else ""))
        if self.manifest["notes"]:
            lines += ["", "## Notes", ""] + ["- " + n for n in self.manifest["notes"]]
        with open(self.p("REPORT.md"), "w") as fh:
            fh.write("\n".join(lines) + "\n")


def level_vram(stem):
    """Army Men levels ship their texture VRAM as STEM.VR1 / STEM.VR2:
    two 256x512-halfword dumps that belong at VRAM x=512 and x=768."""
    from .tim import Vram
    v1, v2 = stem + ".VR1", stem + ".VR2"
    if not (os.path.exists(v1) and os.path.exists(v2)):
        return None
    vram = Vram()
    with open(v1, "rb") as fh:
        vram.load_raw(512, 0, 256, 512, fh.read())
    with open(v2, "rb") as fh:
        vram.load_raw(768, 0, 256, 512, fh.read())
    return vram


def model_names(dat_path, models):
    """Map model offsets to names from a level .DAT's object table.

    The table at 0x2C has u32 count at offset 8 and 52-byte entries:
    char name[8] followed by 11 u32s, the sixth of which is the byte size
    of the object's model. Models appear in the same order as the table.
    """
    import struct
    names = {}
    if not dat_path.upper().endswith(".DAT"):
        return names
    with open(dat_path, "rb") as fh:
        head = fh.read(0x10000)
    if len(head) < 12:
        return names
    count = struct.unpack_from("<I", head, 8)[0]
    if not 0 < count < 1000 or 0x2C + count * 52 > len(head):
        return names
    entries = []
    for i in range(count):
        o = 0x2C + i * 52
        name = head[o:o + 8].split(b"\0")[0].decode("latin-1").strip()
        size = struct.unpack_from("<I", head, o + 28)[0]
        if name and all(32 < ord(c) < 127 for c in name):
            entries.append((name, size))
    j = 0
    for off, m in models:
        if j < len(entries) and entries[j][1] == m.size:
            names[off] = entries[j][0]
            j += 1
    return names


def write_contact_sheet(obj_paths, dest, tile=160, cols=8):
    """Render every OBJ into a grid PNG (skipped if numpy isn't installed)."""
    try:
        import numpy as np
        from .preview import load_obj_tris, render
    except ImportError:
        return
    pngs = {}

    def load_png(path):
        if path not in pngs:
            pngs[path] = read_png_rgba(path)
        return pngs[path]

    tiles = []
    for path in obj_paths:
        tris, textures = load_obj_tris(path, load_png)
        tiles.append(np.frombuffer(render(tris, textures, tile), np.uint8).reshape(tile, tile, 4))
    rows = (len(tiles) + cols - 1) // cols
    sheet = np.zeros((rows * tile, cols * tile, 4), np.uint8)
    sheet[..., 3] = 255
    for i, t in enumerate(tiles):
        r, c = divmod(i, cols)
        sheet[r * tile:(r + 1) * tile, c * tile:(c + 1) * tile] = t
    write_png(dest, cols * tile, rows * tile, sheet.tobytes())


def read_png_rgba(path):
    """Read back an 8-bit RGBA PNG written by write_png."""
    import struct
    import zlib
    import numpy as np
    with open(path, "rb") as fh:
        data = fh.read()
    w, h = struct.unpack(">II", data[16:24])
    idat = b""
    p = 8
    while p < len(data):
        n = struct.unpack(">I", data[p:p + 4])[0]
        if data[p + 4:p + 8] == b"IDAT":
            idat += data[p + 8:p + 8 + n]
        p += 12 + n
    raw = np.frombuffer(zlib.decompress(idat), np.uint8).reshape(h, w * 4 + 1)
    return raw[:, 1:].reshape(h, w, 4)


def _walk(root):
    for d, _, names in os.walk(root):
        for n in sorted(names):
            yield os.path.relpath(os.path.join(d, n), root)


def main(argv=None):
    ap = argparse.ArgumentParser(
        prog="amww", description="Extract PlayStation 1 game assets to usable files.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    ex = sub.add_parser("extract", help="extract a disc image (.chd/.cue/.bin/.iso)")
    ex.add_argument("image")
    sc = sub.add_parser("scan", help="scan a folder of already-extracted files")
    sc.add_argument("folder")
    for p in (ex, sc):
        p.add_argument("-o", "--out", default="extracted")
        p.add_argument("--vab-rate", type=int, default=22050,
                       help="sample rate for VAB bank samples (not stored in the file)")
        p.add_argument("--no-media", action="store_true",
                       help="skip STR/XA conversion with ffmpeg")
    args = ap.parse_args(argv)

    t0 = time.time()
    ex = Extractor(args.out, args.vab_rate, not args.no_media)
    os.makedirs(args.out, exist_ok=True)
    if args.cmd == "extract":
        root, normal, raw = ex.extract_disc(args.image)
    else:
        root = args.folder
        normal, raw = [], []
        for rel in _walk(root):
            with open(os.path.join(root, rel), "rb") as fh:
                head = fh.read(16)
            path = os.path.join(root, rel)
            if head[:12] == disc.SYNC:
                raw.append((rel, path))
            elif disc.is_packed_xa(path):
                conv = os.path.join(args.out, "_streams", rel)
                disc.packed_xa_to_raw(path, conv)
                raw.append((rel, conv))
            else:
                normal.append(rel)
    log("[scan] scanning %d files for TIM/TMD/VAG/VAB/SEQ ..." % len(normal))
    for i, rel in enumerate(normal, 1):
        ex.scan_file(root, rel)
        if i % 50 == 0:
            log("[scan] %d/%d" % (i, len(normal)))
    log("[models] exporting %d TMD model(s) ..." % len(ex.pending_models))
    ex.write_models()
    n3 = sum(len(v) for v in ex.pending_mdl3.values())
    log("[models] exporting %d 3MDL model(s) ..." % n3)
    ex.write_mdl3(root)
    if raw:
        log("[media] converting %d XA/STR stream file(s) ..." % len(raw))
        ex.convert_streams(raw)
    ex.write_report()
    log("[done] %s in %.0fs -> %s" % (
        ", ".join("%s=%d" % kv for kv in sorted(ex.counts.items())) or "nothing found",
        time.time() - t0, os.path.join(args.out, "REPORT.md")))
    return 0


if __name__ == "__main__":
    sys.exit(main())
