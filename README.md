# amww: Army Men: World War – Team Assault asset extractor

A dependency-free Python toolkit that turns a PlayStation disc image of
*Army Men: World War – Team Assault* (SLUS-01435) into ordinary files:

| Game data | Output |
|---|---|
| `3MDL` models in the level `.DAT` files (~2,550) | Wavefront **OBJ + MTL + PNG**, textured, named (`GN_BOX02`, `G1_ROK01`, …) |
| `AMDL` soldiers and vehicles-with-driver | Rigged **glTF (.glb)** with the 23-bone skeleton, plus OBJ, per team colour |
| Level layout (23 levels) | **glTF scene** per level: textured terrain heightmesh + every placed object; heightmap/tilemap PNGs and a placements CSV |
| Level texture VRAM (`.VR1` / `.VR2`) | PNG texture pages, decoded with the right palette per model |
| `TIM` images | PNG (every palette) |
| `3HEX` menu textures (4bpp) | PNG |
| `VAB` sound banks | WAV per sample |
| `XA` music (`AMFF*.XA`) | WAV per channel/track |
| `STR` movies | MP4 (via ffmpeg) |
| CD-DA tracks | WAV |

It also handles generic PS1 formats (TMD models, VAG, SEQ → MIDI), so it is
useful on other PS1 games too, and it writes a `REPORT.md` listing anything
it couldn't decode.

**No game data is included in this repository**, and none should be committed.
You need your own disc image (`.chd`, `.cue/.bin` or `.iso`).

## Requirements

* Python 3.8+
* `chdman` for `.chd` images (`apt install mame-tools`, `brew install rom-tools`,
  or `chdman.exe` from a MAME release)
* `ffmpeg` for STR/XA conversion (optional)
* `numpy` for the `_preview.png` contact sheets (optional)

## Usage

```sh
python -m amww extract "Army Men - World War - Team Assault (USA).chd" -o extracted
```

Or, if you've already extracted the disc's files with another tool:

```sh
python -m amww scan path/to/disc/files -o extracted
```

Output layout:

```
extracted/
  REPORT.md, manifest.json      what was found / what wasn't
  files/                        the disc's filesystem
  models/<LEVEL>/NNN_NAME.obj   3MDL models (+ .mtl, tex/*.png, _preview.png)
  models/characters/<name>/     AMDL characters: <name>_<colour>.glb (rigged,
                                every part a separate mesh), .obj (default
                                look), _allparts.obj (every part as an object)
  levels/<LEVEL>/<LEVEL>.glb    whole level: terrain + placed objects (instanced)
  levels/<LEVEL>/heightmap.png  256x128 height image (white = high)
  levels/<LEVEL>/tilemap.png    top-down picture of the ground textures
  levels/<LEVEL>/placements.csv every object: name, position, yaw, scale
  textures/                     TIM and 3HEX images
  audio/vab/                    sound effects
  music/XA/                     soundtrack, one WAV per XA channel
  video/                        FMV as MP4
```

OBJ files are Y-up (PS1's Y-down axis is flipped) in the game's native units.
Import them into Blender with *File → Import → Wavefront (.obj)*.

## The 3MDL format

Worked out from the disc data; documented in [`amww/mdl3.py`](amww/mdl3.py).
In short: a header, a vertex list, then a stream of 32-bit register writes
(UVs at `0x20–0x2C`, colours at `0x00–0x0C`, vertex indices at `0x30–0x4C`,
texture slot `0x58`, render mode `0x5C`). A write with either low flag bit set
emits a polygon. The model is followed by 20-byte texture-slot records (UV
offset, texture page, CLUT). Level VRAM comes from `.VR1`/`.VR2`, which are
256×512-halfword dumps placed at VRAM x=512 and x=768.

## The AMDL format (characters)

Documented in [`amww/amdl.py`](amww/amdl.py). It's the 3MDL mesh format with
a bone index on every vertex, followed by a skeleton (parent table plus a
3×3 rotation and translation per bone) and one set of texture slots per team
colour (green and tan share textures and differ only in palette). The mesh is
split into named, toggleable parts: body (`torso`, `arms`, `legs`, `helmet`,
`face`…), weapons (`m16 side`, `sniper`, `bzooka sd`, `mortar`, `mine swpr`,
`flmr tnks`…) and damage pieces (`frnt blwwy`, `brkawy sd`, `neck cap`…).
The `.obj` shows the body plus the M16; the `.glb` and `_allparts.obj`
contain every part so you can switch weapons or damage on and off.

There are three detail levels (328/261/173 polygons) in two sizes, shared by
every level, plus rigid single-bone vehicle-with-driver models on some
levels (their wheels and guns are separate 3MDL models).

## Level layout

Documented in [`amww/level.py`](amww/level.py). A level `.DAT` is a header,
the object name table, then a chain of `(id, size)` sections: 8 = object
placements (position, yaw in 1/256 turns, scale, type), 10 = the 256×128
terrain grid (height byte ×64, ground-tile index, flags, shade), 15 = the
collision map, 17 = path nodes, 18 = the models. Terrain cells are 64 units;
model vertices and heights are in units 16× finer, so the exported scene
scales models by 1/16 and heights by 1/16 to keep everything in proportion.
Ground tiles are 48×48 4bpp textures, 25 per page in texture pages 13–15
and 29–31, with palette row 12 + tile.

## Known gaps

* Vehicle parts render grey: they likely use palettes the game fills in at
  runtime (team colours).
* Type-2 `3HEX` images (the 18 full-screen menu backgrounds) aren't decoded yet.
* Character animations haven't been found yet; soldiers export in their
  T-pose bind pose (the rig is included, so they can be posed in Blender).
* Terrain texturing is approximate: tile choice and palettes come out right
  for most of the ground (water, grass, sand, roads), but the per-cell flag
  bits (likely tile rotation/flip) aren't decoded, so transition tiles such
  as shorelines can look patchy, and a few tiles may use a neighbouring
  palette. Water is the lake bed; the game draws a water surface on top.
* Level sections 9 (scripts/units), 13, 14 and 16 aren't decoded, so units
  and vehicles that spawn via scripts aren't placed in the level scenes.
* VAB samples are written at 22,050 Hz (`--vab-rate` to change); VAB files
  don't store the original rate.

## Tests

```sh
python -m unittest discover tests
```

The tests build a synthetic PS1 disc and synthetic 3MDL/3HEX data; no game
data is needed.
