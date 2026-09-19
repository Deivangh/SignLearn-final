#!/usr/bin/env python
"""Shrink a Mixamo/Blender GLB down to something a browser can load quickly.

A 35 MB character is fine in Blender and painful in a web page — Streamlit
re-mounts the component on every rerun, so the model is fetched again each
time. Almost all of that weight is texture data the render never uses:

* every ``normalTexture`` in this export carries ``scale: 0``, which means the
  normal map contributes nothing at all;
* the specular and metallic-roughness maps buy very little on a figure lit by
  three soft lights and viewed from chest up.

So we keep the base colour map, drop the rest, downscale what is left and
re-encode it — PNG only where the material actually blends alpha, JPEG
everywhere else. Geometry, the skeleton and skin weights are untouched.

    python scripts/optimise_avatar.py avatar.glb static/avatar.glb
"""
from __future__ import annotations

import io
import json
import struct
import sys
from pathlib import Path

from PIL import Image

MAX_TEXTURE = 1024        # plenty for a figure framed from the chest up
JPEG_QUALITY = 86


def _read_glb(path: Path):
    raw = path.read_bytes()
    magic, version, _ = struct.unpack_from("<4sII", raw, 0)
    if magic != b"glTF":
        raise SystemExit(f"{path} is not a GLB file")
    off, gltf, blob = 12, None, b""
    while off < len(raw):
        clen, ctype = struct.unpack_from("<I4s", raw, off)
        off += 8
        chunk = raw[off:off + clen]
        if ctype == b"JSON":
            gltf = json.loads(chunk.decode("utf-8"))
        elif ctype.rstrip(b"\x00") == b"BIN":
            blob = chunk
        off += clen + (-clen % 4)
    if gltf is None:
        raise SystemExit("no JSON chunk found")
    return gltf, blob


def _write_glb(path: Path, gltf: dict, blob: bytes) -> None:
    js = json.dumps(gltf, separators=(",", ":")).encode("utf-8")
    js += b" " * (-len(js) % 4)
    blob += b"\x00" * (-len(blob) % 4)
    total = 12 + 8 + len(js) + 8 + len(blob)
    with open(path, "wb") as fh:
        fh.write(struct.pack("<4sII", b"glTF", 2, total))
        fh.write(struct.pack("<I4s", len(js), b"JSON"))
        fh.write(js)
        fh.write(struct.pack("<I4s", len(blob), b"BIN\x00"))
        fh.write(blob)


def _recompress(data: bytes, keep_alpha: bool) -> tuple[bytes, str]:
    img = Image.open(io.BytesIO(data))
    if max(img.size) > MAX_TEXTURE:
        scale = MAX_TEXTURE / max(img.size)
        img = img.resize((max(1, int(img.width * scale)),
                          max(1, int(img.height * scale))), Image.LANCZOS)
    out = io.BytesIO()
    if keep_alpha and img.mode in ("RGBA", "LA", "P"):
        img.convert("RGBA").save(out, format="PNG", optimize=True)
        return out.getvalue(), "image/png"
    img.convert("RGB").save(out, format="JPEG", quality=JPEG_QUALITY, optimize=True)
    return out.getvalue(), "image/jpeg"


def main(src: Path, dst: Path) -> int:
    gltf, blob = _read_glb(src)
    views = gltf.get("bufferViews", [])
    images = gltf.get("images", [])
    textures = gltf.get("textures", [])

    # 1. Strip everything but base colour, and note which textures survive.
    alpha_tex: set[int] = set()
    for mat in gltf.get("materials", []):
        mat.pop("normalTexture", None)
        mat.pop("occlusionTexture", None)
        mat.pop("emissiveTexture", None)
        mat.pop("emissiveFactor", None)
        ext = mat.get("extensions", {})
        ext.pop("KHR_materials_specular", None)
        if not ext:
            mat.pop("extensions", None)
        pbr = mat.get("pbrMetallicRoughness", {})
        pbr.pop("metallicRoughnessTexture", None)
        pbr.setdefault("metallicFactor", 0.0)
        pbr.setdefault("roughnessFactor", 0.85)
        base = pbr.get("baseColorTexture")
        if base and mat.get("alphaMode") in ("BLEND", "MASK"):
            alpha_tex.add(base["index"])

    gltf["extensionsUsed"] = [
        e for e in gltf.get("extensionsUsed", []) if e != "KHR_materials_specular"
    ] or None
    if gltf["extensionsUsed"] is None:
        gltf.pop("extensionsUsed")

    kept_tex = sorted({
        t["index"]
        for m in gltf.get("materials", [])
        for t in [m.get("pbrMetallicRoughness", {}).get("baseColorTexture")]
        if t
    })
    tex_remap = {old: new for new, old in enumerate(kept_tex)}
    for mat in gltf.get("materials", []):
        base = mat.get("pbrMetallicRoughness", {}).get("baseColorTexture")
        if base:
            base["index"] = tex_remap[base["index"]]
    # Copy, so remapping indices below cannot write through into `textures`
    # and corrupt the original-index lookups we still need.
    new_textures = [dict(textures[i]) for i in kept_tex]

    # Which *images* carry an alpha channel that a material actually blends.
    # Must be read before any remapping, using original indices.
    alpha_img = {
        src
        for tex_old in alpha_tex
        for src in [textures[tex_old].get("source")]
        if src is not None
    }

    kept_img = sorted({t["source"] for t in new_textures if "source" in t})
    img_remap = {old: new for new, old in enumerate(kept_img)}
    for t in new_textures:
        if "source" in t:
            t["source"] = img_remap[t["source"]]

    # 2. Re-encode the surviving images.
    new_images = []
    before = after = 0
    for old in kept_img:
        im = images[old]
        view = views[im["bufferView"]]
        start = view.get("byteOffset", 0)
        data = blob[start:start + view["byteLength"]]
        before += len(data)
        enc, mime = _recompress(data, keep_alpha=old in alpha_img)
        after += len(enc)
        print(f"  {im.get('name','?'):40s} {len(data)/1024:8.0f} KB -> "
              f"{len(enc)/1024:7.0f} KB  {mime}")
        new_images.append({"name": im.get("name", ""), "mimeType": mime,
                           "_data": enc})

    # 3. Rebuild the binary chunk: geometry views first, then the new images.
    #    Every offset shifts, so bufferView indices have to be remapped too.
    geometry_views = [i for i, _ in enumerate(views)
                      if i not in {images[o]["bufferView"] for o in range(len(images))
                                   if "bufferView" in images[o]}]
    out_views, out_blob, view_remap = [], bytearray(), {}
    for old in geometry_views:
        v = dict(views[old])
        start = v.get("byteOffset", 0)
        data = blob[start:start + v["byteLength"]]
        out_blob += b"\x00" * (-len(out_blob) % 4)
        v["byteOffset"] = len(out_blob)
        v["byteLength"] = len(data)
        view_remap[old] = len(out_views)
        out_views.append(v)
        out_blob += data

    for img in new_images:
        data = img.pop("_data")
        out_blob += b"\x00" * (-len(out_blob) % 4)
        out_views.append({"buffer": 0, "byteOffset": len(out_blob),
                          "byteLength": len(data)})
        img["bufferView"] = len(out_views) - 1
        out_blob += data

    for acc in gltf.get("accessors", []):
        if "bufferView" in acc:
            acc["bufferView"] = view_remap[acc["bufferView"]]

    gltf["bufferViews"] = out_views
    gltf["images"] = new_images
    gltf["textures"] = new_textures
    gltf["buffers"] = [{"byteLength": len(out_blob)}]

    _write_glb(dst, gltf, bytes(out_blob))
    print(f"\ntextures {before/1e6:.1f} MB -> {after/1e6:.1f} MB")
    print(f"file     {src.stat().st_size/1e6:.1f} MB -> {dst.stat().st_size/1e6:.1f} MB")
    return 0


if __name__ == "__main__":
    if len(sys.argv) < 3:
        raise SystemExit(__doc__)
    dst_path = Path(sys.argv[2])
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    raise SystemExit(main(Path(sys.argv[1]), dst_path))
