#!/usr/bin/env python3
"""
風力ポテンシャルマップの可視化 — 軽量 Leaflet HTML 生成

各県のスコアPNGをImageOverlayで表示する静的HTMLを生成。
Foliumを使わず直接HTMLを出力することでファイルサイズを抑える。

Usage:
    python src/build_map.py -p all
    python src/build_map.py -p fukui
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import rasterio

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import PREFECTURES, PROJECT_ROOT

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

LAYERS = [
    ("total", "総合スコア"),
    ("wind_speed", "風速"),
    ("slope", "傾斜"),
    ("elevation", "標高"),
    ("grid_dist", "送電線距離"),
    ("sub_dist", "変電所距離"),
    ("land_use", "土地利用"),
    ("residential_dist", "居住地距離"),
]


def get_bounds(pref: str) -> dict:
    """スコアTIFからboundsを取得。"""
    tif = PROJECT_ROOT / "output" / pref / "score_total_rgba.tif"
    with rasterio.open(tif) as ds:
        b = ds.bounds
    return {"south": b.bottom, "west": b.left, "north": b.top, "east": b.right}


def create_overlay_png(pref: str, docs_dir: Path):
    """RGBA TIF → PNG に変換。"""
    from PIL import Image

    output_dir = PROJECT_ROOT / "output" / pref
    for layer_name, _ in LAYERS:
        rgba_tif = output_dir / f"score_{layer_name}_rgba.tif"
        png_path = docs_dir / f"score_{layer_name}.png"
        if png_path.exists():
            continue
        if not rgba_tif.exists():
            continue
        with rasterio.open(rgba_tif) as ds:
            rgba = ds.read()
        img = np.moveaxis(rgba, 0, -1)
        pil_img = Image.fromarray(img, "RGBA")
        max_dim = 800
        if max(pil_img.size) > max_dim:
            ratio = max_dim / max(pil_img.size)
            pil_img = pil_img.resize((int(pil_img.width * ratio), int(pil_img.height * ratio)), Image.LANCZOS)
        docs_dir.mkdir(parents=True, exist_ok=True)
        pil_img.save(png_path, optimize=True)


def generate_map_html(pref: str, bounds: dict, name_ja: str) -> str:
    center_lat = (bounds["south"] + bounds["north"]) / 2
    center_lon = (bounds["west"] + bounds["east"]) / 2
    bounds_json = json.dumps([[bounds["south"], bounds["west"]], [bounds["north"], bounds["east"]]])
    layers_json = json.dumps(LAYERS)

    return f"""<!DOCTYPE html>
<html><head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{name_ja} 風力ポテンシャル</title>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9/dist/leaflet.css"/>
<script src="https://unpkg.com/leaflet@1.9/dist/leaflet.js"></script>
<style>
html,body{{margin:0;padding:0;height:100%}}
#map{{height:100%;width:100%}}
.layer-control{{position:absolute;top:10px;right:10px;z-index:1000;background:rgba(255,255,255,0.95);
  border-radius:8px;padding:10px 14px;box-shadow:0 2px 8px rgba(0,0,0,0.2);font:13px/1.5 sans-serif;max-width:200px}}
.layer-control h3{{margin:0 0 6px;font-size:14px}}
.layer-control label{{display:block;cursor:pointer;padding:2px 0}}
.legend{{position:absolute;bottom:20px;left:10px;z-index:1000;background:rgba(255,255,255,0.92);
  border-radius:6px;padding:8px 12px;font:12px sans-serif;box-shadow:0 1px 6px rgba(0,0,0,0.15)}}
.legend i{{width:18px;height:12px;display:inline-block;margin-right:4px;border-radius:2px}}
</style>
</head><body>
<div id="map"></div>
<div class="layer-control" id="ctrl"></div>
<div class="legend">
  <b>スコア</b><br>
  <i style="background:rgb(0,100,0)"></i>最適 (80-100)<br>
  <i style="background:rgb(34,139,34)"></i>高 (60-79)<br>
  <i style="background:rgb(218,165,32)"></i>中 (40-59)<br>
  <i style="background:rgb(255,140,0)"></i>低 (20-39)<br>
  <i style="background:rgb(220,20,60)"></i>不適 (1-19)
</div>
<script>
const bounds = {bounds_json};
const layers = {layers_json};
const map = L.map('map').setView([{center_lat},{center_lon}], 9);
L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png',{{
  attribution:'&copy; OpenStreetMap',maxZoom:18}}).addTo(map);

const overlays = {{}};
const ctrl = document.getElementById('ctrl');
ctrl.innerHTML = '<h3>レイヤー</h3>';
layers.forEach(([id, label], i) => {{
  const img = L.imageOverlay('score_'+id+'.png', bounds, {{opacity:0.7}});
  if (i===0) img.addTo(map);
  overlays[id] = img;
  const lbl = document.createElement('label');
  const cb = document.createElement('input');
  cb.type = 'radio'; cb.name = 'layer'; cb.value = id;
  if (i===0) cb.checked = true;
  cb.onchange = () => {{
    Object.values(overlays).forEach(o => map.removeLayer(o));
    overlays[id].addTo(map);
  }};
  lbl.appendChild(cb);
  lbl.appendChild(document.createTextNode(' '+label));
  ctrl.appendChild(lbl);
}});
</script>
</body></html>"""


def build_all(prefs: list[str]):
    for pref in prefs:
        if pref not in PREFECTURES:
            log.warning("Unknown: %s, skipping", pref)
            continue
        cfg = PREFECTURES[pref]
        name_ja = cfg["name_ja"]
        output_dir = PROJECT_ROOT / "output" / pref
        if not (output_dir / "score_total_rgba.tif").exists():
            log.warning("No scores for %s, skipping", pref)
            continue

        docs_dir = PROJECT_ROOT / "docs" / pref
        docs_dir.mkdir(parents=True, exist_ok=True)

        log.info("Building %s (%s)", pref, name_ja)
        create_overlay_png(pref, docs_dir)
        bounds = get_bounds(pref)
        html = generate_map_html(pref, bounds, name_ja)
        (docs_dir / "wind_potential_map.html").write_text(html, encoding="utf-8")
        log.info("  -> %s (%.1f KB)", docs_dir / "wind_potential_map.html",
                 len(html) / 1024)


def main():
    parser = argparse.ArgumentParser(description="風力ポテンシャルマップ生成 (軽量版)")
    parser.add_argument("-p", "--prefecture", default="all")
    args = parser.parse_args()

    if args.prefecture == "all":
        output_root = PROJECT_ROOT / "output"
        prefs = [d.name for d in sorted(output_root.iterdir())
                 if d.is_dir() and (d / "score_total.tif").exists()]
        log.info("Auto-detected %d prefectures", len(prefs))
    else:
        prefs = [p.strip() for p in args.prefecture.split(",")]

    build_all(prefs)
    log.info("All done. %d maps generated.", len(prefs))


if __name__ == "__main__":
    main()
