#!/usr/bin/env python3
"""
全国風力ポテンシャル — ドキュメント生成

1. prefectures.json — 県別bounds/center/grid_area
2. 県別PNG — score_total_rgba.tif → {pref}.png (原寸)
3. スコア別PNG — score_{type}_rgba.tif → {pref}/score_{type}.png
4. 送電線GeoJSON — All-Japan-Grid → docs/grid/{area}_{kind}.geojson
5. 全国概観PNG — 全県merge → japan_{type}.png

Usage:
    python src/build_map.py                # 全県
    python src/build_map.py -p fukui       # 特定県のみ
    python src/build_map.py --grid-only    # GeoJSON コピーのみ
"""

import argparse
import json
import logging
import math
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import rasterio
from rasterio.coords import BoundingBox
from rasterio.enums import Resampling
from rasterio.merge import merge
from rasterio.transform import Affine
from rasterio.warp import reproject

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import PREFECTURES, PROJECT_ROOT

# PNG 共通グリッド:
# 全県の native TIF は 1/3600° (SRTM原点整列) で出力されている前提。
# PNG はこの 4倍粗いグリッド (4/3600° ≒ 約120m) にスナップ。
# 全県が同じ大域粗グリッドのサブウィンドウになるため、隣接PNG の画素境界が完全一致し、
# Leaflet で重畳しても seam/overlap が発生しない。
# coarse_factor を変えると PNG 解像度と容量が変わる: 8=60m(超高解像,容量大) 4=120m(既定) 2=220m(粗)
COARSE_FACTOR = 4

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

ALL_JAPAN_GRID = Path(os.environ.get(
    "ALL_JAPAN_GRID_DIR",
    str(PROJECT_ROOT.parent / "All-Japan-Grid" / "data"),
))

SCORE_TYPES = ["total", "wind_speed", "slope", "elevation",
               "grid_dist", "sub_dist", "land_use", "residential_dist"]

GRID_AREAS = ["hokkaido", "tohoku", "tokyo", "chubu", "hokuriku",
              "kansai", "chugoku", "shikoku", "kyushu", "okinawa"]


# ── 1. prefectures.json ────────────────────────────────────
def generate_prefectures_json(docs_dir: Path, prefs: list[str], png_bounds: dict[str, list] | None = None):
    """PNG の実効 bounds (coarse-snapped) から prefectures.json を生成。
    png_bounds が与えられれば使用し、なければ TIF bounds にフォールバック。
    """
    png_bounds = png_bounds or {}
    data = {}
    for pref in prefs:
        cfg = PREFECTURES[pref]
        if pref in png_bounds:
            bounds = png_bounds[pref]
        else:
            tif = PROJECT_ROOT / "output" / pref / "score_total_rgba.tif"
            if tif.exists():
                with rasterio.open(tif) as ds:
                    b = ds.bounds
                bounds = [[b.bottom, b.left], [b.top, b.right]]
            else:
                bbox = cfg["bbox"]
                bounds = [[bbox[1], bbox[0]], [bbox[3], bbox[2]]]

        data[pref] = {
            "name_ja": cfg["name_ja"],
            "bounds": bounds,
            "center": cfg["center"],
            "grid_area": cfg["grid_area"],
        }

    out = docs_dir / "prefectures.json"
    out.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    log.info("prefectures.json: %d entries", len(data))


# ── 2. 県別PNG (粗グリッド整列) ─────────────────────────────
def generate_pref_pngs(docs_dir: Path, prefs: list[str]) -> dict[str, list]:
    """全県の score_total_rgba.tif → {pref}.png。
    戻り値: {pref: [[south,west],[north,east]]} — PNG の実効 bounds。
    """
    bounds_map: dict[str, list] = {}
    for pref in prefs:
        output_dir = PROJECT_ROOT / "output" / pref
        rgba_tif = output_dir / "score_total_rgba.tif"
        if not rgba_tif.exists():
            continue

        png_path = docs_dir / f"{pref}.png"
        b = _tif_to_png(rgba_tif, png_path)
        bounds_map[pref] = [[b.bottom, b.left], [b.top, b.right]]

        pref_dir = docs_dir / pref
        pref_dir.mkdir(parents=True, exist_ok=True)
        for stype in SCORE_TYPES:
            src = output_dir / f"score_{stype}_rgba.tif"
            dst = pref_dir / f"score_{stype}.png"
            if src.exists():
                _tif_to_png(src, dst)

    log.info("Prefecture PNGs: %d done (coarse factor=%d)", len(prefs), COARSE_FACTOR)
    return bounds_map


def _tif_to_png(tif_path: Path, png_path: Path, coarse_factor: int = COARSE_FACTOR) -> BoundingBox:
    """RGBA GeoTIFF → PNG。
    全国統一の粗グリッド (res_coarse = res_native * coarse_factor) にスナップして
    nearest-neighbor で再サンプル。隣接県の PNG 画素が完全整列する。
    戻り値: 粗グリッドの BoundingBox (Leaflet にそのまま渡せる)。
    """
    from PIL import Image

    with rasterio.open(tif_path) as ds:
        native_tr = ds.transform
        crs = ds.crs
        bounds = ds.bounds
        rgba_native = ds.read()

    res_native = abs(native_tr.a)
    res_coarse = res_native * coarse_factor

    # 粗グリッドは原点 (lon=0, lat=0) に整列: 全県で共有
    col_min = math.floor(bounds.left  / res_coarse)
    col_max = math.ceil (bounds.right / res_coarse)
    row_min = math.floor(-bounds.top    / res_coarse)
    row_max = math.ceil (-bounds.bottom / res_coarse)
    coarse_w = col_max - col_min
    coarse_h = row_max - row_min
    coarse_tr = Affine(
        res_coarse, 0.0, col_min * res_coarse,
        0.0, -res_coarse, -row_min * res_coarse,
    )
    coarse_bounds = BoundingBox(
        left=col_min * res_coarse,
        bottom=-row_max * res_coarse,
        right=col_max * res_coarse,
        top=-row_min * res_coarse,
    )

    # 各チャネルを粗グリッドに nearest で再サンプル
    # 粗画素中心は native 画素中心と決定的に一致するため、隣県で同じ global 粗画素は
    # 同じ native 画素をサンプリングし、admin マスク判定も一意に決まる。
    dst = np.zeros((4, coarse_h, coarse_w), dtype=np.uint8)
    for ch in range(4):
        reproject(
            source=rgba_native[ch], destination=dst[ch],
            src_transform=native_tr, src_crs=crs,
            dst_transform=coarse_tr, dst_crs=crs,
            resampling=Resampling.nearest,
        )

    arr = np.moveaxis(dst, 0, -1)  # (H, W, 4)
    pil_img = Image.fromarray(arr, "RGBA")
    png_path.parent.mkdir(parents=True, exist_ok=True)
    pil_img.save(png_path, optimize=True)
    return coarse_bounds


# ── 3. 送電線GeoJSON コピー ─────────────────────────────────
def copy_grid_geojson(docs_dir: Path):
    """All-Japan-Grid → docs/grid/{area}_{kind}.geojson。"""
    grid_dir = docs_dir / "grid"
    grid_dir.mkdir(parents=True, exist_ok=True)

    copied = 0
    for area in GRID_AREAS:
        for kind in ["lines", "substations"]:
            src = ALL_JAPAN_GRID / f"{area}_{kind}.geojson"
            dst = grid_dir / f"{area}_{kind}.geojson"
            if src.exists() and not dst.exists():
                shutil.copy2(src, dst)
                copied += 1
            elif src.exists():
                copied += 1  # already exists

    log.info("Grid GeoJSON: %d files in docs/grid/", copied)


# ── 4. 全国概観PNG ─────────────────────────────────────────
def generate_japan_png(docs_dir: Path, prefs: list[str]):
    """全県のRGBA TIFをmerge → japan_{type}.png (~4000x4400)。"""
    from PIL import Image

    for stype in SCORE_TYPES:
        tifs = []
        for pref in prefs:
            p = PROJECT_ROOT / "output" / pref / f"score_{stype}_rgba.tif"
            if p.exists():
                tifs.append(p)

        if not tifs:
            continue

        log.info("Merging %d TIFs for japan_%s.png ...", len(tifs), stype)

        datasets = [rasterio.open(str(t)) for t in tifs]
        try:
            mosaic, mosaic_transform = merge(datasets)
        finally:
            for ds in datasets:
                ds.close()

        # (4, H, W) → PIL
        img = np.moveaxis(mosaic, 0, -1)
        pil_img = Image.fromarray(img, "RGBA")

        # Resize to ~4000px max for web
        max_dim = 4000
        if max(pil_img.size) > max_dim:
            ratio = max_dim / max(pil_img.size)
            pil_img = pil_img.resize(
                (int(pil_img.width * ratio), int(pil_img.height * ratio)),
                Image.LANCZOS,
            )

        out = docs_dir / f"japan_{stype}.png"
        pil_img.save(out, optimize=True)
        log.info("  -> %s (%dx%d, %.1f MB)",
                 out.name, pil_img.width, pil_img.height,
                 out.stat().st_size / 1e6)


# ── main ────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="全国風力ポテンシャル ドキュメント生成")
    parser.add_argument("-p", "--prefecture", default="all")
    parser.add_argument("--grid-only", action="store_true")
    parser.add_argument("--skip-japan-png", action="store_true",
                        help="全国merge PNGをスキップ (時間短縮)")
    args = parser.parse_args()

    docs_dir = PROJECT_ROOT / "docs"
    docs_dir.mkdir(parents=True, exist_ok=True)

    if args.grid_only:
        copy_grid_geojson(docs_dir)
        return

    # 対象県
    if args.prefecture == "all":
        output_root = PROJECT_ROOT / "output"
        prefs = [d.name for d in sorted(output_root.iterdir())
                 if d.is_dir() and (d / "score_total.tif").exists()]
    else:
        prefs = [p.strip() for p in args.prefecture.split(",")]

    log.info("Processing %d prefectures", len(prefs))

    # 2. 県別PNG (先に生成して実効 bounds を取得)
    png_bounds = generate_pref_pngs(docs_dir, prefs)

    # 1. prefectures.json (PNG の粗グリッド bounds で)
    generate_prefectures_json(docs_dir, prefs, png_bounds=png_bounds)

    # 3. Grid GeoJSON
    copy_grid_geojson(docs_dir)

    # 4. 全国概観PNG
    if not args.skip_japan_png:
        generate_japan_png(docs_dir, prefs)

    log.info("All done.")


if __name__ == "__main__":
    main()
