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
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import rasterio
from rasterio.merge import merge

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import PREFECTURES, PROJECT_ROOT

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
def generate_prefectures_json(docs_dir: Path, prefs: list[str]):
    """config.py bboxからprefectures.jsonを生成。
    TIF boundsはSRTMタイル単位で大きいので、config bboxを使う。
    """
    data = {}
    for pref in prefs:
        cfg = PREFECTURES[pref]
        bbox = cfg["bbox"]  # (west, south, east, north)
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


# ── 2. 県別PNG (原寸) ──────────────────────────────────────
def generate_pref_pngs(docs_dir: Path, prefs: list[str]):
    """各県の score_total_rgba.tif → {pref}.png (config bboxでクリップ)。"""
    from PIL import Image

    for pref in prefs:
        output_dir = PROJECT_ROOT / "output" / pref
        rgba_tif = output_dir / "score_total_rgba.tif"
        if not rgba_tif.exists():
            continue

        cfg = PREFECTURES[pref]
        bbox = cfg["bbox"]  # (west, south, east, north)

        png_path = docs_dir / f"{pref}.png"
        _tif_to_png(rgba_tif, png_path, clip_bbox=bbox)

        pref_dir = docs_dir / pref
        pref_dir.mkdir(parents=True, exist_ok=True)
        for stype in SCORE_TYPES:
            src = output_dir / f"score_{stype}_rgba.tif"
            dst = pref_dir / f"score_{stype}.png"
            if src.exists():
                _tif_to_png(src, dst, clip_bbox=bbox)

    log.info("Prefecture PNGs: %d done", len(prefs))


def _tif_to_png(tif_path: Path, png_path: Path, max_dim: int = 0,
                clip_bbox: tuple = None):
    """RGBA GeoTIFF → PNG。clip_bboxでピクセル範囲をクリップ。"""
    from PIL import Image
    from rasterio.windows import from_bounds

    with rasterio.open(tif_path) as ds:
        if clip_bbox:
            west, south, east, north = clip_bbox
            window = from_bounds(west, south, east, north, ds.transform)
            # 整数ピクセルに丸める
            row_off = max(0, int(window.row_off))
            col_off = max(0, int(window.col_off))
            win_h = min(int(window.height), ds.height - row_off)
            win_w = min(int(window.width), ds.width - col_off)
            from rasterio.windows import Window
            window = Window(col_off, row_off, win_w, win_h)
            rgba = ds.read(window=window)
        else:
            rgba = ds.read()

    img = np.moveaxis(rgba, 0, -1)
    pil_img = Image.fromarray(img, "RGBA")

    if max_dim > 0 and max(pil_img.size) > max_dim:
        ratio = max_dim / max(pil_img.size)
        pil_img = pil_img.resize(
            (int(pil_img.width * ratio), int(pil_img.height * ratio)),
            Image.LANCZOS,
        )

    png_path.parent.mkdir(parents=True, exist_ok=True)
    pil_img.save(png_path, optimize=True)


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

    # 1. prefectures.json
    generate_prefectures_json(docs_dir, prefs)

    # 2. 県別PNG
    generate_pref_pngs(docs_dir, prefs)

    # 3. Grid GeoJSON
    copy_grid_geojson(docs_dir)

    # 4. 全国概観PNG
    if not args.skip_japan_png:
        generate_japan_png(docs_dir, prefs)

    log.info("All done.")


if __name__ == "__main__":
    main()
