#!/usr/bin/env python3
"""
各都道府県スコア RGBA TIF → PNG + docs/prefectures.json 更新 (風力版)

output/{pref}/score_{mode}_rgba.tif → docs/{pref}_{mode}.png
totalモードのとき docs/prefectures.json のbounds情報も更新する。

Usage:
    python gen_score_pngs_wind.py               # 全県・全モード
    python gen_score_pngs_wind.py --mode total  # totalのみ
    python gen_score_pngs_wind.py --pref fukui  # 福井県のみ
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import rasterio
from PIL import Image
from rasterio.enums import Resampling

sys.path.insert(0, "src")
from config import PREFECTURES

MODES = [
    "total", "wind_speed", "slope", "elevation",
    "grid_dist", "sub_dist", "land_use", "residential_dist",
]
MAX_PX = 2000
DOCS = Path("docs")
OUTPUT = Path("output")


def tif_to_png(pref: str, mode: str, bounds_data: dict) -> bool:
    suffix = "" if mode == "total" else f"_{mode}"
    tif_hi = OUTPUT / pref / f"score_{mode}_5m_rgba.tif"
    tif_lo = OUTPUT / pref / f"score_{mode}_rgba.tif"
    tif = tif_hi if tif_hi.exists() else tif_lo if tif_lo.exists() else None

    if tif is None:
        print(f"  SKIP {pref}/{mode}: RGBA TIF なし")
        return False

    try:
        with rasterio.open(tif) as ds:
            b = ds.bounds
            if mode == "total":
                cfg = PREFECTURES[pref]
                bounds_data[pref] = {
                    "name_ja": cfg["name_ja"],
                    "bounds": [[b.bottom, b.left], [b.top, b.right]],
                    "center": cfg["center"],
                    "grid_area": cfg["grid_area"],
                }
            h, w = ds.height, ds.width
            scale = MAX_PX / w if w > MAX_PX else 1.0
            new_w = min(w, MAX_PX)
            new_h = int(h * scale)

            data = ds.read(
                out_shape=(4, new_h, new_w),
                resampling=Resampling.nearest,
            )

        # アルファをバイナリ化（半透明フリンジを防止）
        alpha = data[3]
        alpha[alpha > 0] = 255
        data[3] = alpha

        rgba = np.moveaxis(data, 0, -1)
        img = Image.fromarray(rgba, "RGBA")
        # 6色パレット量子化で圧縮
        img = img.quantize(colors=6, method=Image.Quantize.FASTOCTREE).convert("RGBA")

        png_path = DOCS / f"{pref}{suffix}.png"
        img.save(png_path, optimize=True)
        print(f"  OK {pref}/{mode}: {new_w}x{new_h} -> {png_path.stat().st_size//1024}KB")
        return True

    except Exception as e:
        print(f"  ERROR {pref}/{mode}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="スコアTIF→PNG変換 (風力版)")
    parser.add_argument("--mode", choices=MODES, default=None,
                        help="変換するモード（省略時は全モード）")
    parser.add_argument("--pref", default=None,
                        help="対象県キー（省略時は全県）")
    args = parser.parse_args()

    DOCS.mkdir(exist_ok=True)

    modes = [args.mode] if args.mode else MODES
    prefs = [args.pref] if args.pref else list(PREFECTURES.keys())

    # 既存のprefectures.jsonを読み込んでboundsを保持
    pref_json = DOCS / "prefectures.json"
    bounds_data: dict = {}
    if pref_json.exists():
        bounds_data = json.loads(pref_json.read_text(encoding="utf-8"))

    ok = 0
    for pref in prefs:
        for mode in modes:
            if tif_to_png(pref, mode, bounds_data):
                ok += 1

    pref_json.write_text(
        json.dumps(bounds_data, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    total_mb = sum(f.stat().st_size for f in DOCS.glob("*.png")) / 1024 / 1024
    print(f"\n完了: {ok} PNG, prefectures.json ({len(bounds_data)}県), {total_mb:.1f} MB")


if __name__ == "__main__":
    main()
