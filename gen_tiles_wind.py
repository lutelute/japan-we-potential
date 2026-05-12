#!/usr/bin/env python3
"""
全国風力適地スコア XYZタイル生成

docs/{pref}_{mode}.png を大面積→小面積順に合成 → GeoTIFF → XYZタイル

改善点 (REプロジェクト gen_tiles.py 比):
  - 全8モード対応 (wind_speed / residential_dist を追加)
  - 全国bbox を正確な日本領土に拡張 (沖縄・北海道含む)
  - gen_score_pngs_wind.py を先に実行してから使用

デフォルト設定:
  - z=5-9 (REプロジェクトと同じ。z=10+ は GitHub Pages サイズ超過のため)
  - RES = 0.003 °/px (333m/px, REと同一。0.0017 にすると7倍重くなる)
  - 高解像度版: --zoom 6-13 (pws-160coreで生成推奨)

Usage:
    python gen_tiles_wind.py               # 全モード z=5-9 (GitHub Pages用)
    python gen_tiles_wind.py --mode total  # totalのみ
    python gen_tiles_wind.py --zoom 6-13   # 高解像度版 (pws-160core推奨)
"""
import argparse
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image

DOCS = Path("docs")
WORK = Path("tmp_geo_wind")
TILES = DOCS / "tiles"
PREFS_JSON = DOCS / "prefectures.json"

# 日本全域 bounds (沖縄〜北海道)
JAPAN = dict(south=24.0, west=122.8, north=45.7, east=154.0)

# 約 333m/pixel (japan-re-potential と同一設定)
# 0.0017 に下げると 7倍重くなり GitHub Pages への push でタイムアウト
RES = 0.003  # °/px

MODES = [
    "total", "wind_speed", "slope", "elevation",
    "grid_dist", "sub_dist", "land_use", "residential_dist",
]


def mode_suffix(mode: str) -> str:
    return "" if mode == "total" else f"_{mode}"


def composite_prefs(mode: str, prefs: dict) -> np.ndarray:
    """都道府県PNGを大面積→小面積順でキャンバスに重ね合わせ合成"""
    suffix = mode_suffix(mode)
    CW = int((JAPAN["east"] - JAPAN["west"]) / RES)
    CH = int((JAPAN["north"] - JAPAN["south"]) / RES)
    canvas = np.zeros((CH, CW, 4), dtype=np.uint8)

    def area(key):
        b = prefs[key]["bounds"]
        return (b[1][0] - b[0][0]) * (b[1][1] - b[0][1])

    keys = sorted(
        [k for k in prefs if (DOCS / f"{k}{suffix}.png").exists()],
        key=area, reverse=True,
    )

    if not keys:
        print(f"  SKIP {mode}: PNG なし (gen_score_pngs_wind.py を先に実行してください)")
        return canvas

    for key in keys:
        b = prefs[key]["bounds"]
        s, w, n, e = b[0][0], b[0][1], b[1][0], b[1][1]

        r0 = max(0, int((JAPAN["north"] - n) / RES))
        r1 = min(CH, int((JAPAN["north"] - s) / RES))
        c0 = max(0, int((w - JAPAN["west"]) / RES))
        c1 = min(CW, int((e - JAPAN["west"]) / RES))
        if r1 <= r0 or c1 <= c0:
            continue

        img = Image.open(DOCS / f"{key}{suffix}.png").convert("RGBA")
        patch = np.array(img.resize((c1 - c0, r1 - r0), Image.LANCZOS))
        canvas[r0:r1, c0:c1] = patch

    print(f"  合成: {len(keys)} 県, canvas {CW}x{CH}px")
    return canvas


def canvas_to_tiles(canvas: np.ndarray, mode: str, z_min: int, z_max: int) -> int:
    """キャンバスPNG → GeoTIFF → XYZタイル"""
    WORK.mkdir(exist_ok=True)
    comp_png = WORK / f"{mode}.png"
    comp_tif = WORK / f"{mode}.tif"

    Image.fromarray(canvas, "RGBA").save(comp_png)

    r = subprocess.run(
        [
            "gdal_translate", "-a_srs", "EPSG:4326",
            "-a_ullr",
            str(JAPAN["west"]), str(JAPAN["north"]),
            str(JAPAN["east"]), str(JAPAN["south"]),
            "-of", "GTiff", str(comp_png), str(comp_tif),
        ],
        capture_output=True, text=True,
    )
    if r.returncode != 0:
        print(f"  gdal_translate エラー: {r.stderr[:200]}")
        return 0

    out_dir = TILES / mode
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    r = subprocess.run(
        [
            "gdal2tiles.py", "--xyz",
            "-z", f"{z_min}-{z_max}",
            "-r", "bilinear",
            "--processes=4", "--no-kml",
            str(comp_tif), str(out_dir),
        ],
        text=True,
    )
    if r.returncode != 0:
        print(f"  gdal2tiles エラー (mode={mode})")
        return 0

    for f in out_dir.glob("*.html"):
        f.unlink()

    n = sum(1 for _ in out_dir.rglob("*.png"))
    mb = sum(f.stat().st_size for f in out_dir.rglob("*.png")) / 1e6
    print(f"  完了: {n} タイル, {mb:.1f}MB → tiles/{mode}/")
    return n


def main():
    parser = argparse.ArgumentParser(description="全国風力スコア XYZタイル生成")
    parser.add_argument("--mode", choices=MODES + ["all"], default="all")
    parser.add_argument(
        "--zoom", default="5-9",
        help="ズームレベル範囲 (例: 5-9). デフォルト 5-9 (GitHub Pages用)。高解像度は 6-13 (pws-160core推奨)",
    )
    args = parser.parse_args()

    z_min, z_max = map(int, args.zoom.split("-"))

    if not PREFS_JSON.exists():
        print("ERROR: docs/prefectures.json が見つかりません。")
        print("       先に gen_score_pngs_wind.py を実行してください。")
        return

    prefs = json.loads(PREFS_JSON.read_text(encoding="utf-8"))
    TILES.mkdir(parents=True, exist_ok=True)

    modes = MODES if args.mode == "all" else [args.mode]

    total_tiles = 0
    for mode in modes:
        print(f"\n── {mode} ──────────────────────────────")
        canvas = composite_prefs(mode, prefs)
        n = canvas_to_tiles(canvas, mode, z_min, z_max)
        total_tiles += n

    shutil.rmtree(WORK, ignore_errors=True)

    total_mb = sum(f.stat().st_size for f in TILES.rglob("*.png")) / 1e6
    print(f"\n=== 完了: {total_tiles} タイル追加, 総計 {total_mb:.1f}MB (z={z_min}-{z_max}) ===")
    print(f"    → docs/index.html の maxNativeZoom を {z_max} に設定してください")


if __name__ == "__main__":
    main()
