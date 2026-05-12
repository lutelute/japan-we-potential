#!/usr/bin/env python3
"""
高解像度 XYZ タイル生成 — 各県 TIF から直接タイルを生成してマージ

【問題の背景】
gen_tiles_wind.py は全県 PNG を 333m/px キャンバスに合成してからタイル化するため、
z=9 で止めないと GitHub Pages サイズ制限に当たる。

【このスクリプトのアプローチ】
output/{pref}/score_{mode}_rgba.tif (30m 解像度の GeoTIFF) から直接
gdal2tiles で z=7-14 のタイルを生成し、docs/tiles/{mode}/ に上書きマージする。

結果: z=5-9 は gen_tiles_wind.py 由来の全国統合タイル (軽量)
      z=10-14 は各県 TIF 由来の高解像度タイル (詳細)
  → Leaflet の maxNativeZoom=14 で z=14 まで鮮明に表示可能

⚠️  実行推奨環境: pws-160core (80C/160T) または pws-gpu (24T)
     このMacで実行すると数時間の高CPU負荷になります。

Usage:
    python gen_hires_tiles.py                   # 全県・全モード z=10-14
    python gen_hires_tiles.py --mode total      # total のみ
    python gen_hires_tiles.py --pref fukui      # 1 県のみ (テスト用)
    python gen_hires_tiles.py --zoom 10-13      # ズーム指定
    python gen_hires_tiles.py --workers 8       # 並列数 (pws-160core 推奨: 16+)
"""
import argparse
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, "src")
from config import PREFECTURES

MODES = [
    "total", "wind_speed", "slope", "elevation",
    "grid_dist", "sub_dist", "land_use", "residential_dist",
]
OUTPUT = Path("output")
TILES = Path("docs/tiles")


def tif_to_tiles(pref: str, mode: str, z_min: int, z_max: int) -> tuple[str, bool]:
    """1 県・1 モードのタイルを生成して docs/tiles/{mode}/ にマージ"""
    suffix = "" if mode == "total" else f"_{mode}"
    tif_hi = OUTPUT / pref / f"score_{mode}_5m_rgba.tif"
    tif_lo = OUTPUT / pref / f"score_{mode}_rgba.tif"
    tif = tif_hi if tif_hi.exists() else tif_lo if tif_lo.exists() else None

    if tif is None:
        return f"{pref}/{mode}", False

    tmp_dir = Path(f"/tmp/tiles_wind_{pref}_{mode}")
    try:
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir)
        tmp_dir.mkdir(parents=True)

        r = subprocess.run(
            [
                "gdal2tiles.py", "--xyz",
                "-z", f"{z_min}-{z_max}",
                "-r", "near",
                "--processes=2", "--no-kml",
                str(tif), str(tmp_dir),
            ],
            capture_output=True, text=True,
        )
        if r.returncode != 0:
            return f"{pref}/{mode}", False

        # HTML ファイル削除
        for f in tmp_dir.glob("*.html"):
            f.unlink()

        # docs/tiles/{mode}/ にマージ (上書き)
        out_dir = TILES / mode
        out_dir.mkdir(parents=True, exist_ok=True)
        for z_dir in tmp_dir.iterdir():
            if not z_dir.is_dir():
                continue
            for x_dir in z_dir.iterdir():
                dest_x = out_dir / z_dir.name / x_dir.name
                dest_x.mkdir(parents=True, exist_ok=True)
                for tile in x_dir.iterdir():
                    shutil.copy2(tile, dest_x / tile.name)

        n = sum(1 for _ in tmp_dir.rglob("*.png"))
        return f"{pref}/{mode} ({n} tiles)", True

    except Exception as e:
        return f"{pref}/{mode} ERROR: {e}", False
    finally:
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir, ignore_errors=True)


def main():
    parser = argparse.ArgumentParser(description="高解像度タイル生成 (pws-160core 推奨)")
    parser.add_argument("--mode", choices=MODES + ["all"], default="all")
    parser.add_argument("--pref", default=None, help="1 県のみ (省略時は全県)")
    parser.add_argument(
        "--zoom", default="10-14",
        help="ズームレベル (デフォルト: 10-14。z=5-9 は gen_tiles_wind.py で生成)",
    )
    parser.add_argument("--workers", type=int, default=4,
                        help="並列数 (pws-160core では 16 以上推奨)")
    args = parser.parse_args()

    z_min, z_max = map(int, args.zoom.split("-"))
    modes = MODES if args.mode == "all" else [args.mode]
    prefs = [args.pref] if args.pref else list(PREFECTURES.keys())

    tasks = [(p, m) for p in prefs for m in modes]
    print(f"タスク数: {len(tasks)} ({len(prefs)} 県 × {len(modes)} モード)")
    print(f"ズーム: z={z_min}-{z_max} | 並列: {args.workers} workers")
    print("⚠️  pws-160core での実行を推奨（このMacはCPU高負荷になります）")
    print()

    ok = fail = 0
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(tif_to_tiles, p, m, z_min, z_max): (p, m)
                   for p, m in tasks}
        for future in as_completed(futures):
            msg, success = future.result()
            if success:
                ok += 1
                print(f"  OK {msg}")
            else:
                fail += 1
                print(f"  SKIP {msg}")

    total_mb = sum(f.stat().st_size for f in TILES.rglob("*.png")) / 1e6
    print(f"\n=== 完了: {ok} OK, {fail} SKIP")
    print(f"    docs/tiles/ 総計: {total_mb:.0f}MB")
    print(f"    maxNativeZoom を {z_max} に変更してください (docs/index.html)")


if __name__ == "__main__":
    main()
