#!/usr/bin/env python3
"""
ERA5 100m風速データ — 日本全域一括ダウンロード + 県別クリップ

1つのNetCDFで日本全域をダウンロードし、各県のbboxでクリップして
data/{pref}/wind/{pref}_wind_speed.tif に出力する。

Usage:
    python src/download_wind_data_bulk.py              # 全県
    python src/download_wind_data_bulk.py -p fukui     # 特定県のみクリップ
    python src/download_wind_data_bulk.py --skip-download  # DL済みならクリップのみ
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import PREFECTURES, PROJECT_ROOT, get_wind_dir

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# 日本全域 bbox (余裕を持って)
JAPAN_BBOX = {
    "north": 46.0,
    "south": 24.0,
    "west": 122.0,
    "east": 146.0,
}

BULK_NC = PROJECT_ROOT / "data" / "era5_japan_wind_100m.nc"
YEARS = list(range(2014, 2024))


def download_era5_japan(output_nc: Path):
    """CDS APIで日本全域の100m風速月次平均をダウンロード。"""
    import cdsapi

    if output_nc.exists():
        log.info("Already downloaded: %s (%.1f MB)", output_nc, output_nc.stat().st_size / 1e6)
        return

    output_nc.parent.mkdir(parents=True, exist_ok=True)

    log.info("Downloading ERA5 100m wind for Japan (%d-%d) ...", YEARS[0], YEARS[-1])
    log.info("  bbox: N%.1f S%.1f W%.1f E%.1f",
             JAPAN_BBOX["north"], JAPAN_BBOX["south"], JAPAN_BBOX["west"], JAPAN_BBOX["east"])

    client = cdsapi.Client()
    client.retrieve(
        "reanalysis-era5-single-levels-monthly-means",
        {
            "product_type": ["monthly_averaged_reanalysis"],
            "variable": [
                "100m_u_component_of_wind",
                "100m_v_component_of_wind",
            ],
            "year": [str(y) for y in YEARS],
            "month": [f"{m:02d}" for m in range(1, 13)],
            "time": ["00:00"],
            "area": [JAPAN_BBOX["north"], JAPAN_BBOX["west"],
                     JAPAN_BBOX["south"], JAPAN_BBOX["east"]],
            "data_format": "netcdf",
        },
        str(output_nc),
    )

    log.info("Downloaded: %s (%.1f MB)", output_nc, output_nc.stat().st_size / 1e6)


def compute_mean_wind_speed(nc_path: Path):
    """NetCDFからu100/v100の年間平均風速を計算。"""
    import xarray as xr

    log.info("Computing mean wind speed from %s ...", nc_path)
    ds = xr.open_dataset(nc_path)

    # 変数名を自動検出
    u_var = v_var = None
    for name in ds.data_vars:
        lower = name.lower()
        if "u100" in lower or ("u_component" in lower and "100" in lower) or name == "u100":
            u_var = name
        elif "v100" in lower or ("v_component" in lower and "100" in lower) or name == "v100":
            v_var = name

    if u_var is None or v_var is None:
        # フォールバック: 最初の2変数を使う
        vars_list = list(ds.data_vars)
        log.warning("Could not auto-detect u/v vars from %s, trying %s", vars_list, vars_list[:2])
        u_var, v_var = vars_list[0], vars_list[1]

    log.info("  u=%s, v=%s", u_var, v_var)

    u = ds[u_var].values  # (time, lat, lon)
    v = ds[v_var].values
    ws = np.sqrt(u**2 + v**2)
    mean_ws = np.nanmean(ws, axis=0)  # (lat, lon)

    lat_name = "latitude" if "latitude" in ds.coords else "lat"
    lon_name = "longitude" if "longitude" in ds.coords else "lon"
    lats = ds[lat_name].values
    lons = ds[lon_name].values

    ds.close()

    log.info("  Mean wind speed: %.1f - %.1f m/s (mean %.1f)",
             np.nanmin(mean_ws), np.nanmax(mean_ws), np.nanmean(mean_ws))
    return mean_ws, lons, lats


def clip_and_save(mean_ws: np.ndarray, lons: np.ndarray, lats: np.ndarray,
                  pref: str) -> bool:
    """県のbboxでクリップしてGeoTIFFに保存。"""
    import rasterio
    from rasterio.transform import from_bounds

    cfg = PREFECTURES[pref]
    bbox = cfg["bbox"]  # (west, south, east, north)
    wind_dir = get_wind_dir(pref)
    wind_dir.mkdir(parents=True, exist_ok=True)
    output_path = wind_dir / f"{pref}_wind_speed.tif"

    # ERA5は北→南の並び確認
    if lats[0] < lats[-1]:
        mean_ws = mean_ws[::-1]
        lats = lats[::-1]

    # bboxでクリップ
    margin = 0.5  # ERA5グリッド間隔分の余裕
    lat_mask = (lats >= bbox[1] - margin) & (lats <= bbox[3] + margin)
    lon_mask = (lons >= bbox[0] - margin) & (lons <= bbox[2] + margin)

    if not lat_mask.any() or not lon_mask.any():
        log.warning("  %s: no data in bbox, skipping", pref)
        return False

    clipped = mean_ws[np.ix_(lat_mask, lon_mask)]
    clipped_lats = lats[lat_mask]
    clipped_lons = lons[lon_mask]

    height, width = clipped.shape
    west = float(clipped_lons.min()) - 0.125
    east = float(clipped_lons.max()) + 0.125
    south = float(clipped_lats.min()) - 0.125
    north = float(clipped_lats.max()) + 0.125

    transform = from_bounds(west, south, east, north, width, height)

    with rasterio.open(
        output_path, "w", driver="GTiff",
        height=height, width=width, count=1,
        dtype="float32", crs="EPSG:4326",
        transform=transform, nodata=np.nan,
        compress="deflate",
    ) as dst:
        dst.write(clipped.astype(np.float32), 1)

    ws_min = np.nanmin(clipped)
    ws_max = np.nanmax(clipped)
    ws_mean = np.nanmean(clipped)
    log.info("  %s: %.1f-%.1f m/s (mean %.1f) -> %s",
             pref, ws_min, ws_max, ws_mean, output_path)
    return True


def main():
    parser = argparse.ArgumentParser(description="ERA5 風速一括ダウンロード + 県別クリップ")
    parser.add_argument("-p", "--prefecture", default="all")
    parser.add_argument("--skip-download", action="store_true",
                        help="ダウンロード済みならクリップのみ実行")
    parser.add_argument("--force", action="store_true",
                        help="既存TIFを上書き")
    args = parser.parse_args()

    # 1. ダウンロード
    if not args.skip_download:
        download_era5_japan(BULK_NC)

    if not BULK_NC.exists():
        log.error("NetCDF not found: %s", BULK_NC)
        log.error("Run without --skip-download first.")
        sys.exit(1)

    # 2. 平均風速計算
    mean_ws, lons, lats = compute_mean_wind_speed(BULK_NC)

    # 3. 県別クリップ
    if args.prefecture == "all":
        prefs = list(PREFECTURES.keys())
    else:
        prefs = [p.strip() for p in args.prefecture.split(",")]

    ok = 0
    for pref in prefs:
        wind_tif = get_wind_dir(pref) / f"{pref}_wind_speed.tif"
        if wind_tif.exists() and not args.force:
            log.info("  [SKIP] %s (use --force to overwrite)", pref)
            ok += 1
            continue
        if clip_and_save(mean_ws, lons, lats, pref):
            ok += 1

    log.info("Done: %d/%d prefectures", ok, len(prefs))


if __name__ == "__main__":
    main()
