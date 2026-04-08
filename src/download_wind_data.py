#!/usr/bin/env python3
"""
ERA5 再解析データから風速を取得し、県別 GeoTIFF に変換する。

100m高さの年間平均風速を10年分 (2014-2023) ダウンロードし、
長期平均 GeoTIFF として出力する。

前提:
  - cdsapi パッケージがインストール済み
  - ~/.cdsapirc に CDS API キーが設定済み
    (https://cds.climate.copernicus.eu/ で登録)

使い方:
    python src/download_wind_data.py -p fukui
    python src/download_wind_data.py -p akita --years 2019-2023
"""

import argparse
import logging
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import PREFECTURES, get_wind_dir, get_pref_config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def download_era5_wind(bbox: tuple, years: list[int], output_nc: Path):
    """CDS API で ERA5 100m風速の月次平均をダウンロード。

    bbox: (west, south, east, north)
    """
    import cdsapi

    west, south, east, north = bbox
    # ERA5 の area は [north, west, south, east]
    area = [north + 0.5, west - 0.5, south - 0.5, east + 0.5]

    log.info("Downloading ERA5 100m wind speed for %s, years=%s", bbox, years)

    client = cdsapi.Client()
    client.retrieve(
        "reanalysis-era5-single-levels-monthly-means",
        {
            "product_type": "monthly_averaged_reanalysis",
            "variable": [
                "100m_u_component_of_wind",
                "100m_v_component_of_wind",
            ],
            "year": [str(y) for y in years],
            "month": [f"{m:02d}" for m in range(1, 13)],
            "time": "00:00",
            "area": area,
            "format": "netcdf",
        },
        str(output_nc),
    )
    log.info("Downloaded: %s (%.1f MB)", output_nc, output_nc.stat().st_size / 1e6)


def compute_mean_wind_speed(nc_path: Path) -> tuple:
    """NetCDF から年間平均風速を計算。

    Returns: (wind_speed_2d, lons, lats)
    """
    import xarray as xr

    ds = xr.open_dataset(nc_path)

    # u100, v100 コンポーネントから風速を計算
    # 変数名は ERA5 バージョンによって異なる場合がある
    u_var = None
    v_var = None
    for name in ds.data_vars:
        lower = name.lower()
        if "u100" in lower or "u_component" in lower:
            u_var = name
        elif "v100" in lower or "v_component" in lower:
            v_var = name

    if u_var is None or v_var is None:
        raise ValueError(f"風速変数が見つかりません。変数一覧: {list(ds.data_vars)}")

    log.info("Using variables: u=%s, v=%s", u_var, v_var)

    u = ds[u_var].values  # (time, lat, lon)
    v = ds[v_var].values

    # 風速 = sqrt(u^2 + v^2) の時間平均
    ws = np.sqrt(u**2 + v**2)
    mean_ws = np.nanmean(ws, axis=0)  # (lat, lon)

    # 座標
    lat_name = "latitude" if "latitude" in ds.coords else "lat"
    lon_name = "longitude" if "longitude" in ds.coords else "lon"
    lats = ds[lat_name].values
    lons = ds[lon_name].values

    ds.close()
    log.info("Mean wind speed: min=%.1f, max=%.1f m/s", np.nanmin(mean_ws), np.nanmax(mean_ws))
    return mean_ws, lons, lats


def save_wind_geotiff(wind_speed: np.ndarray, lons: np.ndarray, lats: np.ndarray,
                      output_path: Path):
    """風速データを GeoTIFF として保存。"""
    import rasterio
    from rasterio.transform import from_bounds

    # ERA5 は北→南の並びなので確認
    if lats[0] < lats[-1]:
        wind_speed = wind_speed[::-1]
        lats = lats[::-1]

    height, width = wind_speed.shape
    west = float(lons.min()) - 0.125  # ERA5 0.25° grid のピクセル中心を考慮
    east = float(lons.max()) + 0.125
    south = float(lats.min()) - 0.125
    north = float(lats.max()) + 0.125

    transform = from_bounds(west, south, east, north, width, height)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        output_path, "w",
        driver="GTiff",
        height=height,
        width=width,
        count=1,
        dtype="float32",
        crs="EPSG:4326",
        transform=transform,
        nodata=np.nan,
        compress="deflate",
    ) as dst:
        dst.write(wind_speed.astype(np.float32), 1)

    log.info("Wrote %s (%d x %d)", output_path, width, height)


def estimate_wind_from_elevation(pref: str, wind_dir: Path) -> bool:
    """ERA5 が使えない場合のフォールバック: 標高ベースの簡易推定。

    標高と緯度から大まかな風速を推定する。
    実際の解析にはERA5を使うべきだが、テスト用。
    """
    import rasterio
    from rasterio.transform import from_bounds

    land_dir = wind_dir.parent / "land"
    elev_path = land_dir / f"{pref}_elevation.tif"
    slope_path = land_dir / f"{pref}_slope.tif"

    # 標高データがあれば使う
    if elev_path.exists():
        ref_path = elev_path
    elif slope_path.exists():
        ref_path = slope_path
    else:
        log.warning("標高/傾斜データがありません。フォールバック不可。")
        return False

    log.info("標高ベースの風速推定 (フォールバック): %s", ref_path)

    with rasterio.open(ref_path) as ds:
        data = ds.read(1)
        transform = ds.transform
        crs = ds.crs
        height, width = data.shape
        bounds = ds.bounds

    if ref_path == elev_path:
        elev = np.nan_to_num(data, nan=0.0)
    else:
        # slope から推定: 高い傾斜 = 尾根 = やや高い風速と仮定
        elev = np.nan_to_num(data, nan=0.0) * 20  # 粗い推定

    # 簡易モデル: base_ws + elevation_bonus + latitude_bonus
    cfg = get_pref_config(pref)
    center_lat = cfg["center"][0]

    # 基本風速: 緯度に依存 (北ほど風が強い傾向)
    base_ws = 3.5 + (center_lat - 33.0) * 0.15  # 3.5-5.3 m/s

    # 標高ボーナス: 対数則的に増加
    elev_bonus = np.log1p(np.maximum(elev, 0) / 100) * 0.8

    # 風速推定 (3-9 m/s 程度)
    wind_speed = np.clip(base_ws + elev_bonus, 2.0, 12.0).astype(np.float32)

    # 水域や無効値はNaN
    wind_speed[data != data] = np.nan  # NaN propagation

    output_path = wind_dir / f"{pref}_wind_speed.tif"
    wind_dir.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        output_path, "w",
        driver="GTiff",
        height=height, width=width, count=1,
        dtype="float32", crs=crs, transform=transform,
        nodata=np.nan, compress="deflate",
    ) as dst:
        dst.write(wind_speed, 1)

    log.warning("推定風速を出力しました (テスト用): %s", output_path)
    log.warning("本番ではERA5データを使用してください。")
    return True


def main():
    parser = argparse.ArgumentParser(description="ERA5 風速データ取得")
    parser.add_argument("-p", "--prefecture", required=True,
                        choices=list(PREFECTURES.keys()))
    parser.add_argument("--years", default="2014-2023",
                        help="取得年範囲 (例: 2019-2023)")
    parser.add_argument("--fallback", action="store_true",
                        help="ERA5が使えない場合、標高ベースの推定を使用")
    args = parser.parse_args()

    pref = args.prefecture
    cfg = get_pref_config(pref)
    name_ja = cfg["name_ja"]
    wind_dir = get_wind_dir(pref)
    wind_dir.mkdir(parents=True, exist_ok=True)

    output_tif = wind_dir / f"{pref}_wind_speed.tif"
    if output_tif.exists():
        log.info("[SKIP] %s already exists", output_tif)
        return

    # 年範囲パース
    start, end = args.years.split("-")
    years = list(range(int(start), int(end) + 1))

    print("=" * 60)
    print(f"{name_ja} ERA5 風速データ取得")
    print(f"  年範囲: {years[0]}-{years[-1]} ({len(years)}年)")
    print("=" * 60)

    bbox = cfg["bbox"]
    nc_path = wind_dir / f"{pref}_era5_wind.nc"

    try:
        if not nc_path.exists():
            download_era5_wind(bbox, years, nc_path)

        mean_ws, lons, lats = compute_mean_wind_speed(nc_path)
        save_wind_geotiff(mean_ws, lons, lats, output_tif)
        log.info("完了!")

    except Exception as e:
        log.error("ERA5ダウンロード失敗: %s", e)
        if args.fallback:
            log.info("フォールバック: 標高ベースの風速推定を使用します。")
            if estimate_wind_from_elevation(pref, wind_dir):
                log.info("フォールバック完了。")
            else:
                log.error("フォールバックも失敗しました。")
                sys.exit(1)
        else:
            log.info("--fallback オプションで標高ベースの推定を使用できます。")
            sys.exit(1)


if __name__ == "__main__":
    main()
