"""
DEM傾斜解析スクリプト（県別対応）
SRTM 1arc-second (~30m) DEMデータから傾斜角を計算。

参考プロジェクトに slope TIF があればスキップ。

使い方:
    python src/slope_analysis.py -p fukui
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import geopandas as gpd
import rasterio
from rasterio.transform import from_bounds
from rasterio.features import geometry_mask
from shapely.ops import unary_union

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import PREFECTURES, get_land_dir, get_pref_config

SRTM1_SIZE = 3601
SLOPE_THRESHOLD_DEG = 20.0  # 風力: 20度以上は不適

SLOPE_CLASSES = [
    (0, 5, "0-5度（平坦）"),
    (5, 10, "5-10度（緩傾斜）"),
    (10, 20, "10-20度（中傾斜）"),
    (20, 30, "20-30度（急傾斜）"),
    (30, 90, "30度以上（険峻）"),
]


def read_srtm_hgt(filepath: Path) -> np.ndarray:
    data = np.fromfile(filepath, dtype=">i2").reshape(SRTM1_SIZE, SRTM1_SIZE)
    return data.astype(np.float32)


def tile_bounds(tile_name: str) -> tuple:
    lat = int(tile_name[1:3])
    lon = int(tile_name[4:7])
    if tile_name[0] == "S":
        lat = -lat
    if tile_name[3] == "W":
        lon = -lon
    return (lon, lat, lon + 1, lat + 1)


def mosaic_srtm(dem_dir: Path, srtm_tiles: list) -> tuple:
    all_bounds = [tile_bounds(t) for t in srtm_tiles]
    min_lon = min(b[0] for b in all_bounds)
    min_lat = min(b[1] for b in all_bounds)
    max_lon = max(b[2] for b in all_bounds)
    max_lat = max(b[3] for b in all_bounds)

    lon_tiles = sorted(set(b[0] for b in all_bounds))
    lat_tiles = sorted(set(b[1] for b in all_bounds), reverse=True)

    n_lat = len(lat_tiles)
    n_lon = len(lon_tiles)

    tile_pixels = SRTM1_SIZE - 1
    total_rows = n_lat * tile_pixels + 1
    total_cols = n_lon * tile_pixels + 1
    mosaic = np.full((total_rows, total_cols), np.nan, dtype=np.float32)

    for tile_name in srtm_tiles:
        filepath = dem_dir / f"{tile_name}.hgt"
        if not filepath.exists():
            print(f"  WARNING: {filepath} が見つかりません。スキップします。")
            continue
        data = read_srtm_hgt(filepath)
        data[data == -32768] = np.nan
        bounds = tile_bounds(tile_name)
        col_idx = lon_tiles.index(bounds[0])
        row_idx = lat_tiles.index(bounds[1])
        r0 = row_idx * tile_pixels
        c0 = col_idx * tile_pixels
        mosaic[r0 : r0 + SRTM1_SIZE, c0 : c0 + SRTM1_SIZE] = data

    transform = from_bounds(min_lon, min_lat, max_lon, max_lat, total_cols, total_rows)
    return mosaic, transform, "EPSG:4326"


def compute_slope(dem: np.ndarray, transform) -> np.ndarray:
    res_x = transform.a
    res_y = abs(transform.e)
    center_lat = transform.f + transform.e * (dem.shape[0] / 2)
    lat_rad = np.radians(center_lat)
    m_per_deg_lat = 111_320.0
    m_per_deg_lon = 111_320.0 * np.cos(lat_rad)
    cell_size_x = res_x * m_per_deg_lon
    cell_size_y = res_y * m_per_deg_lat
    dy, dx = np.gradient(dem, cell_size_y, cell_size_x)
    slope_rad = np.arctan(np.sqrt(dx**2 + dy**2))
    return np.degrees(slope_rad)


def clip_to_prefecture(data, transform, crs, pref, land_dir):
    cfg = get_pref_config(pref)
    admin_dir = land_dir / "admin_boundary"
    if not admin_dir.exists():
        print(f"WARNING: 境界ファイルが見つかりません。")
        return data, transform

    shp_files = list(admin_dir.rglob("*.shp"))
    if not shp_files:
        return data, transform

    gdf = gpd.read_file(shp_files[0])
    if gdf.crs and str(gdf.crs) != crs:
        gdf = gdf.to_crs(crs)

    pref_geom = unary_union(gdf.geometry)
    pref_bounds = pref_geom.bounds
    inv_transform = ~transform
    col_min, row_max = inv_transform * (pref_bounds[0], pref_bounds[1])
    col_max, row_min = inv_transform * (pref_bounds[2], pref_bounds[3])
    row_min = max(0, int(np.floor(row_min)) - 1)
    row_max = min(data.shape[0], int(np.ceil(row_max)) + 1)
    col_min = max(0, int(np.floor(col_min)) - 1)
    col_max = min(data.shape[1], int(np.ceil(col_max)) + 1)

    clipped = data[row_min:row_max, col_min:col_max].copy()
    new_transform = transform * rasterio.Affine.translation(col_min, row_min)
    mask = geometry_mask([pref_geom], out_shape=clipped.shape, transform=new_transform, invert=False)
    clipped[mask] = np.nan
    return clipped, new_transform


def save_geotiff(data, transform, crs, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        output_path, "w", driver="GTiff",
        height=data.shape[0], width=data.shape[1], count=1,
        dtype="float32", crs=crs, transform=transform,
        nodata=np.nan, compress="deflate",
    ) as dst:
        dst.write(data, 1)
    print(f"  保存完了: {output_path} ({data.shape[1]} x {data.shape[0]})")


def main():
    parser = argparse.ArgumentParser(description="DEM傾斜解析 (風力版)")
    parser.add_argument("-p", "--prefecture", default="fukui", choices=list(PREFECTURES.keys()))
    args = parser.parse_args()

    pref = args.prefecture
    cfg = get_pref_config(pref)
    name_ja = cfg["name_ja"]
    land_dir = get_land_dir(pref)
    output_path = land_dir / f"{pref}_slope.tif"

    # 既にあればスキップ
    if output_path.exists():
        print(f"[SKIP] {output_path} already exists")
        return

    dem_dir = land_dir / "dem"
    srtm_tiles = cfg["srtm_tiles"]

    print("=" * 60)
    print(f"{name_ja} DEM傾斜解析 (風力版)")
    print("=" * 60)

    missing = [t for t in srtm_tiles if not (dem_dir / f"{t}.hgt").exists()]
    if missing:
        print(f"ERROR: DEMファイル不足: {missing}")
        sys.exit(1)

    dem, transform, crs = mosaic_srtm(dem_dir, srtm_tiles)
    slope = compute_slope(dem, transform)
    slope_clipped, clip_transform = clip_to_prefecture(slope, transform, crs, pref, land_dir)
    save_geotiff(slope_clipped, clip_transform, crs, output_path)

    # 標高 TIF も保存 (風速補正に使用)
    elev_path = land_dir / f"{pref}_elevation.tif"
    if not elev_path.exists():
        elev_clipped, _ = clip_to_prefecture(dem, transform, crs, pref, land_dir)
        save_geotiff(elev_clipped, clip_transform, crs, elev_path)

    print("\n完了!")


if __name__ == "__main__":
    main()
