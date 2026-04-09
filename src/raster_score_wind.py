#!/usr/bin/env python3
"""
風力発電ポテンシャル ラスタースコア計算

GIS-MCDA (AHP) 手法で、風力発電の適地スコアを計算する。
太陽光版 (japan-re-potential) との主な違い:
  - 風速が最重要評価基準 (30%)
  - 居住地距離による騒音バッファ (8%)
  - 標高は中程度で最適 (尾根風)、高すぎると不利
  - 傾斜しきい値が太陽光より緩い

Usage:
    python src/raster_score_wind.py -p fukui
    python src/raster_score_wind.py -p akita --resolution 10
"""

import argparse
import logging
import shutil
import subprocess
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.features import geometry_mask, rasterize
from rasterio.merge import merge
from rasterio.transform import from_bounds
from rasterio.warp import reproject
from scipy.ndimage import distance_transform_edt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import PREFECTURES, WEIGHTS, PROJECT_ROOT

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── カラーマップ (スコア -> RGBA) ────────────────────────────
COLOUR_BANDS = [
    ((1, 20), (220, 20, 60, 180)),     # crimson
    ((20, 40), (255, 140, 0, 180)),     # darkorange
    ((40, 60), (218, 165, 32, 180)),    # goldenrod
    ((60, 80), (34, 139, 34, 180)),     # forestgreen
    ((80, 101), (0, 100, 0, 180)),      # darkgreen
]


def score_to_rgba(score_arr: np.ndarray) -> np.ndarray:
    h, w = score_arr.shape
    rgba = np.zeros((4, h, w), dtype=np.uint8)
    for (lo, hi), (r, g, b, a) in COLOUR_BANDS:
        mask = (score_arr >= lo) & (score_arr < hi)
        rgba[0][mask] = r
        rgba[1][mask] = g
        rgba[2][mask] = b
        rgba[3][mask] = a
    return rgba


# ── リファレンスグリッド ─────────────────────────────────────
def load_reference_grid(pref: str, resolution_m: int = 30):
    slope_path = PROJECT_ROOT / "data" / pref / "land" / f"{pref}_slope.tif"
    with rasterio.open(slope_path) as ds:
        native_transform = ds.transform
        native_w, native_h = ds.width, ds.height
        crs = ds.crs
        bounds = ds.bounds

    native_res_x = abs(native_transform.a)
    centre_lat = (bounds.top + bounds.bottom) / 2
    m_per_deg = 111320 * np.cos(np.radians(centre_lat))
    native_m = native_res_x * m_per_deg

    if resolution_m >= native_m * 0.9:
        log.info("Using native grid (%d x %d, ~%.0fm)", native_w, native_h, native_m)
        return native_transform, native_w, native_h, crs, bounds

    scale = native_m / resolution_m
    new_w = int(np.ceil(native_w * scale))
    new_h = int(np.ceil(native_h * scale))
    new_transform = from_bounds(bounds.left, bounds.bottom, bounds.right, bounds.top, new_w, new_h)
    log.info("Custom grid: %d x %d (~%dm)", new_w, new_h, resolution_m)
    return new_transform, new_w, new_h, crs, bounds


def _resample_to_grid(src_path: Path, transform, width, height, crs,
                      resampling=Resampling.bilinear, band=1) -> np.ndarray:
    with rasterio.open(src_path) as ds:
        src_data = ds.read(band)
        src_transform = ds.transform
        src_crs = ds.crs
        if ds.width == width and ds.height == height and ds.transform.almost_equals(transform):
            return src_data
    dst = np.zeros((height, width), dtype=src_data.dtype)
    reproject(
        source=src_data, destination=dst,
        src_transform=src_transform, src_crs=src_crs,
        dst_transform=transform, dst_crs=crs,
        resampling=resampling,
    )
    return dst


# ── 距離スコア共通 ──────────────────────────────────────────
def _distance_score(geometries, transform, width, height, crs,
                    breakpoints: list[tuple[float, int]]) -> np.ndarray:
    if len(geometries) == 0:
        log.warning("    no geometries, returning default 50")
        return np.full((height, width), 50, dtype=np.uint8)

    burned = rasterize(
        [(g, 1) for g in geometries],
        out_shape=(height, width), transform=transform,
        fill=0, dtype=np.uint8, all_touched=True,
    )

    centre_lat = transform.f - abs(transform.e) * height / 2
    lat_rad = np.radians(centre_lat)
    m_per_deg_lon = 111320 * np.cos(lat_rad)
    m_per_deg_lat = 110540
    pixel_dx = abs(transform.a) * m_per_deg_lon
    pixel_dy = abs(transform.e) * m_per_deg_lat

    dist_m = distance_transform_edt(burned == 0, sampling=[pixel_dy, pixel_dx]).astype(np.float32)

    score = np.zeros((height, width), dtype=np.float32)
    bp = sorted(breakpoints, key=lambda x: x[0])
    for i in range(len(bp)):
        d, s = bp[i]
        if i == 0:
            score[dist_m <= d] = s
        else:
            d_prev, s_prev = bp[i - 1]
            mask = (dist_m > d_prev) & (dist_m <= d)
            frac = (dist_m[mask] - d_prev) / (d - d_prev)
            score[mask] = s_prev + frac * (s - s_prev)
    d_last, s_last = bp[-1]
    score[dist_m > d_last] = s_last

    return np.clip(score, 0, 100).astype(np.uint8)


# ── 風速スコア ──────────────────────────────────────────────
def compute_score_wind_speed(pref: str, transform, width, height, crs) -> np.ndarray:
    """風速 -> 適地スコア (0-100)。"""
    wind_path = PROJECT_ROOT / "data" / pref / "wind" / f"{pref}_wind_speed.tif"
    log.info("  wind_speed: reading %s", wind_path)

    if not wind_path.exists():
        log.warning("  wind_speed: not found, returning default 50")
        return np.full((height, width), 50, dtype=np.uint8)

    ws = _resample_to_grid(wind_path, transform, width, height, crs,
                           resampling=Resampling.cubic)
    ws = np.nan_to_num(ws, nan=0.0)

    # 風速スコアリング (ERA5 0.25°格子 → 連続線形補間)
    # ERA5は空間平均のため実測より低い (日本域: 1-5 m/s)
    # 1.5 m/s 以下 = 0, 5.0 m/s 以上 = 100, 間は線形
    score = np.interp(ws, [1.5, 5.0], [0, 100]).astype(np.uint8)

    return score


# ── 傾斜スコア (風力版) ──────────────────────────────────────
def compute_score_slope(pref: str, transform, width, height, crs) -> np.ndarray:
    slope_path = PROJECT_ROOT / "data" / pref / "land" / f"{pref}_slope.tif"
    log.info("  slope: reading %s", slope_path)
    slope = _resample_to_grid(slope_path, transform, width, height, crs)
    slope = np.nan_to_num(slope, nan=0.0)

    # 風力は太陽光より傾斜に寛容だが、急斜面は建設不可
    score = np.zeros((height, width), dtype=np.uint8)
    score[slope < 5] = 100
    score[(slope >= 5) & (slope < 10)] = 80
    score[(slope >= 10) & (slope < 15)] = 60
    score[(slope >= 15) & (slope < 20)] = 30
    score[(slope >= 20) & (slope < 30)] = 10
    score[slope >= 30] = 0

    return score


# ── 標高スコア (風力版) ──────────────────────────────────────
def compute_score_elevation(pref: str, transform, width, height, crs, bounds) -> np.ndarray:
    dem_dir = PROJECT_ROOT / "data" / pref / "land" / "dem"
    elev_path = PROJECT_ROOT / "data" / pref / "land" / f"{pref}_elevation.tif"

    if elev_path.exists():
        log.info("  elevation: reading %s", elev_path)
        elev = _resample_to_grid(elev_path, transform, width, height, crs)
    else:
        hgt_files = sorted(dem_dir.glob("*.hgt")) if dem_dir.exists() else []
        if not hgt_files:
            log.warning("  elevation: no data, using default 70")
            return np.full((height, width), 70, dtype=np.uint8)

        log.info("  elevation: mosaicking %d HGT files", len(hgt_files))
        datasets = [rasterio.open(str(f)) for f in hgt_files]
        mosaic, mosaic_transform = merge(datasets)
        for ds in datasets:
            ds.close()
        elev = np.zeros((height, width), dtype=np.float32)
        reproject(
            source=mosaic[0], destination=elev,
            src_transform=mosaic_transform, src_crs="EPSG:4326",
            dst_transform=transform, dst_crs=crs,
            resampling=Resampling.bilinear,
        )

    elev = np.nan_to_num(elev, nan=0.0)

    # 風力: 中程度の標高が最適 (尾根風), 高すぎると厳しい
    score = np.zeros((height, width), dtype=np.uint8)
    score[elev <= 200] = 70       # 平野部: まあまあ
    score[(elev > 200) & (elev <= 500)] = 90   # 丘陵・尾根: 好条件
    score[(elev > 500) & (elev <= 1000)] = 100  # 山間尾根: 最適
    score[(elev > 1000) & (elev <= 1500)] = 60  # やや厳しい
    score[(elev > 1500) & (elev <= 2000)] = 30  # 建設困難
    score[elev > 2000] = 10       # 高山: ほぼ不可

    return score


# ── 送電線距離 ──────────────────────────────────────────────
def compute_score_grid_dist(pref: str, transform, width, height, crs) -> np.ndarray:
    lines_path = PROJECT_ROOT / "data" / pref / "grid" / f"{pref}_lines.geojson"
    log.info("  grid_dist: reading %s", lines_path)
    gdf = gpd.read_file(lines_path)
    if "voltage_kv" in gdf.columns:
        gdf = gdf[gdf["voltage_kv"] >= 154]
    log.info("  grid_dist: %d lines >= 154kV", len(gdf))

    breakpoints = [
        (0, 100), (1000, 90), (3000, 70),
        (5000, 50), (10000, 20), (20000, 0),
    ]
    return _distance_score(gdf.geometry.tolist(), transform, width, height, crs, breakpoints)


# ── 変電所距離 ──────────────────────────────────────────────
def compute_score_sub_dist(pref: str, transform, width, height, crs) -> np.ndarray:
    subs_path = PROJECT_ROOT / "data" / pref / "grid" / f"{pref}_substations.geojson"
    log.info("  sub_dist: reading %s", subs_path)
    gdf = gpd.read_file(subs_path)
    if "voltage_kv" in gdf.columns:
        gdf = gdf[gdf["voltage_kv"] >= 66]
    log.info("  sub_dist: %d substations >= 66kV", len(gdf))

    geometries = [g.centroid if g.geom_type in ("Polygon", "MultiPolygon") else g
                  for g in gdf.geometry]
    breakpoints = [
        (0, 100), (2000, 80), (5000, 50), (10000, 20), (20000, 0),
    ]
    return _distance_score(geometries, transform, width, height, crs, breakpoints)


# ── 土地利用 ────────────────────────────────────────────────
def compute_score_land_use(pref: str, transform, width, height, crs) -> np.ndarray:
    lu_dir = PROJECT_ROOT / "data" / pref / "land" / "land_use"
    osm_path = lu_dir / "osm_land_use.tif" if lu_dir.exists() else None

    if osm_path is not None and osm_path.exists():
        log.info("  land_use: using OSM data %s", osm_path)
        score = _resample_to_grid(osm_path, transform, width, height, crs,
                                  resampling=Resampling.nearest)
        return score

    log.warning("  land_use: no data, using default 70")
    return np.full((height, width), 70, dtype=np.uint8)


# ── 居住地距離 (騒音バッファ) ────────────────────────────────
def compute_score_residential_dist(pref: str, transform, width, height, crs) -> np.ndarray:
    """居住地からの距離スコア。500m以内は排除。"""
    res_path = PROJECT_ROOT / "data" / pref / "land" / "land_use" / "residential_mask.tif"

    if not res_path.exists():
        log.warning("  residential_dist: no residential mask, using default 70")
        return np.full((height, width), 70, dtype=np.uint8)

    log.info("  residential_dist: reading %s", res_path)
    res_mask = _resample_to_grid(res_path, transform, width, height, crs,
                                 resampling=Resampling.nearest)

    # 居住地ピクセルからの距離を計算
    centre_lat = transform.f - abs(transform.e) * height / 2
    lat_rad = np.radians(centre_lat)
    pixel_dx = abs(transform.a) * 111320 * np.cos(lat_rad)
    pixel_dy = abs(transform.e) * 110540

    dist_m = distance_transform_edt(res_mask == 0, sampling=[pixel_dy, pixel_dx]).astype(np.float32)

    # スコアリング: 500m以内は排除
    score = np.zeros((height, width), dtype=np.uint8)
    score[dist_m < 500] = 0       # 排除ゾーン
    score[(dist_m >= 500) & (dist_m < 1000)] = 30
    score[(dist_m >= 1000) & (dist_m < 2000)] = 70
    score[(dist_m >= 2000) & (dist_m < 3000)] = 90
    score[dist_m >= 3000] = 100

    return score


# ── 総合スコア ──────────────────────────────────────────────
def compute_total_score(scores: dict) -> np.ndarray:
    w = WEIGHTS
    total = (
        scores["wind_speed"].astype(np.float32) * w["wind_speed"]
        + scores["slope"].astype(np.float32) * w["slope"]
        + scores["grid_dist"].astype(np.float32) * w["grid_distance"]
        + scores["sub_dist"].astype(np.float32) * w["substation_distance"]
        + scores["land_use"].astype(np.float32) * w["land_use"]
        + scores["elevation"].astype(np.float32) * w["elevation"]
        + scores["residential_dist"].astype(np.float32) * w["residential_distance"]
        + 50.0 * w["road_distance"]      # デフォルト道路スコア
        + 80.0 * w["protection"]          # デフォルト保護区スコア
    )
    return np.clip(total, 0, 100).astype(np.uint8)


# ── 出力ヘルパー ────────────────────────────────────────────
def write_score_tif(arr: np.ndarray, path: Path, transform, crs):
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        path, "w", driver="GTiff",
        height=arr.shape[0], width=arr.shape[1], count=1,
        dtype="uint8", crs=crs, transform=transform, compress="deflate",
    ) as dst:
        dst.write(arr, 1)
    log.info("  wrote %s (%d x %d)", path, arr.shape[1], arr.shape[0])


def write_rgba_tif(score_arr: np.ndarray, path: Path, transform, crs):
    rgba = score_to_rgba(score_arr)
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        path, "w", driver="GTiff",
        height=score_arr.shape[0], width=score_arr.shape[1], count=4,
        dtype="uint8", crs=crs, transform=transform, compress="deflate",
    ) as dst:
        dst.write(rgba)
    log.info("  wrote RGBA %s", path)


def generate_tiles(rgba_tif: Path, tiles_dir: Path, zoom="7-14"):
    if tiles_dir.exists():
        shutil.rmtree(tiles_dir)
    tiles_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "gdal2tiles.py", "-z", zoom, "-w", "none", "--xyz", "-r", "near",
        str(rgba_tif), str(tiles_dir),
    ]
    log.info("  generating tiles: %s", rgba_tif.name)
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        log.error("  gdal2tiles failed: %s", result.stderr[:500])


# ── メインパイプライン ──────────────────────────────────────
def process_prefecture(pref: str, resolution_m: int = 30, skip_tiles: bool = False):
    log.info("=" * 60)
    log.info("Processing %s @ %dm (WIND)", pref.upper(), resolution_m)
    log.info("=" * 60)

    transform, width, height, crs, bounds = load_reference_grid(pref, resolution_m)
    log.info("Grid: %d x %d pixels, CRS=%s", width, height, crs)

    res_suffix = f"_{resolution_m}m" if resolution_m != 30 else ""
    output_dir = PROJECT_ROOT / "output" / pref
    docs_dir = PROJECT_ROOT / "docs" / pref

    scores = {}

    # 1) 風速
    log.info("[1/7] Wind speed score...")
    scores["wind_speed"] = compute_score_wind_speed(pref, transform, width, height, crs)

    # 2) 傾斜
    log.info("[2/7] Slope score...")
    scores["slope"] = compute_score_slope(pref, transform, width, height, crs)

    # 3) 標高
    log.info("[3/7] Elevation score...")
    scores["elevation"] = compute_score_elevation(pref, transform, width, height, crs, bounds)

    # 4) 送電線距離
    log.info("[4/7] Grid distance score...")
    scores["grid_dist"] = compute_score_grid_dist(pref, transform, width, height, crs)

    # 5) 変電所距離
    log.info("[5/7] Substation distance score...")
    scores["sub_dist"] = compute_score_sub_dist(pref, transform, width, height, crs)

    # 6) 土地利用
    log.info("[6/7] Land use score...")
    scores["land_use"] = compute_score_land_use(pref, transform, width, height, crs)

    # 7) 居住地距離
    log.info("[7/7] Residential distance score...")
    scores["residential_dist"] = compute_score_residential_dist(pref, transform, width, height, crs)

    # 総合スコア
    log.info("Computing total score...")
    scores["total"] = compute_total_score(scores)

    # 県境マスク
    log.info("Masking to prefecture boundary...")
    admin_dir = PROJECT_ROOT / "data" / pref / "land" / "admin_boundary"
    shp_files = list(admin_dir.rglob("*.shp")) if admin_dir.exists() else []
    if shp_files:
        admin = gpd.read_file(shp_files[0])
        boundary_geom = admin.union_all()
        outside_mask = geometry_mask(
            [boundary_geom], transform=transform,
            out_shape=(height, width), invert=False,
        )
        for name in scores:
            scores[name][outside_mask] = 0
        log.info("  Masked %d pixels outside boundary", outside_mask.sum())

    # タイルズームレベル
    zoom = "7-17" if resolution_m <= 5 else "7-16" if resolution_m <= 10 else "7-14"

    # スコア TIF 出力
    score_names = ["total", "wind_speed", "slope", "elevation",
                   "grid_dist", "sub_dist", "land_use", "residential_dist"]
    for name in score_names:
        write_score_tif(scores[name], output_dir / f"score_{name}{res_suffix}.tif",
                        transform, crs)
        write_rgba_tif(scores[name], output_dir / f"score_{name}{res_suffix}_rgba.tif",
                       transform, crs)
        if not skip_tiles:
            tile_name = f"tiles_{name}{res_suffix}" if name != "total" else f"tiles{res_suffix}"
            generate_tiles(output_dir / f"score_{name}{res_suffix}_rgba.tif",
                           docs_dir / tile_name, zoom=zoom)

    log.info("DONE: %s @ %dm (WIND)", pref, resolution_m)


# ── CLI ──────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="風力発電ポテンシャル ラスタースコア計算")
    parser.add_argument("-p", "--prefecture", default="all",
                        choices=list(PREFECTURES.keys()) + ["all"])
    parser.add_argument("-r", "--resolution", type=int, default=30)
    parser.add_argument("--skip-tiles", action="store_true")
    args = parser.parse_args()

    prefs = list(PREFECTURES.keys()) if args.prefecture == "all" else [args.prefecture]
    for pref in prefs:
        try:
            process_prefecture(pref, resolution_m=args.resolution, skip_tiles=args.skip_tiles)
        except Exception:
            log.exception("FAILED: %s", pref)

    log.info("All done.")


if __name__ == "__main__":
    main()
