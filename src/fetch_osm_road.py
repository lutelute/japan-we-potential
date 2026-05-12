#!/usr/bin/env python3
"""
OSM から主要道路を取得し、距離スコア GeoTIFF を生成する。

道路距離スコアは wind turbine の建設・保守アクセス性を表す。
対象道路: motorway / trunk / primary / secondary (とそれぞれのリンク道路)

スコア化:
  0 m    = 100  (道路上)
  500 m  =  90
  2000 m =  70
  5000 m =  40
  10000m =  10
  15000m+ = 0   (過疎地・建設困難)

出力:
  data/{pref}/land/road/{pref}_road_score.tif  (uint8)
  data/{pref}/grid/{pref}_roads.geojson        (参照用)

Usage:
    python src/fetch_osm_road.py -p fukui
    python src/fetch_osm_road.py -p fukui -r 30
"""
import argparse
import logging
import sys
import time
from pathlib import Path

import geopandas as gpd
import numpy as np
import requests
import rasterio
from rasterio.features import rasterize
from scipy.ndimage import distance_transform_edt
from shapely.geometry import shape

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import PREFECTURES, get_grid_dir, get_land_dir, get_pref_config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

OVERPASS_URL = "https://overpass-api.de/api/interpreter"
ROAD_TYPES = [
    "motorway", "motorway_link",
    "trunk", "trunk_link",
    "primary", "primary_link",
    "secondary", "secondary_link",
]

# 道路距離 → スコア ブレークポイント
ROAD_BP = [
    (0,     100),
    (500,    90),
    (2000,   70),
    (5000,   40),
    (10000,  10),
    (15000,   0),
]


def fetch_osm_roads(bbox: tuple, timeout: int = 120, retries: int = 3) -> gpd.GeoDataFrame:
    """Overpass API で主要道路を取得する。"""
    west, south, east, north = bbox
    road_filter = "|".join(ROAD_TYPES)
    query = f"""
[out:json][timeout:{timeout}];
(
  way["highway"~"^({road_filter})$"]({south},{west},{north},{east});
);
out geom;
"""
    for attempt in range(retries):
        try:
            log.info("  Overpass クエリ (試行 %d/%d)...", attempt + 1, retries)
            resp = requests.post(
                OVERPASS_URL,
                data={"data": query},
                timeout=timeout + 30,
            )
            resp.raise_for_status()
            elements = resp.json().get("elements", [])

            rows, geoms = [], []
            for e in elements:
                if e.get("type") != "way" or "geometry" not in e:
                    continue
                coords = [(p["lon"], p["lat"]) for p in e["geometry"]]
                if len(coords) < 2:
                    continue
                geoms.append(shape({"type": "LineString", "coordinates": coords}))
                rows.append({
                    "highway": e.get("tags", {}).get("highway", ""),
                    "name":    e.get("tags", {}).get("name", ""),
                })

            if not geoms:
                log.warning("  道路: 0 件取得")
                return gpd.GeoDataFrame(columns=["geometry"])

            gdf = gpd.GeoDataFrame(rows, geometry=geoms, crs="EPSG:4326")
            log.info("  道路取得: %d 本", len(gdf))
            return gdf

        except Exception as exc:
            log.warning("  Overpass エラー (試行 %d): %s", attempt + 1, exc)
            if attempt < retries - 1:
                wait = 60 * (2 ** attempt)
                log.info("  %d 秒待機...", wait)
                time.sleep(wait)

    return gpd.GeoDataFrame(columns=["geometry"])


def roads_to_score_tif(
    roads: gpd.GeoDataFrame,
    pref: str,
    resolution_m: int = 30,
) -> Path | None:
    """道路 GeoDataFrame → 距離スコア GeoTIFF"""
    import math

    land_dir = get_land_dir(pref)
    slope_path = land_dir / f"{pref}_slope.tif"
    if not slope_path.exists():
        log.error("  参照 slope.tif なし: %s", slope_path)
        return None

    with rasterio.open(slope_path) as ref:
        bounds = ref.bounds
        crs = ref.crs
        width = ref.width
        height = ref.height
        transform = ref.transform

    # 解像度スケール調整
    if resolution_m != 30:
        from rasterio.transform import from_bounds
        factor = 30.0 / resolution_m
        width = int(width * factor)
        height = int(height * factor)
        transform = from_bounds(
            bounds.left, bounds.bottom, bounds.right, bounds.top, width, height
        )

    road_dir = land_dir / "road"
    road_dir.mkdir(parents=True, exist_ok=True)
    out_path = road_dir / f"{pref}_road_score.tif"

    if len(roads) == 0:
        log.warning("  道路データなし → デフォルト50で保存")
        score = np.full((height, width), 50, dtype=np.uint8)
    else:
        roads_proj = roads.to_crs(crs)
        burned = rasterize(
            [(g, 1) for g in roads_proj.geometry if g is not None],
            out_shape=(height, width),
            transform=transform,
            fill=0, dtype=np.uint8, all_touched=True,
        )

        # ピクセル物理サイズ (m)
        lat_c = (bounds.bottom + bounds.top) / 2
        px_dx = abs(transform.a) * 111320 * math.cos(math.radians(lat_c))
        px_dy = abs(transform.e) * 110540

        dist_m = distance_transform_edt(
            burned == 0, sampling=[px_dy, px_dx]
        ).astype(np.float32)

        score = np.zeros((height, width), dtype=np.float32)
        for i, (d, s) in enumerate(ROAD_BP):
            if i == 0:
                score[dist_m <= d] = s
            else:
                d0, s0 = ROAD_BP[i - 1]
                mask = (dist_m > d0) & (dist_m <= d)
                frac = (dist_m[mask] - d0) / (d - d0)
                score[mask] = s0 + frac * (s - s0)
        score[dist_m > ROAD_BP[-1][0]] = ROAD_BP[-1][1]
        score = np.clip(score, 0, 100).astype(np.uint8)

    with rasterio.open(
        out_path, "w", driver="GTiff",
        height=height, width=width, count=1,
        dtype="uint8", crs=crs, transform=transform,
        compress="deflate",
    ) as dst:
        dst.write(score, 1)

    log.info("  道路スコア TIF: %s (mean=%.1f)", out_path, score.mean())
    return out_path


def main():
    parser = argparse.ArgumentParser(description="OSM 道路取得 + 距離スコア TIF 生成")
    parser.add_argument("-p", "--prefecture", required=True,
                        choices=list(PREFECTURES.keys()))
    parser.add_argument("-r", "--resolution", type=int, default=30)
    args = parser.parse_args()

    pref = args.prefecture
    cfg = get_pref_config(pref)
    log.info("=" * 60)
    log.info("%s  道路データ取得 + スコア TIF 生成", cfg["name_ja"])
    log.info("=" * 60)

    roads = fetch_osm_roads(cfg["bbox"])

    # GeoJSON 保存 (参照用・メッシュマップで表示可能)
    grid_dir = get_grid_dir(pref)
    grid_dir.mkdir(parents=True, exist_ok=True)
    if len(roads) > 0:
        roads.to_file(grid_dir / f"{pref}_roads.geojson", driver="GeoJSON")
        log.info("  GeoJSON 保存: %s_roads.geojson (%d 本)", pref, len(roads))

    roads_to_score_tif(roads, pref, args.resolution)
    log.info("完了!")


if __name__ == "__main__":
    main()
