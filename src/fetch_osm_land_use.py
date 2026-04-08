#!/usr/bin/env python3
"""
OSM 土地利用データの取得とラスター化。
japan-re-potential から流用。居住地ポリゴンも別途抽出する (風力の騒音バッファ用)。

使い方:
    python src/fetch_osm_land_use.py -p fukui
"""

import argparse
import json
import logging
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
import rasterio
from rasterio.features import rasterize
from shapely.geometry import Polygon
from shapely.ops import unary_union

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import PREFECTURES, PROJECT_ROOT

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

OVERPASS_URL = "https://overpass-api.de/api/interpreter"

# OSM tag -> score code (風力版: 建物と居住地を厳しくスコア 0)
TAG_SCORE = {
    "building": 0,
    "landuse=residential": 0,
    "landuse=commercial": 0,
    "landuse=industrial": 0,
    "landuse=retail": 0,
    "landuse=railway": 0,
    "landuse=forest": 20,
    "natural=wood": 20,
    "landuse=farmland": 60,      # 風力は農地と共存可能
    "landuse=meadow": 70,
    "landuse=orchard": 40,
    "landuse=vineyard": 40,
    "landuse=paddy": 50,
    "landuse=recreation_ground": 30,
    "leisure=golf_course": 30,   # 風力はゴルフ場との共存困難
    "landuse=brownfield": 90,
    "landuse=greenfield": 90,
    "landuse=quarry": 80,
    "natural=water": 0,
    "landuse=reservoir": 0,
}


def get_score_for_element(tags: dict) -> int:
    if "building" in tags and tags["building"] != "no":
        return TAG_SCORE["building"]
    if tags.get("leisure") == "golf_course":
        return TAG_SCORE["leisure=golf_course"]
    lu = tags.get("landuse", "")
    key = f"landuse={lu}"
    if key in TAG_SCORE:
        return TAG_SCORE[key]
    nat = tags.get("natural", "")
    key = f"natural={nat}"
    if key in TAG_SCORE:
        return TAG_SCORE[key]
    return -1


def query_overpass(bbox, timeout=180):
    west, south, east, north = bbox
    bbox_str = f"{south},{west},{north},{east}"

    query = f"""
[out:json][timeout:{timeout}];
(
  way["landuse"]({bbox_str});
  relation["landuse"]({bbox_str});
  way["natural"="wood"]({bbox_str});
  way["natural"="water"]({bbox_str});
  way["building"]({bbox_str});
  way["leisure"="golf_course"]({bbox_str});
  relation["leisure"="golf_course"]({bbox_str});
);
out geom;
"""

    data = urllib.parse.urlencode({"data": query}).encode("utf-8")

    max_attempts = 12
    for attempt in range(max_attempts):
        try:
            req = urllib.request.Request(OVERPASS_URL, data=data)
            req.add_header("User-Agent", "japan-we-potential/1.0")
            with urllib.request.urlopen(req, timeout=timeout + 30) as resp:
                result = json.loads(resp.read().decode("utf-8"))
            log.info("  Got %d elements from Overpass", len(result.get("elements", [])))
            return result
        except Exception as e:
            wait = min(30 * (2 ** attempt), 300)
            log.warning("  Overpass attempt %d/%d failed: %s — waiting %ds",
                        attempt + 1, max_attempts, e, wait)
            if attempt < max_attempts - 1:
                time.sleep(wait)
            else:
                raise


def elements_to_geometries(elements):
    results = []
    residential_geoms = []

    for elem in elements:
        tags = elem.get("tags", {})
        score = get_score_for_element(tags)
        if score == -1:
            score = 70

        geom = None
        etype = elem.get("type")

        if etype == "way" and "geometry" in elem:
            coords = [(n["lon"], n["lat"]) for n in elem["geometry"]]
            if len(coords) >= 4:
                if coords[0] != coords[-1]:
                    coords.append(coords[0])
                try:
                    geom = Polygon(coords)
                except Exception:
                    pass

        elif etype == "relation" and "members" in elem:
            outer_rings = []
            for member in elem.get("members", []):
                if member.get("role") == "outer" and "geometry" in member:
                    coords = [(n["lon"], n["lat"]) for n in member["geometry"]]
                    if len(coords) >= 4:
                        if coords[0] != coords[-1]:
                            coords.append(coords[0])
                        try:
                            outer_rings.append(Polygon(coords))
                        except Exception:
                            pass
            if outer_rings:
                try:
                    geom = unary_union(outer_rings)
                except Exception:
                    pass

        if geom is not None and geom.is_valid and not geom.is_empty:
            results.append((geom, score))
            # 居住地ポリゴンを別途記録 (騒音バッファ用)
            if tags.get("landuse") == "residential" or "building" in tags:
                residential_geoms.append(geom)

    return results, residential_geoms


def split_bbox(bbox, n_splits=2):
    west, south, east, north = bbox
    lon_step = (east - west) / n_splits
    lat_step = (north - south) / n_splits
    sub_bboxes = []
    for i in range(n_splits):
        for j in range(n_splits):
            sub_bboxes.append((
                west + j * lon_step,
                south + i * lat_step,
                west + (j + 1) * lon_step,
                south + (i + 1) * lat_step,
            ))
    return sub_bboxes


def fetch_land_use_for_prefecture(pref: str):
    cfg = PREFECTURES[pref]
    bbox = cfg["bbox"]
    sub_bboxes = split_bbox(bbox, n_splits=3)

    log.info("Fetching OSM land use for %s (%d sub-queries)", pref, len(sub_bboxes))

    all_geom_scores = []
    all_residential = []
    for i, sub_bbox in enumerate(sub_bboxes):
        log.info("  Sub-query %d/%d", i + 1, len(sub_bboxes))
        try:
            result = query_overpass(sub_bbox, timeout=180)
            geom_scores, residential = elements_to_geometries(result.get("elements", []))
            all_geom_scores.extend(geom_scores)
            all_residential.extend(residential)
        except Exception as e:
            log.error("  Failed sub-query %d: %s", i + 1, e)
        if i < len(sub_bboxes) - 1:
            time.sleep(5)

    log.info("Total polygons: %d, residential: %d", len(all_geom_scores), len(all_residential))

    if not all_geom_scores:
        log.error("No OSM data for %s!", pref)
        return

    # Load reference grid from slope TIF
    slope_path = PROJECT_ROOT / "data" / pref / "land" / f"{pref}_slope.tif"
    with rasterio.open(slope_path) as ds:
        ref_transform = ds.transform
        ref_width = ds.width
        ref_height = ds.height
        ref_crs = ds.crs

    # Rasterize land use
    score_levels = sorted(set(s for _, s in all_geom_scores), reverse=True)
    raster = np.full((ref_height, ref_width), 70, dtype=np.uint8)

    for score_val in score_levels:
        shapes = [(g, score_val) for g, s in all_geom_scores if s == score_val]
        if not shapes:
            continue
        try:
            layer = rasterize(shapes, out_shape=(ref_height, ref_width),
                              transform=ref_transform, fill=255, dtype=np.uint8,
                              all_touched=True)
            mask = layer != 255
            raster[mask] = layer[mask]
        except Exception as e:
            log.error("  Failed to rasterize score=%d: %s", score_val, e)

    # 土地利用出力
    out_dir = PROJECT_ROOT / "data" / pref / "land" / "land_use"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "osm_land_use.tif"

    with rasterio.open(
        out_path, "w", driver="GTiff",
        height=ref_height, width=ref_width, count=1,
        dtype="uint8", crs=ref_crs, transform=ref_transform,
        compress="deflate",
    ) as dst:
        dst.write(raster, 1)
    log.info("Wrote land use: %s", out_path)

    # 居住地マスク出力 (騒音バッファ用: 1=居住地, 0=それ以外)
    if all_residential:
        res_raster = rasterize(
            [(g, 1) for g in all_residential],
            out_shape=(ref_height, ref_width),
            transform=ref_transform,
            fill=0, dtype=np.uint8, all_touched=True,
        )
        res_path = out_dir / "residential_mask.tif"
        with rasterio.open(
            res_path, "w", driver="GTiff",
            height=ref_height, width=ref_width, count=1,
            dtype="uint8", crs=ref_crs, transform=ref_transform,
            compress="deflate",
        ) as dst:
            dst.write(res_raster, 1)
        log.info("Wrote residential mask: %s (%d residential polygons)",
                 res_path, len(all_residential))


def main():
    parser = argparse.ArgumentParser(description="Fetch OSM land use (風力版)")
    parser.add_argument("-p", "--prefecture", default="all",
                        choices=list(PREFECTURES.keys()) + ["all"])
    args = parser.parse_args()

    prefs = list(PREFECTURES.keys()) if args.prefecture == "all" else [args.prefecture]
    for pref in prefs:
        try:
            fetch_land_use_for_prefecture(pref)
        except Exception:
            log.exception("FAILED: %s", pref)
    log.info("All done.")


if __name__ == "__main__":
    main()
