"""
電力系統データを All-Japan-Grid から抽出。
参考プロジェクトにデータがあればリンクで共有。
"""
import argparse
import os
import sys
from pathlib import Path

import geopandas as gpd
import pandas as pd
from shapely.geometry import box

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import PREFECTURES, get_grid_dir, get_pref_config, get_re_data_dir

ALL_JAPAN_GRID = Path(os.environ.get("ALL_JAPAN_GRID_DIR", "/tmp/All-Japan-Grid-ref/data"))


def try_link_grid_from_re_project(pref: str, grid_dir: Path) -> bool:
    """参考プロジェクトのグリッドデータをリンク。"""
    re_grid_dir = get_re_data_dir(pref) / "grid"
    if not re_grid_dir.exists():
        return False

    grid_dir.mkdir(parents=True, exist_ok=True)
    linked = False
    for geojson in re_grid_dir.glob(f"{pref}_*.geojson"):
        local = grid_dir / geojson.name
        if not local.exists():
            local.symlink_to(geojson)
            print(f"  [LINK] {geojson.name} -> {geojson}")
            linked = True
    return linked


def load_area_geojson(grid_area: str, kind: str) -> gpd.GeoDataFrame:
    path = ALL_JAPAN_GRID / f"{grid_area}_{kind}.geojson"
    print(f"  Loading {path.name} ...", end=" ")
    gdf = gpd.read_file(path)
    print(f"{len(gdf)} features")
    return gdf


def filter_by_bbox(gdf: gpd.GeoDataFrame, bbox_tuple: tuple,
                   pref_name_ja: str, buffer_km: float = 5.0) -> gpd.GeoDataFrame:
    xmin, ymin, xmax, ymax = bbox_tuple
    pref_bbox = box(xmin, ymin, xmax, ymax)
    buf_deg = buffer_km / 111.0
    bbox_buffered = pref_bbox.buffer(buf_deg)
    mask = gdf.geometry.intersects(bbox_buffered)
    filtered = gdf[mask].copy()
    print(f"  Filtered to {pref_name_ja} area: {len(filtered)} features")
    return filtered


def extract_voltage_kv(voltage_str) -> float:
    if pd.isna(voltage_str) or voltage_str == "":
        return 0
    try:
        v = float(str(voltage_str).replace(",", ""))
        return v / 1000 if v > 1000 else v
    except ValueError:
        return 0


def main():
    parser = argparse.ArgumentParser(description="電力系統データ抽出 (風力版)")
    parser.add_argument("-p", "--prefecture", required=True,
                        choices=list(PREFECTURES.keys()))
    args = parser.parse_args()

    pref = args.prefecture
    cfg = get_pref_config(pref)
    grid_dir = get_grid_dir(pref)
    name_ja = cfg["name_ja"]

    print("=" * 60)
    print(f"{name_ja} 電力系統データ抽出 (風力版)")
    print("=" * 60)

    # 参考プロジェクトからリンク
    print("\n[1] 参考プロジェクトからリンク")
    if try_link_grid_from_re_project(pref, grid_dir):
        print("  参考プロジェクトのデータをリンクしました。")
        # 既にデータがあれば抽出はスキップ
        expected = [f"{pref}_lines.geojson", f"{pref}_substations.geojson"]
        if all((grid_dir / f).exists() for f in expected):
            print("  必要なデータは全て揃っています。")
            return

    # All-Japan-Gridから抽出
    print(f"\n[2] All-Japan-Grid から抽出")
    bbox_tuple = cfg["bbox"]
    grid_area = cfg["grid_area"]

    subs = load_area_geojson(grid_area, "substations")
    lines = load_area_geojson(grid_area, "lines")
    plants = load_area_geojson(grid_area, "plants")

    # 名前なしを除外
    subs = subs[subs["name"].notna() & (subs["name"] != "")]
    plants = plants[plants["name"].notna() & (plants["name"] != "")]

    subs_f = filter_by_bbox(subs, bbox_tuple, name_ja)
    lines_f = filter_by_bbox(lines, bbox_tuple, name_ja)
    plants_f = filter_by_bbox(plants, bbox_tuple, name_ja)

    if "voltage" in subs_f.columns:
        subs_f["voltage_kv"] = subs_f["voltage"].apply(extract_voltage_kv)
    if "voltage" in lines_f.columns:
        lines_f["voltage_kv"] = lines_f["voltage"].apply(extract_voltage_kv)

    # 統計
    print(f"\n  変電所: {len(subs_f)}, 送電線: {len(lines_f)}, 発電所: {len(plants_f)}")

    # GeoJSON 出力
    grid_dir.mkdir(parents=True, exist_ok=True)
    subs_f.to_file(grid_dir / f"{pref}_substations.geojson", driver="GeoJSON")
    lines_f.to_file(grid_dir / f"{pref}_lines.geojson", driver="GeoJSON")
    plants_f.to_file(grid_dir / f"{pref}_plants.geojson", driver="GeoJSON")
    print(f"\n完了!")


if __name__ == "__main__":
    main()
