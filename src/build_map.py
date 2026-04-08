#!/usr/bin/env python3
"""
風力ポテンシャルマップの可視化 (Folium + GeoTIFF)

Usage:
    python src/build_map.py -p fukui
    python src/build_map.py -p fukui,akita
"""

import argparse
import logging
import sys
from pathlib import Path

import folium
import numpy as np
import rasterio
from rasterio.warp import transform_bounds

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import PREFECTURES, PROJECT_ROOT

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


def create_overlay_image(score_tif: Path, output_png: Path):
    """スコア GeoTIFF を透過PNG に変換 (Folium ImageOverlay 用)。"""
    from PIL import Image

    with rasterio.open(score_tif) as ds:
        rgba = ds.read()  # (4, H, W) for RGBA TIF
        bounds = ds.bounds

    # (4, H, W) -> (H, W, 4)
    img = np.moveaxis(rgba, 0, -1)
    pil_img = Image.fromarray(img, "RGBA")

    # リサイズ (大きすぎる場合)
    max_dim = 2000
    if max(pil_img.size) > max_dim:
        ratio = max_dim / max(pil_img.size)
        new_size = (int(pil_img.width * ratio), int(pil_img.height * ratio))
        pil_img = pil_img.resize(new_size, Image.LANCZOS)

    output_png.parent.mkdir(parents=True, exist_ok=True)
    pil_img.save(output_png)
    log.info("  PNG: %s (%d x %d)", output_png, pil_img.width, pil_img.height)
    return bounds


def build_prefecture_map(pref: str) -> folium.Map:
    """1県分のFoliumマップを作成。"""
    cfg = PREFECTURES[pref]
    center = cfg["center"]
    name_ja = cfg["name_ja"]

    output_dir = PROJECT_ROOT / "output" / pref
    docs_dir = PROJECT_ROOT / "docs" / pref
    docs_dir.mkdir(parents=True, exist_ok=True)

    m = folium.Map(location=center, zoom_start=9, tiles="OpenStreetMap")

    # レイヤー定義
    layers = [
        ("total", "総合スコア", True),
        ("wind_speed", "風速スコア", False),
        ("slope", "傾斜スコア", False),
        ("elevation", "標高スコア", False),
        ("grid_dist", "送電線距離", False),
        ("sub_dist", "変電所距離", False),
        ("land_use", "土地利用", False),
        ("residential_dist", "居住地距離", False),
    ]

    try:
        from PIL import Image
        has_pil = True
    except ImportError:
        has_pil = False
        log.warning("Pillow not installed, skipping image overlay")

    for layer_name, layer_label, show in layers:
        rgba_tif = output_dir / f"score_{layer_name}_rgba.tif"
        if not rgba_tif.exists():
            continue

        if has_pil:
            png_path = docs_dir / f"score_{layer_name}.png"
            bounds = create_overlay_image(rgba_tif, png_path)

            # Folium ImageOverlay
            overlay = folium.raster_layers.ImageOverlay(
                image=str(png_path),
                bounds=[[bounds.bottom, bounds.left], [bounds.top, bounds.right]],
                name=f"{layer_label} ({name_ja})",
                opacity=0.7,
                show=show,
            )
            overlay.add_to(m)
        else:
            # GeoTIFF のメタ情報だけ表示
            with rasterio.open(rgba_tif) as ds:
                bounds = ds.bounds
            folium.Rectangle(
                bounds=[[bounds.bottom, bounds.left], [bounds.top, bounds.right]],
                color="blue", weight=1, fill=False,
                tooltip=f"{layer_label} ({name_ja})",
            ).add_to(m)

    # 送電線オーバーレイ
    grid_file = PROJECT_ROOT / "data" / pref / "grid" / f"{pref}_lines.geojson"
    if grid_file.exists():
        import geopandas as gpd
        lines = gpd.read_file(grid_file)
        if "voltage_kv" in lines.columns:
            hv = lines[lines["voltage_kv"] >= 154]
        else:
            hv = lines

        if len(hv) > 0:
            fg = folium.FeatureGroup(name="送電線 (154kV+)", show=False)
            folium.GeoJson(
                hv.__geo_interface__,
                style_function=lambda x: {"color": "red", "weight": 2, "opacity": 0.6},
            ).add_to(fg)
            fg.add_to(m)

    # 変電所マーカー
    subs_file = PROJECT_ROOT / "data" / pref / "grid" / f"{pref}_substations.geojson"
    if subs_file.exists():
        import geopandas as gpd
        subs = gpd.read_file(subs_file)
        if "voltage_kv" in subs.columns:
            subs = subs[subs["voltage_kv"] >= 66]

        if len(subs) > 0:
            fg = folium.FeatureGroup(name="変電所 (66kV+)", show=False)
            for _, row in subs.iterrows():
                pt = row.geometry.centroid if row.geometry.geom_type != "Point" else row.geometry
                name = row.get("name", "")
                v = row.get("voltage_kv", "?")
                folium.CircleMarker(
                    location=[pt.y, pt.x],
                    radius=4, color="blue", fill=True,
                    tooltip=f"{name} ({v}kV)",
                ).add_to(fg)
            fg.add_to(m)

    folium.LayerControl().add_to(m)
    return m


def main():
    parser = argparse.ArgumentParser(description="風力ポテンシャルマップ生成")
    parser.add_argument("-p", "--prefecture", default="fukui")
    args = parser.parse_args()

    prefs = [p.strip() for p in args.prefecture.split(",")]

    for pref in prefs:
        if pref not in PREFECTURES:
            log.error("Unknown: %s", pref)
            continue

        name_ja = PREFECTURES[pref]["name_ja"]
        log.info("Building map for %s (%s)", pref, name_ja)

        m = build_prefecture_map(pref)
        docs_dir = PROJECT_ROOT / "docs" / pref
        html_path = docs_dir / "wind_potential_map.html"
        m.save(str(html_path))
        log.info("Saved: %s", html_path)

    log.info("All done.")


if __name__ == "__main__":
    main()
