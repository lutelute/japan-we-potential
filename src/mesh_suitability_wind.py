#!/usr/bin/env python3
"""
風力発電適地評価 — メッシュ分析

500m/1km メッシュで各セルの AHP スコアを算出し、
GeoJSON + Folium インタラクティブマップを出力する。

AHP 重み (config.WEIGHTS 準拠):
  風速:       30%  (ERA5 100m高さ)
  送電線距離:  15%  (154kV 以上)
  傾斜:       12%
  変電所距離:  10%  (66kV 以上)
  土地利用:    10%
  居住地距離:   8%  (騒音バッファ, 500m 以内排除)
  標高:         5%  (500-1000m 尾根が最適)
  道路距離:     5%  (デフォルト固定。fetch_osm_road.py で更新可)
  保護区域:     5%  (デフォルト固定)

高速化:
  rasterio.sample() によるベクトル化。for ループ廃止。
  1000 セル → 10 万セルでも数秒で完了。

Usage:
    python src/mesh_suitability_wind.py -p fukui
    python src/mesh_suitability_wind.py -p akita -r 500
    python src/mesh_suitability_wind.py -p fukui -r 0   # 1000m + 500m 両方
"""
import argparse
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.merge import merge
from rasterio.transform import rowcol
from shapely.geometry import box

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import (
    PREFECTURES, WEIGHTS,
    get_grid_dir, get_land_dir, get_output_dir, get_pref_config, PROJECT_ROOT,
)

SCORE_BINS = [0, 20, 40, 60, 80, 100]
SCORE_LABELS = ["不適(0-20)", "低(20-40)", "中(40-60)", "良(60-80)", "最適(80-100)"]


# ── ユーティリティ ─────────────────────────────────────────────

def _mesh_color(score: float) -> str:
    if score >= 80: return "#006400"
    if score >= 60: return "#228B22"
    if score >= 40: return "#DAA520"
    if score >= 20: return "#FF8C00"
    return "#DC143C"


def _extract_voltage_kv(v) -> float:
    if pd.isna(v) or v == "":
        return 0
    try:
        val = float(str(v).replace(",", ""))
        return val / 1000 if val > 1000 else val
    except ValueError:
        return 0


def _sample_tif(tif_path: Path, coords: list) -> np.ndarray:
    """ベクトル化サンプリング。for ループなし。"""
    with rasterio.open(tif_path) as src:
        sampled = src.sample(coords, masked=False)
        return np.array([v[0] for v in sampled], dtype=np.float32)


def _dist_score(dist_m: np.ndarray, breakpoints: list) -> np.ndarray:
    """距離配列 → スコア配列 (線形補間)"""
    bp = sorted(breakpoints, key=lambda x: x[0])
    score = np.zeros(len(dist_m), dtype=np.float32)
    for i, (d, s) in enumerate(bp):
        if i == 0:
            score[dist_m <= d] = s
        else:
            d0, s0 = bp[i - 1]
            mask = (dist_m > d0) & (dist_m <= d)
            frac = (dist_m[mask] - d0) / (d - d0)
            score[mask] = s0 + frac * (s - s0)
    score[dist_m > bp[-1][0]] = bp[-1][1]
    return np.clip(score, 0, 100)


# ── メッシュ生成 ───────────────────────────────────────────────

def create_mesh(pref: str, resolution_m: int) -> gpd.GeoDataFrame:
    cfg = get_pref_config(pref)
    xmin, ymin, xmax, ymax = cfg["bbox"]
    clat = cfg["center"][0]

    dy = resolution_m / 111000.0
    dx = resolution_m / (111000.0 * np.cos(np.radians(clat)))

    xs = np.arange(xmin, xmax, dx)
    ys = np.arange(ymin, ymax, dy)
    XX, YY = np.meshgrid(xs, ys)
    cx = (XX + dx / 2).ravel()
    cy = (YY + dy / 2).ravel()

    geoms = [box(x, y, x + dx, y + dy) for x, y in zip(XX.ravel(), YY.ravel())]
    mesh = gpd.GeoDataFrame({"cx": cx, "cy": cy}, geometry=geoms, crs="EPSG:4326")
    print(f"  メッシュ生成: {len(mesh):,} セル ({resolution_m}m)")
    return mesh


def clip_to_prefecture(mesh: gpd.GeoDataFrame, pref: str) -> gpd.GeoDataFrame:
    land_dir = get_land_dir(pref)
    admin_dir = land_dir / "admin_boundary"
    shp_files = (
        [p for p in admin_dir.rglob("*.shp") if "_subprefecture" not in p.name]
        if admin_dir.exists() else []
    )
    if not shp_files:
        print("  WARNING: 行政区域データなし。クリップなし。")
        return mesh

    admin = gpd.read_file(shp_files[0])
    boundary = admin.union_all()
    clipped = mesh[mesh.geometry.intersects(boundary)].copy().reset_index(drop=True)
    print(f"  クリップ後: {len(clipped):,} セル")
    return clipped


# ── スコア関数群 (すべてベクトル化) ────────────────────────────

def score_wind_speed(mesh: gpd.GeoDataFrame, pref: str) -> np.ndarray:
    wind_tif = PROJECT_ROOT / "data" / pref / "wind" / f"{pref}_wind_speed.tif"
    if not wind_tif.exists():
        print("  風速: データなし → デフォルト50")
        return np.full(len(mesh), 50.0)

    coords = list(zip(mesh["cx"], mesh["cy"]))
    ws = _sample_tif(wind_tif, coords)
    ws = np.nan_to_num(ws, nan=0.0)
    score = np.interp(ws, [1.5, 5.0], [0, 100])
    print(f"  風速: {ws.min():.2f}–{ws.max():.2f} m/s → スコア mean={score.mean():.1f}")
    return score


def score_slope(mesh: gpd.GeoDataFrame, pref: str) -> np.ndarray:
    tif = get_land_dir(pref) / f"{pref}_slope.tif"
    if not tif.exists():
        print("  傾斜: データなし → デフォルト50")
        return np.full(len(mesh), 50.0)

    coords = list(zip(mesh["cx"], mesh["cy"]))
    slp = np.clip(np.nan_to_num(_sample_tif(tif, coords), nan=0.0), 0, 90)
    score = np.where(slp < 5, 100,
            np.where(slp < 10, 80,
            np.where(slp < 15, 60,
            np.where(slp < 20, 30,
            np.where(slp < 30, 10, 0))))).astype(np.float32)
    print(f"  傾斜: mean={score.mean():.1f}")
    return score


def score_elevation(mesh: gpd.GeoDataFrame, pref: str) -> np.ndarray:
    elev_tif = get_land_dir(pref) / f"{pref}_elevation.tif"
    dem_dir = get_land_dir(pref) / "dem"

    if elev_tif.exists():
        coords = list(zip(mesh["cx"], mesh["cy"]))
        elev = np.nan_to_num(_sample_tif(elev_tif, coords), nan=0.0)
    elif dem_dir.exists():
        hgt_files = sorted(dem_dir.glob("*.hgt"))
        if not hgt_files:
            print("  標高: データなし → デフォルト70")
            return np.full(len(mesh), 70.0)
        datasets = [rasterio.open(str(f)) for f in hgt_files]
        mosaic, mosaic_tf = merge(datasets)
        for ds in datasets:
            ds.close()
        rows, cols = rowcol(mosaic_tf, mesh["cx"].values, mesh["cy"].values)
        h, w = mosaic.shape[1], mosaic.shape[2]
        valid = (rows >= 0) & (rows < h) & (cols >= 0) & (cols < w)
        elev = np.full(len(mesh), 500.0)
        elev[valid] = mosaic[0, rows[valid], cols[valid]].astype(np.float32)
    else:
        print("  標高: データなし → デフォルト70")
        return np.full(len(mesh), 70.0)

    # 風力: 500-1000m の尾根が最適
    score = np.where(elev <= 200, 70,
            np.where(elev <= 500, 90,
            np.where(elev <= 1000, 100,
            np.where(elev <= 1500, 60,
            np.where(elev <= 2000, 30, 10))))).astype(np.float32)
    print(f"  標高: mean={score.mean():.1f}")
    return score


def score_grid_distance(mesh: gpd.GeoDataFrame,
                        lines: gpd.GeoDataFrame, epsg: int) -> np.ndarray:
    hv = lines[lines["voltage_kv"] >= 154].copy()
    if len(hv) == 0:
        print("  送電線: 154kV なし → デフォルト50")
        return np.full(len(mesh), 50.0)

    mesh_p = mesh.to_crs(epsg=epsg)
    hv_union = hv.to_crs(epsg=epsg).union_all()
    dist = mesh_p.geometry.centroid.distance(hv_union).values

    bp = [(0, 100), (1000, 90), (3000, 70), (5000, 50), (10000, 20), (20000, 0)]
    score = _dist_score(dist, bp)
    print(f"  送電線距離: mean={score.mean():.1f}")
    return score


def score_substation_distance(mesh: gpd.GeoDataFrame,
                               subs: gpd.GeoDataFrame, epsg: int) -> np.ndarray:
    hv = subs[(subs["voltage_kv"] >= 66) | (subs["voltage_kv"] == 0)].copy()
    if len(hv) == 0:
        print("  変電所: データなし → デフォルト50")
        return np.full(len(mesh), 50.0)

    mesh_p = mesh.to_crs(epsg=epsg)
    sub_union = hv.to_crs(epsg=epsg).geometry.centroid.union_all()
    dist = mesh_p.geometry.centroid.distance(sub_union).values

    bp = [(0, 100), (2000, 80), (5000, 50), (10000, 20), (20000, 0)]
    score = _dist_score(dist, bp)
    print(f"  変電所距離: mean={score.mean():.1f}")
    return score


def score_land_use(mesh: gpd.GeoDataFrame, pref: str) -> np.ndarray:
    tif = get_land_dir(pref) / "land_use" / "osm_land_use.tif"
    if not tif.exists():
        print("  土地利用: データなし → デフォルト70")
        return np.full(len(mesh), 70.0)

    coords = list(zip(mesh["cx"], mesh["cy"]))
    score = _sample_tif(tif, coords)
    score = np.nan_to_num(score, nan=70.0)
    print(f"  土地利用: mean={score.mean():.1f}")
    return score


def score_residential_distance(mesh: gpd.GeoDataFrame,
                                pref: str, epsg: int) -> np.ndarray:
    """騒音バッファ: 500m 以内は完全排除 (スコア 0)"""
    tif = get_land_dir(pref) / "land_use" / "residential_mask.tif"
    if not tif.exists():
        print("  居住地距離: データなし → デフォルト70")
        return np.full(len(mesh), 70.0)

    coords = list(zip(mesh["cx"], mesh["cy"]))
    flag = _sample_tif(tif, coords)
    res_idx = np.where(flag > 0)[0]

    if len(res_idx) == 0:
        print("  居住地距離: 居住地なし → デフォルト70")
        return np.full(len(mesh), 70.0)

    mesh_p = mesh.to_crs(epsg=epsg)
    res_union = mesh_p.iloc[res_idx].geometry.centroid.union_all()
    dist = mesh_p.geometry.centroid.distance(res_union).values.astype(np.float32)

    score = np.where(dist < 500, 0,
            np.where(dist < 1000, 30,
            np.where(dist < 2000, 70,
            np.where(dist < 3000, 90, 100)))).astype(np.float32)

    n_ex = (score == 0).sum()
    print(f"  居住地距離: mean={score.mean():.1f}, 排除={n_ex} ({n_ex/len(mesh)*100:.1f}%)")
    return score


def score_road_distance(mesh: gpd.GeoDataFrame, pref: str) -> np.ndarray:
    """道路距離スコア。fetch_osm_road.py で生成した TIF を使用。"""
    road_tif = get_land_dir(pref) / "road" / f"{pref}_road_score.tif"
    if not road_tif.exists():
        print("  道路距離: データなし → デフォルト50 (fetch_osm_road.py を実行してください)")
        return np.full(len(mesh), 50.0)

    coords = list(zip(mesh["cx"], mesh["cy"]))
    score = np.nan_to_num(_sample_tif(road_tif, coords), nan=50.0)
    print(f"  道路距離: mean={score.mean():.1f}")
    return score


# ── メッシュ計算メイン ─────────────────────────────────────────

def compute_mesh_wind(pref: str, resolution_m: int) -> gpd.GeoDataFrame:
    cfg = get_pref_config(pref)
    epsg = cfg.get("epsg_jpc", 6677)
    out_dir = get_output_dir(pref)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"{cfg['name_ja']}  {resolution_m}m メッシュ (風力 AHP)")
    print(f"{'='*60}")

    mesh = create_mesh(pref, resolution_m)
    mesh = clip_to_prefecture(mesh, pref)
    if len(mesh) == 0:
        print("  WARNING: クリップ後メッシュが空")
        return mesh

    # 系統データ
    grid_dir = get_grid_dir(pref)

    def _load(name):
        try:
            gdf = gpd.read_file(grid_dir / f"{pref}_{name}.geojson")
            gdf["voltage_kv"] = gdf["voltage"].apply(_extract_voltage_kv)
            return gdf
        except Exception as e:
            print(f"  WARNING: {name} 読み込み失敗: {e}")
            return gpd.GeoDataFrame(columns=["geometry", "voltage_kv"])

    lines = _load("lines")
    subs = _load("substations")

    # AHP スコア計算
    print("\n[スコア計算]")
    W = WEIGHTS
    mesh["score_wind_speed"]        = score_wind_speed(mesh, pref)
    mesh["score_slope"]             = score_slope(mesh, pref)
    mesh["score_elevation"]         = score_elevation(mesh, pref)
    mesh["score_grid_dist"]         = score_grid_distance(mesh, lines, epsg)
    mesh["score_sub_dist"]          = score_substation_distance(mesh, subs, epsg)
    mesh["score_land_use"]          = score_land_use(mesh, pref)
    mesh["score_residential_dist"]  = score_residential_distance(mesh, pref, epsg)
    mesh["score_road"]              = score_road_distance(mesh, pref)
    mesh["score_protection"]        = 80.0  # デフォルト固定

    mesh["total_score"] = np.clip(
        mesh["score_wind_speed"]       * W["wind_speed"]
        + mesh["score_slope"]          * W["slope"]
        + mesh["score_elevation"]      * W["elevation"]
        + mesh["score_grid_dist"]      * W["grid_distance"]
        + mesh["score_sub_dist"]       * W["substation_distance"]
        + mesh["score_land_use"]       * W["land_use"]
        + mesh["score_residential_dist"] * W["residential_distance"]
        + mesh["score_road"]           * W["road_distance"]
        + mesh["score_protection"]     * W["protection"],
        0, 100,
    ).round(1)

    print(f"\n  総合スコア: "
          f"min={mesh['total_score'].min():.1f}  "
          f"max={mesh['total_score'].max():.1f}  "
          f"mean={mesh['total_score'].mean():.1f}")

    mesh["score_class"] = pd.cut(mesh["total_score"], bins=SCORE_BINS, labels=SCORE_LABELS)
    for cls in SCORE_LABELS:
        n = int((mesh["score_class"] == cls).sum())
        print(f"    {cls}: {n:,} ({n/len(mesh)*100:.1f}%)")

    out = out_dir / f"{pref}_mesh_{resolution_m}m.geojson"
    mesh.to_file(out, driver="GeoJSON")
    print(f"  → {out.name}")
    return mesh


# ── Folium マップ ──────────────────────────────────────────────

def build_wind_map(
    pref: str,
    meshes: dict,
    lines: gpd.GeoDataFrame,
    subs: gpd.GeoDataFrame,
    plants: gpd.GeoDataFrame | None = None,
) -> str:
    import folium

    cfg = get_pref_config(pref)
    out_dir = get_output_dir(pref)

    m = folium.Map(location=cfg["center"], zoom_start=9, tiles=None)

    for name, url, attr in [
        ("CartoDB",       "cartodbpositron",
         "CartoDB"),
        ("OSM",           "OpenStreetMap",
         "OSM"),
        ("Google 地形図", "https://mt1.google.com/vt/lyrs=p&x={x}&y={y}&z={z}",
         "Google"),
        ("Google 衛星",   "https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}",
         "Google"),
        ("国土地理院",    "https://cyberjapandata.gsi.go.jp/xyz/std/{z}/{x}/{y}.png",
         "GSI"),
        ("国土地理院 航空写真",
         "https://cyberjapandata.gsi.go.jp/xyz/seamlessphoto/{z}/{x}/{y}.jpg",
         "GSI"),
    ]:
        folium.TileLayer(tiles=url, name=name, attr=attr).add_to(m)

    # メッシュレイヤー
    for res, mesh in sorted(meshes.items()):
        show = res == min(meshes.keys())
        fg = folium.FeatureGroup(name=f"適地スコア {res}m (風力)", show=show)

        disp = mesh[[
            "geometry", "total_score",
            "score_wind_speed", "score_slope", "score_elevation",
            "score_grid_dist", "score_sub_dist", "score_land_use",
            "score_residential_dist",
        ]].copy()
        disp["_color"] = disp["total_score"].apply(_mesh_color)
        disp["_tip"] = disp.apply(
            lambda r: (
                f"総合:{r['total_score']:.0f} "
                f"| 風速:{r['score_wind_speed']:.0f}"
                f" 傾斜:{r['score_slope']:.0f}"
                f" 送電:{r['score_grid_dist']:.0f}"
                f" 変電:{r['score_sub_dist']:.0f}"
                f" 土地:{r['score_land_use']:.0f}"
                f" 居住:{r['score_residential_dist']:.0f}"
                f" 標高:{r['score_elevation']:.0f}"
            ),
            axis=1,
        )

        folium.GeoJson(
            disp[["geometry", "_color", "_tip"]].to_json(),
            style_function=lambda feat: {
                "fillColor":   feat["properties"]["_color"],
                "color":       feat["properties"]["_color"],
                "weight":      0.2,
                "fillOpacity": 0.55,
            },
            tooltip=folium.GeoJsonTooltip(fields=["_tip"], aliases=[""], labels=False),
        ).add_to(fg)
        fg.add_to(m)

    # 送電線 (電圧別色分け)
    VCOL = {500: "#cc0000", 275: "#e65100", 220: "#ff8f00",
            154: "#1565c0", 110: "#0288d1", 66: "#2e7d32"}

    def _vcolor(v: float) -> str:
        for kv, col in sorted(VCOL.items(), reverse=True):
            if v >= kv:
                return col
        return "#9e9e9e"

    fg_lines = folium.FeatureGroup(name="送電線 (66kV+)", show=True)
    for _, row in lines[(lines["voltage_kv"] >= 66) | (lines["voltage_kv"] == 0)].iterrows():
        v = row["voltage_kv"]
        name = str(row.get("name", "") or "")
        geom = row.geometry
        if geom is None:
            continue
        coords = []
        if geom.geom_type == "LineString":
            coords = [(c[1], c[0]) for c in geom.coords]
        elif geom.geom_type == "MultiLineString":
            for part in geom.geoms:
                coords.extend([(c[1], c[0]) for c in part.coords])
        if not coords:
            continue
        w_ = 4 if v >= 500 else 3 if v >= 275 else 2.5 if v >= 154 else 1.5
        folium.PolyLine(
            coords, color=_vcolor(v), weight=w_, opacity=0.75,
            tooltip=f"{name} ({v:.0f}kV)" if name else None,
        ).add_to(fg_lines)
    fg_lines.add_to(m)

    # 変電所
    fg_subs = folium.FeatureGroup(name="変電所 (66kV+)", show=True)
    for _, row in subs[(subs["voltage_kv"] >= 66) | (subs["voltage_kv"] == 0)].iterrows():
        v = row["voltage_kv"]
        name = str(row.get("name", "") or f"変電所 ({v:.0f}kV)")
        ctr = row.geometry.centroid
        col = _vcolor(v)
        r_ = 8 if v >= 275 else 6 if v >= 154 else 4
        folium.CircleMarker(
            [ctr.y, ctr.x], radius=r_,
            color=col, fill=True, fill_color=col, fill_opacity=0.65, weight=1.5,
            tooltip=name,
        ).add_to(fg_subs)
        if v >= 66:
            fs = 11 if v >= 275 else 10 if v >= 154 else 8
            folium.Marker(
                [ctr.y, ctr.x],
                icon=folium.DivIcon(
                    html=(
                        f'<div style="font-size:{fs}px;font-weight:bold;color:{col};'
                        f'white-space:nowrap;text-shadow:1px 1px 1px white,'
                        f'-1px -1px 1px white,1px -1px 1px white,-1px 1px 1px white;">'
                        f'{name}</div>'
                    ),
                    icon_size=(0, 0), icon_anchor=(0, -10),
                ),
            ).add_to(fg_subs)
    fg_subs.add_to(m)

    # 発電所 (風力を強調)
    if plants is not None and len(plants) > 0:
        FUEL_COLORS = {
            "wind": "#66bb6a", "solar": "#f5c542", "hydro": "#42a5f5",
            "coal": "#ef5350", "gas": "#ef5350", "nuclear": "#ab47bc",
        }
        fg_wind_p = folium.FeatureGroup(name="既設風力発電所", show=True)
        fg_other_p = folium.FeatureGroup(name="その他発電所", show=False)
        for _, row in plants.iterrows():
            fuel = (row.get("fuel_type", "") or "").lower()
            name = str(row.get("_display_name", row.get("name", "")) or f"発電所 ({fuel})")
            pt = row.geometry
            if pt is None:
                continue
            if pt.geom_type != "Point":
                pt = pt.centroid
            col = FUEL_COLORS.get(fuel, "#78909c")
            mk = folium.CircleMarker(
                [pt.y, pt.x],
                radius=6 if fuel == "wind" else 3,
                color=col, fill=True, fill_color=col, fill_opacity=0.85, weight=1,
                tooltip=f"<b>{name}</b> ({fuel})",
            )
            (fg_wind_p if fuel == "wind" else fg_other_p).add_child(mk)
        fg_wind_p.add_to(m)
        fg_other_p.add_to(m)

    # 送電線バッファ (系統近接性)
    hv = lines[lines["voltage_kv"] >= 154].copy()
    if len(hv) > 0:
        fg_buf = folium.FeatureGroup(name="送電線バッファ (154kV+)", show=False)
        hv_p = hv.to_crs(epsg=cfg["epsg_jpc"])
        prev = None
        for dist_m, col, lbl in [
            (20000, "#DC143C", "20km"), (10000, "#DAA520", "10km"),
            (5000,  "#228B22",  "5km"), (1000,  "#006400",  "1km"),
        ]:
            buf_g = gpd.GeoDataFrame(
                geometry=[hv_p.buffer(dist_m).union_all()],
                crs=f"EPSG:{cfg['epsg_jpc']}",
            ).to_crs(epsg=4326)
            ring = buf_g.geometry.iloc[0]
            if prev is not None:
                ring = ring.difference(prev)
            folium.GeoJson(
                gpd.GeoDataFrame(geometry=[ring.simplify(0.0003)], crs="EPSG:4326").to_json(),
                style_function=lambda x, c=col: {
                    "fillColor": c, "color": c, "weight": 0.3, "fillOpacity": 0.18,
                },
                tooltip=f"154kV+ から {lbl} 圏内",
            ).add_to(fg_buf)
            prev = buf_g.geometry.iloc[0]
        fg_buf.add_to(m)

    # 凡例
    res_text = " / ".join(f"{r}m" for r in sorted(meshes.keys()))
    legend_html = f"""
<div style="position:fixed;bottom:30px;left:30px;z-index:1000;
            background:white;padding:12px;border:2px solid #333;
            border-radius:6px;font-size:11px;opacity:0.93;max-width:270px;">
  <b style="font-size:12px">{cfg['name_ja']} 風力適地スコア</b><br>
  <small>メッシュ: {res_text} | GIS-MCDA (AHP)</small>
  <hr style="margin:4px 0">
  <i style="background:#006400;width:14px;height:14px;display:inline-block;border-radius:2px"></i> 80-100 最適<br>
  <i style="background:#228B22;width:14px;height:14px;display:inline-block;border-radius:2px"></i> 60-79 良好<br>
  <i style="background:#DAA520;width:14px;height:14px;display:inline-block;border-radius:2px"></i> 40-59 中程度<br>
  <i style="background:#FF8C00;width:14px;height:14px;display:inline-block;border-radius:2px"></i> 20-39 低い<br>
  <i style="background:#DC143C;width:14px;height:14px;display:inline-block;border-radius:2px"></i> 0-19 不適<br>
  <hr style="margin:4px 0">
  <b>AHP 重み</b><br>
  <small>
  風速 30% | 送電線距離 15% | 傾斜 12%<br>
  変電所距離 10% | 土地利用 10%<br>
  居住地距離 8% | 標高 5%<br>
  道路距離 5% | 保護区域 5%
  </small>
  <hr style="margin:4px 0">
  <b>送電線</b>
  <span style="color:#cc0000">━ 500kV</span>
  <span style="color:#e65100">━ 275kV</span>
  <span style="color:#1565c0">━ 154kV</span>
  <span style="color:#2e7d32">━ 66kV</span>
</div>
"""
    m.get_root().html.add_child(folium.Element(legend_html))
    folium.LayerControl(collapsed=False).add_to(m)

    out_path = out_dir / f"{pref}_wind_mesh_map.html"
    m.save(str(out_path))
    print(f"  マップ保存: {out_path}")
    return str(out_path)


# ── CLI ───────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="風力発電適地メッシュ評価")
    parser.add_argument("-p", "--prefecture", required=True,
                        choices=list(PREFECTURES.keys()))
    parser.add_argument(
        "-r", "--resolution", type=int, default=1000,
        help="メッシュ解像度 (m)。0 = 1000m + 500m 両方生成",
    )
    parser.add_argument("--no-map", action="store_true",
                        help="Folium マップ生成をスキップ")
    args = parser.parse_args()

    pref = args.prefecture
    resolutions = [1000, 500] if args.resolution == 0 else [args.resolution]

    meshes = {}
    for res in resolutions:
        meshes[res] = compute_mesh_wind(pref, res)

    if not args.no_map and any(len(m) > 0 for m in meshes.values()):
        grid_dir = get_grid_dir(pref)

        def _load(name):
            try:
                gdf = gpd.read_file(grid_dir / f"{pref}_{name}.geojson")
                gdf["voltage_kv"] = gdf["voltage"].apply(_extract_voltage_kv)
                return gdf
            except Exception:
                return gpd.GeoDataFrame(columns=["geometry", "voltage_kv"])

        lines = _load("lines")
        subs = _load("substations")
        plants_path = grid_dir / f"{pref}_plants.geojson"
        plants = gpd.read_file(plants_path) if plants_path.exists() else None

        try:
            build_wind_map(pref, meshes, lines, subs, plants)
        except Exception as e:
            print(f"  マップ生成エラー: {e}")


if __name__ == "__main__":
    main()
