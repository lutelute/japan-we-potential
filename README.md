# 全国風力発電適地ポテンシャル評価

GIS-MCDA (AHP) 手法による全国47都道府県の陸上風力発電 導入・適地ポテンシャル評価。

**ライブマップ**: https://lutelute.github.io/japan-we-potential/

---

## AHP 評価基準

| 基準 | 重み | データソース | スコア化ロジック |
|------|------|------------|----------------|
| 風速 | **30%** | ERA5 100m高さ月次平均 (2014-2023) | 1.5→0, 5.0 m/s→100 (線形) |
| 送電線距離 | 15% | All-Japan-Grid (OSM由来) 154kV+ | 0m→100, 5km→50, 20km→0 |
| 傾斜 | 12% | SRTM 30m DEM | <5°→100, 15-20°→30, >30°→0 |
| 変電所距離 | 10% | All-Japan-Grid 66kV+ | 0m→100, 5km→50, 20km→0 |
| 土地利用 | 10% | OSM Overpass API | 建物→0, 農地→60, 荒地→90 |
| 居住地距離 | 8% | OSM (building/residential) | <500m→0 (排除), >3km→100 |
| 標高 | 5% | SRTM DEM | 500-1000m→100 (尾根最適) |
| 道路距離 | 5% | OSM (motorway〜secondary) | 0m→100, 15km→0 |
| 保護区域 | 5% | — | デフォルト固定 (要実装) |

太陽光版 ([japan-re-potential](https://github.com/lutelute/japan-re-potential)) との主な違い:
風速が最重要基準 (30%)、騒音バッファとして居住地距離 (8%) を追加、標高は尾根優先に逆転。

---

## セットアップ

```bash
# 仮想環境
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# All-Japan-Grid (送電線データ)
export ALL_JAPAN_GRID_DIR=/path/to/All-Japan-Grid/data
```

---

## 実行方法

### 1. 単県テスト (ローカルPC)

```bash
# 風速フォールバック使用 (CDS APIキー不要)
python src/batch_wind.py -p fukui --resolution 30 --wind-fallback

# ERA5実データ使用 (要~/.cdsapirc)
python src/batch_wind.py -p fukui,akita --resolution 30

# 道路データ取得
python src/fetch_osm_road.py -p fukui
```

### 2. 全国実行 (**pws-160core 推奨**)

```bash
# pws-160coreでSSH後
python src/batch_wind.py --resolution 30 --workers 8 --resume
```

> ⚠️ このMacで全国実行すると数時間のCPU高負荷になります。
> pws-160core (80C/160T) または pws-gpu (24T + RTX4090) を使用してください。

### 3. タイル生成 (**pws-160coreまたはpws-gpuで実行推奨**)

```bash
# PNG生成 (全県 × 8モード → docs/*.png)
python gen_score_pngs_wind.py

# 全国統合XYZタイル生成
python gen_tiles_wind.py               # z=6-13 (デフォルト)
python gen_tiles_wind.py --zoom 6-9    # 軽量版 (GitHub Pages用)
python gen_tiles_wind.py --mode total  # totalモードのみ
```

### 4. メッシュ適地評価 (市町村・地域分析用)

```bash
# 1kmメッシュ + Foliumマップ生成
python src/mesh_suitability_wind.py -p fukui

# 1km + 500m 両方
python src/mesh_suitability_wind.py -p fukui -r 0
```

---

## リソース分担ガイド

| タスク | 推定時間 | 推奨環境 |
|--------|---------|---------|
| 単県スコア計算 (30m) | 5-10分 | ローカル可 |
| 全国スコア計算 (30m) | 3-6時間 | **pws-160core** (workers=8以上) |
| 全国スコア計算 (5m) | 数日 | **pws-160core** (workers=2-4) |
| タイル生成 z=6-9 | 10-20分 | ローカル可 |
| タイル生成 z=6-13 | 1-3時間 | **pws-160core** |
| ERA5ダウンロード | 数時間 | ローカル可 (I/O待ち) |
| OSMデータ取得 | 数時間 | ローカル可 (API律速) |

pws-160core 接続:
```bash
ssh pws-ubuntu-server@10.0.70.42     # LAN
ssh pws-ubuntu-server@100.104.225.55  # Tailscale
```

---

## ファイル構成

```
src/
  config.py              — 県設定 (60ユニット) + AHP重み
  batch_wind.py          — 全国バッチオーケストレーター (3フェーズ)
  raster_score_wind.py   — ラスタースコア計算エンジン
  mesh_suitability_wind.py — メッシュ適地評価 (ベクトル化)
  fetch_osm_road.py      — OSM道路データ取得 + 距離スコアTIF生成
  fetch_osm_land_use.py  — OSM土地利用データ取得
  download_wind_data.py  — ERA5風速データ取得
  download_land_data.py  — DEM/行政区域ダウンロード
  extract_grid.py        — All-Japan-Grid から電力系統データ抽出
  slope_analysis.py      — SRTM DEM から傾斜角計算

gen_score_pngs_wind.py   — output/TIF → docs/PNG 変換
gen_tiles_wind.py        — 全国統合 XYZ タイル生成
docs/                    — GitHub Pages (Leaflet.js ビューアー)
output/                  — 計算結果 GeoTIFF (.gitignore)
data/                    — 入力データ (.gitignore)
```

---

## ERA5 セットアップ

1. [CDS](https://cds.climate.copernicus.eu/) でアカウント作成
2. `pip install cdsapi xarray`
3. `~/.cdsapirc` にAPIキーを設定:
   ```
   url: https://cds.climate.copernicus.eu/api/v2
   key: <uid>:<api-key>
   ```
4. フォールバック: `--wind-fallback` で標高ベース推定 (テスト用)

---

## 参考プロジェクト

- **[japan-re-potential](../japan-re-potential)**: 太陽光版。DEM/slope/grid データを共有利用
- **[All-Japan-Grid](../All-Japan-Grid)**: OSM由来の全国送電網データセット
