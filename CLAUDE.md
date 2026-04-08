# japan-we-potential — 全国風力発電ポテンシャル評価

## 概要
GIS-MCDA (AHP) 手法による全国47都道府県の陸上風力発電 導入ポテンシャル・適地ポテンシャル評価。
参考プロジェクト `japan-re-potential` (太陽光版) のインフラを共有利用し、風力固有の評価基準を追加。

## AHP 評価基準 (風力版)

| 基準 | 重み | 説明 | スコアリング |
|------|------|------|------------|
| 風速 | 30% | ERA5 100m高さ年間平均 | <4m/s=0, 5-6=55, 7-8=90, >8=100 |
| 送電線距離 | 15% | 154kV+への距離 | 0m=100, 5km=50, 20km=0 |
| 傾斜 | 12% | SRTM由来 | <5°=100, 15-20°=30, >30°=0 |
| 変電所距離 | 10% | 66kV+への距離 | 0m=100, 5km=50, 20km=0 |
| 土地利用 | 10% | OSM由来 | 建物=0, 森林=20, 農地=60, 荒地=90 |
| 居住地距離 | 8% | 騒音バッファ | <500m=0(排除), 1-2km=70, >3km=100 |
| 標高 | 5% | 尾根風考慮 | 500-1000m=100(最適), >2000m=10 |
| 道路距離 | 5% | デフォルト50 | (未実装) |
| 保護区域 | 5% | デフォルト80 | (未実装) |

## データソース

| レイヤー | ソース | 形式 |
|----------|--------|------|
| 風速 | ERA5 再解析 (CDS API) | NetCDF → GeoTIFF |
| DEM/傾斜 | SRTM 30m | HGT → GeoTIFF |
| 送電線/変電所 | All-Japan-Grid | GeoJSON |
| 土地利用 | OSM Overpass API | GeoJSON → GeoTIFF |
| 行政区域 | 国土数値情報 N03 | Shapefile |
| 居住地マスク | OSM (building/residential) | GeoTIFF |

## ファイル構成

```
src/
  config.py              — 県設定 + AHP重み
  download_land_data.py  — DEM/行政区域ダウンロード
  extract_grid.py        — 送電線データ抽出
  slope_analysis.py      — 傾斜計算
  fetch_osm_land_use.py  — OSM土地利用 + 居住地マスク
  download_wind_data.py  — ERA5風速データ取得
  raster_score_wind.py   — 風力AHPスコア計算
  batch_wind.py          — バッチオーケストレーター
```

## 実行方法

### 単県テスト (30m)
```bash
python src/batch_wind.py -p fukui --resolution 30 --wind-fallback
```

### ERA5使用 (要CDS APIキー)
```bash
python src/batch_wind.py -p fukui,akita --resolution 30
```

### 全国実行 (サーバー)
```bash
python src/batch_wind.py --resolution 5 --resume --workers 2
```

## 参考プロジェクトとのデータ共有
- `japan-re-potential/data/{pref}/` のDEM, slope, grid, land_use を
  シンボリックリンクで共有利用
- 風速データのみ新規取得が必要

## ERA5 セットアップ
1. https://cds.climate.copernicus.eu/ でアカウント作成
2. `pip install cdsapi`
3. `~/.cdsapirc` にAPIキー設定
4. `--wind-fallback` で標高ベース推定も可能 (テスト用)
