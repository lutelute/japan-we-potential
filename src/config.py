"""
全国47都道府県 風力発電ポテンシャル評価 — 県別設定

AHP重みは陸上風力発電向けに設定:
  - 風速が最重要 (30%)
  - 送電線・変電所距離 (25%)
  - 傾斜・土地利用 (22%)
  - 居住地距離・標高・保護区 (23%)
"""
import math
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# 参考プロジェクト (japan-re-potential) のデータを共有利用
RE_POTENTIAL_ROOT = PROJECT_ROOT.parent / "japan-re-potential"


def compute_srtm_tiles(bbox: tuple) -> list[str]:
    """bboxからSRTMタイル名を自動計算する。"""
    west, south, east, north = bbox
    lat_min = int(math.floor(south))
    lat_max = int(math.floor(north))
    lon_min = int(math.floor(west))
    lon_max = int(math.floor(east))

    tiles = []
    for lat in range(lat_min, lat_max + 1):
        for lon in range(lon_min, lon_max + 1):
            ns = "N" if lat >= 0 else "S"
            ew = "E" if lon >= 0 else "W"
            tiles.append(f"{ns}{abs(lat):02d}{ew}{abs(lon):03d}")
    return sorted(tiles)


PREFECTURES = {
    # ============================================================
    # 北海道 — 14振興局に分割
    # ============================================================
    "hokkaido_ishikari": {
        "name_ja": "北海道（石狩）",
        "code": "01",
        "bbox": (140.94, 42.63, 141.92, 43.79),
        "center": [43.20, 141.35],
        "epsg_jpc": 6680,
        "grid_area": "hokkaido",
    },
    "hokkaido_sorachi": {
        "name_ja": "北海道（空知）",
        "code": "01",
        "bbox": (141.44, 42.79, 142.42, 44.1),
        "center": [43.45, 142.05],
        "epsg_jpc": 6680,
        "grid_area": "hokkaido",
    },
    "hokkaido_shiribeshi": {
        "name_ja": "北海道（後志）",
        "code": "01",
        "bbox": (139.77, 42.49, 141.34, 43.43),
        "center": [43.00, 140.40],
        "epsg_jpc": 6680,
        "grid_area": "hokkaido",
    },
    "hokkaido_iburi": {
        "name_ja": "北海道（胆振）",
        "code": "01",
        "bbox": (140.48, 42.25, 142.38, 43.04),
        "center": [42.55, 141.05],
        "epsg_jpc": 6680,
        "grid_area": "hokkaido",
    },
    "hokkaido_hidaka": {
        "name_ja": "北海道（日高）",
        "code": "01",
        "bbox": (141.92, 41.87, 143.38, 43.06),
        "center": [42.45, 142.65],
        "epsg_jpc": 6680,
        "grid_area": "hokkaido",
    },
    "hokkaido_oshima": {
        "name_ja": "北海道（渡島）",
        "code": "01",
        "bbox": (139.28, 41.3, 141.24, 42.68),
        "center": [41.80, 140.50],
        "epsg_jpc": 6680,
        "grid_area": "hokkaido",
    },
    "hokkaido_hiyama": {
        "name_ja": "北海道（檜山）",
        "code": "01",
        "bbox": (139.35, 41.53, 140.52, 42.67),
        "center": [42.10, 139.90],
        "epsg_jpc": 6680,
        "grid_area": "hokkaido",
    },
    "hokkaido_kamikawa": {
        "name_ja": "北海道（上川）",
        "code": "01",
        "bbox": (141.93, 42.83, 143.23, 44.96),
        "center": [43.65, 142.75],
        "epsg_jpc": 6680,
        "grid_area": "hokkaido",
    },
    "hokkaido_rumoi": {
        "name_ja": "北海道（留萌）",
        "code": "01",
        "bbox": (141.24, 43.63, 142.17, 45.04),
        "center": [44.05, 141.80],
        "epsg_jpc": 6680,
        "grid_area": "hokkaido",
    },
    "hokkaido_soya": {
        "name_ja": "北海道（宗谷）",
        "code": "01",
        "bbox": (140.91, 44.53, 142.89, 45.58),
        "center": [45.05, 142.10],
        "epsg_jpc": 6680,
        "grid_area": "hokkaido",
    },
    "hokkaido_okhotsk": {
        "name_ja": "北海道（オホーツク）",
        "code": "01",
        "bbox": (142.53, 43.41, 145.39, 44.74),
        "center": [44.20, 143.75],
        "epsg_jpc": 6680,
        "grid_area": "hokkaido",
    },
    "hokkaido_tokachi": {
        "name_ja": "北海道（十勝）",
        "code": "01",
        "bbox": (142.62, 42.11, 144.08, 43.69),
        "center": [42.90, 143.05],
        "epsg_jpc": 6680,
        "grid_area": "hokkaido",
    },
    "hokkaido_kushiro": {
        "name_ja": "北海道（釧路）",
        "code": "01",
        "bbox": (143.65, 42.79, 145.38, 43.75),
        "center": [43.15, 144.30],
        "epsg_jpc": 6680,
        "grid_area": "hokkaido",
    },
    "hokkaido_nemuro": {
        "name_ja": "北海道（根室）",
        "code": "01",
        "bbox": (144.52, 43.11, 148.94, 45.61),
        "center": [43.30, 145.30],
        "epsg_jpc": 6680,
        "grid_area": "hokkaido",
    },
    # ============================================================
    # 東北
    # ============================================================
    "aomori": {
        "name_ja": "青森県",
        "code": "02",
        "bbox": (139.45, 40.17, 141.73, 41.61),
        "center": [40.82, 140.74],
        "epsg_jpc": 6678,
        "grid_area": "tohoku",
    },
    "iwate": {
        "name_ja": "岩手県",
        "code": "03",
        "bbox": (140.6, 38.7, 142.12, 40.5),
        "center": [39.70, 141.15],
        "epsg_jpc": 6678,
        "grid_area": "tohoku",
    },
    "miyagi": {
        "name_ja": "宮城県",
        "code": "04",
        "bbox": (140.22, 37.72, 141.73, 39.05),
        "center": [38.27, 140.87],
        "epsg_jpc": 6678,
        "grid_area": "tohoku",
    },
    "akita": {
        "name_ja": "秋田県",
        "code": "05",
        "bbox": (139.64, 38.82, 141.05, 40.56),
        "center": [39.72, 140.10],
        "epsg_jpc": 6678,
        "grid_area": "tohoku",
    },
    "yamagata": {
        "name_ja": "山形県",
        "code": "06",
        "bbox": (139.47, 37.68, 140.7, 39.27),
        "center": [38.24, 140.33],
        "epsg_jpc": 6678,
        "grid_area": "tohoku",
    },
    "fukushima": {
        "name_ja": "福島県",
        "code": "07",
        "bbox": (139.11, 36.74, 141.1, 38.03),
        "center": [37.75, 140.47],
        "epsg_jpc": 6677,
        "grid_area": "tohoku",
    },
    # ============================================================
    # 関東
    # ============================================================
    "ibaraki": {
        "name_ja": "茨城県",
        "code": "08",
        "bbox": (139.64, 35.69, 140.9, 37.0),
        "center": [36.34, 140.45],
        "epsg_jpc": 6677,
        "grid_area": "tokyo",
    },
    "tochigi": {
        "name_ja": "栃木県",
        "code": "09",
        "bbox": (139.28, 36.15, 140.34, 37.21),
        "center": [36.65, 139.88],
        "epsg_jpc": 6677,
        "grid_area": "tokyo",
    },
    "gunma": {
        "name_ja": "群馬県",
        "code": "10",
        "bbox": (138.35, 35.94, 139.72, 37.11),
        "center": [36.39, 139.06],
        "epsg_jpc": 6677,
        "grid_area": "tokyo",
    },
    "saitama": {
        "name_ja": "埼玉県",
        "code": "11",
        "bbox": (138.66, 35.7, 139.95, 36.33),
        "center": [35.86, 139.65],
        "epsg_jpc": 6677,
        "grid_area": "tokyo",
    },
    "chiba": {
        "name_ja": "千葉県",
        "code": "12",
        "bbox": (139.69, 34.85, 140.93, 36.15),
        "center": [35.60, 140.12],
        "epsg_jpc": 6677,
        "grid_area": "tokyo",
    },
    "tokyo": {
        "name_ja": "東京都",
        "code": "13",
        "bbox": (138.94, 35.50, 139.92, 35.90),
        "center": [35.68, 139.69],
        "epsg_jpc": 6677,
        "grid_area": "tokyo",
    },
    "kanagawa": {
        "name_ja": "神奈川県",
        "code": "14",
        "bbox": (138.87, 35.08, 139.89, 35.72),
        "center": [35.45, 139.64],
        "epsg_jpc": 6677,
        "grid_area": "tokyo",
    },
    # ============================================================
    # 中部
    # ============================================================
    "niigata": {
        "name_ja": "新潟県",
        "code": "15",
        "bbox": (137.58, 36.69, 139.95, 38.6),
        "center": [37.90, 139.02],
        "epsg_jpc": 6676,
        "grid_area": "tohoku",
    },
    "toyama": {
        "name_ja": "富山県",
        "code": "16",
        "bbox": (136.72, 36.22, 137.81, 37.03),
        "center": [36.70, 137.21],
        "epsg_jpc": 6675,
        "grid_area": "hokuriku",
    },
    "ishikawa": {
        "name_ja": "石川県",
        "code": "17",
        "bbox": (136.19, 36.02, 137.42, 37.91),
        "center": [36.59, 136.63],
        "epsg_jpc": 6675,
        "grid_area": "hokuriku",
    },
    "fukui": {
        "name_ja": "福井県",
        "code": "18",
        "bbox": (135.4, 35.29, 136.88, 36.35),
        "center": [35.85, 136.22],
        "epsg_jpc": 6675,
        "grid_area": "hokuriku",
    },
    "yamanashi": {
        "name_ja": "山梨県",
        "code": "19",
        "bbox": (138.13, 35.12, 139.18, 36.02),
        "center": [35.66, 138.57],
        "epsg_jpc": 6676,
        "grid_area": "tokyo",
    },
    "nagano": {
        "name_ja": "長野県",
        "code": "20",
        "bbox": (137.27, 35.15, 138.79, 37.08),
        "center": [36.23, 138.18],
        "epsg_jpc": 6676,
        "grid_area": "chubu",
    },
    "gifu": {
        "name_ja": "岐阜県",
        "code": "21",
        "bbox": (136.23, 35.08, 137.7, 36.52),
        "center": [35.39, 136.72],
        "epsg_jpc": 6675,
        "grid_area": "chubu",
    },
    "shizuoka": {
        "name_ja": "静岡県",
        "code": "22",
        "bbox": (137.42, 34.52, 139.23, 35.7),
        "center": [34.98, 138.38],
        "epsg_jpc": 6676,
        "grid_area": "chubu",
    },
    "aichi": {
        "name_ja": "愛知県",
        "code": "23",
        "bbox": (136.62, 34.52, 137.89, 35.47),
        "center": [35.18, 136.91],
        "epsg_jpc": 6675,
        "grid_area": "chubu",
    },
    # ============================================================
    # 近畿
    # ============================================================
    "mie": {
        "name_ja": "三重県",
        "code": "24",
        "bbox": (135.8, 33.67, 137.04, 35.31),
        "center": [34.73, 136.51],
        "epsg_jpc": 6674,
        "grid_area": "chubu",
    },
    "shiga": {
        "name_ja": "滋賀県",
        "code": "25",
        "bbox": (135.71, 34.74, 136.51, 35.75),
        "center": [35.00, 135.87],
        "epsg_jpc": 6674,
        "grid_area": "kansai",
    },
    "kyoto": {
        "name_ja": "京都府",
        "code": "26",
        "bbox": (134.8, 34.66, 136.11, 35.83),
        "center": [35.02, 135.76],
        "epsg_jpc": 6674,
        "grid_area": "kansai",
    },
    "osaka": {
        "name_ja": "大阪府",
        "code": "27",
        "bbox": (135.04, 34.22, 135.8, 35.1),
        "center": [34.69, 135.52],
        "epsg_jpc": 6674,
        "grid_area": "kansai",
    },
    "hyogo": {
        "name_ja": "兵庫県",
        "code": "28",
        "bbox": (134.2, 34.11, 135.52, 35.72),
        "center": [34.69, 135.18],
        "epsg_jpc": 6674,
        "grid_area": "kansai",
    },
    "nara": {
        "name_ja": "奈良県",
        "code": "29",
        "bbox": (135.49, 33.81, 136.28, 34.83),
        "center": [34.69, 135.83],
        "epsg_jpc": 6674,
        "grid_area": "kansai",
    },
    "wakayama": {
        "name_ja": "和歌山県",
        "code": "30",
        "bbox": (134.95, 33.38, 136.06, 34.43),
        "center": [33.95, 135.17],
        "epsg_jpc": 6674,
        "grid_area": "kansai",
    },
    # ============================================================
    # 中国
    # ============================================================
    "tottori": {
        "name_ja": "鳥取県",
        "code": "31",
        "bbox": (133.09, 35.01, 134.57, 35.66),
        "center": [35.50, 134.24],
        "epsg_jpc": 6673,
        "grid_area": "chugoku",
    },
    "shimane": {
        "name_ja": "島根県",
        "code": "32",
        "bbox": (131.62, 34.25, 133.44, 37.3),
        "center": [35.47, 132.77],
        "epsg_jpc": 6672,
        "grid_area": "chugoku",
    },
    "okayama": {
        "name_ja": "岡山県",
        "code": "33",
        "bbox": (133.22, 34.25, 134.46, 35.4),
        "center": [34.66, 133.93],
        "epsg_jpc": 6673,
        "grid_area": "chugoku",
    },
    "hiroshima": {
        "name_ja": "広島県",
        "code": "34",
        "bbox": (131.99, 33.98, 133.52, 35.16),
        "center": [34.40, 132.46],
        "epsg_jpc": 6672,
        "grid_area": "chugoku",
    },
    "yamaguchi": {
        "name_ja": "山口県",
        "code": "35",
        "bbox": (130.72, 33.66, 132.54, 34.85),
        "center": [34.19, 131.47],
        "epsg_jpc": 6672,
        "grid_area": "chugoku",
    },
    # ============================================================
    # 四国
    # ============================================================
    "tokushima": {
        "name_ja": "徳島県",
        "code": "36",
        "bbox": (133.61, 33.49, 134.87, 34.3),
        "center": [34.07, 134.56],
        "epsg_jpc": 6673,
        "grid_area": "shikoku",
    },
    "kagawa": {
        "name_ja": "香川県",
        "code": "37",
        "bbox": (133.4, 33.96, 134.5, 34.61),
        "center": [34.34, 134.04],
        "epsg_jpc": 6673,
        "grid_area": "shikoku",
    },
    "ehime": {
        "name_ja": "愛媛県",
        "code": "38",
        "bbox": (131.96, 32.83, 133.74, 34.35),
        "center": [33.84, 132.77],
        "epsg_jpc": 6673,
        "grid_area": "shikoku",
    },
    "kochi": {
        "name_ja": "高知県",
        "code": "39",
        "bbox": (132.43, 32.65, 134.36, 33.93),
        "center": [33.56, 133.53],
        "epsg_jpc": 6673,
        "grid_area": "shikoku",
    },
    # ============================================================
    # 九州
    # ============================================================
    "fukuoka": {
        "name_ja": "福岡県",
        "code": "40",
        "bbox": (129.93, 32.95, 131.24, 34.3),
        "center": [33.61, 130.42],
        "epsg_jpc": 6671,
        "grid_area": "kyushu",
    },
    "saga": {
        "name_ja": "佐賀県",
        "code": "41",
        "bbox": (129.69, 32.9, 130.59, 33.67),
        "center": [33.25, 130.30],
        "epsg_jpc": 6671,
        "grid_area": "kyushu",
    },
    "nagasaki": {
        "name_ja": "長崎県",
        "code": "42",
        "bbox": (128.05, 31.92, 130.44, 34.78),
        "center": [32.75, 129.87],
        "epsg_jpc": 6670,
        "grid_area": "kyushu",
    },
    "kumamoto": {
        "name_ja": "熊本県",
        "code": "43",
        "bbox": (129.89, 32.04, 131.38, 33.25),
        "center": [32.79, 130.74],
        "epsg_jpc": 6671,
        "grid_area": "kyushu",
    },
    "oita": {
        "name_ja": "大分県",
        "code": "44",
        "bbox": (130.77, 32.66, 132.23, 33.79),
        "center": [33.24, 131.61],
        "epsg_jpc": 6671,
        "grid_area": "kyushu",
    },
    "miyazaki": {
        "name_ja": "宮崎県",
        "code": "45",
        "bbox": (130.65, 31.31, 131.94, 32.89),
        "center": [31.91, 131.42],
        "epsg_jpc": 6671,
        "grid_area": "kyushu",
    },
    "kagoshima": {
        "name_ja": "鹿児島県",
        "code": "46",
        "bbox": (128.35, 26.97, 131.26, 32.36),
        "center": [31.56, 130.56],
        "epsg_jpc": 6671,
        "grid_area": "kyushu",
    },
    # ============================================================
    # 沖縄
    # ============================================================
    "okinawa": {
        "name_ja": "沖縄県",
        "code": "47",
        "bbox": (122.88, 24.0, 131.38, 27.94),
        "center": [26.33, 127.80],
        "epsg_jpc": 6691,
        "grid_area": "okinawa",
    },
}

# srtm_tiles は bbox から自動計算
for _key, _cfg in PREFECTURES.items():
    if "srtm_tiles" not in _cfg:
        _cfg["srtm_tiles"] = compute_srtm_tiles(_cfg["bbox"])

# ── 風力発電向け AHP重み ──────────────────────────────────────
# 文献参考: Villacreses et al. (2017), Ayodele et al. (2018)
WEIGHTS = {
    "wind_speed": 0.30,              # 年間平均風速 (最重要)
    "slope": 0.12,                   # 傾斜 (建設コスト)
    "grid_distance": 0.15,           # 送電線距離 (154kV+)
    "substation_distance": 0.10,     # 変電所距離 (66kV+)
    "land_use": 0.10,                # 土地利用制限
    "elevation": 0.05,               # 標高 (尾根風で有利だが高すぎると不利)
    "road_distance": 0.05,           # 道路距離 (建設・保守アクセス)
    "residential_distance": 0.08,    # 居住地距離 (騒音バッファ)
    "protection": 0.05,              # 保護区域
}

# ERA5 風速データ設定
ERA5_VARIABLE = "100m_wind_speed"  # 100m高さの風速
ERA5_YEARS = list(range(2014, 2024))  # 10年平均


def get_data_dir(pref: str) -> Path:
    return PROJECT_ROOT / "data" / pref


def get_grid_dir(pref: str) -> Path:
    return get_data_dir(pref) / "grid"


def get_land_dir(pref: str) -> Path:
    return get_data_dir(pref) / "land"


def get_wind_dir(pref: str) -> Path:
    return get_data_dir(pref) / "wind"


def get_output_dir(pref: str) -> Path:
    return PROJECT_ROOT / "output" / pref


def get_docs_dir(pref: str) -> Path:
    return PROJECT_ROOT / "docs" / pref


def get_re_data_dir(pref: str) -> Path:
    """参考プロジェクトのデータディレクトリ (共有利用)"""
    return RE_POTENTIAL_ROOT / "data" / pref


def get_pref_config(pref: str) -> dict:
    if pref not in PREFECTURES:
        raise ValueError(f"Unknown prefecture: {pref}. Choose from {list(PREFECTURES.keys())}")
    return PREFECTURES[pref]
