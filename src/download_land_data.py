"""
土地利用・DEM データダウンロード（県別対応）

japan-re-potential と同じデータソースを使用。
既に参考プロジェクトにデータがあればシンボリックリンクで共有する。
"""
import argparse
import gzip
import os
import ssl
import sys
import urllib.request
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import PREFECTURES, get_land_dir, get_pref_config, get_re_data_dir

ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE

SRTM1_SIZE = 3601


def download_file(url: str, dest: Path, desc: str) -> bool:
    if dest.exists():
        print(f"  [SKIP] {desc} - already exists")
        return True
    print(f"  [DL] {desc}")
    print(f"       {url}")
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, context=ctx, timeout=120) as resp:
            data = resp.read()
        dest.write_bytes(data)
        size_mb = len(data) / (1024 * 1024)
        print(f"       -> {dest.name} ({size_mb:.1f} MB)")
        return True
    except Exception as e:
        print(f"       ERROR: {e}")
        return False


def extract_zip(zip_path: Path, extract_dir: Path):
    if not zip_path.exists():
        return
    print(f"  [EXTRACT] {zip_path.name}")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_dir)


def try_link_from_re_project(pref: str, land_dir: Path) -> bool:
    """参考プロジェクトのデータがあればシンボリックリンクで共有する。"""
    re_land_dir = get_re_data_dir(pref) / "land"
    if not re_land_dir.exists():
        return False

    linked = False
    # DEM ディレクトリ
    re_dem = re_land_dir / "dem"
    local_dem = land_dir / "dem"
    if re_dem.exists() and not local_dem.exists():
        local_dem.symlink_to(re_dem)
        print(f"  [LINK] dem -> {re_dem}")
        linked = True

    # slope TIF
    re_slope = re_land_dir / f"{pref}_slope.tif"
    local_slope = land_dir / f"{pref}_slope.tif"
    if re_slope.exists() and not local_slope.exists():
        local_slope.symlink_to(re_slope)
        print(f"  [LINK] slope -> {re_slope}")
        linked = True

    # admin_boundary
    re_admin = re_land_dir / "admin_boundary"
    local_admin = land_dir / "admin_boundary"
    if re_admin.exists() and not local_admin.exists():
        local_admin.symlink_to(re_admin)
        print(f"  [LINK] admin_boundary -> {re_admin}")
        linked = True

    # land_use
    re_lu = re_land_dir / "land_use"
    local_lu = land_dir / "land_use"
    if re_lu.exists() and not local_lu.exists():
        local_lu.symlink_to(re_lu)
        print(f"  [LINK] land_use -> {re_lu}")
        linked = True

    return linked


def download_srtm(pref: str, land_dir: Path) -> bool:
    cfg = get_pref_config(pref)
    tiles = cfg["srtm_tiles"]
    dem_dir = land_dir / "dem"
    dem_dir.mkdir(parents=True, exist_ok=True)

    all_ok = True
    for tile in tiles:
        hgt_path = dem_dir / f"{tile}.hgt"
        if hgt_path.exists():
            print(f"  [SKIP] DEM {tile} - already exists")
            continue

        url = f"https://s3.amazonaws.com/elevation-tiles-prod/skadi/{tile[:3]}/{tile}.hgt.gz"
        gz_path = dem_dir / f"{tile}.hgt.gz"
        print(f"  [DL] DEM {tile}")
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, context=ctx, timeout=180) as resp:
                data = resp.read()
            gz_path.write_bytes(data)
            with gzip.open(gz_path, "rb") as f_in:
                hgt_path.write_bytes(f_in.read())
            gz_path.unlink()
            print(f"       -> {hgt_path.name}")
        except Exception as e:
            print(f"       ERROR: {e}")
            if gz_path.exists():
                gz_path.unlink()
            all_ok = False

    return all_ok


def download_admin_boundary(pref: str, land_dir: Path) -> bool:
    cfg = get_pref_config(pref)
    code = cfg["code"]
    name_ja = cfg["name_ja"]

    admin_dir = land_dir / "admin_boundary"
    if admin_dir.exists() and any(admin_dir.rglob("*.shp")):
        print(f"  [SKIP] 行政区域 ({name_ja}) - already exists")
        return True

    url = f"https://nlftp.mlit.go.jp/ksj/gml/data/N03/N03-2024/N03-20240101_{code}_GML.zip"
    dest = land_dir / "admin_boundary.zip"
    if download_file(url, dest, f"行政区域 ({name_ja})"):
        admin_dir.mkdir(parents=True, exist_ok=True)
        try:
            extract_zip(dest, admin_dir)
        except Exception as e:
            print(f"  Extract error: {e}")
            return False
        return True
    return False


def main():
    parser = argparse.ArgumentParser(description="土地利用・DEM データダウンロード (風力版)")
    parser.add_argument(
        "--prefecture", "-p",
        type=str,
        default="fukui",
        choices=list(PREFECTURES.keys()),
    )
    parser.add_argument("--skip-dem", action="store_true")
    args = parser.parse_args()

    pref = args.prefecture
    cfg = get_pref_config(pref)
    name_ja = cfg["name_ja"]
    land_dir = get_land_dir(pref)
    land_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print(f"{name_ja} 土地利用・DEM データダウンロード (風力版)")
    print("=" * 60)

    # 参考プロジェクトからリンク
    print("\n[1] 参考プロジェクト (japan-re-potential) からリンク")
    try_link_from_re_project(pref, land_dir)

    # 行政区域
    print("\n[2] 行政区域データ")
    download_admin_boundary(pref, land_dir)

    # SRTM DEM
    if not args.skip_dem:
        dem_dir = land_dir / "dem"
        if dem_dir.is_symlink():
            print(f"\n[3] DEM - symlinked from re-potential")
        else:
            print(f"\n[3] SRTM DEMタイル (bbox の全 tile をチェック)")
            download_srtm(pref, land_dir)

    print(f"\n完了! データ保存先: {land_dir}")


if __name__ == "__main__":
    main()
