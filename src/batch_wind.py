#!/usr/bin/env python3
"""
全国風力発電ポテンシャル バッチオーケストレーター (3フェーズ方式)

Phase 1: 並列で DEM/slope + extract_grid を実行 (外部API不要)
Phase 2: Overpass API (土地利用) + ERA5 (風速) を順次/並列で実行
Phase 3: 並列で raster_score_wind を実行

Usage:
    python src/batch_wind.py -p fukui --resolution 30
    python src/batch_wind.py -p fukui,akita --resolution 30 --wind-fallback
    python src/batch_wind.py --resolution 5 --resume --workers 2
"""

import argparse
import json
import logging
import os
import subprocess
import sys
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import PREFECTURES, PROJECT_ROOT

LOG_DIR = PROJECT_ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = LOG_DIR / f"batch_wind_{timestamp}.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.FileHandler(log_file, encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)

CHECKPOINT_FILE = PROJECT_ROOT / "data" / "batch_wind_checkpoint.json"
PROGRESS_FILE = PROJECT_ROOT / "data" / "batch_wind_progress.txt"

_checkpoint_lock = threading.Lock()
_progress_lock = threading.Lock()


def load_checkpoint() -> dict:
    with _checkpoint_lock:
        if CHECKPOINT_FILE.exists():
            return json.loads(CHECKPOINT_FILE.read_text(encoding="utf-8"))
        return {}


def save_checkpoint(cp: dict):
    with _checkpoint_lock:
        CHECKPOINT_FILE.parent.mkdir(parents=True, exist_ok=True)
        CHECKPOINT_FILE.write_text(json.dumps(cp, indent=2, ensure_ascii=False), encoding="utf-8")


def update_progress(pref: str, step: str, status: str, detail: str = ""):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{now}] {pref:25s} | {step:20s} | {status:10s} | {detail}\n"
    with _progress_lock:
        with open(PROGRESS_FILE, "a", encoding="utf-8") as f:
            f.write(line)


def mark_step_done(cp, pref, step, resolution):
    with _checkpoint_lock:
        entry = cp.setdefault(pref, {"completed_steps": []})
        if step not in entry.get("completed_steps", []):
            entry.setdefault("completed_steps", []).append(step)
        entry["last_update"] = datetime.now().isoformat()
        entry["resolution"] = resolution
    save_checkpoint(cp)


def is_step_done(cp, pref, step):
    with _checkpoint_lock:
        return step in cp.get(pref, {}).get("completed_steps", [])


def mark_completed(cp, pref):
    with _checkpoint_lock:
        cp.setdefault(pref, {})["status"] = "completed"
        cp[pref]["last_update"] = datetime.now().isoformat()
    save_checkpoint(cp)


def run_step(pref: str, step: str, resolution: int, wind_fallback: bool = False) -> bool:
    src_dir = PROJECT_ROOT / "src"
    python = sys.executable

    env = os.environ.copy()
    if "ALL_JAPAN_GRID_DIR" not in env:
        candidate = Path.home() / "All-Japan-Grid-ref" / "data"
        if candidate.exists():
            env["ALL_JAPAN_GRID_DIR"] = str(candidate)

    cmd_map = {
        "download": [python, str(src_dir / "download_land_data.py"), "-p", pref],
        "extract_grid": [python, str(src_dir / "extract_grid.py"), "-p", pref],
        "slope": [python, str(src_dir / "slope_analysis.py"), "-p", pref],
        "osm_land_use": [python, str(src_dir / "fetch_osm_land_use.py"), "-p", pref],
        "wind_data": [python, str(src_dir / "download_wind_data.py"), "-p", pref]
                    + (["--fallback"] if wind_fallback else []),
        "raster_score": [python, str(src_dir / "raster_score_wind.py"), "-p", pref,
                         "--resolution", str(resolution), "--skip-tiles"],
    }

    cmd = cmd_map.get(step)
    if not cmd:
        log.error("Unknown step: %s", step)
        return False

    step_log = LOG_DIR / f"{pref}_{step}_{timestamp}.log"
    log.info("  [%s] Running %s", pref, step)

    try:
        with open(step_log, "w", encoding="utf-8") as flog:
            result = subprocess.run(
                cmd, stdout=flog, stderr=subprocess.STDOUT,
                timeout=7200, cwd=str(PROJECT_ROOT), env=env,
            )
        if result.returncode == 0:
            log.info("  [%s] OK: %s", pref, step)
            return True
        else:
            log.error("  [%s] FAIL: %s (rc=%d)", pref, step, result.returncode)
            try:
                lines = step_log.read_text(encoding="utf-8").strip().split("\n")
                for line in lines[-3:]:
                    log.error("    | %s", line)
            except Exception:
                pass
            return False
    except subprocess.TimeoutExpired:
        log.error("  [%s] TIMEOUT: %s (>2h)", pref, step)
        return False
    except Exception as e:
        log.error("  [%s] ERROR: %s: %s", pref, step, e)
        return False


def run_phase_parallel(phase_name, pref_list, steps, resolution, cp, workers,
                       wind_fallback=False):
    log.info("=" * 60)
    log.info("PHASE: %s (%d prefectures, %d workers)", phase_name, len(pref_list), workers)
    log.info("=" * 60)

    failed = []

    def _process_one(pref):
        for step in steps:
            if is_step_done(cp, pref, step):
                log.info("  [%s] SKIP %s (done)", pref, step)
                continue
            update_progress(pref, step, "running")
            ok = run_step(pref, step, resolution, wind_fallback)
            if ok:
                mark_step_done(cp, pref, step, resolution)
                update_progress(pref, step, "completed")
            else:
                update_progress(pref, step, "FAILED")
                return False
        return True

    if workers <= 1:
        for pref in pref_list:
            if not _process_one(pref):
                failed.append(pref)
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(_process_one, p): p for p in pref_list}
            for future in as_completed(futures):
                pref = futures[future]
                try:
                    if not future.result():
                        failed.append(pref)
                except Exception as e:
                    log.exception("  [%s] Error: %s", pref, e)
                    failed.append(pref)

    log.info("PHASE %s: %d OK, %d failed", phase_name, len(pref_list) - len(failed), len(failed))
    return failed


def run_phase_overpass(pref_list, resolution, cp):
    step = "osm_land_use"
    remaining = [p for p in pref_list if not is_step_done(cp, p, step)]
    if not remaining:
        return []

    log.info("=" * 60)
    log.info("PHASE: Overpass API (%d prefectures)", len(remaining))
    log.info("=" * 60)

    max_rounds = 3
    for round_num in range(max_rounds):
        if not remaining:
            break
        if round_num > 0:
            wait = 300 * round_num
            log.info("  Round %d: waiting %ds...", round_num + 1, wait)
            time.sleep(wait)

        still_failed = []
        for i, pref in enumerate(remaining):
            update_progress(pref, step, "running")
            ok = run_step(pref, step, resolution)
            if ok:
                mark_step_done(cp, pref, step, resolution)
                update_progress(pref, step, "completed")
                if i < len(remaining) - 1:
                    time.sleep(30)
            else:
                update_progress(pref, step, "FAILED")
                still_failed.append(pref)
                time.sleep(min(120 * (2 ** round_num), 600))
        remaining = still_failed

    return remaining


def main():
    parser = argparse.ArgumentParser(description="風力ポテンシャル バッチ計算")
    parser.add_argument("-p", "--prefecture", default=None)
    parser.add_argument("-r", "--resolution", type=int, default=30)
    parser.add_argument("-w", "--workers", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--reset", action="store_true")
    parser.add_argument("--wind-fallback", action="store_true",
                        help="ERA5が使えない場合、標高ベースの風速推定を使用")
    parser.add_argument("--skip-osm", action="store_true",
                        help="OSM土地利用取得をスキップ (デフォルト値で代替)")
    args = parser.parse_args()

    workers = max(1, args.workers)

    log.info("=" * 60)
    log.info("風力ポテンシャル バッチ計算")
    log.info("  Resolution: %dm | Workers: %d | Fallback: %s",
             args.resolution, workers, args.wind_fallback)
    log.info("=" * 60)

    if args.reset and CHECKPOINT_FILE.exists():
        CHECKPOINT_FILE.unlink()

    cp = load_checkpoint() if args.resume else {}

    if args.prefecture:
        pref_list = [p.strip() for p in args.prefecture.split(",")]
        for p in pref_list:
            if p not in PREFECTURES:
                log.error("Unknown: %s", p)
                sys.exit(1)
    else:
        pref_list = list(PREFECTURES.keys())

    if args.resume:
        already = [p for p in pref_list if cp.get(p, {}).get("status") == "completed"]
        pref_list = [p for p in pref_list if p not in already]
        if already:
            log.info("Skipping %d completed", len(already))

    total = len(pref_list)
    start_time = time.time()

    # Phase 1: download + extract_grid + slope (並列)
    phase1_failed = run_phase_parallel(
        "1-local", pref_list, ["download", "extract_grid", "slope"],
        args.resolution, cp, workers,
    )

    phase2_list = [p for p in pref_list if p not in phase1_failed]

    # Phase 2a: Overpass API (順次)
    if args.skip_osm:
        log.info("OSM skipped (--skip-osm)")
        phase2a_failed = []
    else:
        phase2a_failed = run_phase_overpass(phase2_list, args.resolution, cp)

    # Phase 2b: Wind data (並列 or 順次)
    phase2b_failed = run_phase_parallel(
        "2b-wind", phase2_list, ["wind_data"],
        args.resolution, cp, workers, wind_fallback=args.wind_fallback,
    )

    # Phase 3: raster_score (並列)
    phase3_list = [p for p in phase2_list]  # OSM/wind失敗でもデフォルト値で実行可能
    phase3_failed = run_phase_parallel(
        "3-raster", phase3_list, ["raster_score"],
        args.resolution, cp, workers,
    )

    # 完了マーク
    all_steps = ["download", "extract_grid", "slope", "osm_land_use", "wind_data", "raster_score"]
    for pref in pref_list:
        steps_done = cp.get(pref, {}).get("completed_steps", [])
        if all(s in steps_done for s in all_steps):
            mark_completed(cp, pref)

    elapsed = time.time() - start_time
    completed = sum(1 for p in pref_list if cp.get(p, {}).get("status") == "completed")
    all_failed = set(phase1_failed) | set(phase2a_failed) | set(phase2b_failed) | set(phase3_failed)

    log.info("=" * 60)
    log.info("BATCH COMPLETE")
    log.info("  Total: %d | Completed: %d | Failed: %d", total, completed, len(all_failed))
    log.info("  Elapsed: %.1f min", elapsed / 60)
    log.info("=" * 60)


if __name__ == "__main__":
    main()
