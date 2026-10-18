"""
Standard Morris elementary-effects GSA for the Pambala pyMANGA setup (v2).

Morris (1991) screening with the Campolongo et al. (2007) trajectory design via
SALib (Herman & Usher 2017). 13 factors (morris_problem.json); r = 20
trajectories, num_levels = 4, sampling seed = 42 -> 280 samples, each run with 3
random seeds (840 sims). Per-factor indices: mu* (importance) and sigma
(non-linearity / interactions). Factor wiring and literature-transfer baselines
live in apply_morris_sample() and baselines.json

Response: Year-100 stand state (stand-level + per-species). Before every sample
the live species .py files and d_2_p.xml are restored from
calibration_snapshot_MANUSCRIPT_FINAL_2026-05-21/, and an atexit/SIGINT hook
restores them again on exit.

Outputs: results/design.csv, results/metrics.csv, results/done.txt, logs/.

Usage:
    python run_morris.py [--resume] [--dry-run] [--trajectories 20]
                         [--seeds 3] [--timeout 300] [--max-sims N]
"""
from __future__ import annotations
import argparse
import atexit
import csv
import json
import re
import shutil
import signal
import subprocess
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd


# --- Paths ---

HERE = Path(__file__).resolve().parent
APP_ROOT = HERE.parent
PROJ_ROOT = APP_ROOT.parents[2]
SPECIES_DIR = PROJ_ROOT / "PopulationLib" / "Species"
LIVE_XML = APP_ROOT / "random.xml"
MODEL_OUTPUT = APP_ROOT / "ModelOutput"
POPULATION_CSV = MODEL_OUTPUT / "Population.csv"
EXTRACT_100 = MODEL_OUTPUT / "extract_year_100.py"
POPULATION_100 = MODEL_OUTPUT / "Population_Year_100.csv"

SNAPSHOT_DIR = PROJ_ROOT / "calibration_snapshot_MANUSCRIPT_FINAL_2026-05-21"

PROBLEM_JSON = HERE / "morris_problem.json"
BASELINES_JSON = HERE / "baselines.json"
RESULTS_DIR = HERE / "results"
LOGS_DIR = HERE / "logs"
DESIGN_CSV = RESULTS_DIR / "design.csv"
METRICS_CSV = RESULTS_DIR / "metrics.csv"
DONE_TXT = RESULTS_DIR / "done.txt"
TOPLEVEL_LOG = LOGS_DIR / "run_morris.log"

SPECIES_PREFIXES = {
    "Rhizophora": "Rhizpop",
    "Avicennia":  "Avipop",
    "Bruguiera":  "Brugpop",
    "Lumnitzera": "Lumnpop",
    "Excoecaria": "Excopop",
}
SPECIES_ORDER = ["Rhizophora", "Avicennia", "Bruguiera", "Lumnitzera", "Excoecaria"]
SEEDS_DEFAULT = [643879, 644879, 645879]


# --- Logging ---

def now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def log(msg: str) -> None:
    line = f"[{now()}] {msg}"
    print(line, flush=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    with TOPLEVEL_LOG.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


# --- Snapshot restore ---

def restore_baselines_from_snapshot() -> None:
    """Copy the MANUSCRIPT_FINAL snapshot over the live species .py files and
    d_2_p.xml. Idempotent."""
    if not SNAPSHOT_DIR.exists():
        raise RuntimeError(
            f"Snapshot folder missing: {SNAPSHOT_DIR}. "
            f"Cannot safely run Morris without a restore source."
        )
    for sp in SPECIES_ORDER:
        src = SNAPSHOT_DIR / f"{sp}.py"
        dst = SPECIES_DIR / sp / f"{sp}.py"
        shutil.copyfile(src, dst)
    shutil.copyfile(SNAPSHOT_DIR / "d_2_p.xml", LIVE_XML)


def ensure_clean_on_exit() -> None:
    """Restore the live files to the MANUSCRIPT_FINAL snapshot on normal exit or
    SIGINT, so an interrupted run never leaves a perturbed tree behind."""
    def _restore():
        try:
            restore_baselines_from_snapshot()
            print(f"[{now()}] Live files restored to MANUSCRIPT_FINAL snapshot.")
        except Exception as e:
            print(f"[{now()}] WARNING: snapshot restore failed: {e}", file=sys.stderr)

    atexit.register(_restore)

    def _sigint(signum, frame):
        _restore()
        sys.exit(130)
    signal.signal(signal.SIGINT, _sigint)


# --- Parameter application ---
# The write_* helpers are inlined here (not imported from calibration/) so the
# sensitivity_analysis_v2 folder is self-contained.

_NUM_PATTERN = r'([-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?)'


def write_species_param(species: str, param_name: str, new_value: float) -> None:
    """Replace parameter["<param_name>"] or geometry["<param_name>"] in
    <species>.py. Uses plain float formatting (not repr) so numpy scalars are
    written as bare numbers (e.g. -5160000.0), not 'np.float64(...)' -- which
    would NameError because the species files do not import numpy."""
    fpath = SPECIES_DIR / species / f"{species}.py"
    text = fpath.read_text(encoding="utf-8")
    pattern = re.compile(
        rf'((?:parameter|geometry)\["{re.escape(param_name)}"\]\s*=\s*){_NUM_PATTERN}'
    )
    new_text, n = pattern.subn(
        lambda m: m.group(1) + format(float(new_value), ".10g"),
        text, count=1,
    )
    if n == 0:
        raise RuntimeError(f"Parameter '{param_name}' not found in {species}.py")
    fpath.write_text(new_text, encoding="utf-8")


def write_xml_salinity(left: float, right: float) -> None:
    text = LIVE_XML.read_text(encoding="utf-8")
    new_text = re.sub(
        r"<salinity>\s*[-+\deE\.\s]+</salinity>",
        f"<salinity> {left} {right} </salinity>",
        text, count=1,
    )
    LIVE_XML.write_text(new_text, encoding="utf-8")


def write_xml_recruitment(recruits_per_species: dict[str, int]) -> None:
    """Set each species block's <n_recruitment_per_step> to the given integer."""
    text = LIVE_XML.read_text(encoding="utf-8")
    for sp, prefix in SPECIES_PREFIXES.items():
        n = recruits_per_species.get(sp)
        if n is None:
            continue
        block_re = re.compile(
            rf"(<name>\s*{prefix}\s*</name>.*?<n_recruitment_per_step>\s*)\d+(\s*</n_recruitment_per_step>)",
            re.DOTALL,
        )
        text, n_subs = block_re.subn(rf"\g<1>{int(n)}\g<2>", text, count=1)
        if n_subs == 0:
            raise RuntimeError(f"n_recruitment_per_step block not found for species {sp}")
    LIVE_XML.write_text(text, encoding="utf-8")


def write_xml_seed(seed: int) -> None:
    """Replace or insert <random_seed> at the top level of d_2_p.xml."""
    text = LIVE_XML.read_text(encoding="utf-8")
    if "<random_seed>" in text:
        text = re.sub(
            r"<random_seed>\s*\d+\s*</random_seed>",
            f"<random_seed>{int(seed)}</random_seed>",
            text, count=1,
        )
    else:
        text = re.sub(
            r"(<\?xml[^?]*\?>\s*<[^>]+>)",
            rf"\g<1>\n    <random_seed>{int(seed)}</random_seed>",
            text, count=1,
        )
    LIVE_XML.write_text(text, encoding="utf-8")


# --- Apply one Morris sample to the live files ---

EXPECTED_FACTOR_NAMES = {
    "growth_factor_mult", "maint_factor_mult",
    "leaf_water_potential_mult", "kf_sap_mult",
    "lp_mult", "k_geom_mult", "sun_c_mult",
    "half_max_h_growth_weight_mult", "h_sigmo_slope_mult",
    "sigmo_slope_mult",
    "salinity_left", "salinity_right",
    "n_recruitment_mult",
}


def apply_morris_sample(sample_values: np.ndarray, names: list[str],
                        baselines: dict) -> None:
    """Write a vector of 13 sampled factor values to the live files.

    The wiring is hard-coded against the 13 known factor names; if the problem
    file diverges we fail loudly rather than silently writing nothing
    (factor_kinds in morris_problem.json is documentary only). Restores from the
    snapshot first so each sample starts from a clean baseline."""
    missing = EXPECTED_FACTOR_NAMES - set(names)
    extra = set(names) - EXPECTED_FACTOR_NAMES
    if missing or extra:
        raise RuntimeError(
            f"Morris factor names diverge from runner expectations. "
            f"Missing: {sorted(missing)}, unexpected: {sorted(extra)}. "
            f"Update apply_morris_sample() or morris_problem.json to match."
        )

    restore_baselines_from_snapshot()
    factor_value = dict(zip(names, sample_values))

    sp_base = baselines["species_baselines"]
    sp_set = baselines["species_set_mapping"]
    rhi_baseline = baselines["species_baselines"]["Rhizophora"]
    avi_baseline = baselines["species_baselines"]["Avicennia"]

    # Shared multipliers: one factor scales each species' own baseline.
    for param in ["growth_factor", "maint_factor", "lp", "k_geom",
                  "sun_c", "half_max_h_growth_weight",
                  "h_sigmo_slope", "sigmo_slope"]:
        m = factor_value[f"{param}_mult"]
        for sp in SPECIES_ORDER:
            new_val = sp_base[sp][param] * m
            write_species_param(sp, param, new_val)

    # Per-set multipliers: one factor per Rhi-like / Avi-like baseline set,
    # preserving the published Avicennia/Rhizophora split.
    for param in ["leaf_water_potential", "kf_sap"]:
        m = factor_value[f"{param}_mult"]
        for sp in SPECIES_ORDER:
            if sp_set[sp] == "set_A_rhi_like":
                base = rhi_baseline[param]
            else:
                base = avi_baseline[param]
            new_val = base * m
            write_species_param(sp, param, new_val)

    # Salinity: absolute boundary values.
    write_xml_salinity(
        float(factor_value["salinity_left"]),
        float(factor_value["salinity_right"]),
    )

    # Recruitment: one multiplier per species' baseline, rounded to int >= 1.
    rec_mult = float(factor_value["n_recruitment_mult"])
    rec_dict = {
        sp: max(1, int(round(baselines["recruitment_baselines"][sp] * rec_mult)))
        for sp in SPECIES_ORDER
    }
    write_xml_recruitment(rec_dict)


# --- Run a single simulation + extract metrics ---

def run_single_sim(sample_idx: int, seed: int, timeout: int) -> dict:
    """Run MANGA.py once for one (sample, seed). A simulation that fails to run,
    or whose Year-100 extraction fails, is reported as failed so it never enters
    metrics.csv (injecting zeros/NaNs would distort the GSA)."""
    write_xml_seed(seed)

    # Clear stale outputs so a non-completing run is detected as a failure.
    if POPULATION_CSV.exists():
        POPULATION_CSV.unlink()
    if POPULATION_100.exists():
        POPULATION_100.unlink()

    t0 = time.time()
    sim_res = subprocess.run(
        [sys.executable, "MANGA.py", "-i", str(LIVE_XML.relative_to(PROJ_ROOT))],
        cwd=str(PROJ_ROOT), capture_output=True, text=True, timeout=timeout,
    )
    sim_runtime = time.time() - t0

    if sim_res.returncode != 0 or not POPULATION_CSV.exists():
        log_file = LOGS_DIR / f"sim_{sample_idx:04d}_seed{seed}_FAILED.log"
        log_file.write_text(
            f"STDOUT:\n{sim_res.stdout}\n\nSTDERR:\n{sim_res.stderr}\n",
            encoding="utf-8",
        )
        return {"status": "sim_failed", "runtime": sim_runtime}

    if not EXTRACT_100.exists():
        return {"status": "extractor_script_missing", "runtime": sim_runtime}

    r = subprocess.run(
        [sys.executable, str(EXTRACT_100)],
        cwd=str(PROJ_ROOT), capture_output=True, text=True, timeout=60,
    )
    if r.returncode != 0 or not POPULATION_100.exists():
        log_file = LOGS_DIR / f"sim_{sample_idx:04d}_seed{seed}_EXTRACT_FAILED.log"
        log_file.write_text(
            f"Extractor: {EXTRACT_100.name}\n"
            f"returncode: {r.returncode}\n"
            f"Population_Year_100.csv exists: {POPULATION_100.exists()}\n"
            f"\nSTDOUT:\n{r.stdout}\n\nSTDERR:\n{r.stderr}\n",
            encoding="utf-8",
        )
        return {"status": "extract_failed", "runtime": sim_runtime}

    return {"status": "ok", "runtime": sim_runtime}


def basal_area_m2(dbh_cm: np.ndarray) -> np.ndarray:
    return np.pi * (dbh_cm / 200.0) ** 2


def extract_metrics(sample_idx: int, seed: int) -> dict:
    """Year-100 response variables. Stand-level: total stems, total basal area,
    Shannon diversity, mean DBH. Per-species: stem count, basal area (m^2),
    mean DBH (cm), mean height (m), max DBH (cm)."""
    metrics: dict = {"sample_idx": sample_idx, "seed": seed}

    if POPULATION_100.exists():
        sim100 = pd.read_csv(POPULATION_100, sep="\t")
        sim100["species_prefix"] = sim100["plant"].str.split("_").str[0]
        sim100["dbh_cm"] = sim100["r_stem"] * 200.0

        total_stems = len(sim100)
        metrics["y100_total_stems"] = int(total_stems)
        if total_stems > 0:
            counts = sim100["species_prefix"].value_counts()
            p = counts / counts.sum()
            shannon = float(-np.sum(p * np.log(p))) if total_stems > 1 else 0.0
            metrics["y100_shannon_div"] = shannon
            metrics["y100_total_basal_area_m2"] = float(
                basal_area_m2(sim100["dbh_cm"].values).sum()
            )
            metrics["y100_mean_dbh_cm"] = float(sim100["dbh_cm"].mean())
        else:
            metrics["y100_shannon_div"] = 0.0
            metrics["y100_total_basal_area_m2"] = 0.0
            metrics["y100_mean_dbh_cm"] = np.nan

        for sp, prefix in SPECIES_PREFIXES.items():
            sub = sim100[sim100["species_prefix"] == prefix]
            n = len(sub)
            metrics[f"y100_{sp}_n"] = int(n)
            if n > 0:
                ba = float(basal_area_m2(sub["dbh_cm"].values).sum())
                metrics[f"y100_{sp}_basal_area_m2"] = ba
                metrics[f"y100_{sp}_mean_dbh_cm"] = float(sub["dbh_cm"].mean())
                metrics[f"y100_{sp}_mean_h_m"] = float(sub["h_stem"].mean()) if "h_stem" in sub else np.nan
                metrics[f"y100_{sp}_max_dbh_cm"] = float(sub["dbh_cm"].max())
            else:
                metrics[f"y100_{sp}_basal_area_m2"] = 0.0
                metrics[f"y100_{sp}_mean_dbh_cm"] = np.nan
                metrics[f"y100_{sp}_mean_h_m"] = np.nan
                metrics[f"y100_{sp}_max_dbh_cm"] = np.nan
    else:
        metrics["y100_total_stems"] = 0
        metrics["y100_shannon_div"] = np.nan

    return metrics


# --- Design generation (SALib) ---

def generate_design(n_trajectories: int) -> tuple[dict, np.ndarray]:
    """Load morris_problem.json and generate the Morris sample matrix."""
    from SALib.sample import morris as morris_sample

    with PROBLEM_JSON.open() as f:
        problem = json.load(f)

    salib_problem = {
        "num_vars": problem["num_vars"],
        "names": problem["names"],
        "bounds": problem["bounds"],
    }

    X = morris_sample.sample(salib_problem, N=n_trajectories, num_levels=4, seed=42)
    return problem, X


def write_design_csv(problem: dict, X: np.ndarray) -> None:
    cols = ["sample_idx"] + problem["names"]
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with DESIGN_CSV.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for i, row in enumerate(X):
            w.writerow([i] + [f"{v:.6g}" for v in row])
    log(f"Wrote design ({X.shape[0]} samples x {X.shape[1]} factors) -> {DESIGN_CSV}")


# --- Resume support ---

def load_done_set() -> set[str]:
    if not DONE_TXT.exists():
        return set()
    return set(line.strip() for line in DONE_TXT.read_text().splitlines() if line.strip())


def mark_done(sample_idx: int, seed: int) -> None:
    with DONE_TXT.open("a", encoding="utf-8") as f:
        f.write(f"{sample_idx:04d}_seed{seed}\n")


# --- Main loop ---

def metrics_fieldnames() -> list[str]:
    """Fixed metrics-CSV schema. Every row carries every column (missing values
    left blank) so a partial/failed row cannot silently change the column set."""
    cols = ["sample_idx", "seed", "sim_runtime_s", "wall_time_s"]
    cols += ["y100_total_stems", "y100_total_basal_area_m2",
             "y100_shannon_div", "y100_mean_dbh_cm"]
    for sp in SPECIES_ORDER:
        for suffix in ("n", "basal_area_m2", "mean_dbh_cm",
                       "mean_h_m", "max_dbh_cm"):
            cols.append(f"y100_{sp}_{suffix}")
    return cols


def append_metrics_row(metrics_row: dict) -> None:
    fieldnames = metrics_fieldnames()
    write_header = not METRICS_CSV.exists()
    with METRICS_CSV.open("a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(fieldnames)
        w.writerow(["" if metrics_row.get(k) is None else metrics_row.get(k, "")
                    for k in fieldnames])


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--trajectories", type=int, default=20,
                    help="Morris trajectories r (default 20)")
    ap.add_argument("--seeds", type=int, default=3,
                    help="Number of seeds per Morris sample (default 3)")
    ap.add_argument("--timeout", type=int, default=300,
                    help="Per-simulation timeout in seconds (default 300)")
    ap.add_argument("--dry-run", action="store_true",
                    help="Generate design only; do not run simulations")
    ap.add_argument("--resume", action="store_true",
                    help="Skip already-completed sample_seeds (read done.txt)")
    ap.add_argument("--max-sims", type=int, default=None,
                    help="Stop after this many successful simulations "
                         "(useful for end-to-end smoke test before full run)")
    args = ap.parse_args(argv)

    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    log(f"=== Morris GSA v2 start ===")
    log(f"r = {args.trajectories} trajectories, seeds = {args.seeds}, "
        f"timeout = {args.timeout}s/sim")

    # Generate a fresh design, or reload the existing one when resuming.
    if DESIGN_CSV.exists() and args.resume:
        log(f"Loading existing design from {DESIGN_CSV}")
        with PROBLEM_JSON.open() as f:
            problem = json.load(f)
        df = pd.read_csv(DESIGN_CSV)
        X = df[problem["names"]].to_numpy()
    else:
        log(f"Generating Morris design ...")
        problem, X = generate_design(args.trajectories)
        write_design_csv(problem, X)

    n_samples = X.shape[0]
    seeds = [SEEDS_DEFAULT[i] for i in range(args.seeds)] \
            if args.seeds <= len(SEEDS_DEFAULT) \
            else [SEEDS_DEFAULT[0] + i * 1000 for i in range(args.seeds)]

    total_sims = n_samples * len(seeds)
    log(f"Plan: {n_samples} Morris samples x {len(seeds)} seeds = {total_sims} sims")
    log(f"Estimated runtime at 104s/sim: ~{total_sims * 104 / 3600:.1f} hours")

    if args.dry_run:
        log("Dry run — exiting before any simulations.")
        return 0

    if not SNAPSHOT_DIR.exists():
        log(f"FATAL: snapshot dir missing: {SNAPSHOT_DIR}")
        return 2

    ensure_clean_on_exit()

    with BASELINES_JSON.open() as f:
        baselines = json.load(f)

    done = load_done_set() if args.resume else set()
    if done:
        log(f"Resume mode: skipping {len(done)} already-completed sims")

    success = 0
    failed = 0
    skipped = 0
    sim_counter = 0

    for sample_idx in range(n_samples):
        sample_vals = X[sample_idx]
        try:
            apply_morris_sample(sample_vals, problem["names"], baselines)
        except Exception as e:
            log(f"  sample {sample_idx}: APPLY FAILED -- {e}")
            failed += len(seeds)
            sim_counter += len(seeds)
            continue

        for seed in seeds:
            sim_counter += 1
            sim_id = f"{sample_idx:04d}_seed{seed}"
            if sim_id in done:
                skipped += 1
                continue

            t0 = time.time()
            try:
                sim_result = run_single_sim(sample_idx, seed, args.timeout)
            except subprocess.TimeoutExpired:
                log(f"[{sim_counter}/{total_sims}] {sim_id}: TIMEOUT after {args.timeout}s")
                failed += 1
                continue
            except Exception:
                log(f"[{sim_counter}/{total_sims}] {sim_id}: CRASHED")
                log(traceback.format_exc())
                failed += 1
                continue

            if sim_result["status"] != "ok":
                log(f"[{sim_counter}/{total_sims}] {sim_id}: {sim_result['status'].upper()}")
                failed += 1
                continue

            metrics = extract_metrics(sample_idx, seed)
            metrics["sim_runtime_s"] = sim_result["runtime"]
            metrics["wall_time_s"] = time.time() - t0
            append_metrics_row(metrics)
            mark_done(sample_idx, seed)
            success += 1

            if args.max_sims is not None and success >= args.max_sims:
                log(f"Reached --max-sims = {args.max_sims}; stopping early.")
                log(f"=== Smoke-test ok: success={success} failed={failed} ===")
                return 0

            log(f"[{sim_counter}/{total_sims}] {sim_id}: ok "
                f"({sim_result['runtime']:.0f}s) "
                f"y100 stems={metrics.get('y100_total_stems', '?')}, "
                f"basal_area={metrics.get('y100_total_basal_area_m2', float('nan')):.2f} m^2, "
                f"H={metrics.get('y100_shannon_div', float('nan')):.2f}")

    log(f"=== Morris GSA v2 done: success={success} failed={failed} skipped={skipped} ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
