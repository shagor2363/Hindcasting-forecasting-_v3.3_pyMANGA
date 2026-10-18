"""
Table 3: evaluation of the simulated year-27 (2024) stand against the 2024 reference, per species.

Reference: the reconstructed 2024 stand (Initialization/reconstructed_adults_2024.csv) and the
plot-census abundance targets (plot_based_targets_2024/targets_2024.csv).
Inclusion rule (census rule, applied to reference and simulation alike): girth read to the whole
centimetre and >= 8 cm (DBH >= 2.546 cm), height >= 1.3 m.

Size structure - two-sample Kolmogorov-Smirnov test with the measured sample size (Table 3)
  The reconstruction's DBH distribution of a species is a weighted copy of the measured PCQM trees:
  tree i is copied m_i times (column source_id of Initialization/reconstructed_raw_2024.csv), so the
  reconstructed stand holds n_rec stems but only n_obs independent DBH measurements. The reference
  ECDF is that of the reconstruction (measured trees weighted by m_i); the sample size is n_obs:
      D_obs = KS distance simulation vs reference,
      D_crit = 1.358 * sqrt((n_obs + n_sim) / (n_obs * n_sim))   (alpha = 0.05),
      p = Q_KS(sqrt(n_e) * D_obs),  n_e = n_obs * n_sim / (n_obs + n_sim)  (asymptotic),  pass: D_obs < D_crit.
  Seed-to-seed variation of the simulation is reported as the range over seeds.

Supplementary check - design-based KS test (bootstrap of the field design)
  The sampling units of the field design are the PCQM points (four quarters each), nested in transects.
  Field-sampling null: resample the PCQM points with replacement within each transect, keep each
  selected tree's multiplicity, and build the resampled reference ECDF; D_field = KS distance between
  the resampled and the original reference (2000 draws, seed 0). p_design = P(D_field >= D_obs);
  D_95null, the 95 % quantile of D_field, is the distance a second field sample of the same design
  would reach. Reported in table3_per_seed.csv and table3_values.csv (columns *_design).

Density
  rho_sim / rho_rec and log(rho_sim / rho_rec) with a Poisson 95 % CI, 1.96 * sqrt(1/n_sim + 1/n_rec),
  from the mean simulated count; ratio to the plot-census target count (pass: within 1/1.3 - 1.3).

Input: output/assessment_year27_seeds_701_706.csv (year-27 stands of the frozen parameter set on the six
assessment seeds 701-706, built by make_assessment_csv.py; columns seed, species, dbh_cm, h_m, initial_1997_tree, age_years).
Writes table3_values.csv (formatted as in the manuscript), table3_per_seed.csv, ks_design_null.npz (cache).

Usage: python model_evaluation.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import kolmogorov

HERE = Path(__file__).resolve().parent
PACKAGE = HERE.parents[1]
RECON = PACKAGE / "Initialization" / "reconstructed_adults_2024.csv"
RAW_RECON = PACKAGE / "Initialization" / "reconstructed_raw_2024.csv"
MEASURED = PACKAGE / "pcqm_2024_quarters.csv"
TARGETS = PACKAGE / "plot_based_targets_2024" / "targets_2024.csv"
SEEDS_CSV = PACKAGE / "output" / "assessment_year27_seeds_701_706.csv"   # six assessment seeds of the pyMANGA v3.3.0 runs
NULL_CACHE = HERE / "ks_design_null.npz"

SPECIES = ["Rhizophora", "Avicennia", "Bruguiera", "Lumnitzera", "Excoecaria"]
PREFIX = {"Rhizpop": "Rhizophora", "Avipop": "Avicennia", "Brugpop": "Bruguiera", "Lumnpop": "Lumnitzera", "Excopop": "Excoecaria"}
GIRTH_RESOLUTION = 1.0            # cm, field girth tape reading
GIRTH_MIN = 8.0                   # cm
DBH_MIN = GIRTH_MIN / np.pi
HEIGHT_MIN = 1.3                  # m
DENSITY_BAND = 1.3
ALPHA = 0.05
C_ALPHA = 1.358                   # c(alpha) of the two-sample KS test at alpha = 0.05
DRAWS, NULL_SEED = 2000, 0
DOMAIN_HA = 300.0 * 113.1 / 1e4


def as_measured(dbh):
    """DBH (cm) as the field protocol records it: girth read to the whole centimetre."""
    return np.round(np.asarray(dbh, dtype=float) * np.pi / GIRTH_RESOLUTION) * GIRTH_RESOLUTION / np.pi


def census_rule(dbh, h):
    return (as_measured(dbh) >= DBH_MIN) & (np.asarray(h, float) >= HEIGHT_MIN)


# ----------------------------------------------------------------------------- design-based KS test
def load_design():
    """Measured PCQM trees with their sampling unit (transect_point) and copy multiplicity m."""
    meas = pd.read_csv(MEASURED)
    meas = meas[meas["status"] == "tree"].copy()
    meas["dbh"] = as_measured(meas["r_stem"].values * 200.0)
    meas["unit"] = meas["transect"].astype(str) + "_" + meas["point"].astype(str)
    meas["source_id"] = meas["unit"] + "_" + meas["quadrant"].astype(str)
    m = pd.read_csv(RAW_RECON)["source_id"].value_counts()
    meas["m"] = meas["source_id"].map(m).fillna(0).astype(float)
    return meas


def weighted_ecdf(values, weights, grid):
    order = np.argsort(values)
    v, w = values[order], weights[order]
    return np.interp(grid, v, np.cumsum(w) / w.sum(), left=0.0)


def ks_grid(sim, grid, f_ref):
    f_sim = np.searchsorted(np.sort(sim), grid, side="right") / len(sim)
    return float(np.max(np.abs(f_sim - f_ref)))


def field_null(meas, species, draws=DRAWS, seed=NULL_SEED):
    """D_field draws, reference grid and reference ECDF for one species."""
    rng = np.random.default_rng(seed)
    t = meas[meas["species"] == species]
    all_units = meas.groupby("transect")["unit"].unique()
    grid = np.unique(t["dbh"].values)
    f_ref = weighted_ecdf(t["dbh"].values, t["m"].values, grid)
    by_unit = {u: g for u, g in t.groupby("unit")}
    d = np.empty(draws)
    for b in range(draws):
        vals, wts = [], []
        for transect, units in all_units.items():
            for u in rng.choice(units, len(units), replace=True):
                if u in by_unit:
                    vals.append(by_unit[u]["dbh"].values)
                    wts.append(by_unit[u]["m"].values)
        v = np.concatenate(vals) if vals else np.array([])
        w = np.concatenate(wts) if wts else np.array([])
        d[b] = np.max(np.abs(weighted_ecdf(v, w, grid) - f_ref)) if len(v) else 1.0
    return d, grid, f_ref


def design_test(sim_dbh, d_field, grid, f_ref):
    """(D_obs, p, D_95null) of a simulated DBH sample (cm, as measured)."""
    d_obs = ks_grid(np.asarray(sim_dbh, float), grid, f_ref)
    p = float((np.sum(d_field >= d_obs - 1e-12) + 1) / (len(d_field) + 1))
    return d_obs, p, float(np.quantile(d_field, 0.95))


_NULL = None


def null():
    """{species: (D_field draws, grid, reference ECDF)}, cached in ks_design_null.npz."""
    global _NULL
    if _NULL is not None:
        return _NULL
    if NULL_CACHE.exists():
        z = np.load(NULL_CACHE)
        _NULL = {s: (z["d_" + s], z["grid_" + s], z["f_" + s]) for s in SPECIES}
        return _NULL
    meas = load_design()
    _NULL, save = {}, {}
    for s in SPECIES:
        _NULL[s] = field_null(meas, s)
        save["d_" + s], save["grid_" + s], save["f_" + s] = _NULL[s]
    np.savez_compressed(NULL_CACHE, **save)
    return _NULL


def design_ks(species, sim_dbh_measured):
    """Design-based test for one species; sim_dbh_measured = census-rule DBH sample, as measured."""
    d, grid, f = null()[species]
    return design_test(sim_dbh_measured, d, grid, f)


def ks_nobs(d_obs, n_obs, n_sim):
    """(D_crit, p) of the two-sample KS test with the measured sample size n_obs."""
    d_crit = C_ALPHA * np.sqrt((n_obs + n_sim) / (n_obs * n_sim))
    n_e = n_obs * n_sim / (n_obs + n_sim)
    return float(d_crit), float(kolmogorov(np.sqrt(n_e) * d_obs))


# ----------------------------------------------------------------------------- stands
def reference_counts():
    rec = pd.read_csv(RECON)
    rec = rec[census_rule(rec["r_stem"] * 200.0, rec["h_stem"])]
    return rec["species"].value_counts().reindex(SPECIES, fill_value=0)


def stand_from_population(path):
    """Census-rule adults of a Population_Year_27.csv (pyMANGA output): species, dbh_cm, h_m."""
    p = pd.read_csv(path, sep="\t", usecols=["plant", "r_stem", "h_stem"])
    return pd.DataFrame(dict(species=p["plant"].str.split("_").str[0].map(PREFIX), dbh_cm=p["r_stem"] * 200.0, h_m=p["h_stem"]))


def measured_counts():
    """n_obs: measured PCQM trees per species (every one is copied at least once in the reconstruction)."""
    meas = load_design()
    return meas[meas["m"] > 0]["species"].value_counts().reindex(SPECIES, fill_value=0)


def evaluate_stand(stand, n_obs, targets):
    """Per-species metrics of one year-27 stand (DataFrame with species, dbh_cm, h_m)."""
    keep = census_rule(stand["dbh_cm"], stand["h_m"])
    out = {}
    for s in SPECIES:
        sim = as_measured(stand.loc[keep & (stand["species"] == s), "dbh_cm"].values)
        n = len(sim)
        d_obs, p_design, q95 = design_ks(s, sim) if n else (np.nan, 0.0, np.nan)
        d_crit, p = ks_nobs(d_obs, n_obs[s], n) if n else (np.nan, 0.0)
        out.update({"n_sim_" + s: n, "D_" + s: d_obs, "Dcrit_" + s: d_crit, "p_" + s: p, "ks_ok_" + s: bool(d_obs < d_crit),
                    "p_design_" + s: p_design, "ks_ok_design_" + s: p_design >= ALPHA,
                    "ratio_target_" + s: n / targets.loc[s, "target_count_domain"],
                    "dens_ok_" + s: 1 / DENSITY_BAND <= n / targets.loc[s, "target_count_domain"] <= DENSITY_BAND})
    out["n_dens_ok"] = int(sum(out["dens_ok_" + s] for s in SPECIES))
    out["n_ks_ok"] = int(sum(out["ks_ok_" + s] for s in SPECIES))
    return out


def main():
    n_rec = reference_counts()
    n_obs = measured_counts()
    targets = pd.read_csv(TARGETS).set_index("species")
    nl = null()
    rows = []
    for seed, stand in pd.read_csv(SEEDS_CSV).groupby("seed"):
        r = evaluate_stand(stand, n_obs, targets)
        r["seed"] = int(seed)
        rows.append(r)
    per_seed = pd.DataFrame(rows).set_index("seed")
    per_seed.to_csv(HERE / "table3_per_seed.csv")

    table = []
    for s in SPECIES:
        n_sim = per_seed["n_sim_" + s]
        mean_n = n_sim.mean()
        log_ratio = np.log(mean_n / n_rec[s])
        half = 1.96 * np.sqrt(1 / mean_n + 1 / n_rec[s])
        fmt = lambda v, dec: ("%%.%df (%%.%df-%%.%df)" % (dec, dec, dec)) % (v.mean(), v.min(), v.max())
        table.append(dict(
            species=s, n_rec=int(n_rec[s]), n_obs=int(n_obs[s]), n_sim="%.1f (%d-%d)" % (mean_n, n_sim.min(), n_sim.max()),
            D_obs=fmt(per_seed["D_" + s], 3), D_crit=fmt(per_seed["Dcrit_" + s], 3), p=fmt(per_seed["p_" + s], 2),
            dens_rec_per_ha=int(round(n_rec[s] / DOMAIN_HA)), dens_sim_per_ha="%.1f (%d-%d)" % (mean_n / DOMAIN_HA, round(n_sim.min() / DOMAIN_HA), round(n_sim.max() / DOMAIN_HA)),
            plot_target_per_ha=int(round(targets.loc[s, "target_density_per_ha"])), ratio_to_target=fmt(per_seed["ratio_target_" + s], 2),
            log_ratio_sim_rec="%+.2f [%+.2f, %+.2f]" % (log_ratio, log_ratio - half, log_ratio + half),
            ks_pass_seeds="%d/%d" % (per_seed["ks_ok_" + s].sum(), len(per_seed)), density_pass_seeds="%d/%d" % (per_seed["dens_ok_" + s].sum(), len(per_seed)),
            D_95null_design=round(float(np.quantile(nl[s][0], 0.95)), 3), p_design=fmt(per_seed["p_design_" + s], 2),
            ks_pass_seeds_design="%d/%d" % (per_seed["ks_ok_design_" + s].sum(), len(per_seed))))
    table = pd.DataFrame(table)
    table.to_csv(HERE / "table3_values.csv", index=False)
    pd.set_option("display.width", 250)
    print(table.to_string(index=False))
    joint = int(((per_seed["n_dens_ok"] == 5) & (per_seed["n_ks_ok"] >= 4)).sum())
    print("\nseeds with all five densities within 1/1.3-1.3 of the plot targets and >= 4 KS passes: %d of %d" % (joint, len(per_seed)))
    main_run = PACKAGE / "output" / "Population_Year_27.csv"
    if main_run.exists():
        r = evaluate_stand(stand_from_population(main_run), n_obs, targets)
        print("main run (output/Population_Year_27.csv): densities ok %d/5, KS passes %d/5; p = %s" % (
            r["n_dens_ok"], r["n_ks_ok"], ", ".join("%s %.2f" % (s[:4], r["p_" + s]) for s in SPECIES)))


if __name__ == "__main__":
    main()
