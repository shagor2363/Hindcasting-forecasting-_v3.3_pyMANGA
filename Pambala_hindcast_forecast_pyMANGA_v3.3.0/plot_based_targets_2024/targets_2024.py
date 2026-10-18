"""
2024 abundance targets per species and band from the plot census.

species density in band b = eligible census stems in the band's plots / sampled plot area (ha)
domain density            = sum_b density_b * band area_b / domain area   (bands weighted by their
                            share of the model domain, not by their number of plots)

Eligibility (census rule): girth rounded to the centimetre >= 8 cm (DBH >= 2.546 cm), height >= 1.3 m
(rows without a height are kept: no measured height is below 1.3 m), alive, adult layer, model species.
Multi-stemmed trees: the largest stem gives the DBH; exact duplicate rows of a plot are counted once.
Uncertainty: bootstrap over plots within each band (plots are the sampling units; one band = one
transect), 2000 draws, 95 % percentile interval. Band geometry = reconstruct_segments.R (zone_y).

Writes targets_2024.csv (domain), targets_2024_bands.csv (per band),
targets_2024_for_reconstruction.csv (the counts handed to reconstruct_targets.R).

Usage: python targets_2024.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import plot_census_2024 as P  # noqa: E402

SPECIES = ["Rhizophora", "Avicennia", "Bruguiera", "Lumnitzera", "Excoecaria"]
X1, X2 = -17.9, 282.1
ZONE_Y = [-18.1, 12.50, 37.49, 62.54, 95.0]
BAND_AREA_HA = {b + 1: (X2 - X1) * (ZONE_Y[b + 1] - ZONE_Y[b]) / 1e4 for b in range(4)}
DOMAIN_HA = sum(BAND_AREA_HA.values())
POINTS_PER_TRANSECT = 26
GIRTH_MIN, HEIGHT_MIN = 8.0, 1.3
DRAWS, SEED = 2000, 0


def eligible_census():
    """(plots, eligible census stems, number of duplicate rows removed)."""
    P.DBH_MIN = 2.0                                   # take rows below the old cut, then apply the rounded-girth rule
    plots, raw = P.load_census()
    raw = raw.copy()
    raw["girth_cm"] = np.round(raw["dbh"] * np.pi)
    keep = (raw["girth_cm"] >= GIRTH_MIN) & (raw["h"].isna() | (raw["h"] >= HEIGHT_MIN)) & raw["species"].isin(SPECIES)
    cen = raw[keep].copy()
    duplicates = cen.duplicated(subset=["Transect", "Plot", "species", "dbh", "h", "px", "py"], keep="first")
    return plots, cen[~duplicates].reset_index(drop=True), int(duplicates.sum())


def main():
    plots, cen, n_dup = eligible_census()
    rng = np.random.default_rng(SEED)
    counts = cen.groupby(["Transect", "Plot", "species"]).size().unstack(fill_value=0).reindex(columns=SPECIES, fill_value=0)
    band_rows, boot = [], {}
    for b in (1, 2, 3, 4):
        pb = plots[plots["Transect"] == b]
        n_plots = len(pb)
        area_ha = n_plots * P.PLOT_SIDE ** 2 / 1e4
        c = counts.reindex([(b, p) for p in pb["Plot"]], fill_value=0)
        dens = c.sum() / area_ha
        draws = np.array([c.iloc[rng.integers(0, n_plots, n_plots)].sum().values for _ in range(DRAWS)]) / area_ha
        boot[b] = draws
        for j, s in enumerate(SPECIES):
            band_rows.append(dict(band=b, species=s, census_stems=int(c[s].sum()), plots=n_plots, sampled_area_ha=round(area_ha, 2),
                                  band_area_ha=round(BAND_AREA_HA[b], 3), points_sampled=n_plots, points_total=POINTS_PER_TRANSECT,
                                  coverage_along_transect=round(n_plots / POINTS_PER_TRANSECT, 2),
                                  strip_width_m=P.PLOT_SIDE, band_width_m=round(ZONE_Y[b] - ZONE_Y[b - 1], 1),
                                  density_per_ha=round(float(dens[s]), 1), ci_low=round(float(np.percentile(draws[:, j], 2.5)), 1),
                                  ci_high=round(float(np.percentile(draws[:, j], 97.5)), 1),
                                  target_count_in_band=int(round(dens[s] * BAND_AREA_HA[b]))))
    bands = pd.DataFrame(band_rows)
    dom_rows = []
    for j, s in enumerate(SPECIES):
        d = bands[bands["species"] == s]
        dens_dom = float((d["density_per_ha"] * d["band_area_ha"]).sum() / DOMAIN_HA)
        draws_dom = sum(boot[b][:, j] * BAND_AREA_HA[b] for b in (1, 2, 3, 4)) / DOMAIN_HA
        dom_rows.append(dict(species=s, census_stems=int(d["census_stems"].sum()), sampled_area_ha=round(float(d["sampled_area_ha"].sum()), 2),
                             domain_area_ha=round(DOMAIN_HA, 3), target_density_per_ha=round(dens_dom, 1),
                             ci_low=round(float(np.percentile(draws_dom, 2.5)), 1), ci_high=round(float(np.percentile(draws_dom, 97.5)), 1),
                             target_count_domain=int(round(dens_dom * DOMAIN_HA))))
    dom = pd.DataFrame(dom_rows)
    bands.to_csv(HERE / "targets_2024_bands.csv", index=False)
    dom.to_csv(HERE / "targets_2024.csv", index=False)
    bands[["band", "species", "target_count_in_band"]].to_csv(HERE / "targets_2024_for_reconstruction.csv", index=False)
    pd.set_option("display.width", 250)
    print(dom.to_string(index=False))
    print("duplicate multi-stem rows removed:", n_dup, "| eligible census stems:", len(cen))


if __name__ == "__main__":
    main()
