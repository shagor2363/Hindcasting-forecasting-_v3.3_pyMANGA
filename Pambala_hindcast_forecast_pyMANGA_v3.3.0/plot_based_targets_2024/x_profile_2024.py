"""
Along-transect density profile of each species per band from the 2024 plot census.

    lambda_s,b(x) = sum_c K_h(x - x_c) n_s,c / sum_c K_h(x - x_c) a_c

over the plots c of band b (Gaussian kernel, plot centre x_c, eligible stems n_s,c, sampled plot
area a_c): a kernel-weighted ratio of stems to sampled area, so sampling effort is accounted for.
The bandwidth h is chosen per band and species by leave-one-plot-out cross-validation of that
species' plot counts (Poisson deviance; candidates 10-120 m, the 10 m floor being the plot
spacing). The profile gives the shape of each species' distribution along x inside its band
(initial positions in reconstruct_targets.R); the band totals remain the band targets.

Writes x_profile_2024.csv (band, species, x, lambda_per_ha), x_profile_bandwidths.csv, x_profile_2024.png.

Usage: python x_profile_2024.py   (after targets_2024.py)
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import targets_2024 as T  # noqa: E402

GRID = np.arange(T.X1, T.X2 + 0.5, 1.0)
BANDWIDTHS = np.array([10, 15, 20, 30, 40, 50, 60, 80, 100, 120], float)


def kernel(x, xc, h):
    return np.exp(-0.5 * ((x[:, None] - xc[None, :]) / h) ** 2)


def profile(x, xc, counts, area, h):
    k = kernel(x, xc, h)
    return (k @ counts) / np.maximum(k @ area, 1e-12)


def loo_deviance(xc, counts, area, h):
    """Leave-one-plot-out Poisson deviance of the predicted plot counts."""
    dev = 0.0
    for i in range(len(xc)):
        keep = np.arange(len(xc)) != i
        lam = np.maximum(profile(xc[[i]], xc[keep], counts[keep], area[keep], h)[0] * area[i], 1e-9)
        n = counts[i]
        dev += 2 * np.sum(np.where(n > 0, n * np.log(n / lam), 0) - (n - lam))
    return dev


def main():
    plots, cen, _ = T.eligible_census()
    counts = cen.groupby(["Transect", "Plot", "species"]).size().unstack(fill_value=0).reindex(columns=T.SPECIES, fill_value=0)
    rows, bw_rows = [], []
    fig, axes = plt.subplots(4, 1, figsize=(9, 10), sharex=True)
    for b, ax in zip((1, 2, 3, 4), axes):
        pb = plots[plots["Transect"] == b].sort_values("xc")
        xc = pb["xc"].values.astype(float)
        c = counts.reindex([(b, p) for p in pb["Plot"]], fill_value=0).values.astype(float)
        area = np.full(len(xc), T.P.PLOT_SIDE ** 2 / 1e4)
        for j, s in enumerate(T.SPECIES):
            devs = [loo_deviance(xc, c[:, [j]], area, h) for h in BANDWIDTHS]
            h = float(BANDWIDTHS[int(np.argmin(devs))])
            bw_rows.append(dict(band=b, species=s, bandwidth_m=h, plots=len(xc), **{"dev_h%d" % hh: round(d, 1) for hh, d in zip(BANDWIDTHS, devs)}))
            lam = profile(GRID, xc, c[:, j], area, h)
            for x, l in zip(GRID, lam):
                rows.append(dict(band=b, species=s, x=x, lambda_per_ha=l))
            ax.plot(GRID, lam, label=s)
            ax.scatter(xc, c[:, j] / area, s=8, alpha=0.5)
        ax.set_ylabel("band %d\nstems ha$^{-1}$" % b)
        ax.set_title("bandwidths (leave-one-plot-out): %s" % ", ".join("%s %d m" % (r["species"][:4], r["bandwidth_m"]) for r in bw_rows if r["band"] == b), fontsize=8, loc="left")
    axes[0].legend(ncol=5, fontsize=8)
    axes[-1].set_xlabel("x (m)")
    fig.tight_layout()
    fig.savefig(HERE / "x_profile_2024.png", dpi=150)
    pd.DataFrame(rows).to_csv(HERE / "x_profile_2024.csv", index=False)
    bw = pd.DataFrame(bw_rows)
    bw.to_csv(HERE / "x_profile_bandwidths.csv", index=False)
    print(bw[["band", "species", "bandwidth_m"]].to_string(index=False))


if __name__ == "__main__":
    main()
