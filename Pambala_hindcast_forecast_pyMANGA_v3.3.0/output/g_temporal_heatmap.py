#!/usr/bin/env python3
"""Pair correlation g(r) per species, year and distance, with per-year CSR tests.

One panel per species. Rows are the simulation years 27, 50, 75 and 100; columns
are distance bins over 0-25 m; colour is the observed g(r) on a diverging scale
centred on g = 1; a dot marks bins where the observed g(r) lies outside that
year's own two-sided 95 % CSR envelope; grey cells hold too few tree pairs to
interpret. Each year is compared against an envelope simulated from that year's
own tree count, so no statistic is averaged across years and every dot is a
per-year, per-distance significance statement.

    g > 1   more pairs of trees than expected under CSR (clustering)
    g = 1   as many pairs of trees as expected (complete spatial randomness)
    g < 1   fewer pairs of trees than expected (regular spacing / inhibition)

Companion figure with the same layout for the mark correlation function:
mcf_temporal_heatmap.py.

Reproducing the figure
----------------------
    python g_temporal_heatmap.py

Runs from any working directory: it reads the archived pyMANGA output
Population_Year_{27,50,75,100}.csv (tab-separated, columns plant, x, y, growth)
from this script's own directory and writes g_temporal_heatmap.png and
g_temporal_heatmap_values.csv beside them.
"""
from pathlib import Path
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm
from matplotlib.lines import Line2D
from scipy.spatial.distance import pdist

HERE = Path(__file__).resolve().parent
INPUT_TEMPLATE = "Population_Year_{year}.csv"
OUT_PNG = HERE / "g_temporal_heatmap.png"
OUT_CSV = HERE / "g_temporal_heatmap_values.csv"

YEARS = (27, 50, 75, 100)
SPECIES = (
    ("Rhizpop", "Rhizophora spp."),
    ("Avipop", "A. officinalis"),
    ("Brugpop", "Bruguiera spp."),
    ("Lumnpop", "L. racemosa"),
    ("Excopop", "E. agallocha"),
)

# Model domain corners, taken from <domain> in ../d_2_p.xml. The CSR replicates
# are drawn uniformly over this rectangle, and AREA sets the expected pair
# density the estimator normalises by.
X1, X2 = -17.9, 282.1
Y1, Y2 = -18.1, 95.0
AREA = (X2 - X1) * (Y2 - Y1)

MAX_DIST = 25.0
N_BINS = 25
EDGES = np.linspace(0, MAX_DIST, N_BINS + 1)
R_CENTERS = (EDGES[:-1] + EDGES[1:]) / 2
BIN_WIDTH = EDGES[1] - EDGES[0]

N_SIM = 199              # CSR replicates per (species, year) envelope
Q_LO, Q_HI = 2.5, 97.5   # pointwise two-sided 95 % envelope
MIN_PAIRS = 5            # bins with fewer observed tree pairs are not shown
SEED = 2024              # one RNG, consumed in a fixed species-by-year order,
                         # so every rerun reproduces the published figure

# Diverging scale with g = 1 at white. The range is log-symmetric, so g and 1/g
# sit equally far from white; clustering in the abundant species reaches ~2. The
# noisy small-N species saturate instead of stretching the scale, and the
# colourbar carries triangular caps to show that.
V_MIN, V_CENTER, V_MAX = 0.5, 1.0, 2.0

PANEL_POSITIONS = ((0, 0), (0, 1), (0, 2), (1, 0), (1, 1))


def estimate_gr(x, y):
    """Return binned g(r) and the observed tree-pair count per bin.

    Deliberately without edge correction: the same bias applies to the observed
    pattern and to every envelope replicate, which keeps the comparison
    internally consistent (Stoyan & Stoyan 1994, sec. 15.1).
    """
    n = len(x)
    if n < 4:
        return np.full(N_BINS, np.nan), np.zeros(N_BINS, dtype=int)

    d = pdist(np.column_stack([x, y]))
    d = d[d <= MAX_DIST]
    bin_idx = np.clip(np.digitize(d, EDGES) - 1, 0, N_BINS - 1)
    counts = np.bincount(bin_idx, minlength=N_BINS)

    expected = (n * (n - 1) / AREA) * 2 * np.pi * R_CENTERS * BIN_WIDTH
    g = 2 * counts / expected      # pdist counts each pair once, expected twice
    g[0] = np.nan                  # bin straddles r = 0, where g is degenerate
    return g, counts


def gr_with_envelope(x, y, rng):
    """g(r) of the observed pattern plus a CSR envelope and the pair counts.

    The envelope comes from N_SIM uniform point patterns of the same tree count
    on the same domain, so it is specific to this species and year.
    """
    g_obs, counts = estimate_gr(x, y)
    n = len(x)
    if n < 4:
        undefined = np.full(N_BINS, np.nan)
        return g_obs, undefined, undefined, counts

    g_sims = np.empty((N_SIM, N_BINS))
    for s in range(N_SIM):
        g_sims[s], _ = estimate_gr(rng.uniform(X1, X2, size=n),
                                   rng.uniform(Y1, Y2, size=n))
    with warnings.catch_warnings():
        # The first bin is NaN in every replicate by construction.
        warnings.simplefilter("ignore", RuntimeWarning)
        g_lo = np.nanpercentile(g_sims, Q_LO, axis=0)
        g_hi = np.nanpercentile(g_sims, Q_HI, axis=0)
    return g_obs, g_lo, g_hi, counts


def load_year(year):
    """Read one snapshot year and label each plant with its species prefix.

    Rows without a growth value are the seedlings recruited in that step; they
    have not been through a growth update yet and are excluded.
    """
    df = pd.read_csv(HERE / INPUT_TEMPLATE.format(year=year), sep="\t")
    df["species_prefix"] = df["plant"].str.split("_").str[0]
    return df[df["growth"].notna()]


def collect_grid():
    """Build {species_prefix: {g, sig, valid}} of (year x bin) arrays."""
    frames = {year: load_year(year) for year in YEARS}
    rng = np.random.default_rng(SEED)
    grid = {}
    for prefix, display in SPECIES:
        g = np.full((len(YEARS), N_BINS), np.nan)
        sig = np.zeros((len(YEARS), N_BINS), dtype=bool)
        valid = np.zeros((len(YEARS), N_BINS), dtype=bool)
        for row, year in enumerate(YEARS):
            df = frames[year]
            sub = df[df["species_prefix"] == prefix]
            x = sub["x"].to_numpy(dtype=float)
            y = sub["y"].to_numpy(dtype=float)
            finite = np.isfinite(x) & np.isfinite(y)
            x, y = x[finite], y[finite]

            g_obs, g_lo, g_hi, n_pairs = gr_with_envelope(x, y, rng)
            enough = n_pairs >= MIN_PAIRS
            g[row] = np.where(enough, g_obs, np.nan)
            valid[row] = enough & np.isfinite(g[row])
            sig[row] = valid[row] & ((g[row] < g_lo) | (g[row] > g_hi))
            print(f"  {display:<16s} year {year:>3d}  N={len(x):4d}  "
                  f"significant bins: {int(sig[row].sum()):2d}/"
                  f"{int(valid[row].sum()):2d}")
        grid[prefix] = {"g": g, "sig": sig, "valid": valid}
    return grid


def plot_heatmap(grid, out_png):
    rc = {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans"],
        "font.size": 13,
        "axes.titlesize": 14,
        "axes.labelsize": 14,
        "axes.linewidth": 1.0,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.size": 4.0,
        "ytick.major.size": 0.0,
    }
    cmap = plt.cm.RdBu_r.copy()
    cmap.set_bad("0.9")                 # bins masked by MIN_PAIRS
    norm = TwoSlopeNorm(vmin=V_MIN, vcenter=V_CENTER, vmax=V_MAX)
    y_edges = np.arange(len(YEARS) + 1)

    with plt.rc_context(rc):
        fig = plt.figure(figsize=(11.6, 6.4))
        gs = fig.add_gridspec(2, 3, hspace=0.55, wspace=0.14,
                              left=0.085, right=0.985, top=0.92, bottom=0.12)

        for i, (prefix, display) in enumerate(SPECIES):
            row, col = PANEL_POSITIONS[i]
            ax = fig.add_subplot(gs[row, col])

            mesh = ax.pcolormesh(EDGES, y_edges,
                                 np.ma.masked_invalid(grid[prefix]["g"]),
                                 cmap=cmap, norm=norm,
                                 edgecolors="white", linewidth=0.6)
            ax.invert_yaxis()           # year 27 on top, year 100 at the bottom

            sig_rows, sig_cols = np.where(grid[prefix]["sig"])
            if len(sig_cols):
                ax.scatter(R_CENTERS[sig_cols], sig_rows + 0.5, s=13, c="black",
                           edgecolors="white", linewidths=0.4, zorder=5)

            ax.set_xlim(0, MAX_DIST)
            ax.set_xticks([0, 5, 10, 15, 20, 25])
            ax.set_yticks(np.arange(len(YEARS)) + 0.5)
            ax.set_yticklabels(YEARS)
            ax.spines[:].set_visible(False)
            if row == 1:
                ax.set_xlabel(r"Distance $r$ (m)")
            if col == 0:
                ax.set_ylabel("Year")

            ax.text(-0.02, 1.14, f"({'ABCDE'[i]})", transform=ax.transAxes,
                    fontweight="bold", fontsize=15, va="bottom", ha="left")
            ax.text(0.5, 1.14, display, transform=ax.transAxes, ha="center",
                    va="bottom", fontstyle="italic", fontsize=14,
                    fontweight="bold")

        # The free sixth cell carries the shared colourbar and the legend.
        ax_leg = fig.add_subplot(gs[1, 2])
        ax_leg.set_axis_off()
        box = ax_leg.get_position()

        cax = fig.add_axes([box.x0 + 0.012, box.y0 + 0.06,
                            0.016, box.height - 0.12])
        cb = fig.colorbar(mesh, cax=cax, extend="both")
        cb.set_label(r"Pair correlation $g(r)$", fontsize=12)
        cb.set_ticks([0.5, 0.75, 1.0, 1.5, 2.0])
        cb.ax.tick_params(labelsize=11)

        for y_pos, text, colour in (
            (0.99, "g > 1  more pairs of trees\n          than expected", "#67001f"),
            (0.81, "g = 1  as many pairs of trees\n          as expected", "0.35"),
            (0.63, "g < 1  fewer pairs of trees\n          than expected", "#053061"),
        ):
            ax_leg.text(0.42, y_pos, text, transform=ax_leg.transAxes,
                        fontsize=11, va="top", color=colour, linespacing=1.1)

        handles = [
            Line2D([0], [0], marker="o", color="none", markerfacecolor="black",
                   markeredgecolor="white", markeredgewidth=0.4, markersize=7,
                   label="significant"),
            Line2D([0], [0], marker="s", color="none", markerfacecolor="0.9",
                   markeredgecolor="0.6", markersize=10,
                   label=f"insufficient data\n(<{MIN_PAIRS} tree pairs)"),
        ]
        ax_leg.legend(handles=handles, loc="lower left",
                      bbox_to_anchor=(0.40, 0.00), frameon=False, fontsize=11,
                      handletextpad=0.6, labelspacing=1.2, borderpad=0.4)

        plt.savefig(out_png, dpi=300, bbox_inches="tight")
        plt.close(fig)
    print(f"Saved: {out_png}")


def save_values_csv(grid, out_csv):
    """Write the g(r), significance and data-sufficiency value behind every cell."""
    blocks = [
        pd.DataFrame({
            "species": display,
            "year": np.repeat(YEARS, N_BINS),
            "r_center_m": np.tile(R_CENTERS.round(2), len(YEARS)),
            "g_r": grid[prefix]["g"].ravel(),
            "significant": grid[prefix]["sig"].ravel(),
            "valid": grid[prefix]["valid"].ravel(),
        })
        for prefix, display in SPECIES
    ]
    pd.concat(blocks, ignore_index=True).to_csv(out_csv, index=False,
                                                float_format="%.4f")
    print(f"Saved: {out_csv}")


def main():
    missing = [INPUT_TEMPLATE.format(year=year) for year in YEARS
               if not (HERE / INPUT_TEMPLATE.format(year=year)).is_file()]
    if missing:
        raise SystemExit(f"Missing pyMANGA output in {HERE}: {', '.join(missing)}")

    print("Computing per-(species, year) g(r) grid with CSR significance ...")
    grid = collect_grid()
    plot_heatmap(grid, OUT_PNG)
    save_values_csv(grid, OUT_CSV)


if __name__ == "__main__":
    main()
