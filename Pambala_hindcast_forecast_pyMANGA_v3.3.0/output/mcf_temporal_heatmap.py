#!/usr/bin/env python3
"""Mark correlation K_mm(r) per species, year and distance, with per-year tests.

One panel per species. Rows are the simulation years 27, 50, 75 and 100; columns
are distance bins over 0-25 m; colour is the observed K_mm(r) on a diverging
scale centred on K = 1; a dot marks bins where the observed K_mm(r) lies outside
that year's own two-sided 95 % mark-random envelope; grey cells hold too few tree
pairs to interpret. The mark is stem radius r_stem, and each year is tested
against permutations of its own marks across its own tree positions, so no
statistic is averaged across years and every dot is a per-year, per-distance
significance statement.

    K > 1   size products of tree pairs above the random-marking reference
    K = 1   size products equal to the random-marking reference
    K < 1   size products of tree pairs below the random-marking reference

Companion figure with the same layout for the pair correlation function:
g_temporal_heatmap.py.

Reproducing the figure
----------------------
    python mcf_temporal_heatmap.py

Runs from any working directory: it reads the archived pyMANGA output
Population_Year_{27,50,75,100}.csv (tab-separated, columns plant, x, y, r_stem,
growth) from this script's own directory and writes mcf_temporal_heatmap.png and
mcf_temporal_heatmap_values.csv beside them.
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
OUT_PNG = HERE / "mcf_temporal_heatmap.png"
OUT_CSV = HERE / "mcf_temporal_heatmap_values.csv"

YEARS = (27, 50, 75, 100)
SPECIES = (
    ("Rhizpop", "Rhizophora spp."),
    ("Avipop", "A. officinalis"),
    ("Brugpop", "Bruguiera spp."),
    ("Lumnpop", "L. racemosa"),
    ("Excopop", "E. agallocha"),
)

MAX_DIST = 25.0
N_BINS = 25
EDGES = np.linspace(0, MAX_DIST, N_BINS + 1)
R_CENTERS = (EDGES[:-1] + EDGES[1:]) / 2

N_SIM = 199              # mark permutations per (species, year) envelope
Q_LO, Q_HI = 2.5, 97.5   # pointwise two-sided 95 % envelope
MIN_PAIRS = 5            # bins with fewer observed tree pairs are not shown
SEED = 2024              # one RNG, consumed in a fixed species-by-year order,
                         # so every rerun reproduces the published figure

# Diverging scale with K = 1 at white. The range is clipped so the noisy
# small-N species (E. agallocha, N ~ 60) saturate instead of stretching the
# scale for everyone else; the colourbar carries triangular caps to show that.
V_MIN, V_CENTER, V_MAX = 0.6, 1.0, 1.6

PANEL_POSITIONS = ((0, 0), (0, 1), (0, 2), (1, 0), (1, 1))


def mcf_with_envelope(x, y, marks, rng):
    """Return observed K_mm(r), its mark-random envelope and the pair count per bin.

    K_mm(r) is the mean product of the marks of all tree pairs a distance r
    apart, normalised by the squared mean mark. The envelope comes from N_SIM
    permutations of the marks over the fixed tree positions, so it holds the
    point pattern constant and tests only the arrangement of sizes on it.
    """
    n = len(marks)
    if n < 4:
        undefined = np.full(N_BINS, np.nan)
        return undefined, undefined.copy(), undefined.copy(), np.zeros(N_BINS, dtype=int)

    # pdist and triu_indices enumerate the pairs in the same order, so dists[k]
    # is the distance between trees i_idx[k] and j_idx[k].
    dists = pdist(np.column_stack([x, y]))
    i_idx, j_idx = np.triu_indices(n, k=1)
    near = dists <= MAX_DIST
    bin_idx = np.clip(np.digitize(dists[near], EDGES) - 1, 0, N_BINS - 1)
    i_idx, j_idx = i_idx[near], j_idx[near]

    n_pairs = np.bincount(bin_idx, minlength=N_BINS)
    members = [np.flatnonzero(bin_idx == b) for b in range(N_BINS)]

    def binned_mean(values):
        out = np.full(N_BINS, np.nan)
        for b, idx in enumerate(members):
            if len(idx):
                out[b] = values[idx].mean()
        return out

    mu_sq = marks.mean() ** 2
    k_obs = binned_mean(marks[i_idx] * marks[j_idx]) / mu_sq

    k_sims = np.empty((N_SIM, N_BINS))
    for s in range(N_SIM):
        shuffled = rng.permutation(marks)
        k_sims[s] = binned_mean(shuffled[i_idx] * shuffled[j_idx]) / mu_sq
    with warnings.catch_warnings():
        # Bins holding no pair at all are NaN in every permutation.
        warnings.simplefilter("ignore", RuntimeWarning)
        k_lo = np.nanpercentile(k_sims, Q_LO, axis=0)
        k_hi = np.nanpercentile(k_sims, Q_HI, axis=0)
    return k_obs, k_lo, k_hi, n_pairs


def load_year(year):
    """Read one snapshot year and label each plant with its species prefix.

    Rows without a growth value are the seedlings recruited in that step; they
    have not been through a growth update yet and are excluded.
    """
    df = pd.read_csv(HERE / INPUT_TEMPLATE.format(year=year), sep="\t")
    df["species_prefix"] = df["plant"].str.split("_").str[0]
    return df[df["growth"].notna()]


def collect_grid():
    """Build {species_prefix: {K, sig, valid}} of (year x bin) arrays."""
    frames = {year: load_year(year) for year in YEARS}
    rng = np.random.default_rng(SEED)
    grid = {}
    for prefix, display in SPECIES:
        k = np.full((len(YEARS), N_BINS), np.nan)
        sig = np.zeros((len(YEARS), N_BINS), dtype=bool)
        valid = np.zeros((len(YEARS), N_BINS), dtype=bool)
        for row, year in enumerate(YEARS):
            df = frames[year]
            sub = df[df["species_prefix"] == prefix]
            x = sub["x"].to_numpy(dtype=float)
            y = sub["y"].to_numpy(dtype=float)
            marks = sub["r_stem"].to_numpy(dtype=float)
            finite = np.isfinite(x) & np.isfinite(y) & np.isfinite(marks)
            x, y, marks = x[finite], y[finite], marks[finite]

            k_obs, k_lo, k_hi, n_pairs = mcf_with_envelope(x, y, marks, rng)
            enough = n_pairs >= MIN_PAIRS
            k[row] = np.where(enough, k_obs, np.nan)
            valid[row] = enough & np.isfinite(k[row])
            sig[row] = valid[row] & ((k[row] < k_lo) | (k[row] > k_hi))
            print(f"  {display:<16s} year {year:>3d}  N={len(marks):4d}  "
                  f"significant bins: {int(sig[row].sum()):2d}/"
                  f"{int(valid[row].sum()):2d}")
        grid[prefix] = {"K": k, "sig": sig, "valid": valid}
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
                                 np.ma.masked_invalid(grid[prefix]["K"]),
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
        cb.set_label(r"Mark correlation $K_{mm}(r)$", fontsize=12)
        cb.set_ticks([0.6, 0.8, 1.0, 1.2, 1.4, 1.6])
        cb.ax.tick_params(labelsize=11)

        for y_pos, text, colour in (
            (0.99, "K > 1  size products above\n          random-marking reference", "#67001f"),
            (0.81, "K = 1  size products equal to\n          random-marking reference", "0.35"),
            (0.63, "K < 1  size products below\n          random-marking reference", "#053061"),
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
    """Write the K_mm, significance and data-sufficiency value behind every cell."""
    blocks = [
        pd.DataFrame({
            "species": display,
            "year": np.repeat(YEARS, N_BINS),
            "r_center_m": np.tile(R_CENTERS.round(2), len(YEARS)),
            "K_mm": grid[prefix]["K"].ravel(),
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

    print("Computing per-(species, year) MCF grid with mark-random significance ...")
    grid = collect_grid()
    plot_heatmap(grid, OUT_PNG)
    save_values_csv(grid, OUT_CSV)


if __name__ == "__main__":
    main()
