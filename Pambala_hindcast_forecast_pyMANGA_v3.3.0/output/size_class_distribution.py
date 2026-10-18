"""
Size-class distribution of the simulated stand (fraction of stems per DBH class, median and min-max range over the
yearly snapshots of output/Population.csv), one panel per species. Manuscript figure script with the shared y-axis
ending at 0.6 (the published version used 0.42, which clips the Bruguiera sapling classes in this run).

Usage (from the package root): python output/size_class_distribution.py output output
Writes output/size_class_distribution.png and size_class_distribution_values.csv.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

HERE = Path(__file__).resolve().parent
POP_CSV = HERE / "Population.csv"
FIG_SIZE_CLASS = HERE / "size_class_distribution.png"

DELTA_T = 3.16e7
SPECIES_LIST = ["Rhizophora", "Avicennia", "Bruguiera", "Lumnitzera", "Excoecaria"]
SPECIES_MAPPING = {"Rhizpop": "Rhizophora", "Avipop": "Avicennia", "Brugpop": "Bruguiera", "Lumnpop": "Lumnitzera", "Excopop": "Excoecaria"}
SPECIES_DISPLAY = {"Rhizophora": "Rhizophora spp.", "Avicennia": "A. officinalis", "Bruguiera": "Bruguiera spp.", "Lumnitzera": "L. racemosa", "Excoecaria": "E. agallocha"}
YEARS = [27, 50, 75, 100]
# Upper edge capped at 50 cm; the few stems above 50 cm DBH are not in the histogram.
DBH_BINS = np.array([0, 1, 2.5, 5, 10, 15, 20, 30, 40, 50], dtype=float)
DBH_BIN_CENTRES = 0.5 * (DBH_BINS[:-1] + DBH_BINS[1:])
OCCUPIED_MIN = 0.05
Y_MAX = 0.63   # shared upper y-limit (user request 2026-10-15: 0.6 scale); Bruguiera 1-2.5 cm ribbon reaches ~0.62 in v33rc1


def load_population(csv_path):
    df = pd.read_csv(csv_path, sep="\t", usecols=["plant", "time", "r_stem"])
    df["species"] = df["plant"].str.extract(r"([A-Za-z]+)_")[0].map(SPECIES_MAPPING)
    df["dbh"] = df["r_stem"] * 200.0
    df["year"] = (df["time"] / DELTA_T).round().astype(int)
    return df


def size_class_fraction(dbh):
    counts, _ = np.histogram(dbh, bins=DBH_BINS)
    return counts / counts.sum() if counts.sum() else np.full(counts.size, np.nan)


def distributions(df):
    out = {sp: {} for sp in SPECIES_LIST}
    for year in YEARS:
        yr = df[df["year"] == year]
        for sp in SPECIES_LIST:
            out[sp][year] = size_class_fraction(yr.loc[yr["species"] == sp, "dbh"].to_numpy(dtype=float))
    return out


def plot(dists, out_png):
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"], "font.size": 14, "axes.titlesize": 15,
                         "axes.labelsize": 15, "axes.linewidth": 1.0, "xtick.labelsize": 13, "ytick.labelsize": 13, "xtick.direction": "out",
                         "ytick.direction": "out", "xtick.major.size": 5.0, "ytick.major.size": 5.0, "xtick.major.width": 1.0,
                         "ytick.major.width": 1.0, "legend.frameon": False, "legend.fontsize": 14})
    fig = plt.figure(figsize=(11.2, 6.6))
    gs = GridSpec(2, 3, figure=fig, hspace=0.30, wspace=0.18, left=0.08, right=0.985, top=0.95, bottom=0.11)
    panel_positions = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1)]
    median_color, ribbon_color = "#5d3a1a", "#d4b48f"
    visible_ticks = [1, 2.5, 5, 10, 25, 50]
    rows = []
    for i, sp in enumerate(SPECIES_LIST):
        ax = fig.add_subplot(gs[panel_positions[i]])
        stacked = np.vstack([dists[sp][y] for y in YEARS if np.isfinite(dists[sp][y]).all()])
        f_min, f_max, f_med = stacked.min(axis=0), stacked.max(axis=0), np.median(stacked, axis=0)
        per_bin_range = f_max - f_min
        occupied = f_med >= OCCUPIED_MIN
        mean_range_pp = float(np.nanmean(per_bin_range[occupied])) * 100.0 if occupied.any() else float("nan")
        max_range_pp = float(np.nanmax(per_bin_range)) * 100.0
        for lo, hi, mn, mx, md, occ in zip(DBH_BINS[:-1], DBH_BINS[1:], f_min, f_max, f_med, occupied):
            rows.append(dict(species=sp, dbh_class_lower_cm=lo, dbh_class_upper_cm=hi, fraction_min=mn, fraction_max=mx, fraction_median=md, occupied=occ))
        rows.append(dict(species=sp, dbh_class_lower_cm=np.nan, dbh_class_upper_cm=np.nan, fraction_min=np.nan, fraction_max=np.nan, fraction_median=np.nan,
                         occupied=np.nan, mean_temporal_range_pp=mean_range_pp, max_temporal_range_pp=max_range_pp))
        ax.fill_between(DBH_BIN_CENTRES, f_min, f_max, color=ribbon_color, alpha=0.65, linewidth=0, zorder=2)
        ax.plot(DBH_BIN_CENTRES, f_med, color=median_color, linewidth=2.2, linestyle="--", marker="o", markersize=4.8, zorder=3)
        ax.text(0.97, 0.97, f"Mean temporal range = {mean_range_pp:.0f} pp", transform=ax.transAxes, ha="right", va="top", fontsize=9.5,
                bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="0.7", alpha=0.92, linewidth=0.6))
        ax.set_xscale("log")
        ax.set_xlim(0.85, 55)
        ax.set_xticks(visible_ticks)
        ax.set_xticklabels([f"{t:g}" for t in visible_ticks])
        ax.minorticks_off()
        ax.set_ylim(0, Y_MAX)          # manuscript script: 0.42, which clips the Bruguiera (and Avicennia, Excoecaria) sapling classes
        ax.set_yticks([0.0, 0.2, 0.4, 0.6])
        ax.tick_params(top=False, right=False)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        row, col = panel_positions[i]
        if row == 1:
            ax.set_xlabel(r"D$_{130}$ class upper limit (cm)")
        if col == 0:
            ax.set_ylabel("Fraction of stems")
        else:
            ax.set_yticklabels([])
        ax.text(-0.04, 1.06, f"({'ABCDE'[i]})", transform=ax.transAxes, fontweight="bold", fontsize=16, va="bottom", ha="left")
        ax.text(0.5, 1.04, SPECIES_DISPLAY[sp], transform=ax.transAxes, ha="center", va="bottom", fontstyle="italic", fontsize=15, fontweight="bold")
    handles = [Line2D([0], [0], color=median_color, lw=2.0, linestyle="--", marker="o", markersize=4.5, label="Median fraction\nacross years"),
               Patch(facecolor=ribbon_color, alpha=0.65, edgecolor="none", label="Min–max\ntemporal range")]
    leg_box = gs[1, 2].get_position(fig)
    leg = fig.legend(handles=handles, loc="center", bbox_to_anchor=(0.5 * (leg_box.x0 + leg_box.x1), 0.5 * (leg_box.y0 + leg_box.y1)),
                     bbox_transform=fig.transFigure, title="Cross-year variation in\nD$_{130}$ size-class structure", title_fontsize=11, fontsize=10,
                     frameon=True, framealpha=1.0, edgecolor="0.7", handlelength=1.8, handletextpad=0.6, labelspacing=0.5, borderpad=0.6)
    leg.get_title().set_fontweight("bold")
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    pd.DataFrame(rows).to_csv(out_png.with_name(out_png.stem + "_values.csv"), index=False)
    print(f"Wrote {out_png}")


if __name__ == "__main__":
    import sys
    pop = Path(sys.argv[1]) / "Population.csv" if len(sys.argv) > 1 else POP_CSV
    out = Path(sys.argv[2]) / "size_class_distribution.png" if len(sys.argv) > 2 else FIG_SIZE_CLASS
    out.parent.mkdir(parents=True, exist_ok=True)
    plot(distributions(load_population(pop)), out)
