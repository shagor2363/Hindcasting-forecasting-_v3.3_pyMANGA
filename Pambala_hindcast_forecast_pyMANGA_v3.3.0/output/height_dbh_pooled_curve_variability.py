"""
Cross-year variability of the height-D130 allometry.

Produces one figure, height_dbh_cross_year_variability_parametric.png: per
species, the ratio of each year's H-D allometric fit to the pooled
four-year fit, summarised across years 27, 50, 75, 100 as a median curve
with IQR and min-max ribbons around the reference line ratio = 1 (the
pooled allometry). The WIDTH of the temporal envelope around ratio = 1 is
the reported quantity: it measures how much the fitted allometry varies
from year to year. It is a descriptive magnitude, not a test of stability
or persistence.

Companion to the g(r), K_mm(r) and size-class panels; see
plot_deviation_parametric_figure for the full method and the
descriptive-reference (not statistical-null) caveat.

Both the per-year and pooled fits use the same Mitscherlich / Chapman-
Richards AICc-selected allometry with boundary screening
(fit_per_year_allometry).

KNOWN BIAS OF THE POOLED REFERENCE: each year's trees also contribute to
the pooled fit in the denominator, so numerator and denominator are not
independent and the ratio is pulled toward 1. The envelope therefore
UNDERSTATES true cross-year variability (by roughly 15-50% depending on
species, measured against a leave-one-year-out reference).

Usage: python height_dbh_pooled_curve_variability.py [DIR with Population_Year_{27,50,75,100}.csv] [OUT_DIR]
(defaults: this directory). Writes height_dbh_cross_year_variability_parametric.png.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy.optimize import curve_fit


YEARS = [27, 50, 75, 100]
DBH_MIN_CM = 2.0

SPECIES_FROM_PREFIX = {
    "Avipop": "Avicennia",
    "Brugpop": "Bruguiera",
    "Excopop": "Excoecaria",
    "Lumnpop": "Lumnitzera",
    "Rhizpop": "Rhizophora",
}

SPECIES_PLOT_LIST = ["Rhizophora", "Avicennia", "Bruguiera",
                     "Lumnitzera", "Excoecaria"]

SPECIES_DISPLAY = {
    "Rhizophora": "Rhizophora spp.",
    "Avicennia":  "A. officinalis",
    "Bruguiera":  "Bruguiera spp.",
    "Lumnitzera": "L. racemosa",
    "Excoecaria": "E. agallocha",
}
PANEL_LETTERS = ["A", "B", "C", "D", "E"]
PANEL_POSITIONS = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1)]


# ----------------------------------------------------------------------
# Data loading
# ----------------------------------------------------------------------

def load_year(model_output: Path, year: int) -> pd.DataFrame:
    f = model_output / f"Population_Year_{year}.csv"
    df = pd.read_csv(f, sep="\t")
    df["dbh_cm"] = 2.0 * df["r_stem"] * 100.0
    df["height_m"] = df["h_stem"]
    df["prefix"] = df["plant"].astype(str).str.extract(r"^([A-Za-z]+)",
                                                        expand=False)
    df["species"] = df["prefix"].map(SPECIES_FROM_PREFIX)
    df["year"] = year
    return df[["species", "year", "dbh_cm", "height_m"]].dropna()


# ----------------------------------------------------------------------
# Figure styling
# ----------------------------------------------------------------------

def _apply_rcparams():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans"],
        "font.size": 14,
        "axes.titlesize": 15,
        "axes.labelsize": 15,
        "axes.linewidth": 1.0,
        "xtick.labelsize": 13,
        "ytick.labelsize": 13,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.size": 5.0,
        "ytick.major.size": 5.0,
        "xtick.major.width": 1.0,
        "ytick.major.width": 1.0,
        "legend.frameon": False,
        "legend.fontsize": 14,
    })


# ----------------------------------------------------------------------
# Allometry model: per-year Mitscherlich/CR fit with boundary screening
# ----------------------------------------------------------------------

def chapman_richards(dbh, a, b, c):
    return 1.3 + a * (1.0 - np.exp(-b * dbh)) ** c


def mitscherlich(dbh, a, b):
    return 1.3 + a * (1.0 - np.exp(-b * dbh))


def _aicc(rss, n, k):
    if n - k - 1 <= 0 or rss <= 0:
        return np.inf
    aic = n * np.log(rss / n) + 2 * k
    return aic + 2 * k * (k + 1) / (n - k - 1)


def _at_boundary(popt, lo, hi, tol=0.02):
    popt = np.asarray(popt, float)
    lo = np.asarray(lo, float)
    hi = np.asarray(hi, float)
    span = np.where(hi > lo, hi - lo, 1.0)
    return bool(np.any(((popt - lo) < tol * span) |
                       ((hi - popt) < tol * span)))


def fit_per_year_allometry(dbh: np.ndarray, h: np.ndarray):
    """Fit Mitscherlich + (if AICc favours) CR with boundary screening on
    BOTH forms. Returns a dict (or None) describing the chosen fit, with
    a boundary_ok flag indicating whether the chosen fit survived
    screening or was retained as a marked-fallback because no screened
    fit existed.
    """
    dbh = np.asarray(dbh, float)
    h = np.asarray(h, float)
    n = len(dbh)
    if n < 8:
        return None
    h_max = float(np.nanmax(h))
    a0 = max(h_max - 1.3, 1.0)
    a_max = 60.0
    cr_lo, cr_hi = [0.5, 1e-3, 0.1], [a_max, 5.0, 5.0]
    m_lo,  m_hi  = [0.5, 1e-3],      [a_max, 5.0]

    cr_pack = None
    try:
        popt_cr, pcov_cr = curve_fit(
            chapman_richards, dbh, h,
            p0=[a0, 0.15, 1.0],
            bounds=(cr_lo, cr_hi),
            maxfev=20000,
        )
        ses_cr = np.sqrt(np.diag(pcov_cr))
        on_bnd = _at_boundary(popt_cr, cr_lo, cr_hi)
        rss = float(np.sum((h - chapman_richards(dbh, *popt_cr)) ** 2))
        tss = float(np.sum((h - np.mean(h)) ** 2))
        r2 = 1.0 - rss / tss if tss > 0 else float("nan")
        cr_pack = dict(form="CR", popt=popt_cr, ses=ses_cr,
                       boundary_ok=(not on_bnd),
                       aicc=_aicc(rss, n, 3), r2=r2)
    except Exception:
        cr_pack = None

    m_pack = None
    try:
        popt_m, pcov_m = curve_fit(
            mitscherlich, dbh, h,
            p0=[a0, 0.15],
            bounds=(m_lo, m_hi),
            maxfev=20000,
        )
        ses_m = np.sqrt(np.diag(pcov_m))
        on_bnd = _at_boundary(popt_m, m_lo, m_hi)
        rss = float(np.sum((h - mitscherlich(dbh, *popt_m)) ** 2))
        tss = float(np.sum((h - np.mean(h)) ** 2))
        r2 = 1.0 - rss / tss if tss > 0 else float("nan")
        m_pack = dict(form="M", popt=popt_m, ses=ses_m,
                      boundary_ok=(not on_bnd),
                      aicc=_aicc(rss, n, 2), r2=r2)
    except Exception:
        m_pack = None

    candidates = [p for p in (cr_pack, m_pack) if p is not None]
    if not candidates:
        return None

    screened = [p for p in candidates if p["boundary_ok"]]
    if screened:
        chosen = min(screened, key=lambda p: p["aicc"])
    else:
        chosen = min(candidates, key=lambda p: p["aicc"])

    def _predict(d_grid):
        d = np.asarray(d_grid, float)
        if chosen["form"] == "CR":
            return chapman_richards(d, *chosen["popt"])
        return mitscherlich(d, *chosen["popt"])

    chosen["predict"] = _predict
    return chosen


# ----------------------------------------------------------------------
# Figure: cross-year deviation from pooled allometry
# ----------------------------------------------------------------------

def plot_deviation_parametric_figure(df_all, out_png):
    """Cross-year variability panel: ratio of per-year H-D fit to the
    pooled four-year H-D fit, framed as departure from a reference
    line at ratio = 1.

    Companion to the spatial (g(r), K_mm(r)) and size-class panels.
    Note the distinction from the
    spatial panels: g(r) and K_mm(r) compare an observed median
    against a formal STATISTICAL NULL (CSR Monte-Carlo envelope,
    mark-random permutation envelope) -- the dashed line in those
    panels IS a hypothesis-test reference. The allometric analogue
    here divides per-year H-D fits by a pooled H-D fit built on all
    four forecast years' trees combined; the ratio = 1 line is a
    DESCRIPTIVE REFERENCE (the across-year pooled allometry), not a
    generative null. The figure shows the magnitude of year-specific
    departures from that reference, not whether those departures
    significantly reject a null hypothesis. The y-axis is
    dimensionless and bounded near unity, so the dominant visual
    signal is HOW WIDE the temporal envelope is around the reference
    -- not the trivial "H rises with D" that an absolute-H plot would
    convey.

    Each year's fit is evaluated only within that year's own observed
    [d_min, d_max]; the ratio, the median, IQR, min-max envelope and
    the headline metric are all computed on the four-year overlap, so
    the panel cannot be dominated by extrapolation past any single
    year's data support.

    """
    _apply_rcparams()
    fig = plt.figure(figsize=(11.2, 6.6))
    gs = GridSpec(2, 3, figure=fig, hspace=0.30, wspace=0.18,
                  left=0.08, right=0.985, top=0.94, bottom=0.11)

    sp_data = {}

    for sp in SPECIES_PLOT_LIST:
        sp_df = df_all[df_all["species"] == sp]
        if sp_df.empty:
            sp_data[sp] = {"empty": True}
            continue

        # Pooled four-year fit: same Mitscherlich/CR AICc-selected form
        # as the per-year fits, just on all years' trees combined. This
        # is a descriptive across-year reference, not a statistical null
        # (and it contains year t's own trees -- see module docstring).
        pooled = fit_per_year_allometry(sp_df["dbh_cm"].to_numpy(),
                                          sp_df["height_m"].to_numpy())
        if pooled is None:
            sp_data[sp] = {"empty": True}
            continue

        # Trim the very-small-D region (D < 3 cm). Parametric H-D fits
        # all start from the same breast-height intercept (1.3 m) so the
        # H_t/H_pool ratio is identifiable only once the fits have
        # diverged enough from 1.3 m: at D=2 the per-year fits all
        # predict H≈1.5-2 m and small numerical differences inflate
        # the ratio to 2-3, producing a panel-edge spike that distracts
        # from the stability signal at biologically meaningful DBH.
        d_lo = max(float(sp_df["dbh_cm"].min()), DBH_MIN_CM, 3.0)
        d_hi = float(sp_df["dbh_cm"].max())
        if not (d_hi > d_lo):
            sp_data[sp] = {"empty": True}
            continue
        d_grid = np.linspace(d_lo, d_hi, 200)

        pool_curve = pooled["predict"](d_grid)
        # Guard against pool_curve <= small value (would inflate ratios).
        pool_curve = np.where(pool_curve > 0.5, pool_curve, np.nan)

        per_year_ratios = {}
        per_year_valid = {}
        for yr in YEARS:
            yr_df = sp_df[sp_df["year"] == yr]
            if len(yr_df) < 8:
                continue
            fit = fit_per_year_allometry(yr_df["dbh_cm"].to_numpy(),
                                          yr_df["height_m"].to_numpy())
            if fit is None:
                continue
            yr_dmin = float(yr_df["dbh_cm"].min())
            yr_dmax = float(yr_df["dbh_cm"].max())
            yr_curve = fit["predict"](d_grid)
            with np.errstate(divide="ignore", invalid="ignore"):
                ratio = yr_curve / pool_curve
            valid = ((d_grid >= yr_dmin) & (d_grid <= yr_dmax)
                     & np.isfinite(ratio))
            per_year_ratios[int(yr)] = ratio
            per_year_valid[int(yr)] = valid

        if len(per_year_ratios) < 2:
            sp_data[sp] = {"empty": True}
            continue

        sorted_years = sorted(per_year_ratios)
        ratio_stack = np.vstack([per_year_ratios[yr] for yr in sorted_years])
        valid_stack = np.vstack([per_year_valid[yr] for yr in sorted_years])
        all_valid = valid_stack.all(axis=0)
        if not all_valid.any():
            sp_data[sp] = {"empty": True}
            continue

        d_overlap = d_grid[all_valid]
        r_over = ratio_stack[:, all_valid]

        median_r = np.median(r_over, axis=0)
        q25_r = np.quantile(r_over, 0.25, axis=0)
        q75_r = np.quantile(r_over, 0.75, axis=0)
        min_r = r_over.min(axis=0)
        max_r = r_over.max(axis=0)

        per_d_range = max_r - min_r
        mean_range = float(np.mean(per_d_range))
        median_of_median = float(np.median(median_r))
        mean_range_pct = (100.0 * mean_range / median_of_median
                          if median_of_median > 0 else float("nan"))

        sp_data[sp] = dict(
            empty=False, d_overlap=d_overlap,
            median_r=median_r, q25_r=q25_r, q75_r=q75_r,
            min_r=min_r, max_r=max_r,
            mean_range=mean_range, mean_range_pct=mean_range_pct,
        )

    # Greyscale palette + single accent colour (Okabe-Ito vermillion),
    # distinct from green / blue / brown used by the three sister
    # figures so the four panels are immediately distinguishable while
    # still reading as a family.
    OUTER_GREY = "#dcdcdc"
    INNER_GREY = "#9e9e9e"
    SIGNAL_COLOR = "#D55E00"
    NULL_COLOR = "0.30"

    # Tight shared y-axis at [0.75, 1.35]: the ratio sits near 1, so this
    # window removes the empty headroom while keeping the null (=1) near the
    # centre and panels comparable. Biologically implausible spikes from the
    # small-DBH boundary regime are cropped rather than driving the scale.
    y_lo_shared, y_hi_shared = 0.75, 1.35

    for i_panel, sp in enumerate(SPECIES_PLOT_LIST):
        row, col = PANEL_POSITIONS[i_panel]
        ax = fig.add_subplot(gs[row, col])
        d = sp_data.get(sp, {"empty": True})

        if d.get("empty", True):
            ax.text(0.5, 0.5, "no data", ha="center", va="center",
                    transform=ax.transAxes, color="#777", fontsize=12)
            ax.set_xticks([]); ax.set_yticks([])
            ax.text(-0.04, 1.04, "(" + PANEL_LETTERS[i_panel] + ")",
                    transform=ax.transAxes, fontweight="bold",
                    fontsize=16, va="bottom", ha="left")
            ax.text(0.5, 1.02,
                    SPECIES_DISPLAY.get(sp, sp),
                    transform=ax.transAxes, ha="center", va="bottom",
                    fontstyle="italic", fontsize=15, fontweight="bold")
            continue

        d_x = d["d_overlap"]
        ax.fill_between(d_x, d["min_r"], d["max_r"],
                        color=OUTER_GREY, edgecolor="none", zorder=1)
        ax.fill_between(d_x, d["q25_r"], d["q75_r"],
                        color=INNER_GREY, edgecolor="none", zorder=2)
        ax.axhline(1.0, color=NULL_COLOR, linestyle="--",
                   linewidth=0.9, zorder=3)
        ax.plot(d_x, d["median_r"], color=SIGNAL_COLOR,
                linewidth=2.1, zorder=4)

        ann = ("%.0f%% of median" % d["mean_range_pct"])
        ax.text(0.97, 0.95, ann,
                transform=ax.transAxes, ha="right", va="top",
                fontsize=9.5,
                bbox=dict(boxstyle="round,pad=0.25",
                          facecolor="white", edgecolor="0.7",
                          alpha=0.95, linewidth=0.6))

        x_min = float(d_x.min())
        x_max = float(d_x.max())
        x_pad = 0.02 * max(x_max - x_min, 1.0)
        ax.set_xlim(x_min - x_pad, x_max + x_pad)
        ax.set_ylim(y_lo_shared, y_hi_shared)

        ax.tick_params(top=False, right=False)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        if row == 1:
            ax.set_xlabel(r"D$_{130}$ (cm)")
        if col == 0:
            ax.set_ylabel(r"$H_t / H_{\mathrm{pool}}$")

        ax.text(-0.04, 1.04, "(" + PANEL_LETTERS[i_panel] + ")",
                transform=ax.transAxes, fontweight="bold",
                fontsize=16, va="bottom", ha="left")
        ax.text(0.5, 1.02,
                SPECIES_DISPLAY.get(sp, sp),
                transform=ax.transAxes, ha="center", va="bottom",
                fontstyle="italic", fontsize=15, fontweight="bold")

    # Single compact shared legend in the freed (empty) sixth grid cell —
    # a figure-level legend rather than a full legend panel.
    legend_handles = [
        Line2D([0], [0], color=SIGNAL_COLOR, lw=2.1,
               label="Median $H_t/H_{\\mathrm{pool}}$\nacross years 27, 50, 75, 100"),
        Patch(facecolor=INNER_GREY, edgecolor="none",
              label="Across-year IQR (Q25-Q75)"),
        Patch(facecolor=OUTER_GREY, edgecolor="none",
              label="Across-year min-max range"),
        Line2D([0], [0], color=NULL_COLOR, lw=0.9, linestyle="--",
               label="Pooled allometry (ratio = 1)"),
    ]
    leg_box = gs[1, 2].get_position(fig)
    leg = fig.legend(
        handles=legend_handles,
        loc="center",
        bbox_to_anchor=(0.5 * (leg_box.x0 + leg_box.x1),
                        0.5 * (leg_box.y0 + leg_box.y1)),
        bbox_transform=fig.transFigure,
        title="Cross-year variability of the\nheight$-$D$_{130}$ allometry",
        title_fontsize=11,
        fontsize=10,
        frameon=True, framealpha=1.0, edgecolor="0.7",
        handlelength=1.8, handletextpad=0.6,
        labelspacing=0.5, borderpad=0.6,
    )
    leg.get_title().set_fontweight("bold")

    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("Wrote " + str(out_png))


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def main(argv):
    here = Path(__file__).resolve().parent
    model_output = Path(argv[1]) if len(argv) > 1 else here
    out_dir = Path(argv[2]) if len(argv) > 2 else here
    out_dir.mkdir(parents=True, exist_ok=True)

    frames = []
    for y in YEARS:
        f = model_output / f"Population_Year_{y}.csv"
        if not f.exists():
            print(f"[skip] {f} not found")
            continue
        frames.append(load_year(model_output, y))
    if not frames:
        sys.exit("No year snapshots found.")
    df_all = pd.concat(frames, ignore_index=True)

    df_all = df_all[(df_all["dbh_cm"] >= DBH_MIN_CM)
                    & np.isfinite(df_all["height_m"])
                    & (df_all["height_m"] > 0)].copy()

    dev_png = out_dir / "height_dbh_cross_year_variability_parametric.png"
    plot_deviation_parametric_figure(df_all, dev_png)


if __name__ == "__main__":
    main(sys.argv)
