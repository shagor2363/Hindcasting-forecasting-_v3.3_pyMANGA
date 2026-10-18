"""
1997 PCQM survey (datasets/PAM1_1_7.csv) -> demo_1997.csv (transect trees in pyMANGA format)

Geometrically constrained PCQM-to-position extraction (manuscript, Methods): for each
recorded adult tree one draw from

    P(r, theta) = delta(r - d_i) * Uniform(theta_min, theta_max)

gives its position: the measured point-to-tree distance d_i is kept (with a Gaussian
measurement error sigma_d = 0.1 m) and the bearing is drawn uniformly within the recorded
quadrant. Sample points lie 10 m apart along transects, transects 25 m apart.

Filters (logged per tree in drop_log_1997.csv): transects 1-4; parseable quadrant code
"<transect>.<point><a-d>"; one of the five model species; distance > 0; DBH > 0 (or GBH).

Output columns: plant, x, y, r_stem (m, = DBH / 200), h_stem (m; 3.0 m where no height was
recorded), species. Random seed 42.

Usage: python pcqm_position_extraction_1997.py
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
SRC = HERE / "datasets" / "PAM1_1_7.csv"
OUT_DEMO = HERE / "demo_1997.csv"
OUT_LOG = HERE / "drop_log_1997.csv"

PLOT_SPACING_M = 10.0
TRANSECT_SPACING_M = 25.0
DISTANCE_ERROR_STD_M = 0.1
RANDOM_SEED = 42

QUADRANT_ANGLE_DEG = {"a": (0.0, 90.0), "b": (90.0, 180.0), "c": (180.0, 270.0), "d": (270.0, 360.0)}

SPECIES_MAP = {
    "R. muc": "Rhizophora", "R.muc": "Rhizophora",
    "R. api": "Rhizophora", "R.api": "Rhizophora", "R. app": "Rhizophora",
    "L. rac": "Lumnitzera", "L.rac": "Lumnitzera",
    "L . rac": "Lumnitzera", "L  rac": "Lumnitzera", "L rac": "Lumnitzera",
    "B. sex": "Bruguiera", "B.sex": "Bruguiera",
    "B. gym": "Bruguiera", "B.gym": "Bruguiera",
    "A. off": "Avicennia", "A.off": "Avicennia",
    "E. aga": "Excoecaria", "E.aga": "Excoecaria",
    "X. gra": "_NonModel_Xylocarpus",
    "A. cor": "_NonModel_Aegiceras", "A.cor": "_NonModel_Aegiceras",
    "H. lit": "_NonModel_Heritiera",
    "R. spp": "_GenusOnly_Rhizophora", "B. spp": "_GenusOnly_Bruguiera",
    "L. spp": "_GenusOnly_Lumnitzera", "A. spp": "_GenusOnly_Avicennia",
    "E. spp": "_GenusOnly_Excoecaria",
}

PYMANGA_PREFIX = {"Rhizophora": "Rhizpop", "Avicennia": "Avipop", "Bruguiera": "Brugpop",
                  "Lumnitzera": "Lumnpop", "Excoecaria": "Excopop"}


def _to_num(s: str) -> float:
    if s is None:
        return np.nan
    s = str(s).strip()
    if s in ("", "?", "-", "NA", "N/A", "n/a", "na"):
        return np.nan
    try:
        return float(s.replace(",", "."))
    except ValueError:
        return np.nan


def _normalise_species(raw: str) -> str | None:
    if not raw:
        return None
    key = str(raw).strip()
    if key in SPECIES_MAP:
        return SPECIES_MAP[key]
    key2 = re.sub(r"\s+", " ", key)
    if key2 in SPECIES_MAP:
        return SPECIES_MAP[key2]
    key3 = re.sub(r"\s*\.\s*", ".", key)
    if key3 in SPECIES_MAP:
        return SPECIES_MAP[key3]
    for k, v in SPECIES_MAP.items():
        if k.lower() == key.lower():
            return v
    return f"_Unknown[{raw}]"


def _parse_quadrant(s: str):
    if not s:
        return (None, None, None)
    m = re.match(r"^\s*(\d+)\s*\.\s*(\d+)\s*([a-d])\s*$", str(s), re.IGNORECASE)
    if not m:
        return (None, None, None)
    return (int(m.group(1)), int(m.group(2)), m.group(3).lower())


def load_raw() -> pd.DataFrame:
    """Read PAM1_1_7.csv (4 header rows; fixed column positions)."""
    raw = pd.read_csv(SRC, skiprows=4, header=None, dtype=str, keep_default_na=False)
    df = pd.DataFrame({
        "tree_marker":   raw.iloc[:, 0].str.strip(),
        "quadrant_full": raw.iloc[:, 1].str.strip(),
        "veg_class":     raw.iloc[:, 2].str.strip(),
        "species_raw":   raw.iloc[:, 11].str.strip(),
        "d_raw":         raw.iloc[:, 12].str.strip(),
        "gbh_raw":       raw.iloc[:, 13].str.strip(),
        "dbh_raw":       raw.iloc[:, 14].str.strip(),
        "height_raw":    raw.iloc[:, 15].str.strip(),
    })
    parsed = df["quadrant_full"].apply(_parse_quadrant)
    df["transect"] = parsed.apply(lambda x: x[0])
    df["plot"] = parsed.apply(lambda x: x[1])
    df["quadrant"] = parsed.apply(lambda x: x[2])
    df["d_num"] = df["d_raw"].apply(_to_num)
    df["gbh_num"] = df["gbh_raw"].apply(_to_num)
    df["dbh_num"] = df["dbh_raw"].apply(_to_num)
    df["height_num"] = df["height_raw"].apply(_to_num)
    df["dbh_final"] = df["dbh_num"]
    need_dbh = df["dbh_final"].isna() & df["gbh_num"].notna() & (df["gbh_num"] > 0)
    df.loc[need_dbh, "dbh_final"] = df.loc[need_dbh, "gbh_num"] / np.pi
    df["species_canonical"] = df["species_raw"].apply(_normalise_species)
    return df


def classify(row) -> str:
    if not row["species_raw"] and not row["quadrant_full"]:
        return "blank_row"
    if row["species_raw"].lower() == "species":
        return "header_repeat"
    if not row["species_raw"]:
        return "no_species_field"
    if row["transect"] is None:
        return "unparseable_quadrant"
    if row["transect"] not in (1, 2, 3, 4):
        return f"transect_{row['transect']}_excluded"
    sp = row["species_canonical"]
    if sp is None:
        return "species_normalisation_failed"
    if sp.startswith("_NonModel_"):
        return f"non_model_species[{row['species_raw']}]"
    if sp.startswith("_GenusOnly_"):
        return f"genus_only[{row['species_raw']}]"
    if sp.startswith("_Unknown["):
        return f"unknown_species[{row['species_raw']}]"
    if pd.isna(row["d_num"]) or row["d_num"] <= 0:
        return "no_distance"
    if pd.isna(row["dbh_final"]) or row["dbh_final"] <= 0:
        return "no_dbh"
    return "INCLUDED"


def position_single_draw(distance_m: float, quadrant: str, plot: int, transect: int,
                         rng: np.random.Generator) -> tuple[float, float]:
    """One draw from P(r, theta) = delta(r - d) * Uniform(theta_min, theta_max), sigma_d = 0.1 m."""
    plot_x = (plot - 1) * PLOT_SPACING_M
    plot_y = (transect - 1) * TRANSECT_SPACING_M
    theta_min, theta_max = QUADRANT_ANGLE_DEG[quadrant]
    angle_rad = np.radians(rng.uniform(theta_min, theta_max))
    d_realised = max(0.05, distance_m + rng.normal(0.0, DISTANCE_ERROR_STD_M))
    return plot_x + d_realised * np.cos(angle_rad), plot_y + d_realised * np.sin(angle_rad)


def main():
    df = load_raw()
    df["status"] = df.apply(classify, axis=1)
    log = df[df["transect"].isin([1, 2, 3, 4])].copy()
    log_cols = ["tree_marker", "quadrant_full", "transect", "plot", "quadrant", "species_raw",
                "species_canonical", "d_num", "gbh_num", "dbh_num", "dbh_final", "height_num", "status"]
    log[log_cols].to_csv(OUT_LOG, index=False)
    print("drop log: %s (%d rows)" % (OUT_LOG.name, len(log)))
    print(log["status"].value_counts(dropna=False).to_string())

    inc = log[log["status"] == "INCLUDED"].copy().reset_index(drop=True)
    rng = np.random.default_rng(RANDOM_SEED)
    rows, counters = [], {sp: 0 for sp in PYMANGA_PREFIX}
    for _, r in inc.iterrows():
        species = r["species_canonical"]
        x, y = position_single_draw(r["d_num"], r["quadrant"], int(r["plot"]), int(r["transect"]), rng)
        counters[species] += 1
        h_stem = float(r["height_num"]) if not pd.isna(r["height_num"]) and r["height_num"] > 0 else 3.0
        rows.append({"plant": "%s_%09d" % (PYMANGA_PREFIX[species], counters[species]),
                     "x": round(x, 2), "y": round(y, 2),
                     "r_stem": round(float(r["dbh_final"]) / 200.0, 4),
                     "h_stem": round(h_stem, 1), "species": species})
    out = pd.DataFrame(rows, columns=["plant", "x", "y", "r_stem", "h_stem", "species"])
    out.to_csv(OUT_DEMO, index=False)
    print("wrote %s: %d trees" % (OUT_DEMO.name, len(out)))
    print(out["species"].value_counts().to_string())


if __name__ == "__main__":
    main()
