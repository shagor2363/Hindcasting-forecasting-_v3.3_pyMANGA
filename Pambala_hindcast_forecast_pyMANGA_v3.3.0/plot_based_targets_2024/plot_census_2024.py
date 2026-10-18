"""
2024 plot census (datasets/Millie_German_Pambala_Raw_Field_Data_2024.xlsx) in the frame of the
PCQM survey, which is the frame of the reconstruction and of the model domain.

Plot numbering differs between the two workbooks: the plot workbook counts plots on transects
1-3 from the opposite end (PCQM point = 27 - plot) and on transect 4 from the same end.
load_census() verifies this by matching individual trees between the workbooks and flags the
census trees that are also PCQM adults (i.e. inputs of the reconstruction).

Used by targets_2024.py, x_profile_2024.py and the evaluation figures.
"""
import math
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
PACKAGE = HERE.parent
RAW_XLSX = PACKAGE / "datasets" / "Millie_German_Pambala_Raw_Field_Data_2024.xlsx"
PCQM_CSV = PACKAGE / "pcqm_2024_normalized.csv"

PLOT_SIDE = 10.0                                # census plots are 10 m x 10 m
DBH_MIN = 2.5                                   # adult threshold shared by both surveys (GBH >= 8 cm)
SPECIES_KEY = {"rmuc": "Rhizophora", "rapi": "Rhizophora", "lrac": "Lumnitzera",
               "bsex": "Bruguiera", "bgym": "Bruguiera", "aoff": "Avicennia",
               "eaga": "Excoecaria"}


def _num(s):
    return pd.to_numeric(s.astype(str).str.replace(",", "."), errors="coerce")


def _key(s):
    return s.astype(str).str.replace(" ", "").str.replace(".", "", regex=False).str.lower()


def load_census():
    """Return (plots, adults).

    plots   one row per censused plot: Transect, Plot, k (PCQM point index), xc, yc (plot centre, model coordinates)
    adults  live adults of the five model species with DBH >= DBH_MIN: species, dbh (cm, largest stem), h (m),
            px, py (in-plot coordinates), x, y (model coordinates), pcqm_input
    """
    raw = pd.read_excel(RAW_XLSX, sheet_name="Sheet 1", header=0)
    raw = raw.rename(columns={"Coordinates": "px", raw.columns[12]: "py"})
    stems = raw[raw.columns[13:37]].apply(_num)             # per-stem girths, multi-stemmed trees
    multi = raw["GBH"].astype(str).str.lower().str.contains("multi")
    raw["dbh"] = np.where(multi, stems.max(axis=1), _num(raw["GBH"])) / math.pi
    raw["h"], raw["d"] = _num(raw["Height"]), _num(raw["d"])
    raw["px"], raw["py"] = _num(raw["px"]), _num(raw["py"])
    raw["key"] = _key(raw["Species"])
    raw["species"] = raw["key"].map(SPECIES_KEY)
    raw["dead"] = raw["Dead(D)"].astype(str).str.strip().str.upper().eq("D")
    raw["k"] = np.where(raw["Transect"] == 4, raw["Plot"], 27 - raw["Plot"])
    raw["xc"], raw["yc"] = (raw["k"] - 1) * 10.0, (raw["Transect"] - 1) * 25.0
    raw["x"] = raw["xc"] + raw["px"] - PLOT_SIDE / 2
    raw["y"] = raw["yc"] + raw["py"] - PLOT_SIDE / 2

    # Flag the census trees that are also PCQM adults.
    pcqm = pd.read_csv(PCQM_CSV)
    pcqm = pcqm[(pcqm["tree_type"] == "A") & pcqm["distance_m"].notna()].copy()
    pcqm["key"] = _key(pcqm["species_raw"])
    raw["pcqm_input"] = False
    for _, r in pcqm.iterrows():
        c = raw[(raw["Transect"] == r["transect"]) & (raw["k"] == r["plot"]) & (raw["key"] == r["key"]) & ~raw["pcqm_input"]]
        if len(c) == 0:
            continue
        score = (c["d"] - r["distance_m"]).abs().fillna(9.0)
        if pd.notna(r["height_m"]):
            score = score + (c["h"] - r["height_m"]).abs().fillna(0.0)
        if score.min() < 0.05:
            raw.loc[score.idxmin(), "pcqm_input"] = True
    matched = int(raw["pcqm_input"].sum())
    assert matched >= 0.95 * len(pcqm), "plot-number mapping between workbooks does not hold"
    print("PCQM adults found in the plot workbook at the mapped point: %d / %d" % (matched, len(pcqm)))

    plot_rows = raw[raw["Method"] == "Plot"]
    plots = (plot_rows.groupby(["Transect", "Plot"], as_index=False).agg(k=("k", "first"), xc=("xc", "first"), yc=("yc", "first")))
    adults = plot_rows[(plot_rows["Layer"] == "AT") & ~plot_rows["dead"] & plot_rows["species"].notna() & (plot_rows["dbh"] >= DBH_MIN)]
    cols = ["Transect", "Plot", "k", "xc", "yc", "species", "dbh", "h", "px", "py", "x", "y", "pcqm_input"]
    return plots, adults[cols].reset_index(drop=True)
