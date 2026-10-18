"""
pcqm_2024_normalized.csv + demo_2024.csv -> pcqm_2024_quarters.csv

One row per PCQM quarter (sample point x quadrant), the input of the 2024 reconstruction
(reconstruct_targets.R): columns as in pcqm_quarters_1997.py (status tree / vacant / unrecorded,
distance_m, model, radius, and the placed tree x, y, r_stem, h_stem, species for the trees of
demo_2024.csv). Positions are redrawn with the generator and seed of
pcqm_position_extraction_2024.py and checked against demo_2024.csv.

Usage: python pcqm_quarters_2024.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

from pcqm_position_extraction_2024 import SEED, compute_plot_center, position_tree

HERE = Path(__file__).resolve().parent
PCQM_CSV = HERE / "pcqm_2024_normalized.csv"
DEMO_CSV = HERE / "demo_2024.csv"
OUT_CSV = HERE / "pcqm_2024_quarters.csv"

pcqm = pd.read_csv(PCQM_CSV)
# Each sample point lists its four adult quarters (a-d) first, then the young trees.
pcqm = pcqm[pcqm.groupby(["transect", "plot"]).cumcount() < 4].copy()
assert set(pcqm["tree_type"]) <= {"A", "0"}

has_distance = pcqm["distance_m"].notna() & (pcqm["distance_m"] > 0)
pcqm["status"] = np.where(pcqm["tree_type"] == "0", "vacant", np.where(has_distance, "tree", "unrecorded"))
pcqm["model"] = (pcqm["status"] == "tree") & pcqm["species"].notna()
centres = [compute_plot_center(t, p) for t, p in zip(pcqm["transect"], pcqm["plot"])]
pcqm["xp"], pcqm["yp"] = zip(*centres)

rng = np.random.default_rng(SEED)
in_demo = pcqm["model"] & (pcqm["dbh_cm"] > 0) & (pcqm["height_m"] > 0)
for col in ("x", "y", "r_stem", "h_stem"):
    pcqm[col] = np.nan
for i, row in pcqm[in_demo].iterrows():
    x, y = position_tree(row["xp"], row["yp"], row["quadrant"], row["distance_m"], rng)
    pcqm.loc[i, ["x", "y"]] = round(x, 2), round(y, 2)
    pcqm.loc[i, "r_stem"] = round(row["dbh_cm"] / 200.0, 4)
    pcqm.loc[i, "h_stem"] = round(row["height_m"], 2)

demo = pd.read_csv(DEMO_CSV)
cols = ["x", "y", "r_stem", "h_stem", "species"]
redrawn = pcqm.loc[in_demo, cols].sort_values(cols).reset_index(drop=True)
assert redrawn.equals(demo[cols].sort_values(cols).reset_index(drop=True)), "redrawn positions do not reproduce demo_2024.csv"

placed = np.hypot(pcqm["x"] - pcqm["xp"], pcqm["y"] - pcqm["yp"])
pcqm["radius"] = np.where(in_demo, placed, np.where(pcqm["status"] == "tree", pcqm["distance_m"], np.nan))

out = pcqm.rename(columns={"plot": "point"})[
    ["transect", "point", "quadrant", "xp", "yp", "status", "distance_m", "model", "radius", "species", "x", "y", "r_stem", "h_stem"]]
out.to_csv(OUT_CSV, index=False)
print("wrote %s: %d quarters (%s), %d demo trees" % (OUT_CSV.name, len(out), out["status"].value_counts().to_dict(), int(in_demo.sum())))
