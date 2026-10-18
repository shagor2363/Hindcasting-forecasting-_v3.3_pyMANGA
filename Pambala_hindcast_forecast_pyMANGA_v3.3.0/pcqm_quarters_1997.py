"""
datasets/PAM1_1_7.csv + demo_1997.csv -> pcqm_1997_quarters.csv

One row per PCQM quarter (sample point x quadrant), the input of the band-wise
reconstruction (reconstruct_segments.R):

  status      tree        adult recorded with a distance
              vacant      no adult in the quarter
              unrecorded  adult present but no distance recorded
              extra       a measured young tree in a young-tree slot of 1997 (kept as an observed
                          tree; it takes no part in the density estimate)
  distance_m  point-to-tree distance of the nearest adult (any species)
  model       the adult is one of the five model species
  radius      radius of the quarter that holds no other adult
  x, y, r_stem, h_stem, species   the tree as placed in demo_1997.csv (empty otherwise)

The positions are redrawn with the generator and seed of pcqm_position_extraction_1997.py
and checked against demo_1997.csv.

Usage: python pcqm_quarters_1997.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

from pcqm_position_extraction_1997 import (PLOT_SPACING_M, RANDOM_SEED, TRANSECT_SPACING_M,
                                           classify, load_raw, position_single_draw)

HERE = Path(__file__).resolve().parent
DEMO_CSV = HERE / "demo_1997.csv"
OUT_CSV = HERE / "pcqm_1997_quarters.csv"

pcqm = load_raw()
pcqm["in_demo"] = pcqm.apply(classify, axis=1) == "INCLUDED"
pcqm = pcqm[pcqm["transect"].isin([1, 2, 3, 4])].reset_index(drop=True)
pcqm[["transect", "plot"]] = pcqm[["transect", "plot"]].astype(int)

# Each sample point lists its four adult quarters (a-d) first, then the young trees.
adult_slot = pcqm.groupby(["transect", "plot"]).cumcount() < 4
assert set(pcqm.loc[adult_slot, "tree_marker"]) <= {"A", "0"}
has_distance = pcqm["d_num"] > 0
pcqm["status"] = np.where(~adult_slot, "extra",
                          np.where(pcqm["tree_marker"] == "0", "vacant",
                                   np.where(has_distance, "tree", "unrecorded")))
canonical = pcqm["species_canonical"].fillna("")
pcqm["model"] = (pcqm["status"] == "tree") & ~canonical.str.startswith(("_NonModel_", "_Unknown")) & (canonical != "")
pcqm["species"] = canonical.where(pcqm["in_demo"])
pcqm["xp"] = (pcqm["plot"] - 1) * PLOT_SPACING_M
pcqm["yp"] = (pcqm["transect"] - 1) * TRANSECT_SPACING_M

rng = np.random.default_rng(RANDOM_SEED)
for col in ("x", "y", "r_stem", "h_stem"):
    pcqm[col] = np.nan
for i, row in pcqm[pcqm["in_demo"]].iterrows():
    x, y = position_single_draw(row["d_num"], row["quadrant"], row["plot"], row["transect"], rng)
    pcqm.loc[i, ["x", "y"]] = round(x, 2), round(y, 2)
    pcqm.loc[i, "r_stem"] = round(row["dbh_final"] / 200.0, 4)
    pcqm.loc[i, "h_stem"] = round(row["height_num"] if row["height_num"] > 0 else 3.0, 1)

demo = pd.read_csv(DEMO_CSV)
cols = ["x", "y", "r_stem", "h_stem", "species"]
redrawn = pcqm.loc[pcqm["in_demo"], cols].sort_values(cols).reset_index(drop=True)
assert redrawn.equals(demo[cols].sort_values(cols).reset_index(drop=True)), "redrawn positions do not reproduce demo_1997.csv"

placed = np.hypot(pcqm["x"] - pcqm["xp"], pcqm["y"] - pcqm["yp"])
pcqm["radius"] = np.where(pcqm["status"] != "tree", np.nan, np.where(pcqm["in_demo"], placed, pcqm["d_num"]))

out = pcqm[adult_slot | pcqm["in_demo"]].rename(columns={"plot": "point", "d_num": "distance_m"})[
    ["transect", "point", "quadrant", "xp", "yp", "status", "distance_m", "model", "radius", "species", "x", "y", "r_stem", "h_stem"]]
out.to_csv(OUT_CSV, index=False)
print("wrote %s: %d rows (%s), %d demo trees" % (OUT_CSV.name, len(out), out["status"].value_counts().to_dict(), int(pcqm["in_demo"].sum())))
