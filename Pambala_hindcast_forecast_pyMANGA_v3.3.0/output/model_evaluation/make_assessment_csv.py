"""
Collect the year-27 (2024) stands of the assessment runs into one table for model_evaluation.py.

Reads output/seed<seed>/Population.csv (pyMANGA OneFile output of the 27-year runs,
seeds 701-706) and writes output/assessment_year27_seeds_701_706.csv with the columns
    seed, species, dbh_cm, h_m, initial_1997_tree, age_years
for every tree alive in year 27 (all sizes; the census rule is applied by model_evaluation.py).
initial_1997_tree: the tree was in the initial stand (present at year 0); age_years: 27 - year of
recruitment for recruits (empty for initial trees, whose age is unknown).

Usage (from the package root): python output/model_evaluation/make_assessment_csv.py [--seeds 701 702 ...]
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
PACKAGE = HERE.parents[1]
RUNS = PACKAGE / "output"
OUT = PACKAGE / "output" / "assessment_year27_seeds_701_706.csv"
YEAR = 3.16e7
EVAL_YEAR = 27
PREFIX = {"Rhizpop": "Rhizophora", "Avipop": "Avicennia", "Brugpop": "Bruguiera", "Lumnpop": "Lumnitzera", "Excopop": "Excoecaria"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="+", type=int, default=[701, 702, 703, 704, 705, 706])
    a = ap.parse_args()
    rows = []
    for seed in a.seeds:
        pop = RUNS / f"seed{seed}" / "Population.csv"
        p = pd.read_csv(pop, sep="\t", usecols=["plant", "time", "r_stem", "h_stem"])
        p = p[p["r_stem"].notna()]
        p["year"] = np.rint(p["time"] / YEAR).astype(int)
        born = p.groupby("plant")["year"].min()
        y = p[p["year"] == EVAL_YEAR].copy()
        y["born"] = y["plant"].map(born)
        out = pd.DataFrame(dict(seed=seed, species=y["plant"].str.split("_").str[0].map(PREFIX),
                                dbh_cm=y["r_stem"] * 200.0, h_m=y["h_stem"],
                                initial_1997_tree=y["born"] == 0,
                                age_years=np.where(y["born"] == 0, np.nan, EVAL_YEAR - y["born"])))
        rows.append(out)
        print(f"seed {seed}: {len(out)} trees in year {EVAL_YEAR} ({int(out.initial_1997_tree.sum())} from the 1997 stand)")
    pd.concat(rows, ignore_index=True).to_csv(OUT, index=False, float_format="%.6g")
    print("wrote", OUT.relative_to(PACKAGE))


if __name__ == "__main__":
    main()
