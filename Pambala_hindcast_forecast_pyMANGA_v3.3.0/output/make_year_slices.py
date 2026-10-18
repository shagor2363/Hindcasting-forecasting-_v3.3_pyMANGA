"""
Cut the yearly snapshots the figure scripts read out of a pyMANGA Population.csv.

Writes Population_Year_{0,27,50,75,100}.csv (same tab-separated columns) next to the input file.
Year = time / 3.16e7 s. Year 27 corresponds to 2024 (hindcast end), year 100 to 2097.

Usage: python make_year_slices.py [PATH/Population.csv]   (default: output/Population.csv)
"""
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
SECONDS_PER_YEAR = 3.16e7
YEARS = (0, 27, 50, 75, 100)

population = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "Population.csv"
p = pd.read_csv(population, sep="\t")
p["year"] = (p["time"] / SECONDS_PER_YEAR).round().astype(int)
for y in YEARS:
    out = population.with_name("Population_Year_%d.csv" % y)
    p[p["year"] == y].drop(columns="year").to_csv(out, sep="\t", index=False)
    print("wrote", out.name, int((p["year"] == y).sum()), "trees")
