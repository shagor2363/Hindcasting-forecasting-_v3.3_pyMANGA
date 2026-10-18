"""
MTPPR output (reconstructed_raw_<year>.csv) -> pyMANGA initial-population CSV.

Clamps positions to the model domain, drops trees with non-positive stem radius or height,
sorts by species and x, and assigns plant IDs with the population-group prefixes of
project.xml (Rhizpop, Avipop, Brugpop, Lumnpop, Excopop).

Output columns: plant, x, y, r_stem (m), h_stem (m), species. Only the two field-measured
sizes travel with the trees; pyMANGA derives the remaining geometry of each initial tree from
its species file when it loads the stand.

Usage: python postprocess_csv.py --input Initialization/reconstructed_raw_1997.csv --output Initialization/reconstructed_adults_1997.csv
"""
import argparse
from pathlib import Path

import pandas as pd

DOMAIN_X = (-17.9, 282.1)
DOMAIN_Y = (-18.1, 95.0)
ZONE_Y = [DOMAIN_Y[0], 12.50, 37.49, 62.54, DOMAIN_Y[1]]
SPECIES_PREFIX = {"Rhizophora": "Rhizpop", "Avicennia": "Avipop", "Bruguiera": "Brugpop",
                  "Lumnitzera": "Lumnpop", "Excoecaria": "Excopop"}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--input", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    args = ap.parse_args()

    df = pd.read_csv(args.input)
    print("read %d trees from %s" % (len(df), args.input.name))
    out_of_domain = int(((df["x"] < DOMAIN_X[0]) | (df["x"] > DOMAIN_X[1]) | (df["y"] < DOMAIN_Y[0]) | (df["y"] > DOMAIN_Y[1])).sum())
    if out_of_domain:
        print("  %d trees outside the domain clamped to its edge" % out_of_domain)
        df["x"] = df["x"].clip(*DOMAIN_X)
        df["y"] = df["y"].clip(*DOMAIN_Y)
    for col in ("r_stem", "h_stem"):
        bad = df[col] <= 0
        if bad.any():
            print("  %d trees with %s <= 0 dropped" % (int(bad.sum()), col))
            df = df[~bad]
    unknown = ~df["species"].isin(SPECIES_PREFIX)
    if unknown.any():
        print("  %d trees of unknown species dropped: %s" % (int(unknown.sum()), sorted(df.loc[unknown, "species"].unique())))
        df = df[~unknown]

    df = df.sort_values(["species", "x"]).reset_index(drop=True)
    counters, ids = {sp: 0 for sp in SPECIES_PREFIX}, []
    for sp in df["species"]:
        counters[sp] += 1
        ids.append("%s_%09d" % (SPECIES_PREFIX[sp], counters[sp]))
    df["plant"] = ids
    out = df[["plant", "x", "y", "r_stem", "h_stem", "species"]]
    out.to_csv(args.output, index=False)
    print("wrote %d trees to %s" % (len(out), args.output))

    area_ha = (DOMAIN_X[1] - DOMAIN_X[0]) * (DOMAIN_Y[1] - DOMAIN_Y[0]) / 1e4
    if "is_original" in df.columns:
        print("  measured (PCQM) trees kept at their positions: %d" % int(df["is_original"].astype(str).str.upper().str.startswith("T").sum()))
    print("  per species: %s" % out["species"].value_counts().reindex(SPECIES_PREFIX).to_dict())
    print("  domain %.3f ha, %d trees per ha" % (area_ha, round(len(out) / area_ha)))
    if "band" in df.columns:
        for b in range(4):
            n = int((df["band"] == b).sum())
            print("  band %d (y %.2f-%.2f): %d trees, %d per ha" % (b + 1, ZONE_Y[b], ZONE_Y[b + 1], n,
                  round(n / ((DOMAIN_X[1] - DOMAIN_X[0]) * (ZONE_Y[b + 1] - ZONE_Y[b]) / 1e4))))


if __name__ == "__main__":
    main()
