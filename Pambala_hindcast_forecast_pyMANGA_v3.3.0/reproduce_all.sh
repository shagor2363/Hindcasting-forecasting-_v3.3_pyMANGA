#!/bin/bash
# Rerun the simulations and the analysis. Run from the repository root:
#     PYMANGA=../pyMANGA_v3.3.0 bash reproduce_all.sh
# A 27-year run takes 30-45 min, the 100-year run about 2.5 h (one core each).
set -eu
PYMANGA=${PYMANGA:-../pyMANGA_v3.3.0}
cd "$(dirname "$0")"

# simulations: six assessment runs (27 years, seeds 701-706) and the full run (100 years, seed 701)
for s in 701 702 703 704 705 706; do
  python "$PYMANGA/MANGA.py" -i Initialization/project_27yr_seed$s.xml
done
python "$PYMANGA/MANGA.py" -i Initialization/project_100yr_seed701.xml

# calibration comparison (Table 3) from the six assessment runs
python output/model_evaluation/make_assessment_csv.py
python output/model_evaluation/model_evaluation.py

# structural results of the 100-year run
python output/make_year_slices.py output/Population.csv
python output/size_class_distribution.py output output
python output/height_dbh_pooled_curve_variability.py output output
python output/mcf_temporal_heatmap.py
python output/g_temporal_heatmap.py
