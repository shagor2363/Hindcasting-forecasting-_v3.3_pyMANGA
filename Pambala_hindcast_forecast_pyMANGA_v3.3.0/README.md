# Pambala mangrove stand modelling: code and data

This repository contains the data, model inputs, scripts and outputs used to simulate mangrove stand development at Pambala, Sri Lanka, from 1997 to 2097. The 1997–2024 period was used for model calibration; subsequent years are projections. The study used the public **pyMANGA v3.3.0** release without modifying its source code.

## Software

- **pyMANGA v3.3.0** (commit `20dcbfcd993af54000d6bdb421c459ceac7a4c05`): https://github.com/pymanga/pyMANGA/releases/tag/v3.3.0
- **MTPPR v2.0.0** (commit `6982c39`), used for stand reconstruction: https://github.com/ChrisWudel/Multi-trait-point-pattern-reconstruction
- **Python 3.13.5** and the packages in `requirements.txt`; **R 4.x** and the packages required by MTPPR for reconstructing the stands.

The original pyMANGA and MTPPR source code is not bundled with this repository. The reconstructed stands are included; the simulation outputs (about 480 MB) are not and are regenerated with the commands below.

## Stand reconstruction

The 1997 and 2024 stands were reconstructed from point-centred quarter method (PCQM) surveys using MTPPR. The reconstruction was conducted separately within four transect bands.

For **1997**, tree abundances were estimated from the PCQM distances. The reconstruction contains 7,318 stems and was used to initialize the model. For **2024**, species-specific abundances were constrained by a census of 64 plots (10 × 10 m), and the spatial reconstruction also used the plot-based density profiles. The reconstructed 2024 stand contains 7,088 stems and served as the calibration reference. Its positions are an estimated spatial representation, not a complete field census.

The original PCQM distances and tree measurements were retained; directions within the recorded quadrants were assigned with a fixed random seed. The processing scripts are in the repository root, the plot-based targets in `plot_based_targets_2024/`, and the reconstruction scripts are `reconstruct_segments.R`, `reconstruct_targets.R` and `postprocess_csv.py`. The resulting stands are in `Initialization/`.

To repeat reconstruction, download MTPPR v2.0.0 and make its R source file available at `MTPPR/func/MTPPR-v.2.0.0.R`, or set `MTPPR_PATH` to that file. The included processed PCQM and plot-target files can be used directly with the reconstruction scripts.

## Simulation and calibration

The model covers **3.393 ha** (300 × 113.1 m) and represents five taxa: *Rhizophora* spp., *Avicennia officinalis*, *Bruguiera* spp., *Lumnitzera racemosa* and *Excoecaria agallocha*. It uses BETTINA growth, above- and below-ground resource competition, a fixed salinity gradient (20–42 ppt), growth-dependent and stochastic mortality, and species-specific annual recruitment. Recruits are placed randomly within the domain. The simulation uses annual time steps.

Species parameters were initially informed by the 1997 height–diameter and crown–diameter relationships. Selected growth multipliers, recruitment inputs, mortality coefficients and recruit dimensions were then adjusted manually to improve agreement with the reconstructed 2024 stand in species-specific density and diameter distributions. The final parameter set was fixed before the six assessment runs (random seeds **701–706**). A Morris elementary-effects screening of the Bettina parameters, salinity bounds and recruitment (`morris/`: design, year-100 responses and the script) guided which parameters were adjusted.

The final species parameters and initial populations are in `Initialization/species/` and `Initialization/initial_stand_1997/`. Project files are in `Initialization/project_*.xml`; the complete calibrated settings are in `Initialization/calibrated_setting_v33rc1.json`. **Year 0 = 1997, year 27 = 2024, and year 100 = 2097.** The 100-year projection and manuscript figures use seed 701.

## Running the simulations

Run these commands from the repository root:

```bash
git clone https://github.com/pymanga/pyMANGA.git ../pyMANGA_v3.3.0
git -C ../pyMANGA_v3.3.0 checkout v3.3.0
pip install -r requirements.txt

# Calibration-period simulation (27 years, seed 701)
python ../pyMANGA_v3.3.0/MANGA.py -i Initialization/project_27yr_seed701.xml

# Full simulation (100 years, seed 701)
python ../pyMANGA_v3.3.0/MANGA.py -i Initialization/project_100yr_seed701.xml
```

To rerun all six assessment simulations, the full 100-year simulation, and the analysis scripts, use `bash reproduce_all.sh`. The simulation outputs are written to `output/` (`output/seed701/` to `output/seed706/` for the assessment runs, `output/Population.csv` for the full run). The analysis scripts in `output/` generate the calibration comparisons and structural results used in the manuscript.

## Repository contents

- `datasets/` — 1997 and 2024 field data.
- `plot_based_targets_2024/` — 2024 census-derived abundance and spatial targets.
- `Initialization/` — reconstructed stands, initial populations, species parameters and pyMANGA project files.
- `morris/` — Morris sensitivity screening (design.csv, metrics.csv, run_morris.py).
- `output/` — simulation results, model assessment and structural-analysis scripts.
- `reproduce_all.sh` and `requirements.txt` — reproduction workflow and Python dependencies.

For the full modelling methodology, including the statistical comparisons, see the accompanying manuscript.
