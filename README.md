# Daily Newhall Simulation Model (D-NSM)

This repository contains the code used to develop, run, and evaluate the Daily Newhall Simulation Model (D-NSM), a daily extension of the original Newhall Simulation Model (NSM) for estimating surface and rootzone soil moisture from precipitation, temperature, and soil properties.

The GitHub repository includes the codebase and a minimal example dataset for repository verification. The full processed dataset used for manuscript-scale analysis will be archived separately in Zenodo and linked here once available.

## Repository contents

```text
daily-newhall-model/
├─ README.md
├─ requirements.txt
├─ environment.yml
├─ CITATION.cff
├─ .gitignore
├─ data_reference/
│  ├─ constants_classic.csv
│  ├─ knorth_classic.csv
│  ├─ ksouth_classic.csv
│  ├─ crn-stations.txt
│  └─ stations_XY.csv
├─ src/
│  ├─ dataset_preprocessing/
│  │  ├─ 01_download_uscrn.py
│  │  ├─ 02_qc_uscrn.py
│  │  ├─ 02b_select_station_year.py
│  │  ├─ 03_extract_gssurgo.py
│  │  ├─ 04_build_soil_params_by_station.py
│  │  ├─ 04b_build_theta_sat.py
│  │  ├─ 05_extract_climate.py
│  │  └─ 06_station_coordinates.py
│  ├─ nsm/
│  │  ├─ build_monthly_inputs.py
│  │  └─ run_nsm.py
│  ├─ dnsm/
│  │  ├─ climate.py
│  │  ├─ moisture_condition.py
│  │  ├─ pet.py
│  │  ├─ soil_profile.py
│  │  └─ run_dnsm.py
│  ├─ pet_screening/
│  │  ├─ modis_filtering.py
│  │  └─ compare_pet_vs_modis.py
│  ├─ evaluation/
│  │  ├─ compare_faw_daily_vs_monthly.py
│  │  ├─ build_pet_8day_from_dnsm.py
│  │  └─ compute_vwc_performance_metrics.py
│  ├─ figures/
│  │  ├─ fig4_boxplot_faw.py
│  │  ├─ fig5_timeseries_faw.py
│  │  ├─ fig6_boxplot_vwc.py
│  │  ├─ fig7_timeseries_vwc.py
│  │  └─ fig8_timeseries_with_without_s2.py
│  └─ utils/
│     ├─ paths.py
│     └─ utils_obs_bulk.py
├─ docs/
├─ examples/
├─ outputs/
├─ uscrn_daily01/
└─ gSSURGO_CONUS.gdb
```

## Minimal example data included in GitHub

This GitHub repository contains only a minimal example dataset for repository verification. In the current version, the minimal example data are limited to station `12987`.

These example files are intended to help users confirm that the repository structure, paths, and main scripts run correctly. They are not the full dataset used for manuscript-scale analysis.

## Full data availability

The full processed dataset used for the manuscript will be archived separately in Zenodo. A DOI and access link will be added here after archiving.

## Required external datasets

Full reproduction of the workflow requires several external datasets that users must obtain separately.

USCRN daily observations  
Place the station files in:

```text
uscrn_daily01/
```

gSSURGO database for the conterminous United States  
Place the geodatabase at:

```text
gSSURGO_CONUS.gdb
```

MODIS PET data for PET screening  
Place the input file at:

```text
outputs/1016/pet_screening/MODIS_dailyPET.csv
```

## Note on `YEAR = 1016`

Several scripts use `YEAR = 1016` as a shorthand label for the 2010–2016 evaluation pool used during station-year screening. It does not represent a calendar year.

## Software requirements

This repository was developed and tested in a local Python environment.  
A reproducible environment file will be added or finalized in a future update.

## Main workflow

## 1. Dataset preprocessing

Scripts in `src/dataset_preprocessing/` prepare station-level soil, climate, and metadata inputs used by the NSM and D-NSM workflows.

Main scripts include:

`02_qc_uscrn.py`  
Quality control of USCRN station records.

`02b_select_station_year.py`  
Selection of screened station-year combinations.

`03_extract_gssurgo.py`  
Extraction of soil information from gSSURGO.

`04_build_soil_params_by_station.py`  
Construction of station-level soil hydraulic parameters.

`04b_build_theta_sat.py`  
Estimation of saturation-related soil parameters.

`05_extract_climate.py`  
Preparation of climate input files.

`06_station_coordinates.py`  
Preparation of station coordinate metadata.

Example commands:

```bash
python src/dataset_preprocessing/02_qc_uscrn.py
python src/dataset_preprocessing/02b_select_station_year.py
python src/dataset_preprocessing/03_extract_gssurgo.py
python src/dataset_preprocessing/04_build_soil_params_by_station.py
python src/dataset_preprocessing/04b_build_theta_sat.py
python src/dataset_preprocessing/05_extract_climate.py
python src/dataset_preprocessing/06_station_coordinates.py
```

## 2. Monthly NSM workflow

Scripts:
- `src/nsm/build_monthly_inputs.py`
- `src/nsm/run_nsm.py`

Purpose:  
Prepare monthly inputs and run the original monthly NSM for comparison against the daily model.

Main inputs:  
- processed station metadata in `outputs/`
- monthly climate inputs
- NSM reference constants in `data_reference/`

Main outputs:  
- monthly NSM results written under `outputs/`

Example commands:

```bash
python src/nsm/build_monthly_inputs.py
python src/nsm/run_nsm.py
```

## 3. Daily D-NSM workflow

Script:
- `src/dnsm/run_dnsm.py`

Purpose:  
Run the Daily Newhall Simulation Model and generate daily surface and rootzone soil moisture outputs.

Main inputs:  
- processed climate inputs in `outputs/1016/daily_climate/`
- one-year warm-up climate inputs in `outputs/1016/daily_climate_spinup/`
- station coordinates in `outputs/1016/coordinates_final_1016.csv`
- layer-based soil parameters in `outputs/1016/layers_by_station_1016.csv`
- optional saturation parameter file in `outputs/1016/layers_by_station_1016_satPTF_only.csv`

Main outputs:  
- daily D-NSM results written to:

```text
outputs/1016/daily_newhall/pet_PP/<station>/daily_results_<station>_PP_spinup_layers_ptf_df_dual.csv
```

Example command:

```bash
python src/dnsm/run_dnsm.py
```

## 4. Evaluation against USCRN observations

### 4.1 VWC-based evaluation

Script:
- `src/evaluation/compute_vwc_performance_metrics.py`

Purpose:  
Compare D-NSM volumetric water content outputs against bulk soil moisture observations derived from USCRN.

Main inputs:  
- D-NSM daily outputs in `outputs/1016/daily_newhall/pet_PP/`
- observed station files in `outputs/1016/per_station_sm/`
- station metadata in `outputs/1016/aws_by_station_final_1016.csv`

Main outputs:  
- summary and station-level VWC performance metrics in:

```text
outputs/1016/metrics/vwc/
```

Example command:

```bash
python src/evaluation/compute_vwc_performance_metrics.py
```

### 4.2 FAW comparison between daily D-NSM and monthly NSM

Script:
- `src/evaluation/compare_faw_daily_vs_monthly.py`

Purpose:  
Compare end-of-month fraction of available water from the daily D-NSM and the monthly NSM.

Main inputs:  
- monthly NSM outputs
- daily D-NSM outputs
- observation-based FAW inputs

Main outputs:  
- comparison tables and files written under `outputs/`

Example command:

```bash
python src/evaluation/compare_faw_daily_vs_monthly.py
```

## 5. Figure generation

Scripts in `src/figures/` reproduce manuscript figures.

Examples:

- `fig4_boxplot_faw.py`
- `fig5_timeseries_faw.py`
- `fig6_boxplot_vwc.py`
- `fig7_timeseries_vwc.py`
- `fig8_timeseries_with_without_s2.py`

Example commands:

```bash
python src/figures/fig4_boxplot_faw.py
python src/figures/fig5_timeseries_faw.py
python src/figures/fig6_boxplot_vwc.py
python src/figures/fig7_timeseries_vwc.py
python src/figures/fig8_timeseries_with_without_s2.py
```

## Supporting analysis: PET screening against MODIS

This workflow is included as supporting analysis and is separate from the main D-NSM evaluation workflow.

Scripts:
- `src/evaluation/build_pet_8day_from_dnsm.py`
- `src/pet_screening/modis_filtering.py`
- `src/pet_screening/compare_pet_vs_modis.py`

Purpose:  
Generate 8-day PET totals from D-NSM outputs and compare them with MODIS PET.

Main inputs:  
- D-NSM daily outputs in `outputs/1016/daily_newhall/pet_PP/`
- MODIS PET input file in `outputs/1016/pet_screening/MODIS_dailyPET.csv`

Main outputs:  
- 8-day PET files in:

```text
outputs/1016/daily_PET/pet_PP/
```

- filtered MODIS PET file:

```text
outputs/1016/pet_screening/MODIS_ETPET_STRICT_8day.csv
```

- PET comparison metrics in:

```text
outputs/1016/pet_screening/pet_batch_metrics/
```

Example commands:

```bash
python src/evaluation/build_pet_8day_from_dnsm.py
python src/pet_screening/modis_filtering.py
python src/pet_screening/compare_pet_vs_modis.py
```

## Example run order

A typical run order for the main workflow is:

```bash
python src/dataset_preprocessing/02_qc_uscrn.py
python src/dataset_preprocessing/02b_select_station_year.py
python src/dataset_preprocessing/03_extract_gssurgo.py
python src/dataset_preprocessing/04_build_soil_params_by_station.py
python src/dataset_preprocessing/04b_build_theta_sat.py
python src/dataset_preprocessing/05_extract_climate.py
python src/dataset_preprocessing/06_station_coordinates.py
python src/nsm/build_monthly_inputs.py
python src/nsm/run_nsm.py
python src/dnsm/run_dnsm.py
python src/evaluation/compute_vwc_performance_metrics.py
python src/evaluation/compare_faw_daily_vs_monthly.py
```

If needed, the supporting PET workflow can then be run separately.

## Citation

If you use this repository, please cite the associated manuscript and the software metadata in `CITATION.cff`.

## License

A project license will be added in a future update.

## Contact

Moonyoung Lee  
Lyles School of Civil and Construction Engineering  
Purdue University
