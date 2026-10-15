# Daily Newhall Simulation Model (D-NSM)

Python code for *A Daily Newhall Simulation Model for Surface and Rootzone Soil Moisture Estimation* (Lee, Merwade, and Levi).

D-NSM estimates daily surface (0–12.5 cm) and rootzone (0–100 cm) soil moisture from precipitation, temperature, and soil properties using a simple water balance with two storage domains.

## Code

### `src/dataset_preprocessing/`

| Script | Description |
| --- | --- |
| `02_qc_uscrn_v2_pt_qc_final.py` | Screen USCRN soil moisture and climate records. |
| `02b_select_station_year_pt_qc.py` | Select one evaluation year per station. |
| `03_extract_gssurgo.py` | Match stations to gSSURGO map units. |
| `04_build_soil_params_by_station_final_qc_w33_dbthirdbar_ptqc.py` | Prepare soil parameters for the main model. |
| `04b_build_theta_sat_final_qc_w33_dbthirdbar_ptqc.py` | Estimate saturated water content for the main model. |
| `05_extract_climate_ptqc.py` | Prepare daily climate and warm-up inputs. |
| `06_station_coordinates_final_qc_w33_dbthirdbar_ptqc.py` | Prepare station coordinates. |
| `04_build_soil_params_by_station_final_qc_fc10_33_dbthirdbar_ptqc.py` | Prepare soil inputs for the field-capacity sensitivity test. |
| `04b_build_theta_sat_final_qc_fc10_33_dbthirdbar_ptqc.py` | Estimate saturated water content for that test. |

### `src/dnsm/`

| Script | Description |
| --- | --- |
| `climate.py` | Process climate forcing. |
| `pet.py` | Calculate daily potential evapotranspiration. |
| `soil_profile.py` | Define soil layers and water storage. |
| `moisture_condition.py` | Handle slot-based moisture updates. |
| `run_dnsm_final_qc_w33_dbthirdbar_ptqc.py` | Run the main D-NSM model. |
| `run_dnsm_final_qc_w33_dbthirdbar_ptqc_single.py` | Run the single-storage comparison. |
| `run_alpha_fast_sensitivity_final_qc_w33_dbthirdbar_ptqc.py` | Test fast-drainage coefficients. |
| `run_dnsm_final_qc_fc10_33_dbthirdbar_ptqc.py` | Run the field-capacity sensitivity test. |

### `src/nsm/`

| Script | Description |
| --- | --- |
| `build_monthly_inputs_dbthirdbar_ptqc.py` | Prepare monthly climate inputs for the original NSM. |
| `run_nsm_dbthirdbar_ptqc.py` | Run the original monthly NSM. |

### `src/evaluation/`

| Script | Description |
| --- | --- |
| `compare_faw_daily_vs_monthly_final_thirdbar_ptqc.py` | Compare fractional available water from the NSM and D-NSM. |
| `compute_vwc_performance_metrics_final_qc_w33_dbthirdbar_ptqc.py` | Calculate main VWC performance metrics. |
| `compute_alpha_fast_sensitivity_metrics_final_qc_w33_dbthirdbar_ptqc.py` | Evaluate fast-drainage sensitivity results. |
| `compute_vwc_performance_metrics_final_qc_fc10_33_dbthirdbar_ptqc.py` | Evaluate field-capacity sensitivity results. |

### `src/figures/`

| Script | Description |
| --- | --- |
| `fig5_timeseries_faw_ptqc.py` | Plot FAW time series. (Figure 4) |
| `fig6_boxplot_vwc_ptqc.py` | Plot VWC performance metrics. (Figure 5) |
| `fig7_timeseries_vwc_ptqc.py` | Plot daily VWC time series. (Figure 6) |
| `fig8_timeseries_with_without_s2_ptqc_batesville.py` | Compare simulations with and without fast-drainage storage. (Figure 7) |

Figure numbers in script names may differ from those in the manuscript.

### Other files

- `src/utils/paths.py`: Data paths.
- `src/utils/utils_obs_bulk.py`: Depth-weighted USCRN soil moisture observations.
- `data_reference/`: Station metadata (`crn-stations.txt`, `stations_XY.csv`) and original NSM parameter tables (`constants_classic.csv`, `knorth_classic.csv`, `ksouth_classic.csv`).
- `requirements.txt`: Python dependencies.

## Data and setup

Processed inputs and model outputs are stored separately on Zenodo (updated DOI to be added). Download the dataset and place its `outputs/` folder in the repository root, keeping the original subfolders and filenames. The main model reads processed inputs from `outputs/1016/`.

Raw data sources:
- [USCRN Daily01](https://www.ncei.noaa.gov/pub/data/uscrn/products/daily01/)
- [USDA-NRCS gSSURGO](https://www.nrcs.usda.gov/resources/data-and-reports/gridded-soil-survey-geographic-gssurgo-database)

Raw USCRN files are expected in `uscrn_daily01/`. For gSSURGO preprocessing, put the database in `external_data/gSSURGO_CONUS.gdb` or change `GSSURGO_GDB` in `src/utils/paths.py` to your local `.gdb` path. The main D-NSM run uses processed soil parameter files and does not require the original `.gdb`.

Install dependencies from the repository root:

```bash
python -m pip install -r requirements.txt
```

Run the main model and its evaluation:

```bash
python src/dnsm/run_dnsm_final_qc_w33_dbthirdbar_ptqc.py
python src/evaluation/compute_vwc_performance_metrics_final_qc_w33_dbthirdbar_ptqc.py
```

Other simulations and evaluations are provided in `src/dnsm/`, `src/nsm/`, and `src/evaluation/`.

**Preprocessing note:** `03_extract_gssurgo.py` currently expects `outputs/selected_station_year.csv`. If using the output of `02b_select_station_year_pt_qc.py`, change that filename to `selected_station_year_1016_pt_qc.csv` in `03_extract_gssurgo.py`.

## Citation

Lee, M., Merwade, V., and Levi, M. R. *A Daily Newhall Simulation Model for Surface and Rootzone Soil Moisture Estimation.* (Publication details pending.)

Software and dataset Zenodo version DOIs will be added after the updated releases.
