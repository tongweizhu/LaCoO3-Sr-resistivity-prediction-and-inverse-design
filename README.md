# LSCO Resistivity Prediction Workbench

A desktop-first scientific workbench for predicting LSCO electrical resistivity with the supplied GWO-75 model. The web interface opens in **English by default**; use the language selector in the header to switch the current browser session to Chinese.

## What is included

- FastAPI prediction API: `http://127.0.0.1:5050` (`/docs` for OpenAPI)
- Rio desktop web interface: `http://127.0.0.1:8000`
- Four fixed workflows: **Single Prediction**, **Batch Prediction**, **Inverse Design**, and **Model Fine-tuning**
- Browser-native downloads for CSV, PNG, XLSX, JSON, and fine-tuned model artifacts

The interface is intentionally designed for a desktop browser. It uses a classic scientific-workstation layout: a flat grey canvas, thin group-box borders, compact label/input rows, a blue active tab, left-side controls, and a wide white results area. The complete workflow stays visible on one screen; no mobile switcher layout is provided.

Typography prefers **Times New Roman** throughout, with `Tinos` and `Liberation Serif` as metrically compatible Linux fallbacks. Titles, tabs, field labels, controls, status messages, and plot text use enlarged desktop sizing for quick reading.

## Start locally

Requirements: Python 3.12. From this directory:

```bash
bash run.sh
```

The first run creates `.venv` and installs the pinned dependencies. To prepare the environment without starting services:

```bash
bash run.sh install
```

To stop services started by the script:

```bash
bash run.sh stop
```

Alternative ports can be supplied without changing the source:

```bash
BACKEND_PORT=5051 FRONTEND_PORT=8080 bash run.sh
```

Logs are written to `.run/backend.log` and `.run/frontend.log`. The supported runtime is the project `.venv`; legacy virtual environments are not used.

## Model and units

The default artifact is `zhu/models/LSCO_GWO75_without_lattice_api_v1.joblib`. It exposes the GWO population-75 model through a 13-field API and does not include the lattice parameters `a` or `c`.

The supplied source artifact uses spreadsheet-style field names and stores its three oxygen-pressure inputs in log space. `zhu/adopt_gwo75_model.py` creates the API adapter without retraining the estimator or changing its tree weights. The adapter records the original source SHA-256 as `56734a4cdc4f3394ee8749a65e974be8469f89b23f46d82ec003ca2b8d14a67b`, along with the public-to-source field mapping and preprocessing metadata.

The saved GWO-75 preprocessors were created with scikit-learn 1.7.2, so the supported environment pins `scikit-learn==1.7.2`. Use the project environment created by `run.sh` rather than an older system or legacy environment.

The API contract is:

- model input fields are read from `GET /features` at runtime;
- `Tmeas` is measurement temperature in K;
- users enter `Psyn`, `PA`, and `Pc` as positive values in mbar; the application applies the model's required `log10` transform internally before imputation and scaling;
- `y_log` is `log10(ρ / (μΩ·cm))`;
- `y = 10 ** y_log`, returned as `μΩ·cm`;
- `rho_pred_uohm_cm` in batch exports uses the same `μΩ·cm` unit.

The active model fields are:

| Field | Meaning | Unit / values |
| --- | --- | --- |
| `Psyn` | Synthesis/growth oxygen pressure | mbar |
| `Oxygen activation` | Oxygen activation method | `No` or `Ozone` |
| `Mismatch` | Lattice mismatch | % |
| `Ts` | Growth/substrate temperature | °C |
| `Growth method` | Film growth method | `PLD` or `MBE` |
| `A` | Annealing status | `0` or `1` |
| `TA` | Annealing temperature | °C |
| `PA` | Annealing oxygen pressure | mbar |
| `tA` | Annealing duration | h |
| `Pc` | Cooling oxygen pressure | mbar |
| `t` | Film thickness | nm |
| `Sr` | Strontium content / doping fraction | dimensionless (`0.05` = 5%) |
| `Tmeas` | Measurement temperature | K |

`Substrate`, `a`, and `c` are not model input columns. Do not guess units for fields added by a different model; use the metadata and scale supplied by that model.

The active model has exactly 13 input fields: the 12 fixed preparation/material fields shown in the Single Prediction form plus `Tmeas`, which is generated from the selected temperature interval.

### Default test data

Single Prediction opens with the following GWO-75 test case:

| Field | Default value |
| --- | ---: |
| `Psyn` | `1.33322e-5` mbar |
| `Oxygen activation` | `No` |
| `Mismatch` | `1.96` % |
| `Ts` | `700` °C |
| `A` | `1` |
| `TA` | `350` °C |
| `PA` | `212.276` mbar |
| `tA` | `2` h |
| `Pc` | `1.33322e-5` mbar |
| `t` | `30` nm |
| `Sr` | `0.50` |
| `Growth method` | `MBE` |
| `Tmeas` | `30–300` K, 60 equally spaced points |

This test case deliberately places `TA`, `PA`, and `tA` outside the saved GWO-75 training ranges. The application therefore reports those fields as extrapolations; the warning is expected and does not prevent prediction. The `30–300 K` measurement interval itself is inside the saved temperature range.

## Workflows

### Single Prediction

Fill every model feature with a finite value. `A` is an explicit 0/1 selection and categorical fields use dropdowns. `Psyn`, `PA`, and `Pc` must additionally be positive because their saved-model preprocessing uses `log10`. The curve contains 60 equally spaced temperatures; the default interval is `30–300 K`, but wider intervals are allowed and are marked as extrapolation when outside the saved training ranges.

### Batch Prediction

1. Download the CSV template.
2. Upload CSV or XLSX data.
3. Review the validation summary and compact preview.
4. Run the prediction and save CSV or XLSX from the browser.

Required columns may appear in any order. Missing columns, empty cells, non-numeric values, invalid categories, non-binary `A` values, and non-positive `Psyn`/`PA`/`Pc` values block execution (the explicit `PA=0` placeholder remains valid when `A=0`). Extra columns are retained in the result and explicitly excluded from model prediction. Out-of-range rows are reported as extrapolation.

### Inverse Design

Enter a positive target resistivity in `μΩ·cm` and define discrete candidate values for `Sr`, `Ts`, `Tmeas`, substrate, `Psyn`, `Pc`, and annealing status. The backend evaluates the bounded Cartesian grid with the currently loaded GWO-75 forward model and ranks candidates by `|Δ log10(ρ)|`; this is a constrained candidate search, not a unique mathematical inverse solution.

The default `500 μΩ·cm` search contains 2,000 candidates and reproduces the supplied developer-package smoke result: the closest default candidate is approximately `853.146 μΩ·cm`. The grid is capped at 20,000 candidates, `Sr` must remain inside the mismatch-table domain `0–0.8`, pressure values must be positive physical values in mbar, and invalid categories or non-finite values are rejected. `Substrate` is only an auxiliary selector used to interpolate `Mismatch`; it is not passed to the model as an input feature.

The results page shows a compact paginated preview and reports training-range extrapolation. Full CSV, a three-sheet XLSX (`Top candidates`, `All candidates`, and `Mismatch table`), and the normalized JSON configuration are generated in memory and saved through the browser. No result path is created or exposed on the server.

### Model Fine-tuning

Upload a trusted `.joblib` model and a CSV/XLSX training file. Preflight requires complete model feature columns, enough valid rows, values compatible with the model-declared input transforms, and a final column containing positive `μΩ·cm` targets. The run reapplies the same pressure preprocessing as prediction and reports Before → After R², RMSE, MAE, and tree count; RMSE and MAE are explicitly log-space metrics.

Temporary files are removed after processing. All user-visible artifacts are saved through the browser's native file dialog; server absolute paths are not exposed or retained as downloads.

## Development checks

Run the complete test suite:

```bash
MPLCONFIGDIR=/tmp/zhu_interface_mpl .venv/bin/python -m unittest discover -s tests -v
```

The repository also includes focused checks for the `/features` metadata contract, the `y = 10 ** y_log` unit transform, inverse-grid construction and ranking, input validation, desktop page construction, and the fixed viewport boundary.

## Packaging

The release archive should contain source, tests, the model artifact, and the pinned dependency files. It should not contain local virtual environments (`.venv`, `interface`, or `zhu/models/backendenv*`), `.run`, caches, source documents/data, or Git metadata. A reproducible local archive can be created from the explicit release file list below:

```bash
mkdir -p dist
tar --exclude='*/__pycache__' --exclude='*.pyc' --exclude='zhu/models/backendenv' --exclude='zhu/models/backendenv~' --exclude='zhu/temp_old_model.json' --exclude='zhu/zhu/new_data_for_fine_tuning.xlsx' --exclude='zhu/zhu/pages/_bak' --exclude='zhu/zhu/inverse_design_archived.py' --exclude='zhu/models/LSCO_pop50_without_lattice.joblib' --exclude='zhu/models/LSCO_pop50_without_lattice_api_v1.joblib' --exclude='zhu/models/NGO_XGBoost_package_full_pipeline.joblib' --exclude='zhu/zhu/NGO_XGBoost_package_full_pipeline.joblib' -czf dist/lsco-resistivity-workbench.tar.gz README.md .gitignore requirements.txt run.sh call_model.py tests zhu
```
