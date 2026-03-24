# Diffusion-Models-ENSO

*Exploring applications of generative diffusion models for  ENSO variability.*

---

## 📦 Project Structure

This project uses:

- ✅ **Pixi** for environment and dependency management
- `src/` for reusable Python package code
- `scripts/` for command-line data preparation workflows
- `analysis/` for exploratory notebooks and marimo apps

---

## 🔧 Setup

Clone the repo and install dependencies with Pixi:

```bash
git clone git@github.com:your-username/Diffusion-Models-ENSO.git
cd Diffusion-Models-ENSO
pixi install
```

This works on macOS, Linux, and Windows.

Optional (macOS/Linux only):

```bash
./setup.sh
```

## VS Code

Pixi creates the project environment under `.pixi/envs/default`, so VS Code may not
auto-detect it as a Jupyter kernel the way it would with a local `.venv`.

To use the project environment in VS Code:

1. Select the interpreter at `.pixi/envs/default/bin/python`.
2. Install the project in editable mode so imports from `src/` work in the kernel:

```bash
pixi run python -m pip install -e .
```

3. If no Jupyter kernel appears, register one explicitly:

```bash
pixi run python -m ipykernel install --user \
  --name diffusion-models-enso \
  --display-name "Python (Diffusion-Models-ENSO)"
```

For marimo-based analysis apps under `analysis/`, you can launch them directly:

```bash
pixi run marimo edit analysis/data_analysis.py
```
or open them in VSCode (after registering the kernel in the previous step).

## Data Preparation

The current analysis pipeline has three Python entry points under `scripts/`.

### 1. Monthly diagnostics

[`scripts/build_monthly_diagnostics.py`](scripts/build_monthly_diagnostics.py)
reads the raw monthly diffusion NetCDF files and builds the first persisted
analysis artifact, `data/monthly_diagnostics.nc`.

It computes, for each `month x year x sample` entry:

- `global_trefht`: weighted global-mean `TREFHT`
- `nino34_trefht`: weighted `TREFHT` averaged over the Niño 3.4 region
- `nino34_index`: the default anomaly definition, computed as
  `nino34_trefht - ensemble mean across sample` for each month/year pair
- `file_name`: the source monthly NetCDF file used for that month/year slot

The output dataset also stores:

- `month` as readable strings such as `December`, `January`, ...
- `month_number` as the numeric month code
- attrs describing the source input directory, file pattern, weighting
  convention, Niño 3.4 bounds, and index definition

Example:

```bash
pixi run python scripts/build_monthly_diagnostics.py \
  --input-dir /path/to/monthly/netcdf/files \
  --start-year 2015 \
  --output data/monthly_diagnostics.nc \
  --overwrite
```

Useful options:

- `--input-dir`: directory containing the raw monthly NetCDF files
- `--months`: month numbers to include; defaults to `12, 1, 2, 6, 7, 8`
- `--start-year`: year assigned to the first sorted file
- `--pattern`: filename pattern template for month discovery
- `--variable`: source variable to read; defaults to `TREFHT`
- `--output`: output path; defaults to `data/monthly_diagnostics.nc`
- `--overwrite`: replace an existing output file

### 2. ENSO event selection

[`scripts/build_enso_selection.py`](scripts/build_enso_selection.py) reads
`data/monthly_diagnostics.nc` and writes `data/enso_selection.nc`.

This second artifact stores the selected ENSO events as an `event` table with:

- `month`, `month_number`, `year`, `sample`
- `nino34_index`
- `file_name`
- `event_type` (`nino` or `nina`)

The dataset attrs also keep the shared selection thresholds and provenance back
to the monthly diagnostics artifact.

Example:

```bash
pixi run python scripts/build_enso_selection.py \
  --input data/monthly_diagnostics.nc \
  --output data/enso_selection.nc \
  --overwrite
```

Useful options:

- `--input`: input monthly diagnostics path
- `--months`: month numbers used for selection; defaults to `12, 1, 2`
- `--quantile-nino`: upper quantile used for El Niño selection
- `--quantile-nina`: lower quantile used for La Niña selection
- `--output`: output path; defaults to `data/enso_selection.nc`
- `--overwrite`: replace an existing output file

### 3. ENSO composites

[`scripts/build_enso_composites.py`](scripts/build_enso_composites.py) reads
`data/enso_selection.nc` and builds year-by-year Niño and Niña composite fields,
writing `data/enso_composites.nc`.

By default it builds composites for:

- years `2015, 2025, 2035, 2045, 2055, 2065, 2075, 2085`
- variables `TREFHT`, `PS`, `PRECT`

The output dataset is organized as:

- `trefht_composite(event_type, year, lat, lon)`
- `ps_composite(event_type, year, lat, lon)`
- `prect_composite(event_type, year, lat, lon)`
- `n_events(event_type, year)`

Parallelism is intentionally minimal: if you pass `--num-workers`, workers are
split by variable, so one worker can build `TREFHT` while another builds `PS`
or `PRECT`.

Example:

```bash
pixi run python scripts/build_enso_composites.py \
  --input data/enso_selection.nc \
  --output data/enso_composites.nc \
  --overwrite
```

Parallel example:

```bash
pixi run python scripts/build_enso_composites.py \
  --input data/enso_selection.nc \
  --num-workers 3 \
  --output data/enso_composites.nc \
  --overwrite
```

Useful options:

- `--input`: input ENSO selection path
- `--variables`: variables to composite; defaults to `TREFHT, PS, PRECT`
- `--years`: years to composite; defaults to `2015, 2025, ..., 2085`
- `--num-workers`: optional worker count; parallelism is across variables
- `--output`: output path; defaults to `data/enso_composites.nc`
- `--overwrite`: replace an existing output file

To inspect the CLIs:

```bash
pixi run python scripts/build_monthly_diagnostics.py --help
pixi run python scripts/build_enso_selection.py --help
pixi run python scripts/build_enso_composites.py --help
```

Generated outputs such as `data/monthly_diagnostics.nc`,
`data/enso_selection.nc`, and `data/enso_composites.nc` are kept out of git.

---

## 🤝 Contributing & Branching

To contribute:

1. Create a feature branch:
   ```bash
   git checkout -b feature/my-new-analysis
   ```

2. Add files and changes

3. Run code checks manually:
   ```bash
   pixi run format
   pixi run typecheck
   pixi run test
   ```

4. If any files were modified (e.g., by `black`), **re-add them**:
   ```bash
   git add <modified-files>
   ```

5. Commit your changes.

6. You can now push changes to your branch:
   ```bash
   git push origin feature/my-new-analysis
   ```

7. When ready, create a pull request.

---

## 🧪 Testing

Run tests locally with:

```bash
pixi run test
```

Add tests under the `tests/` folder using descriptive names.

---

## 🚀 Development

After activating the environment:

```bash
pixi install
pixi run test
pixi run format
pixi run typecheck
pixi run notebook
```
