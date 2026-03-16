# Diffusion-Models-ENSO

*Exploring applications of generative diffusion models for  ENSO variability.*

---

## 📦 Project Structure

This project uses:

- ✅ **Pixi** for environment and dependency management
- `src/` for reusable Python package code
- `scripts/` for command-line data preparation workflows
- `notebooks/` for exploratory analysis

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

## Data Preparation

The first reproducible data-preparation entry point is
[`scripts/build_diffusion_month.py`](scripts/build_diffusion_month.py).
It scans monthly diffusion NetCDF files, computes global and Nino 3.4
temperature diagnostics, and writes a combined dataset to
`data/diffusion.nc`.

Example:

```bash
pixi run python scripts/build_diffusion_month.py \
  --data-dir /path/to/monthly/netcdf/files \
  --start-year 2015
```

Generated outputs such as `data/diffusion.nc` are kept out of git.

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
