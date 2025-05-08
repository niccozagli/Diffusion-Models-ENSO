# Diffusion-Models-ENSO

Where's ENZO

## 🔧 Setup

```bash
./setup.sh
```

## 🚀 Development

```bash
conda activate Diffusion-Models-ENSO
poetry run pytest
poetry run black src tests
poetry run isort src tests
```

## ✅ Pre-commit

Hooks run automatically before each commit. You can also trigger them manually:

```bash
poetry run pre-commit run --all-files
```

## ⚙️ CI/CD

This project uses [GitHub Actions](https://github.com/features/actions) to automatically lint and test all code on every push.
