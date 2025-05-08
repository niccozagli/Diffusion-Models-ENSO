# Diffusion-Models-ENSO

Where's ENZO!!!  
*Exploring applications of generative diffusion models for  ENSO variability.*

Test the PR
---

## 📦 Project Structure

This project uses:

- ✅ **Conda** for environment and system-level package management
- ✅ **Poetry** for Python dependencies and packaging
- ✅ **pre-commit** for formatting, linting, and static analysis
- ✅ **GitHub Actions** for CI (code quality checks)
- ✅ **Branch-based development** for safe collaboration

---

## 🔧 Setup

Clone the repo and run the setup script:

```bash
git clone git@github.com:your-username/Diffusion-Models-ENSO.git
cd Diffusion-Models-ENSO
./setup.sh
```

This will:

- Create a Conda environment
- Install Python dependencies via Poetry

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
   poetry run pre-commit run
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

7. When ready, create a pull  request. All changes will go through pull request review and CI checks before merging.

---

## 🧪 Testing

Run tests locally with:

```bash
poetry run pytest
```

Add tests under the `tests/` folder using descriptive names.

---

## 🚀 Development

After activating the environment:

```bash
conda activate Diffusion-Models-ENSO

# Run tests
poetry run pytest

# Format and lint
poetry run black src tests
poetry run isort src tests
poetry run mypy src
```
