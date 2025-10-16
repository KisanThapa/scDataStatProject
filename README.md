# Python project with uv environment

This project uses Astral's uv to manage Python versions, virtual environments, and dependencies. Follow the steps below to install uv, initialize the project, install Python 3.12, and run the `roc_plot.ipynb` notebook.

## 1) Install uv

- macOS (Homebrew):
  ```bash
  brew install uv
  ```

- macOS/Linux (official installer):
  ```bash
  curl -LsSf https://astral.sh/uv/install.sh | sh
  ```

- Windows (PowerShell):
  ```powershell
  irm https://astral.sh/uv/install.ps1 | iex
  ```

Verify:
```bash
uv --version
```

## 2) Initialize this project

If this repository is already set up with uv (i.e., it includes a `pyproject.toml`), you can skip to Step 3.

Otherwise, initialize uv in the project directory:
```bash
uv init
```

This creates a basic `pyproject.toml` and sets up the project structure. You can add dependencies later with `uv add`.

## 3) Install and pin Python 3.12

uv can download and manage Python for the project:

```bash
# Install Python 3.12 (once on your machine)
uv python install 3.12

# Pin this project to Python 3.12 (creates/updates a local version file)
uv python pin 3.12
```

Alternatively, you can create a virtual environment specifying Python 3.12 directly:
```bash
uv venv --python 3.12
```

Notes:
- `uv python install` downloads a managed Python if it is not present.
- `uv python pin 3.12` ensures commands like `uv run` use Python 3.12 in this project.

## 4) Create the virtual environment and install dependencies

If the project already defines dependencies in `pyproject.toml`:
```bash
# Create/refresh the .venv and install project dependencies
uv sync
```

If you are starting fresh and just want to run the notebook, add the typical notebook and plotting dependencies (adjust as needed):
```bash
uv add jupyter ipykernel
uv add numpy pandas scikit-learn matplotlib seaborn
```

This updates `pyproject.toml` and installs the packages into the project’s `.venv`.

Optional: Activate the virtual environment (not required if you use `uv run`):
- macOS/Linux:
  ```bash
  source .venv/bin/activate
  ```
- Windows:
  ```powershell
  .\.venv\Scripts\activate
  ```

## 5) Register a Jupyter kernel for this project (recommended)

Register a named kernel so the notebook runs with the project’s environment:
```bash
uv run python -m ipykernel install --user --name uv-py312 --display-name "Python (uv, 3.12)"
```

You’ll later select this kernel inside Jupyter.

## 6) Run the `roc_plot.ipynb` notebook




## Troubleshooting

- If Jupyter doesn’t see your kernel, re-run the kernel registration step:
  ```bash
  uv run python -m ipykernel install --user --name uv-py312 --display-name "Python (uv, 3.12)"
  ```

- Ensure you’re in the project directory (where `pyproject.toml` and `.venv` live) when running uv commands.

- If you prefer not to activate `.venv`, always use `uv run <command>` to execute tools with the project environment.

## Project structure (example)

```
.
├─ pyproject.toml
├─ uv.lock
├─ .venv/               # created by uv sync or uv venv
├─ roc_plot.ipynb
```

You’re ready to work with Python 3.12 and run `roc_plot.ipynb` using uv!