# STA314 / Cancer Type Classification (Team Project)

This repo contains our course project for multiclass classification of cancer types
(GBM / LUSC / OV) using gene expression features.

## Project Structure

- `src/`
  - `train_baseline.py` : baseline training + (optional) prediction
  - `evaluate_cv.py`    : cross-validation evaluation scripts
- `data/` (NOT tracked in Git)
  - `train.csv`
  - `test.csv`
  - `sample_submission.csv`
- `notebooks/`
  - EDA / exploration notebooks (optional)
- `report/`
  - `figures/` : figures used in the report
  - report source files (pdf is not tracked)

## Data Setup (Required)

Place the Kaggle files into `data/`:

- `data/train.csv`
- `data/test.csv`
- `data/sample_submission.csv`

Note: `data/` is ignored by git to avoid pushing large files.

## Environment Setup

Recommended: create a virtual environment.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -U numpy pandas scikit-learn matplotlib