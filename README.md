# STA314 / Cancer Type Classification (Team Project)

This repository contains the code for our STA314 final project on multiclass cancer type prediction using gene-expression data.

## Project overview

We study a three-class classification problem based on the TCGA cancer dataset provided for the course project. The goal is to predict the cancer type from high-dimensional gene-expression measurements.

The three cancer classes are:

- GBM (glioblastoma multiforme)
- LUSC (lung squamous cell carcinoma)
- OV (ovarian cancer)

The dataset contains 886 labeled training samples and 12,043 identifiers/features. The gene-expression values are provided in log space.

## Project Links

- Kaggle competition page: https://www.kaggle.com/competitions/classification-of-cancer-types
- Final report: [Add report file or link if applicable]

## Repository purpose

This repository is organized to make our analysis easier to read and reproduce. It includes:

- shared data loading and preprocessing utilities
- model-specific training and tuning scripts
- code used to generate validation results
- code used to generate Kaggle submission files

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
```

## Shared Preprocessing

The shared preprocessing steps are implemented in `src/preprocessing.py`.

These steps include:

- reading the training and test data
- removing the non-predictive sample ID column
- separating predictors and response
- basic sanity checks, including:
  - training and test data shapes
  - missing values
  - duplicated sample IDs
  - class counts in the training set

## Models Considered

Our project compares several model classes, including:

- multinomial logistic regression
- LDA / QDA
- support vector machines
- random forest

Different models use different model-specific pipelines, but they follow the same overall project structure.

## Validation Strategy

We used a common validation design across models.

The labeled training data were split into an 80% training portion and a 20% holdout set for model comparison and sanity checking.

Within the training portion, we used 5-fold cross-validation for tuning and model assessment.

Final predictive performance was also evaluated using Kaggle submission results.

## How to Run

Make sure the project root contains the following files in the `data/` folder:

- `train.csv`
- `test.csv`
- `sample_submission.csv`

Example commands:

```bash
python src/logistic/train_baseline.py
python src/logistic/train_balanced_logistic.py
python src/logistic/evaluate_cv.py
python src/logistic/evaluate_cv_balanced.py
python src/logistic/evaluate_cv_refined.py
```

## Reproducibility Notes

- Random seeds are fixed where appropriate.
- Shared preprocessing is centralized in `src/preprocessing.py`.
- Training scripts and tuning scripts are separated for clarity.
- Submission files are generated directly from the training scripts.

## Final Report Connection

This repository supports the code appendix of our final report. It is intended to document the analysis workflow and reproduce the main results reported in the paper, including preprocessing, model tuning, holdout evaluation, and final Kaggle submissions.

## Team Information

- Team name: [Human Learning]
- Team members:
  - Xiaotong Zhu
  - Ada Fu
  - Veronica Yu
  - Liying He

## Notes

This repository is primarily for course project organization and reproducibility. Some scripts correspond to intermediate experiments, while others correspond to the final models discussed in the report.