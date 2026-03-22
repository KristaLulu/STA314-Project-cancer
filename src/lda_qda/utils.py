import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import (
    LinearDiscriminantAnalysis,
    QuadraticDiscriminantAnalysis,
)


RANDOM_STATE = 42


def build_lda_baseline():
    return Pipeline([
        ("variance_filter", VarianceThreshold(threshold=0.0)),
        ("scaler", StandardScaler()),
        ("lda", LinearDiscriminantAnalysis())
    ])


def build_qda_baseline():
    return Pipeline([
        ("variance_filter", VarianceThreshold(threshold=0.0)),
        ("scaler", StandardScaler()),
        ("qda", QuadraticDiscriminantAnalysis(reg_param=0.1))
    ])


def build_lda_selected(k=100):
    return Pipeline([
        ("variance_filter", VarianceThreshold(threshold=0.0)),
        ("scaler", StandardScaler()),
        ("select_k", SelectKBest(score_func=f_classif, k=k)),
        ("lda", LinearDiscriminantAnalysis(
            solver="lsqr",
            shrinkage="auto"
        ))
    ])


def get_lda_summary_row(model_name, cv_acc, cv_f1, holdout_acc, holdout_f1):
    return {
        "model": model_name,
        "cv_accuracy": cv_acc,
        "cv_macro_f1": cv_f1,
        "holdout_accuracy": holdout_acc,
        "holdout_macro_f1": holdout_f1
    }