from sklearn.pipeline import Pipeline
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, LinearSVC
from sklearn.decomposition import PCA

RANDOM_STATE = 314


def build_linear_svm():
    return Pipeline([
        ("variance_filter", VarianceThreshold(threshold=0.0)),
        ("scaler", StandardScaler()),
        ("feature_select", SelectKBest(score_func=f_classif)),
        ("svm", LinearSVC(
            penalty="l2",
            loss="squared_hinge",
            dual=True,
            class_weight="balanced",
            max_iter=10000,
            random_state=RANDOM_STATE
        ))
    ])


def build_rbf_svm():
    return Pipeline([
        ("variance_filter", VarianceThreshold(threshold=0.0)),
        ("scaler", StandardScaler()),
        ("feature_select", SelectKBest(score_func=f_classif)),
        ("svm", SVC(
            kernel="rbf",
            class_weight="balanced",
            random_state=RANDOM_STATE
        ))
    ])


def build_poly_svm():
    return Pipeline([
        ("variance_filter", VarianceThreshold(threshold=0.0)),
        ("scaler", StandardScaler()),
        ("feature_select", SelectKBest(score_func=f_classif)),
        ("svm", SVC(
            kernel="poly",
            class_weight="balanced",
            random_state=RANDOM_STATE
        ))
    ])


def build_plain_linear_svm():
    return Pipeline([
        ("variance_filter", VarianceThreshold(threshold=0.0)),
        ("scaler", StandardScaler()),
        ("svm", LinearSVC(
            penalty="l2",
            loss="squared_hinge",
            dual=True,
            class_weight="balanced",
            max_iter=10000,
            random_state=RANDOM_STATE
        ))
    ])


def build_pca_linear_svm():
    return Pipeline([
        ("variance_filter", VarianceThreshold(threshold=0.0)),
        ("scaler", StandardScaler()),
        ("pca", PCA()),
        ("svm", LinearSVC(
            penalty="l2",
            loss="squared_hinge",
            dual=True,
            class_weight="balanced",
            max_iter=10000,
            random_state=RANDOM_STATE
        ))
    ])