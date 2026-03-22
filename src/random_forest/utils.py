from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier

RANDOM_STATE = 314


def build_rf_pipeline():
    rf = RandomForestClassifier(random_state=RANDOM_STATE)

    return Pipeline([
        ("select", SelectKBest(score_func=f_classif)),
        ("rf", rf)
    ])