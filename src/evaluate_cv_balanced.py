from pathlib import Path
import warnings

import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

warnings.filterwarnings("ignore")


def main():
    root = Path(__file__).resolve().parents[1]

    train = pd.read_csv(root / "data" / "train.csv")

    X = train.drop(columns=["id", "cancer"])
    y = train["cancer"]

    pipe = Pipeline([
        ("variance_filter", VarianceThreshold(threshold=0.0)),
        ("select_k", SelectKBest(score_func=f_classif)),
        ("scaler", StandardScaler()),
        ("logit", LogisticRegression(
            solver="lbfgs",
            class_weight="balanced",
            max_iter=5000,
            random_state=314
        ))
    ])

    param_grid = {
        "select_k__k": [100, 200, 500],
        "logit__C": [0.001, 0.01, 0.1, 1]
    }

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=314
    )

    grid = GridSearchCV(
        estimator=pipe,
        param_grid=param_grid,
        scoring="accuracy",
        cv=cv,
        n_jobs=-1,
        verbose=1
    )

    grid.fit(X, y)

    print("Best parameters:", grid.best_params_)
    print("Best CV accuracy:", grid.best_score_)

    results = pd.DataFrame(grid.cv_results_)
    results = results[[
        "param_select_k__k",
        "param_logit__C",
        "mean_test_score",
        "std_test_score",
        "rank_test_score"
    ]].sort_values("rank_test_score")

    print(results.head(10))

    results.to_csv(root / "cv_results_balanced_logistic.csv", index=False)
    print("Saved CV results to cv_results_balanced_logistic.csv")


if __name__ == "__main__":
    main()