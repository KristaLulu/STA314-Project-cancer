from pathlib import Path
import warnings

import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from preprocessing import load_raw_data, split_features_target, run_sanity_checks, get_project_root

warnings.filterwarnings("ignore")


def main():
    # 1. Read shared data
    train, test = load_raw_data()
    X, y, _, _ = split_features_target(train, test)

    # 2. Shared sanity checks
    run_sanity_checks(train, test)

    # 3. Model-specific pipeline
    pipe = Pipeline([
        ("variance_filter", VarianceThreshold(threshold=0.0)),
        ("select_k", SelectKBest(score_func=f_classif)),
        ("scaler", StandardScaler()),
        ("logit", LogisticRegression(
            solver="lbfgs",
            max_iter=5000,
            random_state=314
        ))
    ])

    # 4. Model-specific tuning grid
    param_grid = {
        "select_k__k": [150, 200, 250, 300, 400, 500],
        "logit__C": [0.003, 0.005, 0.01, 0.02, 0.05]
    }

    # 5. Model-specific CV setup
    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=314
    )

    # 6. Grid search
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

    # 7. Save CV results
    results = pd.DataFrame(grid.cv_results_)
    results = results[[
        "param_select_k__k",
        "param_logit__C",
        "mean_test_score",
        "std_test_score",
        "rank_test_score"
    ]].sort_values("rank_test_score")

    print(results.head(15))

    root = get_project_root()
    results.to_csv(root / "cv_results_refined_logistic.csv", index=False)
    print("Saved CV results to cv_results_refined_logistic.csv")


if __name__ == "__main__":
    main()