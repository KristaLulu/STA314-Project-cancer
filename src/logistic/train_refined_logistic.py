from pathlib import Path
import warnings

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score

from preprocessing import load_raw_data, split_features_target, run_sanity_checks, get_project_root

warnings.filterwarnings("ignore")


def main():
    # 1. Read shared data
    train, test = load_raw_data()
    X, y, X_test, test_ids = split_features_target(train, test)

    # 2. Shared sanity checks
    run_sanity_checks(train, test)

    # 3. Model-specific validation split
    X_train, X_valid, y_train, y_valid = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=314,
        stratify=y
    )

    # 4. Model-specific pipeline
    model = Pipeline([
        ("variance_filter", VarianceThreshold(threshold=0.0)),
        ("select_k", SelectKBest(score_func=f_classif, k=200)),
        ("scaler", StandardScaler()),
        ("logit", LogisticRegression(
            solver="lbfgs",
            C=0.003,
            max_iter=5000,
            random_state=314
        ))
    ])

    # 5. Fit model
    model.fit(X_train, y_train)

    # 6. Validate
    y_valid_pred = model.predict(X_valid)

    print("Validation accuracy:", accuracy_score(y_valid, y_valid_pred))
    print("Validation macro-F1:", f1_score(y_valid, y_valid_pred, average="macro"))
    print(classification_report(y_valid, y_valid_pred))

    # 7. Refit on full training data
    model.fit(X, y)

    # 8. Predict test data
    test_pred = model.predict(X_test)

    submission = pd.DataFrame({
        "id": test_ids,
        "cancer": test_pred
    })

    root = get_project_root()
    submission.to_csv(root / "logistic_refined_submission.csv", index=False)
    print("Saved submission to logistic_refined_submission.csv")


if __name__ == "__main__":
    main()