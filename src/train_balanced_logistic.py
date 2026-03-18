from pathlib import Path
import warnings

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

warnings.filterwarnings("ignore")


def main():
    root = Path(__file__).resolve().parents[1]

    train = pd.read_csv(root / "data" / "train.csv")
    test = pd.read_csv(root / "data" / "test.csv")

    X = train.drop(columns=["id", "cancer"])
    y = train["cancer"]

    X_test = test.drop(columns=["id"])
    test_ids = test["id"]

    print("train shape:", train.shape)
    print("test shape:", test.shape)
    print("missing in train:", train.isna().sum().sum())
    print("missing in test:", test.isna().sum().sum())
    print("duplicate train ids:", train["id"].duplicated().sum())
    print("duplicate test ids:", test["id"].duplicated().sum())
    print("class counts:\n", y.value_counts().sort_index())

    X_train, X_valid, y_train, y_valid = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=314,
        stratify=y
    )

    model = Pipeline([
        ("variance_filter", VarianceThreshold(threshold=0.0)),
        ("select_k", SelectKBest(score_func=f_classif, k=100)),
        ("scaler", StandardScaler()),
        ("logit", LogisticRegression(
            solver="lbfgs",
            C=0.001,
            class_weight="balanced",
            max_iter=5000,
            random_state=314
        ))
    ])

    model.fit(X_train, y_train)

    y_valid_pred = model.predict(X_valid)

    print("Validation accuracy:", accuracy_score(y_valid, y_valid_pred))
    print(classification_report(y_valid, y_valid_pred))

    model.fit(X, y)

    test_pred = model.predict(X_test)

    submission = pd.DataFrame({
        "id": test_ids,
        "cancer": test_pred
    })

    submission.to_csv(root / "balanced_logistic_submission.csv", index=False)
    print("Saved submission to balanced_logistic_submission.csv")


if __name__ == "__main__":
    main()