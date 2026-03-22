import warnings
import pandas as pd

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_val_score,
)
from sklearn.metrics import accuracy_score, f1_score, classification_report

from preprocessing import (
    load_raw_data,
    split_features_target,
    run_sanity_checks,
    get_project_root,
)
from lda_qda.lda_qua_utils import build_lda_selected, RANDOM_STATE

warnings.filterwarnings("ignore")


def main():
    train, test = load_raw_data()
    X, y, X_test, test_ids = split_features_target(train, test)

    run_sanity_checks(train, test)

    X_train, X_holdout, y_train, y_holdout = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y
    )

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=RANDOM_STATE
    )

    # Final chosen LDA model
    model = build_lda_selected(k=100)

    cv_acc = cross_val_score(
        model,
        X_train,
        y_train,
        cv=cv,
        scoring="accuracy",
        n_jobs=-1
    ).mean()

    cv_f1 = cross_val_score(
        model,
        X_train,
        y_train,
        cv=cv,
        scoring="f1_macro",
        n_jobs=-1
    ).mean()

    print("Optimized LDA")
    print(f"CV accuracy: {cv_acc:.4f}")
    print(f"CV macro-F1: {cv_f1:.4f}")

    model.fit(X_train, y_train)
    y_holdout_pred = model.predict(X_holdout)

    holdout_acc = accuracy_score(y_holdout, y_holdout_pred)
    holdout_f1 = f1_score(y_holdout, y_holdout_pred, average="macro")

    print(f"Holdout accuracy: {holdout_acc:.4f}")
    print(f"Holdout macro-F1: {holdout_f1:.4f}")
    print("\nClassification report:")
    print(classification_report(y_holdout, y_holdout_pred))

    # Save summary
    summary_df = pd.DataFrame([{
        "model": "Optimized LDA (k=100, lsqr + auto shrinkage)",
        "cv_accuracy": cv_acc,
        "cv_macro_f1": cv_f1,
        "holdout_accuracy": holdout_acc,
        "holdout_macro_f1": holdout_f1
    }])

    root = get_project_root()
    summary_df.to_csv(root / "optimized_lda_results.csv", index=False)
    print("\nSaved summary to optimized_lda_results.csv")

    # Refit on full labeled data
    model.fit(X, y)

    # Generate Kaggle submission
    test_pred = model.predict(X_test)
    submission = pd.DataFrame({
        "id": test_ids,
        "cancer": test_pred
    })

    submission.to_csv(root / "lda_submission.csv", index=False)
    print("Saved submission to lda_submission.csv")


if __name__ == "__main__":
    main()