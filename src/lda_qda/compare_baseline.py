import warnings
import pandas as pd

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_val_score,
)
from sklearn.metrics import accuracy_score, f1_score

from preprocessing import (
    load_raw_data,
    split_features_target,
    run_sanity_checks,
    get_project_root,
)
from lda_qda.utils import (
    build_lda_baseline,
    build_qda_baseline,
    get_lda_summary_row,
    RANDOM_STATE,
)

warnings.filterwarnings("ignore")


def evaluate_model(name, model, X_train, y_train, X_holdout, y_holdout, cv):
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

    model.fit(X_train, y_train)
    y_pred = model.predict(X_holdout)

    holdout_acc = accuracy_score(y_holdout, y_pred)
    holdout_f1 = f1_score(y_holdout, y_pred, average="macro")

    print(f"\n{name}")
    print(f"CV accuracy: {cv_acc:.4f}")
    print(f"CV macro-F1: {cv_f1:.4f}")
    print(f"Holdout accuracy: {holdout_acc:.4f}")
    print(f"Holdout macro-F1: {holdout_f1:.4f}")

    return get_lda_summary_row(
        model_name=name,
        cv_acc=cv_acc,
        cv_f1=cv_f1,
        holdout_acc=holdout_acc,
        holdout_f1=holdout_f1
    )


def main():
    train, test = load_raw_data()
    X, y, _, _ = split_features_target(train, test)

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

    lda_base = build_lda_baseline()
    qda_base = build_qda_baseline()

    results = []
    results.append(
        evaluate_model(
            "Baseline LDA",
            lda_base,
            X_train,
            y_train,
            X_holdout,
            y_holdout,
            cv
        )
    )
    results.append(
        evaluate_model(
            "Baseline QDA",
            qda_base,
            X_train,
            y_train,
            X_holdout,
            y_holdout,
            cv
        )
    )

    results_df = pd.DataFrame(results)
    print("\nSummary table:")
    print(results_df)

    root = get_project_root()
    results_df.to_csv(root / "lda_qda_baseline_results.csv", index=False)
    print("\nSaved summary to lda_qda_baseline_results.csv")


if __name__ == "__main__":
    main()