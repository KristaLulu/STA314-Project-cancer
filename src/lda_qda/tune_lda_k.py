import warnings
import pandas as pd

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_val_score,
)

from preprocessing import (
    load_raw_data,
    split_features_target,
    run_sanity_checks,
    get_project_root,
)
from lda_qda.utils import build_lda_selected, RANDOM_STATE

warnings.filterwarnings("ignore")


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

    k_list = [50, 100, 200, 500, 1000]
    results = []

    for k in k_list:
        model = build_lda_selected(k=k)

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

        results.append({
            "k": k,
            "mean_cv_accuracy": cv_acc,
            "mean_cv_macro_f1": cv_f1
        })

        print(
            f"k = {k:4d} | "
            f"CV accuracy = {cv_acc:.4f} | "
            f"CV macro-F1 = {cv_f1:.4f}"
        )

    results_df = pd.DataFrame(results).sort_values(
        by="mean_cv_macro_f1",
        ascending=False
    )

    print("\nTop tuning results:")
    print(results_df)

    root = get_project_root()
    results_df.to_csv(root / "cv_results_lda_k.csv", index=False)
    print("\nSaved tuning results to cv_results_lda_k.csv")


if __name__ == "__main__":
    main()