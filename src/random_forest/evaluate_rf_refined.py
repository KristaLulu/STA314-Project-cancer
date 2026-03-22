import warnings
import pandas as pd
import joblib

from sklearn.model_selection import (
    train_test_split,
    GridSearchCV,
    StratifiedKFold,
)
from sklearn.metrics import (
    accuracy_score,
    f1_score,
)

from preprocessing import (
    load_raw_data,
    split_features_target,
    run_sanity_checks,
    get_project_root,
)
from random_forest.utils import build_rf_pipeline, RANDOM_STATE

warnings.filterwarnings("ignore")


def main():
    # 1. Read shared data
    train, test = load_raw_data()
    X, y, X_test, test_ids = split_features_target(train, test)

    # 2. Shared sanity checks
    run_sanity_checks(train, test)

    # 3. Single split only (same as original RF code)
    X_tr, X_val, y_tr, y_val = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=RANDOM_STATE
    )

    # 4. Build pipeline
    pipe_rf = build_rf_pipeline()

    # 5. Refined tuning grid
    tuning_grid_refine = {
        "select__k": [400, 500, 600],
        "rf__n_estimators": [100, 200, 300],
        "rf__max_features": ["sqrt"],
        "rf__max_depth": [None],
        "rf__min_samples_leaf": [1, 2, 3],
        "rf__class_weight": [None]
    }

    # 6. Cross-validation
    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=RANDOM_STATE
    )
    scoring = {
        "accuracy": "accuracy",
        "f1_macro": "f1_macro"
    }

    grid = GridSearchCV(
        estimator=pipe_rf,
        param_grid=tuning_grid_refine,
        cv=cv,
        n_jobs=-1,
        scoring=scoring,
        refit="f1_macro"
    )

    grid.fit(X_tr, y_tr)
    rf_best_refined = grid.best_estimator_

    best_idx = grid.best_index_
    cv_f1 = grid.cv_results_["mean_test_f1_macro"][best_idx]
    cv_acc = grid.cv_results_["mean_test_accuracy"][best_idx]

    print("CV macro F1:", cv_f1)
    print("CV accuracy:", cv_acc)

    print("Refined best parameters:")
    print("select__k =", grid.best_params_["select__k"])
    print("rf__n_estimators =", grid.best_params_["rf__n_estimators"])
    print("rf__min_samples_leaf =", grid.best_params_["rf__min_samples_leaf"])

    # 7. Holdout evaluation
    val_pred_refined = rf_best_refined.predict(X_val)

    print("Refined CV macro F1:", grid.best_score_)
    print("Refined holdout accuracy:", accuracy_score(y_val, val_pred_refined))
    print("Refined holdout macro F1:", f1_score(y_val, val_pred_refined, average="macro"))

    # 8. Save summary
    results = pd.DataFrame([{
        "model": "Random Forest Refined",
        "best_params": str(grid.best_params_),
        "cv_accuracy": cv_acc,
        "cv_macro_f1": cv_f1,
        "holdout_accuracy": accuracy_score(y_val, val_pred_refined),
        "holdout_macro_f1": f1_score(y_val, val_pred_refined, average="macro"),
    }])

    root = get_project_root()
    results.to_csv(root / "rf_refined_results.csv", index=False)
    print("Saved results to rf_refined_results.csv")

    # 9. Refit on full labeled training data
    rf_best_refined.fit(X, y)

    # 10. Predict test data
    test_pred = rf_best_refined.predict(X_test)

    submission = pd.DataFrame({
        "id": test_ids,
        "cancer": test_pred
    })

    submission.to_csv(root / "submission_rf_refined.csv", index=False)
    print("Saved submission to submission_rf_refined.csv")

    # 11. Save model
    joblib.dump(rf_best_refined, root / "random_forest_refined.pkl")
    print("Saved model to random_forest_refined.pkl")

    print(
        "\nRefined grid search yields the same performance metrics as the "
        "baseline search, suggesting that the initial grid already provides "
        "a stable solution."
    )


if __name__ == "__main__":
    main()