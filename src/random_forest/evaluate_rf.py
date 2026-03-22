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
    confusion_matrix,
    classification_report,
)

from preprocessing import (
    load_raw_data,
    split_features_target,
    run_sanity_checks,
    get_project_root,
)
from random_forest.rf_utils import build_rf_pipeline, RANDOM_STATE

warnings.filterwarnings("ignore")


def main():
    # 1. Read shared data
    train, test = load_raw_data()
    X, y, X_test, test_ids = split_features_target(train, test)

    # 2. Shared sanity checks
    run_sanity_checks(train, test)

    # 3. Validation split
    X_tr, X_val, y_tr, y_val = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=RANDOM_STATE
    )

    # 4. Build pipeline
    pipe_rf = build_rf_pipeline()

    # 5. Tuning grid
    tuning_grid = {
        "select__k": [500, 1000],
        "rf__n_estimators": [200, 400],
        "rf__max_features": ["sqrt", "log2"],
        "rf__max_depth": [None, 20],
        "rf__min_samples_leaf": [2, 4],
        "rf__class_weight": [None, "balanced"],
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
        param_grid=tuning_grid,
        cv=cv,
        n_jobs=-1,
        scoring=scoring,
        refit="f1_macro"
    )

    grid.fit(X_tr, y_tr)
    rf_best = grid.best_estimator_

    best_idx = grid.best_index_
    cv_f1 = grid.cv_results_["mean_test_f1_macro"][best_idx]
    cv_acc = grid.cv_results_["mean_test_accuracy"][best_idx]

    print("CV macro F1:", cv_f1)
    print("CV accuracy:", cv_acc)

    print("Best parameters:")
    print("select__k =", grid.best_params_["select__k"])
    print("rf__n_estimators =", grid.best_params_["rf__n_estimators"])
    print("rf__max_features =", grid.best_params_["rf__max_features"])
    print("rf__max_depth =", grid.best_params_["rf__max_depth"])
    print("rf__min_samples_leaf =", grid.best_params_["rf__min_samples_leaf"])
    print("rf__class_weight =", grid.best_params_["rf__class_weight"])

    # 7. Holdout evaluation
    val_pred = rf_best.predict(X_val)

    print("CV macro F1:", grid.best_score_)
    print("Holdout Accuracy:", accuracy_score(y_val, val_pred))
    print("Holdout macro F1:", f1_score(y_val, val_pred, average="macro"))
    print(confusion_matrix(y_val, val_pred))
    print(classification_report(y_val, val_pred))

    # 8. Save summary
    results = pd.DataFrame([{
        "model": "Random Forest Baseline",
        "best_params": str(grid.best_params_),
        "cv_accuracy": cv_acc,
        "cv_macro_f1": cv_f1,
        "holdout_accuracy": accuracy_score(y_val, val_pred),
        "holdout_macro_f1": f1_score(y_val, val_pred, average="macro"),
    }])

    root = get_project_root()
    results.to_csv(root / "rf_results.csv", index=False)
    print("Saved results to rf_results.csv")

    # 9. Refit on full labeled training data
    rf_best.fit(X, y)

    # 10. Predict test data
    test_pred = rf_best.predict(X_test)

    submission = pd.DataFrame({
        "id": test_ids,
        "cancer": test_pred
    })

    submission.to_csv(root / "submission_rf.csv", index=False)
    print("Saved submission to submission_rf.csv")

    # 11. Save model
    joblib.dump(rf_best, root / "random_forest.pkl")
    print("Saved model to random_forest.pkl")


if __name__ == "__main__":
    main()