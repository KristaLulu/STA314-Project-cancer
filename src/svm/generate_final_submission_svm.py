import warnings
import pandas as pd

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    GridSearchCV,
)

from preprocessing import (
    load_raw_data,
    split_features_target,
    run_sanity_checks,
    get_project_root,
)
from svm.svm_utils import build_linear_svm, RANDOM_STATE

warnings.filterwarnings("ignore")


def main():
    train, test = load_raw_data()
    X, y, X_test, test_ids = split_features_target(train, test)
    run_sanity_checks(train, test)

    # Keep same workflow as original code:
    # tune on training portion only, compare on holdout, then refit final winner on full data.
    X_tr, X_hold, y_tr, y_hold = train_test_split(
        X,
        y,
        test_size=0.20,
        stratify=y,
        random_state=RANDOM_STATE
    )

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=RANDOM_STATE
    )

    scoring = {
        "accuracy": "accuracy",
        "f1_macro": "f1_macro"
    }

    linear_pipe = build_linear_svm()
    linear_grid = {
        "feature_select__k": [100, 200, 300, 500],
        "svm__C": [0.0005, 0.001, 0.005, 0.01]
    }

    grid_linear = GridSearchCV(
        estimator=linear_pipe,
        param_grid=linear_grid,
        scoring=scoring,
        refit="f1_macro",
        cv=cv,
        n_jobs=-1,
        verbose=1,
        return_train_score=False
    )

    grid_linear.fit(X_tr, y_tr)

    print("Final submission model best params:", grid_linear.best_params_)

    final_model = grid_linear.best_estimator_
    final_model.fit(X, y)

    test_pred = final_model.predict(X_test)

    submission = pd.DataFrame({
        "id": test_ids,
        "cancer": test_pred
    })

    root = get_project_root()
    submission.to_csv(root / "submission_svm_final.csv", index=False)
    print("Saved submission to submission_svm_final.csv")


if __name__ == "__main__":
    main()