import warnings
import pandas as pd

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    GridSearchCV,
)
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
)

from preprocessing import (
    load_raw_data,
    split_features_target,
    run_sanity_checks,
    get_project_root,
)
from svm.utils import build_linear_svm, RANDOM_STATE

warnings.filterwarnings("ignore")


def main():
    train, test = load_raw_data()
    X, y, _, _ = split_features_target(train, test)
    run_sanity_checks(train, test)

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

    linear_grid_refined = {
        "feature_select__k": [25, 50, 75, 100, 150],
        "svm__C": [0.00005, 0.0001, 0.00025, 0.0005, 0.001]
    }

    grid = GridSearchCV(
        estimator=linear_pipe,
        param_grid=linear_grid_refined,
        scoring=scoring,
        refit="f1_macro",
        cv=cv,
        n_jobs=-1,
        verbose=1,
        return_train_score=False
    )

    grid.fit(X_tr, y_tr)

    best_idx = grid.best_index_
    cv_f1 = grid.cv_results_["mean_test_f1_macro"][best_idx]
    cv_acc = grid.cv_results_["mean_test_accuracy"][best_idx]

    y_pred = grid.predict(X_hold)
    hold_acc = accuracy_score(y_hold, y_pred)
    hold_f1 = f1_score(y_hold, y_pred, average="macro")
    hold_precision = precision_score(y_hold, y_pred, average="macro")
    hold_recall = recall_score(y_hold, y_pred, average="macro")
    cm = confusion_matrix(y_hold, y_pred)

    print("Refined Linear SVM best params:", grid.best_params_)
    print("Refined Linear SVM CV F1 Macro:", round(cv_f1, 4))
    print("Refined Linear SVM CV Accuracy:", round(cv_acc, 4))
    print("Refined Linear SVM Holdout Accuracy:", round(hold_acc, 4))
    print("Refined Linear SVM Holdout Macro F1:", round(hold_f1, 4))
    print("Refined Linear SVM Macro Precision:", round(hold_precision, 4))
    print("Refined Linear SVM Macro Recall:", round(hold_recall, 4))
    print("Refined Linear SVM Confusion Matrix:")
    print(cm)
    print("Refined Linear SVM Classification Report:")
    print(classification_report(y_hold, y_pred, digits=4))

    results = pd.DataFrame([{
        "model": "Refined Linear SVM",
        "best_params": str(grid.best_params_),
        "cv_f1_macro": round(cv_f1, 4),
        "cv_accuracy": round(cv_acc, 4),
        "holdout_accuracy": round(hold_acc, 4),
        "holdout_macro_f1": round(hold_f1, 4),
        "holdout_macro_precision": round(hold_precision, 4),
        "holdout_macro_recall": round(hold_recall, 4)
    }])

    root = get_project_root()
    results.to_csv(root / "refined_linear_svm_results.csv", index=False)
    print("\nSaved results to refined_linear_svm_results.csv")


if __name__ == "__main__":
    main()