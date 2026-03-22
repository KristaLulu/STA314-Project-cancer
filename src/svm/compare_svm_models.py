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
from svm.svm_utils import (
    build_linear_svm,
    build_rbf_svm,
    build_poly_svm,
    RANDOM_STATE,
)

warnings.filterwarnings("ignore")


def summarize_grid_result(model_name, grid, X_hold, y_hold):
    best_idx = grid.best_index_

    cv_f1 = grid.cv_results_["mean_test_f1_macro"][best_idx]
    cv_acc = grid.cv_results_["mean_test_accuracy"][best_idx]

    y_pred = grid.predict(X_hold)
    hold_acc = accuracy_score(y_hold, y_pred)
    hold_f1 = f1_score(y_hold, y_pred, average="macro")
    macro_precision = precision_score(y_hold, y_pred, average="macro")
    macro_recall = recall_score(y_hold, y_pred, average="macro")
    cm = confusion_matrix(y_hold, y_pred)

    print(f"\n{model_name} best params:", grid.best_params_)
    print(f"{model_name} CV F1 Macro:", round(cv_f1, 4))
    print(f"{model_name} CV Accuracy:", round(cv_acc, 4))
    print(f"{model_name} Holdout Accuracy:", round(hold_acc, 4))
    print(f"{model_name} Holdout Macro F1:", round(hold_f1, 4))
    print(f"{model_name} Macro Precision:", round(macro_precision, 4))
    print(f"{model_name} Macro Recall:", round(macro_recall, 4))
    print(f"{model_name} Confusion Matrix:")
    print(cm)
    print(f"{model_name} Classification Report:")
    print(classification_report(y_hold, y_pred, digits=4))

    return {
        "Model": model_name,
        "CV F1 Macro": round(cv_f1, 4),
        "CV Accuracy": round(cv_acc, 4),
        "Holdout Accuracy": round(hold_acc, 4),
        "Holdout Macro F1": round(hold_f1, 4),
    }


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

    # Linear SVM
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

    # Refined linear SVM
    linear_grid_refined = {
        "feature_select__k": [25, 50, 75, 100, 150],
        "svm__C": [0.00005, 0.0001, 0.00025, 0.0005, 0.001]
    }

    grid_linear_refined = GridSearchCV(
        estimator=linear_pipe,
        param_grid=linear_grid_refined,
        scoring=scoring,
        refit="f1_macro",
        cv=cv,
        n_jobs=-1,
        verbose=1,
        return_train_score=False
    )
    grid_linear_refined.fit(X_tr, y_tr)

    # RBF SVM
    rbf_pipe = build_rbf_svm()
    rbf_grid = {
        "feature_select__k": [50, 100, 200, 300],
        "svm__C": [0.1, 1, 10, 50],
        "svm__gamma": ["scale", 0.001, 0.01, 0.1]
    }

    grid_rbf = GridSearchCV(
        estimator=rbf_pipe,
        param_grid=rbf_grid,
        scoring=scoring,
        refit="f1_macro",
        cv=cv,
        n_jobs=-1,
        verbose=1,
        return_train_score=False
    )
    grid_rbf.fit(X_tr, y_tr)

    # Refined RBF SVM
    rbf_grid_refined = {
        "feature_select__k": [25, 50, 75, 100],
        "svm__C": [0.01, 0.05, 0.1, 0.5, 1],
        "svm__gamma": ["scale", 0.0001, 0.0005, 0.001]
    }

    grid_rbf_refined = GridSearchCV(
        estimator=rbf_pipe,
        param_grid=rbf_grid_refined,
        scoring=scoring,
        refit="f1_macro",
        cv=cv,
        n_jobs=-1,
        verbose=1,
        return_train_score=False
    )
    grid_rbf_refined.fit(X_tr, y_tr)

    # Polynomial SVM
    poly_pipe = build_poly_svm()
    poly_grid = {
        "feature_select__k": [25, 50, 100],
        "svm__C": [0.1, 1, 10],
        "svm__degree": [2, 3],
        "svm__gamma": ["scale", 0.001],
        "svm__coef0": [0, 1]
    }

    grid_poly = GridSearchCV(
        estimator=poly_pipe,
        param_grid=poly_grid,
        scoring=scoring,
        refit="f1_macro",
        cv=cv,
        n_jobs=-1,
        verbose=1,
        return_train_score=False
    )
    grid_poly.fit(X_tr, y_tr)

    comparison = pd.DataFrame([
        summarize_grid_result("Linear SVM", grid_linear, X_hold, y_hold),
        summarize_grid_result("Refined Linear SVM", grid_linear_refined, X_hold, y_hold),
        summarize_grid_result("RBF SVM", grid_rbf, X_hold, y_hold),
        summarize_grid_result("Refined RBF SVM", grid_rbf_refined, X_hold, y_hold),
        summarize_grid_result("Polynomial SVM", grid_poly, X_hold, y_hold),
    ])

    print("\nComparison table:")
    print(comparison)

    root = get_project_root()
    comparison.to_csv(root / "svm_model_comparison.csv", index=False)
    print("\nSaved comparison table to svm_model_comparison.csv")


if __name__ == "__main__":
    main()