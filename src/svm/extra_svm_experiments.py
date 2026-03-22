import warnings
import pandas as pd

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    GridSearchCV,
)
from sklearn.metrics import accuracy_score, f1_score

from preprocessing import (
    load_raw_data,
    split_features_target,
    run_sanity_checks,
    get_project_root,
)
from svm.svm_utils import (
    build_plain_linear_svm,
    build_pca_linear_svm,
    RANDOM_STATE,
)

warnings.filterwarnings("ignore")


def main():
    train, test = load_raw_data()
    X, y, X_test, test_ids = split_features_target(train, test)
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

    # Plain linear SVM
    plain_linear_pipe = build_plain_linear_svm()
    plain_linear_grid = {
        "svm__C": [0.0001, 0.0003, 0.001, 0.003, 0.01]
    }

    grid_plain_linear = GridSearchCV(
        estimator=plain_linear_pipe,
        param_grid=plain_linear_grid,
        scoring=scoring,
        refit="f1_macro",
        cv=cv,
        n_jobs=-1,
        verbose=1,
        return_train_score=False
    )
    grid_plain_linear.fit(X_tr, y_tr)

    best_idx_plain = grid_plain_linear.best_index_
    plain_cv_f1 = grid_plain_linear.cv_results_["mean_test_f1_macro"][best_idx_plain]
    plain_cv_acc = grid_plain_linear.cv_results_["mean_test_accuracy"][best_idx_plain]

    y_pred_plain = grid_plain_linear.predict(X_hold)
    plain_hold_acc = accuracy_score(y_hold, y_pred_plain)
    plain_hold_f1 = f1_score(y_hold, y_pred_plain, average="macro")

    print("Plain Linear SVM best params:", grid_plain_linear.best_params_)
    print("Plain Linear SVM CV F1 Macro:", round(plain_cv_f1, 4))
    print("Plain Linear SVM CV Accuracy:", round(plain_cv_acc, 4))
    print("Plain Linear SVM Holdout Accuracy:", round(plain_hold_acc, 4))
    print("Plain Linear SVM Holdout Macro F1:", round(plain_hold_f1, 4))

    # PCA + Linear SVM
    pca_linear_pipe = build_pca_linear_svm()
    pca_linear_grid = {
        "pca__n_components": [10, 25, 50, 100, 200],
        "svm__C": [0.0001, 0.0003, 0.001, 0.003, 0.01]
    }

    grid_pca_linear = GridSearchCV(
        estimator=pca_linear_pipe,
        param_grid=pca_linear_grid,
        scoring=scoring,
        refit="f1_macro",
        cv=cv,
        n_jobs=-1,
        verbose=1,
        return_train_score=False
    )
    grid_pca_linear.fit(X_tr, y_tr)

    best_idx_pca = grid_pca_linear.best_index_
    pca_cv_f1 = grid_pca_linear.cv_results_["mean_test_f1_macro"][best_idx_pca]
    pca_cv_acc = grid_pca_linear.cv_results_["mean_test_accuracy"][best_idx_pca]

    y_pred_pca = grid_pca_linear.predict(X_hold)
    pca_hold_acc = accuracy_score(y_hold, y_pred_pca)
    pca_hold_f1 = f1_score(y_hold, y_pred_pca, average="macro")

    print("PCA + Linear SVM best params:", grid_pca_linear.best_params_)
    print("PCA + Linear SVM CV F1 Macro:", round(pca_cv_f1, 4))
    print("PCA + Linear SVM CV Accuracy:", round(pca_cv_acc, 4))
    print("PCA + Linear SVM Holdout Accuracy:", round(pca_hold_acc, 4))
    print("PCA + Linear SVM Holdout Macro F1:", round(pca_hold_f1, 4))

    results = pd.DataFrame([
        {
            "Model": "Plain Linear SVM",
            "CV F1 Macro": round(plain_cv_f1, 4),
            "CV Accuracy": round(plain_cv_acc, 4),
            "Holdout Accuracy": round(plain_hold_acc, 4),
            "Holdout Macro F1": round(plain_hold_f1, 4),
        },
        {
            "Model": "PCA + Linear SVM",
            "CV F1 Macro": round(pca_cv_f1, 4),
            "CV Accuracy": round(pca_cv_acc, 4),
            "Holdout Accuracy": round(pca_hold_acc, 4),
            "Holdout Macro F1": round(pca_hold_f1, 4),
        }
    ])

    print("\nExtra SVM experiment summary:")
    print(results)

    root = get_project_root()
    results.to_csv(root / "svm_extra_experiments.csv", index=False)
    print("\nSaved results to svm_extra_experiments.csv")

    # Save optional submissions
    final_plain = grid_plain_linear.best_estimator_
    final_plain.fit(X, y)
    test_pred_plain = final_plain.predict(X_test)
    submission_plain = pd.DataFrame({
        "id": test_ids,
        "cancer": test_pred_plain
    })
    submission_plain.to_csv(root / "submission_plain_linear_svm.csv", index=False)
    print("Saved submission_plain_linear_svm.csv")

    final_pca = grid_pca_linear.best_estimator_
    final_pca.fit(X, y)
    test_pred_pca = final_pca.predict(X_test)
    submission_pca = pd.DataFrame({
        "id": test_ids,
        "cancer": test_pred_pca
    })
    submission_pca.to_csv(root / "submission_pca_linear_svm.csv", index=False)
    print("Saved submission_pca_linear_svm.csv")


if __name__ == "__main__":
    main()