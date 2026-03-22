import warnings

from preprocessing import (
    load_raw_data,
    split_features_target,
    run_sanity_checks,
)

from eda.eda_utils import (
    apply_zero_variance_filter,
    standardize_features,
    plot_target_distribution,
    run_pca_2d,
)

warnings.filterwarnings("ignore")


def main():
    # 1. Read shared data
    train, test = load_raw_data()
    X_train, y_train, X_test, test_ids = split_features_target(train, test)

    # 2. Shared sanity checks
    run_sanity_checks(train, test)

    # 3. Data description
    print("\nData description:")
    print("train shape:", train.shape)
    print("test shape:", test.shape)
    print("\nTrain info:")
    print(train.info())

    # 4. Initial observations
    print("\nInitial observations:")
    print("missing in train:", train.isna().sum().sum())
    print("missing in test:", test.isna().sum().sum())
    print("duplicate train ids:", train["id"].duplicated().sum())
    print("duplicate test ids:", test["id"].duplicated().sum())
    print("number of features:", X_train.shape[1])

    # 5. Zero-variance filtering
    X_var, X_test_var, kept_cols, var_selector = apply_zero_variance_filter(
        X_train,
        X_test
    )
    print("\nfeatures after variance filter:", X_var.shape[1])

    # 6. Standardization
    X_scaled, X_test_scaled, scaler = standardize_features(X_var, X_test_var)

    # 7. Target class distribution plot
    plot_target_distribution(y_train)

    # 8. PCA plot
    pca, X_pca = run_pca_2d(X_scaled, y_train)
    print("Explained variance:", pca.explained_variance_ratio_)

    print("Class counts:\n", y_train.value_counts().sort_index())


if __name__ == "__main__":
    main()