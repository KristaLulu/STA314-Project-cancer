import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


def apply_zero_variance_filter(X_train, X_test):
    selector = VarianceThreshold(threshold=0.0)
    X_train_var = selector.fit_transform(X_train)
    X_test_var = selector.transform(X_test)
    kept_cols = X_train.columns[selector.get_support()]
    return X_train_var, X_test_var, kept_cols, selector


def standardize_features(X_train, X_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, scaler


def plot_target_distribution(y):
    plt.figure(figsize=(6, 4))
    sns.countplot(x=y)
    plt.title("Target Class Distribution")
    plt.xlabel("Cancer Type")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.show()


def run_pca_2d(X_scaled, y):
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    plt.figure(figsize=(8, 6))
    sns.scatterplot(
        x=X_pca[:, 0],
        y=X_pca[:, 1],
        hue=y,
        palette="Set1"
    )
    plt.title("PCA Projection (2D)")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.tight_layout()
    plt.show()

    return pca, X_pca