import warnings
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from sklearn.decomposition import PCA

from preprocessing import (
    load_raw_data,
    split_features_target,
    run_sanity_checks,
)
from lda_qda.lda_qua_utils import build_lda_selected, RANDOM_STATE

warnings.filterwarnings("ignore")


def plot_confusion_matrix(model, X_holdout, y_holdout):
    y_pred = model.predict(X_holdout)
    cm = confusion_matrix(y_holdout, y_pred)

    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.title("Optimized LDA Confusion Matrix")
    plt.tight_layout()
    plt.show()


def plot_pca_decision_boundary(X, y):
    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    X_pca = pca.fit_transform(X)

    model_2d = build_lda_selected(k=100)
    model_2d.fit(X, y)

    x_min, x_max = X_pca[:, 0].min() - 1, X_pca[:, 0].max() + 1
    y_min, y_max = X_pca[:, 1].min() - 1, X_pca[:, 1].max() + 1

    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 200),
        np.linspace(y_min, y_max, 200)
    )

    # This is only for visualization in PCA space
    plt.figure(figsize=(8, 6))
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, alpha=0.7)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("PCA Projection of Cancer Classes")
    plt.tight_layout()
    plt.show()


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

    model = build_lda_selected(k=100)
    model.fit(X_train, y_train)

    plot_confusion_matrix(model, X_holdout, y_holdout)
    plot_pca_decision_boundary(X_train, y_train)


if __name__ == "__main__":
    main()