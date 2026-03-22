from pathlib import Path
import pandas as pd


def get_project_root():
    return Path(__file__).resolve().parents[1]


def load_raw_data():
    root = get_project_root()
    train = pd.read_csv(root / "data" / "train.csv")
    test = pd.read_csv(root / "data" / "test.csv")
    return train, test


def split_features_target(train, test):
    X = train.drop(columns=["id", "cancer"])
    y = train["cancer"]
    X_test = test.drop(columns=["id"])
    test_ids = test["id"]
    return X, y, X_test, test_ids


def run_sanity_checks(train, test):
    print("train shape:", train.shape)
    print("test shape:", test.shape)
    print("missing in train:", train.isna().sum().sum())
    print("missing in test:", test.isna().sum().sum())
    print("duplicate train ids:", train["id"].duplicated().sum())
    print("duplicate test ids:", test["id"].duplicated().sum())
    print("class counts:\n", train["cancer"].value_counts().sort_index())