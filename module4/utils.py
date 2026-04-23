"""Utility helpers for Module 4 preprocessing steps."""

import numpy as np
import pandas as pd


def handle_missing(X):
    """Handle missing values in the feature matrix.

    Args:
        X: Input feature matrix.

    Returns:
        Processed feature matrix with missing values handled.
    """
    X_filled = X.copy()

    numerical_columns = X_filled.select_dtypes(include=[np.number]).columns
    categorical_columns = X_filled.select_dtypes(include=["object", "category"]).columns

    for col in numerical_columns:
        median_value = X_filled[col].median()
        if pd.isna(median_value):
            median_value = 0
        X_filled[col] = X_filled[col].fillna(median_value)

    for col in categorical_columns:
        mode_series = X_filled[col].mode(dropna=True)
        fill_value = mode_series.iloc[0] if not mode_series.empty else "missing"
        X_filled[col] = X_filled[col].fillna(fill_value)

    return X_filled


def encode_categorical(X):
    """Encode categorical features for model compatibility.

    Args:
        X: Input feature matrix.

    Returns:
        Encoded feature matrix.
    """
    categorical_columns = X.select_dtypes(include=["object", "category"]).columns
    if len(categorical_columns) == 0:
        return X.copy()

    return pd.get_dummies(X, columns=list(categorical_columns), drop_first=True)


def align_columns(X_train, X_test):
    """Align training and test columns after preprocessing.

    Args:
        X_train: Training feature matrix.
        X_test: Test feature matrix.

    Returns:
        Tuple of aligned (X_train, X_test).
    """
    aligned_train, aligned_test = X_train.align(
        X_test,
        join="left",
        axis=1,
        fill_value=0,
    )
    return aligned_train, aligned_test
