"""Training entrypoint for Module 4 model training layer."""

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

from module4.predict import generate_predictions
from module4.utils import align_columns, encode_categorical, handle_missing


def train_model(data_contract):
    """Train a Decision Tree model from a DataContract payload.

    Args:
        data_contract: Dictionary containing X, Y, and metadata.

    Returns:
        Tuple of (model, X_train, X_test, y_train, y_test).

    Raises:
        ValueError: If input features or target are missing/empty.
    """
    X = data_contract["X"]
    y = data_contract["Y"]

    if X is None or y is None:
        raise ValueError("X and y must not be None.")
    if X.empty or y.empty:
        raise ValueError("X and y must not be empty.")

    X = handle_missing(X)
    X = encode_categorical(X)

    can_stratify = False
    if isinstance(y, pd.Series):
        value_counts = y.value_counts(dropna=False)
        can_stratify = len(value_counts) > 1 and (value_counts >= 2).all()

    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42,
            stratify=y if can_stratify else None,
        )
    except ValueError:
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42,
            stratify=None,
        )

    X_train, X_test = align_columns(X_train, X_test)

    model = DecisionTreeClassifier(
        max_depth=5,
        min_samples_split=10,
        min_samples_leaf=5,
        random_state=42,
    )
    model.fit(X_train, y_train)

    return model, X_train, X_test, y_train, y_test


def run_training(data_contract):
    """Run end-to-end training and prediction for Module 4.

    Args:
        data_contract: DataContract dictionary containing model inputs.

    Returns:
        Dictionary with the trained model, held-out test data, predicted labels,
        class probabilities used by downstream bias analysis modules, and
        evaluation metrics for model validation and comparison.
    """
    model, X_train, X_test, y_train, y_test = train_model(data_contract)
    preds, probs = generate_predictions(model, X_test)
    accuracy = accuracy_score(y_test, preds)
    precision = precision_score(y_test, preds, average="weighted", zero_division=0)
    recall = recall_score(y_test, preds, average="weighted", zero_division=0)
    f1 = f1_score(y_test, preds, average="weighted", zero_division=0)

    return {
        "model": model,
        "X_test": X_test,
        "y_test": y_test,
        "predictions": preds,
        "probabilities": probs,
        "metrics": {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
        },
    }
