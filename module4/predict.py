"""Inference helpers for Module 4 model serving."""

import numpy as np


def generate_predictions(model, X):
    """Generate predicted labels and class probabilities.

    Args:
        model: Trained classifier with `predict` and `predict_proba` methods.
        X: Preprocessed feature matrix for inference.

    Returns:
        Tuple of (preds, probs), where `preds` are predicted class labels
        and `probs` are class probabilities required for downstream bias
        analysis.

    Raises:
        ValueError: If model is None, or X is None/empty.
    """
    if model is None:
        raise ValueError("model must not be None.")
    if X is None:
        raise ValueError("X must not be None.")
    if len(X) == 0:
        raise ValueError("X must not be empty.")

    preds = model.predict(X)
    probs = model.predict_proba(X)
    return preds, probs
