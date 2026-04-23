"""Training API: CSV upload, LogisticRegression, JSON-safe responses."""

import io
import traceback
from typing import Any

import numpy as np
import pandas as pd
from fastapi import APIRouter, File, HTTPException, UploadFile
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

from module4.utils import align_columns, encode_categorical, handle_missing

router = APIRouter()

_MIN_ROWS_FOR_SPLIT = 10


@router.post("/train")
async def train_model(file: UploadFile = File(...)) -> Any:
    """
    Accept a CSV upload, train LogisticRegression, return metrics and
    predictions/probabilities for all rows.
    """
    if file.filename is None or not str(file.filename).lower().endswith(".csv"):
        raise HTTPException(
            status_code=400,
            detail="Only .csv files are accepted.",
        )

    try:
        raw = await file.read()
    except OSError as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Could not read upload: {exc}",
        ) from exc

    if not raw or (isinstance(raw, bytes) and not raw.strip()):
        raise HTTPException(status_code=400, detail="Upload is empty.")

    try:
        text = (
            raw.decode("utf-8-sig")
            if isinstance(raw, bytes)
            else str(raw)
        )
        df = pd.read_csv(io.StringIO(text))
    except Exception as exc:  # noqa: BLE001 — surface CSV parse issues
        raise HTTPException(
            status_code=400,
            detail=f"Invalid or unreadable CSV: {exc}",
        ) from exc

    if df.empty or len(df.columns) < 2:
        raise HTTPException(
            status_code=400,
            detail="CSV must be non-empty and have at least two columns (features + target).",
        )

    print("DataFrame shape:", df.shape)
    print("DataFrame head:\n", df.head())

    target_column = df.columns[-1]
    X = df.drop(columns=[target_column])
    y = df[target_column].copy()

    if X.empty or y.empty:
        raise HTTPException(
            status_code=400,
            detail="Features or target column is empty.",
        )

    try:
        X_proc = handle_missing(X)
        X_proc = encode_categorical(X_proc)
        n_rows = len(X_proc)

        if n_rows < _MIN_ROWS_FOR_SPLIT:
            X_train = X_proc
            y_train = y
            model = LogisticRegression(
                max_iter=1000,
                random_state=42,
                n_jobs=None,
            )
            model.fit(X_train, y_train)
            preds = model.predict(X_proc)
            probs = model.predict_proba(X_proc)
            accuracy = float(accuracy_score(y, preds))
            precision = float(
                precision_score(
                    y,
                    preds,
                    average="weighted",
                    zero_division=0,
                )
            )
            recall = float(
                recall_score(
                    y,
                    preds,
                    average="weighted",
                    zero_division=0,
                )
            )
            f1 = float(
                f1_score(
                    y,
                    preds,
                    average="weighted",
                    zero_division=0,
                )
            )
        else:
            can_stratify = False
            if isinstance(y, pd.Series):
                vc = y.value_counts(dropna=False)
                can_stratify = len(vc) > 1 and (vc >= 2).all()

            try:
                X_train, X_test, y_train, y_test = train_test_split(
                    X_proc,
                    y,
                    test_size=0.2,
                    random_state=42,
                    stratify=y if can_stratify else None,
                )
            except ValueError:
                X_train, X_test, y_train, y_test = train_test_split(
                    X_proc,
                    y,
                    test_size=0.2,
                    random_state=42,
                    stratify=None,
                )

            X_train, X_test = align_columns(X_train, X_test)

            model = LogisticRegression(
                max_iter=1000,
                random_state=42,
                n_jobs=None,
            )
            model.fit(X_train, y_train)

            preds = model.predict(X_proc)
            probs = model.predict_proba(X_proc)

            pred_test = model.predict(X_test)
            accuracy = float(accuracy_score(y_test, pred_test))
            precision = float(
                precision_score(
                    y_test,
                    pred_test,
                    average="weighted",
                    zero_division=0,
                )
            )
            recall = float(
                recall_score(
                    y_test,
                    pred_test,
                    average="weighted",
                    zero_division=0,
                )
            )
            f1 = float(
                f1_score(
                    y_test,
                    pred_test,
                    average="weighted",
                    zero_division=0,
                )
            )

        metrics = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
        }

        return {
            "metrics": metrics,
            "predictions": np.asarray(preds).tolist(),
            "probabilities": np.asarray(probs).tolist(),
        }
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        err_msg = f"{type(exc).__name__}: {exc!s}"
        print("Training error:\n", traceback.format_exc())
        raise HTTPException(status_code=500, detail=err_msg) from exc
