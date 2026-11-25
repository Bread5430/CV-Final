"""HOG + color features and a linear SVM (SGD hinge) on letterboxed LISA crops.

The detector (``acf_boost.py``) only proposes boxes. This module labels a
crop as ``background`` or one of the logical sign classes from
``lisa_coco.CLASSIFY_CLASSES``.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

# OpenCV HOG through OpenCL can stall for minutes on CPU-only Windows.
cv2.ocl.setUseOpenCL(False)
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import hinge_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from lisa_coco import CLASSIFY_CLASSES

HOG_WIN = 64
HOG_CELL = 8
HOG_BLOCK = 16
HOG_STRIDE = 8
HOG_BINS = 9
HSV_BINS = (8, 4, 4)
LUV_BINS = (8, 4, 4)


def _hog_descriptor() -> cv2.HOGDescriptor:
    win = (HOG_WIN, HOG_WIN)
    return cv2.HOGDescriptor(
        win,
        (HOG_BLOCK, HOG_BLOCK),
        (HOG_STRIDE, HOG_STRIDE),
        (HOG_CELL, HOG_CELL),
        HOG_BINS,
    )


_HOG = _hog_descriptor()


def hog_vector(bgr: np.ndarray) -> np.ndarray:
    """Fine HOG on a 64x64 BGR crop (shape / inner edges)."""
    img = np.asarray(bgr)
    if img.shape[0] != HOG_WIN or img.shape[1] != HOG_WIN:
        img = cv2.resize(img, (HOG_WIN, HOG_WIN), interpolation=cv2.INTER_LINEAR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    vec = _HOG.compute(gray)
    if vec is None:
        return np.zeros(
            ((HOG_WIN - HOG_BLOCK) // HOG_STRIDE + 1) ** 2 * 4 * HOG_BINS,
            dtype=np.float32,
        )
    return np.asarray(vec, dtype=np.float32).reshape(-1)


def _hist3(image: np.ndarray, bins: Tuple[int, int, int]) -> np.ndarray:
    hist = cv2.calcHist(
        [image],
        [0, 1, 2],
        None,
        bins,
        [0, 256, 0, 256, 0, 256],
    )
    hist = np.asarray(hist, dtype=np.float32).reshape(-1)
    total = float(hist.sum())
    if total > 0:
        hist /= total
    return hist


def color_histogram(bgr: np.ndarray) -> np.ndarray:
    """Coarse HSV + LUV histograms (superclass color)."""
    img = np.asarray(bgr)
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    luv = cv2.cvtColor(img, cv2.COLOR_BGR2LUV)
    return np.concatenate([_hist3(hsv, HSV_BINS), _hist3(luv, LUV_BINS)], axis=0)


def crop_features(bgr: np.ndarray) -> np.ndarray:
    return np.concatenate([hog_vector(bgr), color_histogram(bgr)], axis=0).astype(
        np.float32
    )


def crops_to_features(
    crops: Sequence[np.ndarray],
    verbose: bool = True,
) -> np.ndarray:
    if not crops:
        n_hog = ((HOG_WIN - HOG_BLOCK) // HOG_STRIDE + 1) ** 2 * 4 * HOG_BINS
        n_color = int(np.prod(HSV_BINS) + np.prod(LUV_BINS))
        return np.zeros((0, n_hog + n_color), dtype=np.float32)
    rows = []
    n = len(crops)
    for i, crop in enumerate(crops, start=1):
        rows.append(crop_features(crop))
        if verbose and (i == 1 or i % 2000 == 0 or i == n):
            print(f"    features {i}/{n}", flush=True)
    return np.stack(rows, axis=0)


@dataclass
class SignCropClassifier:
    """Linear SVM (SGD hinge) over HOG+color crop vectors."""

    pipeline: Pipeline
    class_names: List[str]
    history: List[Dict[str, float]] = field(default_factory=list)

    @property
    def n_classes(self) -> int:
        return len(self.class_names)

    def predict(self, crops: Sequence[np.ndarray]) -> np.ndarray:
        x = crops_to_features(crops)
        return np.asarray(self.pipeline.predict(x), dtype=np.int64)

    def predict_proba(self, crops: Sequence[np.ndarray]) -> np.ndarray:
        x = crops_to_features(crops)
        scores = np.asarray(self.pipeline.decision_function(x), dtype=np.float64)
        if scores.ndim == 1:
            scores = np.stack([-scores, scores], axis=1)
        scores = scores - scores.max(axis=1, keepdims=True)
        exp = np.exp(scores)
        return exp / exp.sum(axis=1, keepdims=True)

    def save(self, path: str) -> None:
        with open(path, "wb") as handle:
            pickle.dump(
                {
                    "pipeline": self.pipeline,
                    "class_names": self.class_names,
                    "history": self.history,
                },
                handle,
            )

    @classmethod
    def load(cls, path: str) -> "SignCropClassifier":
        with open(path, "rb") as handle:
            payload = pickle.load(handle)
        return cls(
            pipeline=payload["pipeline"],
            class_names=list(payload["class_names"]),
            history=list(payload.get("history") or []),
        )


def _multiclass_hinge(y: np.ndarray, scores: np.ndarray, labels: np.ndarray) -> float:
    if y.size == 0:
        return 0.0
    return float(hinge_loss(y, scores, labels=labels))


def _error_rate(y: np.ndarray, pred: np.ndarray) -> float:
    if y.size == 0:
        return 0.0
    return float(np.mean(y != pred))


def fit_linear_svm(
    x: np.ndarray,
    y: np.ndarray,
    class_names: Optional[Sequence[str]] = None,
    C: float = 1.0,
    max_iter: int = 200,
    x_val: Optional[np.ndarray] = None,
    y_val: Optional[np.ndarray] = None,
    eval_every: int = 1,
) -> SignCropClassifier:
    """Standardize features and fit a class-weighted linear SVM.

    ``sklearn.svm.LinearSVC`` (liblinear) hangs on this environment
    (Python 3.13 / sklearn 1.6). SGD with a hinge loss is the same
    linear SVM. One epoch is logged per step for the convergence plot.
    """
    del C  # SGD uses ``alpha``; keep the name so train_classify --C still parses.
    names = list(class_names) if class_names is not None else list(CLASSIFY_CLASSES)
    x = np.asarray(x, dtype=np.float32)
    y = np.asarray(y, dtype=np.int64)
    labels = np.arange(len(names), dtype=np.int64)
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)
    x_val_scaled = None
    y_val_arr = None
    if x_val is not None and y_val is not None and len(y_val):
        x_val_scaled = scaler.transform(np.asarray(x_val, dtype=np.float32))
        y_val_arr = np.asarray(y_val, dtype=np.int64)

    svm = SGDClassifier(
        loss="hinge",
        class_weight="balanced",
        max_iter=1,
        tol=None,
        warm_start=True,
        shuffle=True,
        random_state=0,
    )
    history: List[Dict[str, float]] = []
    n_epochs = max(1, int(max_iter))
    every = max(1, int(eval_every))
    for epoch in range(1, n_epochs + 1):
        svm.fit(x_scaled, y)
        if epoch == 1 or epoch % every == 0 or epoch == n_epochs:
            train_scores = svm.decision_function(x_scaled)
            train_pred = svm.predict(x_scaled)
            row: Dict[str, float] = {
                "epoch": float(epoch),
                "train_error": _error_rate(y, train_pred),
                "train_hinge": _multiclass_hinge(y, train_scores, labels),
            }
            if x_val_scaled is not None and y_val_arr is not None:
                val_scores = svm.decision_function(x_val_scaled)
                val_pred = svm.predict(x_val_scaled)
                row["val_error"] = _error_rate(y_val_arr, val_pred)
                row["val_hinge"] = _multiclass_hinge(y_val_arr, val_scores, labels)
            history.append(row)
            if epoch == 1 or epoch % max(every, 10) == 0 or epoch == n_epochs:
                extra = ""
                if "val_error" in row:
                    extra = f"  val_err={row['val_error']:.3f}"
                print(
                    f"  epoch {epoch}/{n_epochs}  train_err={row['train_error']:.3f}"
                    f"{extra}",
                    flush=True,
                )

    pipeline = Pipeline([("scale", scaler), ("svm", svm)])
    return SignCropClassifier(pipeline=pipeline, class_names=names, history=history)


def confusion_counts(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_classes: int,
) -> np.ndarray:
    matrix = np.zeros((n_classes, n_classes), dtype=np.int64)
    for t, p in zip(y_true.tolist(), y_pred.tolist()):
        if 0 <= t < n_classes and 0 <= p < n_classes:
            matrix[t, p] += 1
    return matrix


def per_class_metrics(matrix: np.ndarray) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for k in range(matrix.shape[0]):
        tp = float(matrix[k, k])
        support = float(matrix[k, :].sum())
        pred = float(matrix[:, k].sum())
        recall = tp / support if support else 0.0
        precision = tp / pred if pred else 0.0
        f1 = (
            2.0 * precision * recall / (precision + recall)
            if (precision + recall)
            else 0.0
        )
        rows.append(
            {
                "class_id": k,
                "support": int(support),
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }
        )
    return rows
