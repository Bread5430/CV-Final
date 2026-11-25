"""Train the 13-way crop classifier (12 logical sign classes + background).

Uses the same ``LISACocoDataset`` as ACF. Ground-truth boxes become
letterboxed crops with ``logical_id``; easy negatives are random patches
with IoU < 0.3 against GT. Writes a confusion matrix and an SGD
convergence curve. After fitting, runs ACF + classifier eval on the
color-only test split unless ``--skip-pipeline``.

    python train_classify.py
    python train_classify.py --max-images 800 --bg-ratio 1.0
    python train_classify.py --skip-pipeline
    python eval_pipeline.py
"""

from __future__ import annotations

import argparse
import os
import time
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np

from classify import (
    SignCropClassifier,
    confusion_counts,
    crops_to_features,
    fit_linear_svm,
    per_class_metrics,
)
from lisa_coco import (
    BACKGROUND_CLASS,
    CLASSIFY_CLASSES,
    LISACocoDataset,
    classify_id_for_logical,
    load_splits,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train HOG+color SVM on logical LISA crops.")
    p.add_argument("--max-images", type=int, default=None, help="Cap images per split.")
    p.add_argument("--size", type=int, default=64)
    p.add_argument("--pad", type=float, default=1.4)
    p.add_argument("--bg-ratio", type=float, default=1.0, help="Background crops / sign crops.")
    p.add_argument("--C", type=float, default=1.0)
    p.add_argument("--epochs", type=int, default=40, help="SGD epochs (logged on the convergence plot).")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=str, default="classify_svm.pkl")
    p.add_argument(
        "--fig",
        type=str,
        default="classify_confusion.png",
        help="Crop-only confusion-matrix figure (valid + test).",
    )
    p.add_argument(
        "--curve",
        type=str,
        default="classify_convergence.png",
        help="Train/val error and hinge loss vs epoch.",
    )
    p.add_argument(
        "--skip-pipeline",
        action="store_true",
        help="Do not run ACF+classifier eval on the test split.",
    )
    p.add_argument("--acf", type=str, default="acf_boosted.pkl")
    return p.parse_args()


def _indices(dataset: LISACocoDataset, max_images: Optional[int]) -> List[int]:
    n = len(dataset)
    if max_images is None or max_images >= n:
        return list(range(n))
    return list(range(int(max_images)))


def collect_split(
    dataset: LISACocoDataset,
    indices: Sequence[int],
    size: int,
    pad: float,
    bg_ratio: float,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    crops: List[np.ndarray] = []
    labels: List[int] = []
    t0 = time.perf_counter()
    n_idx = len(indices)
    for step, (crop, logical_id, _meta) in enumerate(
        dataset.iter_sign_crops(pad=pad, size=size, clahe=True, indices=indices),
        start=1,
    ):
        crops.append(crop)
        labels.append(classify_id_for_logical(logical_id))
        if step % 500 == 0:
            print(
                f"  {dataset.split} signs {step}  {time.perf_counter() - t0:.1f}s",
                flush=True,
            )
    n_sign = len(crops)
    n_bg = int(round(n_sign * float(bg_ratio)))
    print(
        f"  {dataset.split}: {n_sign} sign crops from {n_idx} images; "
        f"sampling {n_bg} background",
        flush=True,
    )
    if n_bg > 0:
        bg = dataset.sample_background_crops(
            n=n_bg,
            size=size,
            rng=rng,
            clahe=True,
            indices=indices,
        )
        crops.extend(bg)
        labels.extend([CLASSIFY_CLASSES.index(BACKGROUND_CLASS)] * len(bg))
    print(
        f"  {dataset.split}: extracting HOG+color on {len(crops)} crops",
        flush=True,
    )
    x = crops_to_features(crops)
    y = np.asarray(labels, dtype=np.int64)
    return x, y


def _accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if y_true.size == 0:
        return 0.0
    return float(np.mean(y_true == y_pred))


def evaluate(
    model: SignCropClassifier,
    x: np.ndarray,
    y: np.ndarray,
) -> Dict[str, object]:
    y_pred = np.asarray(model.pipeline.predict(x), dtype=np.int64)
    matrix = confusion_counts(y, y_pred, n_classes=model.n_classes)
    metrics = per_class_metrics(matrix)
    for row, name in zip(metrics, model.class_names):
        row["name"] = name
    return {
        "accuracy": _accuracy(y, y_pred),
        "confusion": matrix,
        "per_class": metrics,
        "y_true": y,
        "y_pred": y_pred,
    }


def _print_split(name: str, result: Dict[str, object]) -> None:
    print(f"\n{name} accuracy: {result['accuracy']:.3f}")
    print(f"{'class':<20} {'n':>5} {'P':>6} {'R':>6} {'F1':>6}")
    for row in result["per_class"]:
        print(
            f"{row['name']:<20} {row['support']:5d} "
            f"{row['precision']:6.3f} {row['recall']:6.3f} {row['f1']:6.3f}"
        )


def plot_confusion(
    results: Dict[str, Dict[str, object]],
    class_names: Sequence[str],
    path: str,
    title: str = "Crop classifier confusion (row-normalized color, raw counts)",
) -> None:
    splits = [name for name in ("valid", "test") if name in results]
    if not splits:
        splits = list(results.keys())
    n = len(splits)
    fig, axes = plt.subplots(1, n, figsize=(6.2 * n, 5.8))
    if n == 1:
        axes = [axes]
    for ax, split in zip(axes, splits):
        matrix = np.asarray(results[split]["confusion"], dtype=np.float64)
        row_sum = matrix.sum(axis=1, keepdims=True)
        with np.errstate(divide="ignore", invalid="ignore"):
            norm = np.divide(matrix, row_sum, out=np.zeros_like(matrix), where=row_sum > 0)
        im = ax.imshow(norm, cmap="Blues", vmin=0.0, vmax=1.0)
        ax.set_title(f"{split}  acc={results[split]['accuracy']:.3f}")
        ticks = np.arange(len(class_names))
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        ax.set_xticklabels(class_names, rotation=75, ha="right", fontsize=8)
        ax.set_yticklabels(class_names, fontsize=8)
        ax.set_xlabel("predicted")
        ax.set_ylabel("true")
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                if matrix[i, j] <= 0:
                    continue
                ax.text(
                    j,
                    i,
                    int(matrix[i, j]),
                    ha="center",
                    va="center",
                    fontsize=6,
                    color="white" if norm[i, j] > 0.55 else "#222222",
                )
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def save_convergence_curve(history: Sequence[Dict[str, float]], path: str) -> None:
    if not history:
        return
    epochs = [row["epoch"] for row in history]
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.2), facecolor="#111111")
    fig.suptitle("Crop classifier convergence", color="white", fontsize=13, fontweight="bold")

    def _style(ax) -> None:
        ax.grid(True, alpha=0.25)
        ax.tick_params(colors="#b0bec5")
        for spine in ax.spines.values():
            spine.set_color("#444")
        ax.set_facecolor("#1a1a1a")

    ax = axes[0]
    ax.plot(epochs, [row["train_error"] for row in history], color="#80cbc4", label="train error")
    if "val_error" in history[0]:
        ax.plot(epochs, [row["val_error"] for row in history], color="#ffd54f", label="val error")
    ax.set_xlabel("epoch", color="#cfd8dc")
    ax.set_ylabel("misclassification", color="#cfd8dc")
    ax.set_title("crop error  (logical class + background)", color="white", fontsize=10)
    err_max = max(row.get("val_error", 0.0) for row in history)
    err_max = max(err_max, max(row["train_error"] for row in history))
    ax.set_ylim(0, max(0.15, 1.05 * err_max))
    ax.legend(facecolor="#222", edgecolor="#444", labelcolor="white")
    _style(ax)

    ax = axes[1]
    ax.plot(epochs, [row["train_hinge"] for row in history], color="#80cbc4", label="train hinge")
    if "val_hinge" in history[0]:
        ax.plot(epochs, [row["val_hinge"] for row in history], color="#ffd54f", label="val hinge")
    ax.set_xlabel("epoch", color="#cfd8dc")
    ax.set_title("multiclass hinge loss", color="white", fontsize=10)
    ax.legend(facecolor="#222", edgecolor="#444", labelcolor="white")
    _style(ax)

    fig.tight_layout(rect=(0, 0.03, 1, 0.92))
    fig.savefig(path, dpi=130, facecolor=fig.get_facecolor())
    plt.close(fig)


def _resolve(path: str, out_dir: str) -> str:
    return path if os.path.isabs(path) else os.path.join(out_dir, path)


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    datasets = load_splits()

    print("Logical class counts (train/valid/test):")
    for split, dataset in datasets.items():
        dist = dataset.logical_class_distribution()
        parts = [f"{row['name']}={row['count']}" for row in dist["classes"] if row["count"]]
        print(f"  {split}: {', '.join(parts)}")

    print("\nCollecting train crops...")
    x_train, y_train = collect_split(
        datasets["train"],
        _indices(datasets["train"], args.max_images),
        size=args.size,
        pad=args.pad,
        bg_ratio=args.bg_ratio,
        rng=rng,
    )
    print(f"train features {x_train.shape}  labels {y_train.shape}", flush=True)

    print("\nCollecting valid crops...")
    x_valid, y_valid = collect_split(
        datasets["valid"],
        _indices(datasets["valid"], args.max_images),
        size=args.size,
        pad=args.pad,
        bg_ratio=args.bg_ratio,
        rng=rng,
    )
    print(f"valid features {x_valid.shape}  labels {y_valid.shape}", flush=True)

    print("Fitting linear SVM (SGD hinge)...", flush=True)
    t0 = time.perf_counter()
    model = fit_linear_svm(
        x_train,
        y_train,
        class_names=CLASSIFY_CLASSES,
        C=args.C,
        max_iter=args.epochs,
        x_val=x_valid,
        y_val=y_valid,
    )
    print(f"fit in {time.perf_counter() - t0:.1f}s", flush=True)

    results: Dict[str, Dict[str, object]] = {
        "valid": evaluate(model, x_valid, y_valid),
    }
    _print_split("valid", results["valid"])

    print("\nCollecting test crops...")
    x_test, y_test = collect_split(
        datasets["test"],
        _indices(datasets["test"], args.max_images),
        size=args.size,
        pad=args.pad,
        bg_ratio=args.bg_ratio,
        rng=rng,
    )
    results["test"] = evaluate(model, x_test, y_test)
    _print_split("test", results["test"])

    out_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = _resolve(args.out, out_dir)
    fig_path = _resolve(args.fig, out_dir)
    curve_path = _resolve(args.curve, out_dir)
    model.save(model_path)
    plot_confusion(results, CLASSIFY_CLASSES, fig_path)
    save_convergence_curve(model.history, curve_path)
    print(f"\nwrote {model_path}")
    print(f"wrote {fig_path}")
    print(f"wrote {curve_path}")

    if not args.skip_pipeline:
        from eval_pipeline import evaluate_detection_classification

        acf_path = _resolve(args.acf, out_dir)
        if not os.path.isfile(acf_path):
            print(f"skip pipeline eval (missing {acf_path})", flush=True)
        else:
            print("\nFull ACF + classifier eval on color-only test...", flush=True)
            evaluate_detection_classification(
                clf_path=model_path,
                acf_path=acf_path,
                detector="acf",
                split="test",
                max_images=args.max_images,
                size=args.size,
                pad=args.pad,
                out_dir=out_dir,
            )


if __name__ == "__main__":
    main()
