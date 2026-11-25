"""Full-pipeline eval: ROI proposals + crop classifier.

Default stage 1 is the per-shape ACF bank (``acf_bank.pkl``), falling back
to a single ``BoostedAcf``. ``--detector roi`` uses MSER/HSV ``propose_rois``.
Each ROI is letterboxed and labeled by the crop SVM. Background predictions
are dropped. Remaining boxes are matched to GT at IoU 0.5 (class-agnostic).

    python eval_pipeline.py
    python eval_pipeline.py --detector acf --acf acf_bank.pkl --split test
    python eval_pipeline.py --detector roi --max-boxes 40
"""

from __future__ import annotations

import argparse
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np

from acf import clahe_bgr
from acf_boost import AcfBank, BoostedAcf, load_acf_stage
from classify import (
    SignCropClassifier,
    confusion_counts,
    crops_to_features,
    per_class_metrics,
)
from lisa_coco import (
    CLASSIFY_CLASSES,
    classify_id_for_logical,
    iou,
    letterbox_crop,
    load_splits,
)
from roi import propose_rois
from train_classify import plot_confusion


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate detector + crop classifier on LISA.")
    p.add_argument("--detector", type=str, default="acf", choices=("acf", "roi"))
    p.add_argument("--acf", type=str, default="acf_bank.pkl")
    p.add_argument("--clf", type=str, default="classify_svm.pkl")
    p.add_argument("--split", type=str, default="test", choices=("train", "valid", "test"))
    p.add_argument("--max-images", type=int, default=None)
    p.add_argument("--size", type=int, default=64)
    p.add_argument("--pad", type=float, default=1.4)
    p.add_argument("--iou", type=float, default=0.5)
    p.add_argument(
        "--max-boxes",
        type=int,
        default=None,
        help="Default 40 for ACF bank and MSER/HSV proposals.",
    )
    p.add_argument("--out-dir", type=str, default=None)
    return p.parse_args()


def greedy_match_pairs(
    pred_boxes: Sequence[Sequence[float]],
    gt_boxes: Sequence[Sequence[float]],
    iou_thresh: float = 0.5,
) -> Tuple[List[Tuple[int, int, float]], List[int], List[int]]:
    """Score-ordered greedy matches. Returns (pairs, unmatched_pred, unmatched_gt)."""
    assigned = set()
    matches: List[Tuple[int, int, float]] = []
    unmatched_pred: List[int] = []
    for i, pred in enumerate(pred_boxes):
        best_j = -1
        best_iou = 0.0
        for j, gt in enumerate(gt_boxes):
            if j in assigned:
                continue
            overlap = iou(pred, gt)
            if overlap > best_iou:
                best_iou = overlap
                best_j = j
        if best_j >= 0 and best_iou >= iou_thresh:
            assigned.add(best_j)
            matches.append((i, best_j, float(best_iou)))
        else:
            unmatched_pred.append(i)
    unmatched_gt = [j for j in range(len(gt_boxes)) if j not in assigned]
    return matches, unmatched_pred, unmatched_gt


def _resolve(path: str, out_dir: str) -> str:
    return path if os.path.isabs(path) else os.path.join(out_dir, path)


def plot_iou_hist(ious: Sequence[float], path: str, iou_thresh: float) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    values = np.asarray(list(ious), dtype=np.float64)
    if values.size:
        ax.hist(values, bins=20, range=(0.0, 1.0), color="#3b6ea5", edgecolor="white")
        ax.axvline(iou_thresh, color="#c0392b", ls="--", label=f"match thresh {iou_thresh:.2f}")
        ax.axvline(float(np.mean(values)), color="#27ae60", ls=":", label=f"mean {np.mean(values):.3f}")
        ax.legend(frameon=False, fontsize=8)
    ax.set_xlabel("IoU of matched boxes")
    ax.set_ylabel("Count")
    ax.set_title(f"Pipeline matched IoU  (n={values.size})")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _propose_boxes(
    detector: str,
    bgr: np.ndarray,
    acf_model: Optional[Union[AcfBank, BoostedAcf]],
    max_boxes: int,
) -> List[Tuple[float, float, float, float]]:
    if detector == "roi":
        boxes, _scores = propose_rois(bgr, max_boxes=int(max_boxes))
        return boxes
    if acf_model is None:
        raise ValueError("ACF detector requested but model is missing")
    boxes, _scores = acf_model.propose(bgr, max_boxes=int(max_boxes), stride=4)
    return boxes


def evaluate_detection_classification(
    clf_path: str,
    acf_path: Optional[str] = None,
    detector: str = "acf",
    split: str = "test",
    max_images: Optional[int] = None,
    size: int = 64,
    pad: float = 1.4,
    iou_thresh: float = 0.5,
    max_boxes: Optional[int] = None,
    out_dir: Optional[str] = None,
) -> Dict[str, Any]:
    if out_dir is None:
        out_dir = os.path.dirname(os.path.abspath(__file__))
    detector = detector.lower()
    if max_boxes is None:
        max_boxes = 40
    acf_model = None
    if detector == "acf":
        if not acf_path:
            raise ValueError("acf_path is required when detector='acf'")
        acf_model = load_acf_stage(acf_path)
    classifier = SignCropClassifier.load(clf_path)
    dataset = load_splits()[split]
    n_images = len(dataset) if max_images is None else min(int(max_images), len(dataset))
    n_classes = len(CLASSIFY_CLASSES)
    y_true: List[int] = []
    y_pred: List[int] = []
    matched_ious: List[float] = []
    n_gt = 0
    n_pred_sign = 0
    n_loc_tp = 0
    n_cls_tp = 0
    stage = "propose_rois" if detector == "roi" else os.path.basename(acf_path or "acf")

    print(
        f"pipeline {split}: {n_images} images  "
        f"detector={detector} ({stage})  clf={os.path.basename(clf_path)}",
        flush=True,
    )
    for index in range(n_images):
        bgr, anns, _image_id = dataset.get(index)
        n_gt += len(anns)
        boxes = _propose_boxes(detector, bgr, acf_model, int(max_boxes))
        if not boxes:
            for ann in anns:
                y_true.append(classify_id_for_logical(int(ann["logical_id"])))
                y_pred.append(0)
            if (index + 1) % 25 == 0 or index + 1 == n_images:
                print(f"  {index + 1}/{n_images} images", flush=True)
            continue

        frame = clahe_bgr(bgr)
        crops = [letterbox_crop(frame, box, pad=pad, size=size) for box in boxes]
        features = crops_to_features(crops, verbose=False)
        pred_ids = np.asarray(classifier.pipeline.predict(features), dtype=np.int64)

        keep = [i for i, cid in enumerate(pred_ids.tolist()) if cid != 0]
        keep_boxes = [boxes[i] for i in keep]
        keep_cls = [int(pred_ids[i]) for i in keep]
        n_pred_sign += len(keep_boxes)

        gt_boxes = [ann["bbox"] for ann in anns]
        gt_cls = [classify_id_for_logical(int(ann["logical_id"])) for ann in anns]
        matches, unmatched_pred, unmatched_gt = greedy_match_pairs(
            keep_boxes, gt_boxes, iou_thresh=iou_thresh
        )
        n_loc_tp += len(matches)
        for pi, gi, overlap in matches:
            matched_ious.append(overlap)
            y_true.append(gt_cls[gi])
            y_pred.append(keep_cls[pi])
            if keep_cls[pi] == gt_cls[gi]:
                n_cls_tp += 1
        for pi in unmatched_pred:
            y_true.append(0)
            y_pred.append(keep_cls[pi])
        for gi in unmatched_gt:
            y_true.append(gt_cls[gi])
            y_pred.append(0)
        if (index + 1) % 25 == 0 or index + 1 == n_images:
            print(f"  {index + 1}/{n_images} images", flush=True)

    y_true_arr = np.asarray(y_true, dtype=np.int64)
    y_pred_arr = np.asarray(y_pred, dtype=np.int64)
    matrix = confusion_counts(y_true_arr, y_pred_arr, n_classes=n_classes)
    metrics = per_class_metrics(matrix)
    for row, name in zip(metrics, CLASSIFY_CLASSES):
        row["name"] = name
    mean_iou = float(np.mean(matched_ious)) if matched_ious else 0.0
    loc_recall = n_loc_tp / max(n_gt, 1)
    loc_precision = n_loc_tp / max(n_pred_sign, 1)
    cls_recall = n_cls_tp / max(n_gt, 1)
    cls_precision = n_cls_tp / max(n_pred_sign, 1)
    matched_acc = n_cls_tp / max(n_loc_tp, 1)

    print(
        f"\n{split} pipeline  "
        f"loc recall={loc_recall:.3f}  loc precision={loc_precision:.3f}  "
        f"mean IoU={mean_iou:.3f}  (n_match={len(matched_ious)})"
    )
    print(
        f"  class-aware  recall={cls_recall:.3f}  precision={cls_precision:.3f}  "
        f"acc on matched boxes={matched_acc:.3f}"
    )
    print(f"{'class':<20} {'n':>5} {'P':>6} {'R':>6} {'F1':>6}")
    for row in metrics:
        print(
            f"{row['name']:<20} {row['support']:5d} "
            f"{row['precision']:6.3f} {row['recall']:6.3f} {row['f1']:6.3f}"
        )

    confusion_path = os.path.join(out_dir, f"pipeline_confusion_{split}.png")
    iou_path = os.path.join(out_dir, f"pipeline_iou_{split}.png")
    plot_confusion(
        {
            split: {
                "accuracy": matched_acc,
                "confusion": matrix,
            }
        },
        CLASSIFY_CLASSES,
        confusion_path,
        title=f"{detector} + classifier confusion (misses → background column)",
    )
    plot_iou_hist(matched_ious, iou_path, iou_thresh)
    print(f"wrote {confusion_path}")
    print(f"wrote {iou_path}")
    return {
        "split": split,
        "n_images": n_images,
        "n_gt": n_gt,
        "n_pred_sign": n_pred_sign,
        "loc_recall": loc_recall,
        "loc_precision": loc_precision,
        "cls_recall": cls_recall,
        "cls_precision": cls_precision,
        "matched_acc": matched_acc,
        "mean_iou": mean_iou,
        "confusion": matrix,
        "per_class": metrics,
        "confusion_path": confusion_path,
        "iou_path": iou_path,
    }


def main() -> None:
    args = parse_args()
    out_dir = args.out_dir or os.path.dirname(os.path.abspath(__file__))
    acf_path = _resolve(args.acf, out_dir)
    if args.detector == "acf" and not os.path.isfile(acf_path):
        fallback = _resolve("acf_boosted.pkl", out_dir)
        if os.path.isfile(fallback):
            print(f"{acf_path} missing; using {fallback}", flush=True)
            acf_path = fallback
    evaluate_detection_classification(
        clf_path=_resolve(args.clf, out_dir),
        acf_path=acf_path,
        detector=args.detector,
        split=args.split,
        max_images=args.max_images,
        size=args.size,
        pad=args.pad,
        iou_thresh=args.iou,
        max_boxes=args.max_boxes,
        out_dir=out_dir,
    )


if __name__ == "__main__":
    main()
