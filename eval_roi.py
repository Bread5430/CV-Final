"""Proposal-only evaluation: recall@K vs LISA GT boxes.

Does not run the crop classifier. Tune stage-1 on train/valid (not test).

    python eval_roi.py
    python eval_roi.py --detector acf --acf acf_bank.pkl --split valid --ks 20,40,80
    python eval_roi.py --detector roi --split valid --max-boxes 40
"""

from __future__ import annotations

import argparse
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

from acf_boost import load_acf_stage
from eval_pipeline import greedy_match_pairs, plot_iou_hist
from lisa_coco import LOGICAL_CLASSES, load_splits
from roi import RoiConfig, propose_rois, visualize_proposals
from shape_detect import CLASS_TO_SHAPE, TEMPLATE_NAMES
from visualize_acf import _bgr_to_rgb

SIZE_BINS: Tuple[Tuple[str, float, float], ...] = (
    ("<16", 0.0, 16.0),
    ("16-32", 16.0, 32.0),
    ("32-64", 32.0, 64.0),
    (">64", 64.0, 1e9),
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate classical ROI proposals on LISA.")
    p.add_argument("--split", type=str, default="valid", choices=("train", "valid", "test"))
    p.add_argument("--detector", type=str, default="roi", choices=("roi", "acf", "acf-bank"))
    p.add_argument("--acf", type=str, default="acf_bank.pkl")
    p.add_argument("--max-images", type=int, default=None)
    p.add_argument("--max-boxes", type=int, default=40)
    p.add_argument("--iou", type=float, default=0.5)
    p.add_argument("--ks", type=str, default="20,40,80")
    p.add_argument("--n-show", type=int, default=8, help="Random frames to draw.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", type=str, default=None)
    return p.parse_args()


def parse_ks(raw: str) -> List[int]:
    values = []
    for part in raw.split(","):
        part = part.strip()
        if part:
            values.append(int(part))
    return values or [20, 40, 80]


def _size_bin(box: Sequence[float]) -> str:
    side = max(float(box[2]), float(box[3]))
    for name, lo, hi in SIZE_BINS:
        if lo <= side < hi:
            return name
    return SIZE_BINS[-1][0]


def evaluate_proposals(
    split: str = "valid",
    max_images: Optional[int] = None,
    max_boxes: int = 40,
    iou_thresh: float = 0.5,
    ks: Optional[Sequence[int]] = None,
    n_show: int = 0,
    seed: int = 0,
    out_dir: Optional[str] = None,
    detector: str = "roi",
    acf_path: Optional[str] = None,
) -> Dict[str, Any]:
    if out_dir is None:
        out_dir = os.path.dirname(os.path.abspath(__file__))
    k_list = sorted({int(k) for k in (ks or (20, 40, 80))})
    budget = max(int(max_boxes), max(k_list) if k_list else int(max_boxes))
    dataset = load_splits()[split]
    n_images = len(dataset) if max_images is None else min(int(max_images), len(dataset))
    cfg = RoiConfig(max_boxes=int(budget))
    stage = None
    if detector in ("acf", "acf-bank"):
        if not acf_path:
            acf_path = os.path.join(out_dir, "acf_bank.pkl")
        if not os.path.isabs(acf_path):
            acf_path = os.path.join(out_dir, acf_path)
        if not os.path.isfile(acf_path):
            fallback = os.path.join(out_dir, "acf_boosted.pkl")
            if os.path.isfile(fallback):
                acf_path = fallback
        print(f"loading {acf_path}", flush=True)
        stage = load_acf_stage(acf_path)

    n_gt = 0
    n_pred = 0
    hits_all = 0
    hits_at: Dict[int, int] = {k: 0 for k in k_list}
    matched_ious: List[float] = []
    class_gt = {name: 0 for name in LOGICAL_CLASSES}
    class_hit = {name: 0 for name in LOGICAL_CLASSES}
    shape_gt = {name: 0 for name in TEMPLATE_NAMES}
    shape_hit = {name: 0 for name in TEMPLATE_NAMES}
    size_gt = {name: 0 for name, _lo, _hi in SIZE_BINS}
    size_hit = {name: 0 for name, _lo, _hi in SIZE_BINS}

    print(
        f"{detector} {split}: {n_images} images  max_boxes={budget}  iou={iou_thresh}",
        flush=True,
    )
    for index in range(n_images):
        bgr, anns, _image_id = dataset.get(index)
        gt_boxes = [ann["bbox"] for ann in anns]
        n_gt += len(gt_boxes)
        if stage is not None:
            boxes, scores = stage.propose(bgr, max_boxes=int(budget), stride=4)
        else:
            boxes, scores = propose_rois(bgr, config=cfg, max_boxes=int(budget))
        n_pred += len(boxes)
        if gt_boxes:
            matches, _unmatched_pred, _unmatched_gt = greedy_match_pairs(
                boxes, gt_boxes, iou_thresh=iou_thresh
            )
            hits_all += len(matches)
            matched_ious.extend(overlap for _p, _g, overlap in matches)
            matched_gt = {gi for _p, gi, _ov in matches}
            for k in k_list:
                k_boxes = boxes[:k]
                k_matches, _, _ = greedy_match_pairs(k_boxes, gt_boxes, iou_thresh=iou_thresh)
                hits_at[k] += len(k_matches)
            for gi, ann in enumerate(anns):
                name = str(ann["logical_name"])
                class_gt[name] = class_gt.get(name, 0) + 1
                hit = gi in matched_gt
                if hit:
                    class_hit[name] = class_hit.get(name, 0) + 1
                shape = CLASS_TO_SHAPE.get(name)
                if shape is not None:
                    shape_gt[shape] = shape_gt.get(shape, 0) + 1
                    if hit:
                        shape_hit[shape] = shape_hit.get(shape, 0) + 1
                bin_name = _size_bin(ann["bbox"])
                size_gt[bin_name] = size_gt.get(bin_name, 0) + 1
                if hit:
                    size_hit[bin_name] = size_hit.get(bin_name, 0) + 1
        if (index + 1) % 25 == 0 or index + 1 == n_images:
            print(f"  {index + 1}/{n_images} images", flush=True)

    mean_iou = float(np.mean(matched_ious)) if matched_ious else 0.0
    mean_prop = n_pred / max(n_images, 1)
    recall_all = hits_all / max(n_gt, 1)
    print(
        f"\n{split} {detector} proposals  recall={recall_all:.3f}  "
        f"mean IoU={mean_iou:.3f}  mean boxes/image={mean_prop:.1f}  "
        f"(n_gt={n_gt}, n_match={hits_all})"
    )
    for k in k_list:
        rec = hits_at[k] / max(n_gt, 1)
        print(f"  recall@{k:<3d} = {rec:.3f}")
    print(f"{'class':<20} {'n':>5} {'R':>6}")
    for name in LOGICAL_CLASSES:
        support = class_gt.get(name, 0)
        rec = class_hit.get(name, 0) / max(support, 1)
        print(f"{name:<20} {support:5d} {rec:6.3f}")
    print(f"{'shape':<20} {'n':>5} {'R':>6}")
    for name in TEMPLATE_NAMES:
        support = shape_gt.get(name, 0)
        rec = shape_hit.get(name, 0) / max(support, 1)
        print(f"{name:<20} {support:5d} {rec:6.3f}")
    print(f"{'size':<20} {'n':>5} {'R':>6}")
    for name, _lo, _hi in SIZE_BINS:
        support = size_gt.get(name, 0)
        rec = size_hit.get(name, 0) / max(support, 1)
        print(f"{name:<20} {support:5d} {rec:6.3f}")

    tag = detector.replace("-", "_")
    iou_path = os.path.join(out_dir, f"roi_iou_{tag}_{split}.png")
    plot_iou_hist(matched_ious, iou_path, iou_thresh)
    print(f"wrote {iou_path}")

    grid_path = None
    if n_show > 0:
        grid_path = os.path.join(out_dir, f"roi_proposals_{tag}_{split}.png")
        _save_proposal_grid(
            dataset, n_images, int(n_show), cfg, int(budget), int(seed), grid_path, stage
        )
        print(f"wrote {grid_path}")

    return {
        "split": split,
        "detector": detector,
        "n_images": n_images,
        "n_gt": n_gt,
        "n_pred": n_pred,
        "recall": recall_all,
        "recall_at": {k: hits_at[k] / max(n_gt, 1) for k in k_list},
        "mean_iou": mean_iou,
        "mean_boxes": mean_prop,
        "iou_path": iou_path,
        "grid_path": grid_path,
    }


def _save_proposal_grid(
    dataset,
    n_images: int,
    n_show: int,
    cfg: RoiConfig,
    max_boxes: int,
    seed: int,
    path: str,
    stage=None,
) -> None:
    rng = np.random.default_rng(seed)
    count = min(int(n_show), n_images)
    indices = rng.choice(n_images, size=count, replace=False)
    cols = min(4, count)
    rows = int(np.ceil(count / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 3.4 * rows))
    axes_list = np.atleast_1d(axes).ravel()
    for ax, index in zip(axes_list, indices):
        bgr, anns, _ = dataset.get(int(index))
        if stage is not None:
            boxes, scores = stage.propose(bgr, max_boxes=max_boxes, stride=4)
        else:
            boxes, scores = propose_rois(bgr, config=cfg, max_boxes=max_boxes)
        vis = visualize_proposals(bgr, boxes, scores)
        ax.imshow(_bgr_to_rgb(vis))
        for ann in anns:
            x, y, w, h = ann["bbox"]
            ax.add_patch(
                Rectangle((x, y), w, h, fill=False, edgecolor="#ff3355", linewidth=1.4)
            )
        name = dataset.image_records()[int(index)].get("file_name", index)
        ax.set_title(os.path.basename(str(name)), fontsize=8)
        ax.set_axis_off()
    for ax in axes_list[count:]:
        ax.set_axis_off()
    fig.suptitle("Green = proposals, red = GT", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    out_dir = args.out_dir or os.path.dirname(os.path.abspath(__file__))
    evaluate_proposals(
        split=args.split,
        max_images=args.max_images,
        max_boxes=args.max_boxes,
        iou_thresh=args.iou,
        ks=parse_ks(args.ks),
        n_show=args.n_show,
        seed=args.seed,
        out_dir=out_dir,
        detector=args.detector,
        acf_path=args.acf,
    )


if __name__ == "__main__":
    main()
