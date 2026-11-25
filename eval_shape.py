"""Evaluate color + silhouette ROI on LISA GT boxes.

Window sizes are fit on **train** ``max(w, h)`` per silhouette, then scored
on the requested split (default valid).

    python eval_shape.py
    python eval_shape.py --split valid --n-show 10 --max-images 200
"""

from __future__ import annotations

import argparse
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

from eval_pipeline import greedy_match_pairs, plot_iou_hist
from lisa_coco import LOGICAL_CLASSES, load_splits
from roi import visualize_proposals
from shape_detect import (
    ShapeConfig,
    build_templates,
    count_train_sides,
    detect_shapes,
    fit_template_windows,
    format_window_table,
)
from visualize_acf import _bgr_to_rgb


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate color+shape ROI on LISA.")
    p.add_argument("--split", type=str, default="valid", choices=("train", "valid", "test"))
    p.add_argument("--max-images", type=int, default=200)
    p.add_argument("--max-boxes", type=int, default=20)
    p.add_argument(
        "--win",
        type=str,
        default="",
        help="Optional comma-separated sizes that override train-fit windows "
        "for every silhouette (e.g. 24,36,52).",
    )
    p.add_argument("--score", type=float, default=0.25)
    p.add_argument("--iou", type=float, default=0.5)
    p.add_argument("--ks", type=str, default="5,10,20")
    p.add_argument("--n-show", type=int, default=10, help="Random frames to draw.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", type=str, default=None)
    return p.parse_args()


def parse_ks(raw: str) -> List[int]:
    values = []
    for part in raw.split(","):
        part = part.strip()
        if part:
            values.append(int(part))
    return values or [5, 10, 20]


def parse_win_override(raw: str) -> Optional[Tuple[int, ...]]:
    raw = (raw or "").strip()
    if not raw:
        return None
    sizes = tuple(int(part.strip()) for part in raw.split(",") if part.strip())
    return sizes or None


def evaluate_shapes(
    split: str = "valid",
    max_images: Optional[int] = None,
    max_boxes: int = 20,
    win_override: Optional[Sequence[int]] = None,
    score_threshold: float = 0.25,
    iou_thresh: float = 0.5,
    ks: Optional[Sequence[int]] = None,
    n_show: int = 10,
    seed: int = 0,
    out_dir: Optional[str] = None,
) -> Dict[str, Any]:
    if out_dir is None:
        out_dir = os.path.dirname(os.path.abspath(__file__))
    k_list = [int(k) for k in (ks or (5, 10, 20))]
    splits = load_splits()
    train = splits["train"]
    dataset = splits[split]
    n_images = len(dataset) if max_images is None else min(int(max_images), len(dataset))

    counts = count_train_sides(train)
    if win_override:
        from shape_detect import TEMPLATE_NAMES

        windows = {name: tuple(int(v) for v in win_override) for name in TEMPLATE_NAMES}
        print(f"window override for all silhouettes: {windows['octagon']}", flush=True)
    else:
        windows = fit_template_windows(train)
        print("train-fit windows (p25/p50/p75 of max(w,h)):", flush=True)
        print(format_window_table(windows, counts), flush=True)

    cfg = ShapeConfig(
        windows=windows,
        score_threshold=float(score_threshold),
        max_boxes=int(max_boxes),
        nms_iou=float(iou_thresh),
    )
    templates = build_templates(windows)
    strides = sorted(
        {
            max(1, int(round(cfg.stride_frac * int(w))))
            for sizes in windows.values()
            for w in sizes
        }
    )

    n_gt = 0
    n_pred = 0
    hits_all = 0
    hits_at: Dict[int, int] = {k: 0 for k in k_list}
    matched_ious: List[float] = []
    class_gt = {name: 0 for name in LOGICAL_CLASSES}
    class_hit = {name: 0 for name in LOGICAL_CLASSES}

    print(
        f"shape {split}: {n_images} images  stride=0.2W {strides}  "
        f"max_boxes={cfg.max_boxes}  score>={cfg.score_threshold}  iou={iou_thresh}",
        flush=True,
    )
    for index in range(n_images):
        bgr, anns, _image_id = dataset.get(index)
        gt_boxes = [ann["bbox"] for ann in anns]
        n_gt += len(gt_boxes)
        boxes, _scores = detect_shapes(bgr, config=cfg, templates=templates)
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
                k_matches, _, _ = greedy_match_pairs(
                    k_boxes, gt_boxes, iou_thresh=iou_thresh
                )
                hits_at[k] += len(k_matches)
            for gi, ann in enumerate(anns):
                name = str(ann["logical_name"])
                class_gt[name] = class_gt.get(name, 0) + 1
                if gi in matched_gt:
                    class_hit[name] = class_hit.get(name, 0) + 1
        if (index + 1) % 25 == 0 or index + 1 == n_images:
            print(f"  {index + 1}/{n_images} images", flush=True)

    mean_iou = float(np.mean(matched_ious)) if matched_ious else 0.0
    mean_prop = n_pred / max(n_images, 1)
    recall_all = hits_all / max(n_gt, 1)
    print(
        f"\n{split} color+shape ROI  recall={recall_all:.3f}  "
        f"mean IoU={mean_iou:.3f}  mean boxes/image={mean_prop:.1f}  "
        f"(n_gt={n_gt}, n_match={hits_all})",
        flush=True,
    )
    for k in k_list:
        rec = hits_at[k] / max(n_gt, 1)
        print(f"  recall@{k:<3d} = {rec:.3f}", flush=True)
    print(f"{'class':<20} {'n':>5} {'R':>6}", flush=True)
    for name in LOGICAL_CLASSES:
        support = class_gt.get(name, 0)
        rec = class_hit.get(name, 0) / max(support, 1)
        print(f"{name:<20} {support:5d} {rec:6.3f}", flush=True)

    iou_path = os.path.join(out_dir, f"shape_iou_{split}.png")
    plot_iou_hist(matched_ious, iou_path, iou_thresh)
    print(f"wrote {iou_path}", flush=True)

    grid_path = None
    if n_show > 0:
        grid_path = os.path.join(out_dir, f"shape_detections_{split}.png")
        _save_detection_grid(
            dataset,
            n_images,
            int(n_show),
            cfg,
            templates,
            int(seed),
            grid_path,
        )
        print(f"wrote {grid_path}", flush=True)

    return {
        "split": split,
        "n_images": n_images,
        "n_gt": n_gt,
        "n_pred": n_pred,
        "recall": recall_all,
        "recall_at": {k: hits_at[k] / max(n_gt, 1) for k in k_list},
        "mean_iou": mean_iou,
        "mean_boxes": mean_prop,
        "windows": {k: list(v) for k, v in windows.items()},
        "iou_path": iou_path,
        "grid_path": grid_path,
    }


def _save_detection_grid(
    dataset,
    n_images: int,
    n_show: int,
    cfg: ShapeConfig,
    templates,
    seed: int,
    path: str,
) -> None:
    rng = np.random.default_rng(seed)
    count = min(int(n_show), n_images)
    indices = rng.choice(n_images, size=count, replace=False)
    cols = min(5, count)
    rows = int(np.ceil(count / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4.0 * cols, 3.2 * rows))
    axes_list = np.atleast_1d(axes).ravel()
    for ax, index in zip(axes_list, indices):
        bgr, anns, _ = dataset.get(int(index))
        boxes, scores = detect_shapes(bgr, config=cfg, templates=templates)
        vis = visualize_proposals(bgr, boxes, scores, max_draw=cfg.max_boxes)
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
    fig.suptitle("Green = color+shape ROI, red = GT", fontsize=11)
    fig.tight_layout()
    directory = os.path.dirname(os.path.abspath(path)) or "."
    os.makedirs(directory, exist_ok=True)
    tmp_path = path + ".tmp.png"
    fig.savefig(tmp_path, dpi=120)
    plt.close(fig)
    try:
        os.replace(tmp_path, path)
    except OSError:
        alt = os.path.splitext(path)[0] + "_grid.png"
        os.replace(tmp_path, alt)
        print(f"  could not overwrite {path}; wrote {alt}", flush=True)


def main() -> None:
    args = parse_args()
    out_dir = args.out_dir or os.path.dirname(os.path.abspath(__file__))
    evaluate_shapes(
        split=args.split,
        max_images=args.max_images,
        max_boxes=args.max_boxes,
        win_override=parse_win_override(args.win),
        score_threshold=args.score,
        iou_thresh=args.iou,
        ks=parse_ks(args.ks),
        n_show=args.n_show,
        seed=args.seed,
        out_dir=out_dir,
    )


if __name__ == "__main__":
    main()
