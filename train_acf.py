"""Train boosted ACF detectors on LISA (per-silhouette bank by default).

Logical names only choose which specialist sees which GT boxes. Inference
is still class-agnostic: ``AcfBank`` unions boxes and NMS.

Pipeline
--------
1. Fit ``(win_w, win_h)`` per US silhouette from train GT aspect / height.
2. Native-aspect ACF crops as positives; cell slices as negatives.
3. RealBoost + hard-negative bootstrap on a dense real scale pyramid.
4. Write ``acf_bank.pkl`` (five specialists) plus per-shape convergence plots.

    python train_acf.py
    python train_acf.py --eval-only --n-show 10
    python train_acf.py --no-bank --win 16     # legacy single square model
    python train_acf.py --recollect
"""

from __future__ import annotations

import argparse
import json
import os
import textwrap
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

from acf_boost import (
    AcfBank,
    BoostedAcf,
    calibrate_cascade,
    collect_windows,
    collect_windows_for_shapes,
    colorfulness,
    evaluate_detection,
    fit_realboost,
    fit_shape_cell_windows,
    greedy_match_iou,
    hard_negatives,
    load_acf_stage,
)
from lisa_coco import LISACocoDataset
from shape_detect import SHAPE_TO_CLASSES, TEMPLATE_NAMES
from visualize_acf import _bgr_to_rgb, _plain


def parse_max_features(value: str) -> Union[int, float, str]:
    lowered = value.strip().lower()
    if lowered in ("sqrt", "log2", "all"):
        return lowered
    if "." in value:
        return float(value)
    return int(value)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train RealBoost on ACF windows.")
    p.add_argument(
        "--max-images",
        type=int,
        default=0,
        help="Train images to sample; 0 uses the full color train split.",
    )
    p.add_argument("--trees", type=int, default=80, help="Trees per silhouette (Nam used 2048).")
    p.add_argument(
        "--bank",
        dest="bank",
        action="store_true",
        default=True,
        help="Train one BoostedAcf per US silhouette (default).",
    )
    p.add_argument(
        "--no-bank",
        dest="bank",
        action="store_false",
        help="Train the legacy single mixed-shape model instead of a bank.",
    )
    p.add_argument("--target-cells", type=int, default=16, help="Approx. cell area side per shape.")
    p.add_argument("--depth", type=int, default=3, help="Nam used 3–5; 2 is a stump-ish tree.")
    p.add_argument("--win", type=int, default=16, help="Aggregated cells on a side (16 → 32 px at shrink=2).")
    p.add_argument("--shrink", type=int, default=2)
    p.add_argument("--neg-per-image", type=int, default=12)
    p.add_argument("--bootstrap", type=int, default=2, help="Hard-negative rounds after the first forest.")
    p.add_argument("--val-frac", type=float, default=0.2)
    p.add_argument(
        "--max-features",
        type=parse_max_features,
        default=256,
        help="Features tried per split: sqrt, log2, an int, or a fraction in (0, 1].",
    )
    p.add_argument(
        "--row-frac",
        type=float,
        default=0.5,
        help="Fraction of negatives used to grow each tree (positives are kept).",
    )
    p.add_argument(
        "--n-bins",
        type=int,
        default=64,
        help="Quantile bins per ACF cell (candidate split thresholds).",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--eval-every",
        type=int,
        default=10,
        help="Run sliding-window detection on held-out train frames every N trees.",
    )
    p.add_argument(
        "--eval-images",
        type=int,
        default=12,
        help="Held-out train frames for image-level recall/IoU during fitting.",
    )
    p.add_argument("--out-dir", default=None)
    p.add_argument(
        "--windows-cache",
        default=None,
        help="Path to an .npz of collected windows. Default is a name derived "
        "from max-images / win / shrink / neg / seed in --out-dir.",
    )
    p.add_argument(
        "--recollect",
        action="store_true",
        help="Ignore an existing windows cache and run ACF collection again.",
    )
    p.add_argument(
        "--no-save-windows",
        action="store_true",
        help="Do not write a windows cache after collection.",
    )
    p.add_argument(
        "--cascade-recall",
        type=float,
        default=0.995,
        help="Fraction of val positives that must pass each cascade stage.",
    )
    p.add_argument(
        "--no-cascade",
        action="store_true",
        help="Skip τ calibration; every window runs every tree.",
    )
    p.add_argument("--eval-only", action="store_true", help="Load acf_boosted.pkl; skip training.")
    p.add_argument("--n-show", type=int, default=10, help="Random test images to draw.")
    return p.parse_args()


WINDOWS_CACHE_VERSION = 5


def default_windows_path(
    out_dir: str,
    args: argparse.Namespace,
    shape: str = "all",
    win_w: Optional[int] = None,
    win_h: Optional[int] = None,
) -> str:
    ww = int(args.win if win_w is None else win_w)
    hh = int(args.win if win_h is None else win_h)
    name = (
        f"acf_windows_v{WINDOWS_CACHE_VERSION}_{shape}_img{args.max_images}"
        f"_w{ww}x{hh}_sh{args.shrink}_neg{args.neg_per_image}_seed{args.seed}.npz"
    )
    return os.path.join(out_dir, name)


def _windows_meta(
    args: argparse.Namespace,
    n_train: int,
    n_det: int,
    shape: str = "all",
    win_w: Optional[int] = None,
    win_h: Optional[int] = None,
) -> Dict[str, Any]:
    ww = int(args.win if win_w is None else win_w)
    hh = int(args.win if win_h is None else win_h)
    return {
        "version": WINDOWS_CACHE_VERSION,
        "win": int(args.win),
        "win_w": ww,
        "win_h": hh,
        "shape": str(shape),
        "shrink": int(args.shrink),
        "neg_per_image": int(args.neg_per_image),
        "seed": int(args.seed),
        "max_images": int(args.max_images),
        "eval_images": int(args.eval_images),
        "jitter_per_gt": 1,
        "feat_norm": "luv_raw_grad_l2",
        "clahe": True,
        "color_only": True,
        "native_aspect": True,
        "n_train_images": int(n_train),
        "n_det_images": int(n_det),
    }


def _meta_mismatches(meta: Dict[str, Any], expected: Dict[str, Any]) -> List[str]:
    keys = (
        "version",
        "win_w",
        "win_h",
        "shape",
        "shrink",
        "neg_per_image",
        "seed",
        "max_images",
        "eval_images",
        "jitter_per_gt",
        "feat_norm",
        "clahe",
        "native_aspect",
    )
    return [k for k in keys if meta.get(k) != expected.get(k)]


def save_windows(
    path: str,
    x: np.ndarray,
    y: np.ndarray,
    train_idx: np.ndarray,
    det_idx: np.ndarray,
    args: argparse.Namespace,
    shape: str = "all",
    win_w: Optional[int] = None,
    win_h: Optional[int] = None,
) -> str:
    """Write collected windows + the image index split that produced them."""
    meta = _windows_meta(
        args, len(train_idx), len(det_idx), shape=shape, win_w=win_w, win_h=win_h
    )
    payload = np.frombuffer(json.dumps(meta).encode("utf-8"), dtype=np.uint8)
    tmp = f"{path}.{os.getpid()}.tmp.npz"
    np.savez_compressed(
        tmp,
        x=np.asarray(x, dtype=np.float32),
        y=np.asarray(y, dtype=np.float64),
        train_idx=np.asarray(train_idx, dtype=np.int32),
        det_idx=np.asarray(det_idx, dtype=np.int32),
        meta=payload,
    )
    try:
        os.replace(tmp, path)
        return path
    except OSError as exc:
        root, ext = os.path.splitext(path)
        alt = f"{root}_{time.strftime('%H%M%S')}{ext}"
        os.replace(tmp, alt)
        print(f"  (could not overwrite {os.path.basename(path)}: {exc})", flush=True)
        return alt


def load_windows(
    path: str,
    args: argparse.Namespace,
    shape: str = "all",
    win_w: Optional[int] = None,
    win_h: Optional[int] = None,
) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """Return ``(x, y, train_idx, det_idx)`` if ``path`` matches this run's args."""
    if not os.path.isfile(path):
        return None
    ww = int(args.win if win_w is None else win_w)
    hh = int(args.win if win_h is None else win_h)
    with np.load(path, allow_pickle=False) as data:
        if "meta" not in data or "x" not in data or "y" not in data:
            print(f"  windows cache {path} is missing fields; collecting again", flush=True)
            return None
        meta = json.loads(bytes(data["meta"]).decode("utf-8"))
        expected = _windows_meta(
            args,
            int(data["train_idx"].shape[0]) if "train_idx" in data else 0,
            int(data["det_idx"].shape[0]) if "det_idx" in data else 0,
            shape=shape,
            win_w=ww,
            win_h=hh,
        )
        bad = _meta_mismatches(meta, expected)
        if bad:
            print(
                f"  windows cache {os.path.basename(path)} does not match "
                f"{', '.join(bad)}; collecting again",
                flush=True,
            )
            return None
        x = np.asarray(data["x"], dtype=np.float32)
        y = np.asarray(data["y"], dtype=np.float64)
        train_idx = np.asarray(data["train_idx"], dtype=np.int32)
        det_idx = np.asarray(data["det_idx"], dtype=np.int32)
    n_feat = int(ww * hh * 10)
    if x.ndim != 2 or x.shape[1] != n_feat:
        print(
            f"  windows cache feature dim {x.shape} != (N, {n_feat}); collecting again",
            flush=True,
        )
        return None
    return x, y, train_idx, det_idx


def split_windows(
    x: np.ndarray,
    y: np.ndarray,
    rng: np.random.Generator,
    val_frac: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Stratified window split so val keeps both signs and background."""
    def _split(idx: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        rng.shuffle(idx)
        n_val = max(1, int(round(len(idx) * val_frac))) if len(idx) > 4 else 0
        return idx[n_val:], idx[:n_val]

    pos = np.flatnonzero(y > 0)
    neg = np.flatnonzero(y < 0)
    tr_p, va_p = _split(pos)
    tr_n, va_n = _split(neg)
    train_idx = np.concatenate([tr_p, tr_n]) if len(va_p) + len(va_n) else np.arange(len(y))
    val_idx = np.concatenate([va_p, va_n]) if len(va_p) + len(va_n) else np.zeros((0,), dtype=int)
    rng.shuffle(train_idx)
    return x[train_idx], y[train_idx], x[val_idx], y[val_idx]


def _style_axis(ax) -> None:
    ax.grid(True, alpha=0.25)
    ax.tick_params(colors="#b0bec5")
    for spine in ax.spines.values():
        spine.set_color("#444")
    ax.set_facecolor("#1a1a1a")


def save_convergence_curve(history: list, path: str) -> None:
    if not history:
        return
    trees = [row["tree"] for row in history]
    det_rows = [row for row in history if "det_recall" in row]
    n_axes = 3 if det_rows else 2
    fig, axes = plt.subplots(1, n_axes, figsize=(5.6 * n_axes, 4.2), facecolor="#111111")
    axes = np.atleast_1d(axes)
    fig.suptitle("RealBoost convergence", color="white", fontsize=13, fontweight="bold")
    starts = sorted({int(row.get("fit_start", 0)) for row in history if row.get("fit_start", 0) > 0})

    ax = axes[0]
    ax.plot(trees, [row["train_error"] for row in history], color="#80cbc4", label="train error")
    if "val_error" in history[0]:
        ax.plot(trees, [row["val_error"] for row in history], color="#ffd54f", label="val error")
    ax.set_xlabel("trees", color="#cfd8dc")
    ax.set_ylabel("misclassification", color="#cfd8dc")
    ax.set_title("window error  (sign(F) vs ±1)", color="white", fontsize=10)
    ax.legend(facecolor="#222", edgecolor="#444", labelcolor="white")
    ax.set_ylim(0, 1)
    for i, start in enumerate(starts):
        for axis in axes:
            axis.axvline(start, color="#ef9a9a", ls=":", alpha=0.8)
        ax.text(
            start,
            0.92,
            f"hard-neg {i + 1}",
            color="#ef9a9a",
            fontsize=7,
            rotation=90,
            va="top",
        )
    _style_axis(ax)

    ax = axes[1]
    ax.plot(trees, [row["train_exp_loss"] for row in history], color="#80cbc4", label="train exp-loss")
    if "val_exp_loss" in history[0]:
        ax.plot(trees, [row["val_exp_loss"] for row in history], color="#ffd54f", label="val exp-loss")
    ax.set_yscale("log")
    ax.plot(trees, [row["train_pos_recall"] for row in history], color="#ef9a9a", ls="--", label="train pos recall")
    ax.plot(trees, [row["train_neg_reject"] for row in history], color="#90caf9", ls="--", label="train neg reject")
    ax.set_xlabel("trees", color="#cfd8dc")
    ax.set_title("window RealBoost loss (not detection)", color="white", fontsize=10)
    ax.legend(facecolor="#222", edgecolor="#444", labelcolor="white", fontsize=8)
    _style_axis(ax)

    if det_rows:
        ax = axes[2]
        dt = [row["tree"] for row in det_rows]
        ax.plot(dt, [row["det_recall"] for row in det_rows], "o-", color="#80cbc4", label="recall @0.5 IoU")
        ax.plot(dt, [row["det_precision"] for row in det_rows], "o-", color="#ffd54f", label="precision")
        ax.plot(dt, [row["det_mean_iou"] for row in det_rows], "o--", color="#ef9a9a", label="mean matched IoU")
        ax.set_xlabel("trees", color="#cfd8dc")
        ax.set_ylim(0, 1)
        ax.set_title("image-level detection (held-out train frames)", color="white", fontsize=10)
        ax.legend(facecolor="#222", edgecolor="#444", labelcolor="white", fontsize=8)
        _style_axis(ax)

    fig.tight_layout(rect=(0, 0.03, 1, 0.92))
    fig.savefig(path, dpi=130, facecolor=fig.get_facecolor())
    plt.close(fig)


def sample_test_frames(
    dataset: LISACocoDataset,
    n: int,
    rng: np.random.Generator,
    min_box: int = 20,
    min_color: float = 16.0,
) -> List[Tuple[np.ndarray, list, int]]:
    order = rng.permutation(len(dataset))
    picked: List[Tuple[np.ndarray, list, int]] = []
    for index in order:
        bgr, anns, image_id = dataset.get(int(index))
        if not anns:
            continue
        if max(a["bbox"][2] * a["bbox"][3] for a in anns) < min_box * min_box:
            continue
        if colorfulness(bgr) < min_color:
            continue
        picked.append((bgr, anns, image_id))
        if len(picked) >= n:
            break
    if len(picked) < n:
        raise RuntimeError(f"only found {len(picked)} usable test frames")
    return picked


def save_png(path: str, writer: Callable[[str], None]) -> str:
    """Write a PNG, falling back if Windows has the previous figure locked.

    Cursor/VS Code keep an open handle on images shown in the editor.
    Overwriting that path raises ``OSError: [Errno 22] Invalid argument``.
    The fallback uses a timestamp so the new file is not the locked one.
    """
    try:
        writer(path)
        return path
    except OSError as exc:
        root, ext = os.path.splitext(path)
        alt = f"{root}_{time.strftime('%H%M%S')}{ext}"
        writer(alt)
        print(f"  (could not overwrite {os.path.basename(path)}: {exc})", flush=True)
        return alt


def save_roi_grid(
    examples: List[Tuple[np.ndarray, list, List[tuple], List[float], int]],
    path: str,
    title: str,
) -> None:
    n = len(examples)
    cols = 5
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(16.5, 3.4 * rows), facecolor="#111111")
    fig.suptitle(
        title,
        color="white",
        fontsize=14,
        fontweight="bold",
        y=0.98,
    )
    axes_flat = np.atleast_1d(axes).ravel()
    for ax, (bgr, anns, boxes, scores, image_id) in zip(axes_flat, examples):
        ax.imshow(_bgr_to_rgb(bgr))
        for ann in anns:
            x, y, w, h = ann["bbox"]
            ax.add_patch(Rectangle((x, y), w, h, fill=False, edgecolor="#00e5ff", linewidth=1.5))
        for box, score in zip(boxes, scores):
            x, y, w, h = box
            ax.add_patch(Rectangle((x, y), w, h, fill=False, edgecolor="#ffeb3b", linewidth=1.5))
            ax.text(x, max(0, y - 2), f"{score:.1f}", color="#ffeb3b", fontsize=6, va="bottom")
        ax.set_title(f"id {image_id}", color="#cfd8dc", fontsize=8)
        _plain(ax)
    for ax in axes_flat[n:]:
        ax.set_visible(False)

    fig.text(
        0.04,
        0.015,
        textwrap.dedent(
            """
            Cyan = ground truth. Yellow = detector ROI after the scale pyramid + NMS.
            Box size comes from which pyramid level fired (scale = model_px / sign_px).
            """
        ).strip(),
        color="#b0bec5",
        fontsize=8.5,
    )
    fig.tight_layout(rect=(0, 0.05, 1, 0.94))
    fig.savefig(path, dpi=130, facecolor=fig.get_facecolor())
    plt.close(fig)


def _split_train_indices(
    n_images: int,
    args: argparse.Namespace,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    n = n_images if int(args.max_images) <= 0 else min(int(args.max_images), n_images)
    indices = rng.choice(n_images, size=n, replace=False)
    n_eval = min(int(args.eval_images), max(0, n - 8))
    det_idx = np.asarray(indices[:n_eval], dtype=np.int32)
    train_idx = np.asarray(indices[n_eval:], dtype=np.int32)
    return train_idx, det_idx


def train_one_model(
    args: argparse.Namespace,
    rng: np.random.Generator,
    dataset: LISACocoDataset,
    windows_path: str,
    train_idx: np.ndarray,
    det_idx: np.ndarray,
    win_w: int,
    win_h: int,
    shape: str = "all",
    logical_names: Optional[Sequence[str]] = None,
    min_object_px: float = 12.0,
    max_object_px: float = 220.0,
) -> BoostedAcf:
    cached = None if args.recollect else load_windows(
        windows_path, args, shape=shape, win_w=win_w, win_h=win_h
    )
    if cached is not None:
        x, y, cached_train, cached_det = cached
        train_idx = cached_train
        det_idx = cached_det
        print(
            f"loaded windows from {windows_path}  "
            f"({x.shape[0]} windows, dim={x.shape[1]}, "
            f"{len(train_idx)} train images, {len(det_idx)} det-eval)",
            flush=True,
        )
    else:
        print(
            f"collecting {shape} windows from {len(train_idx)} train images "
            f"({win_w}x{win_h} cells; {len(det_idx)} det-eval frames)...",
            flush=True,
        )
        x, y = collect_windows(
            dataset,
            train_idx,
            shrink=args.shrink,
            win_h=int(win_h),
            win_w=int(win_w),
            negatives_per_image=args.neg_per_image,
            rng=rng,
            logical_names=logical_names,
        )
        print(
            f"  {int(np.sum(y > 0))} positives, {int(np.sum(y < 0))} negatives, "
            f"dim={x.shape[1]}",
            flush=True,
        )
        if not args.no_save_windows:
            saved = save_windows(
                windows_path, x, y, train_idx, det_idx, args,
                shape=shape, win_w=win_w, win_h=win_h,
            )
            print(f"  wrote windows cache {saved}", flush=True)

    return _fit_from_windows(
        args,
        rng,
        dataset,
        x,
        y,
        train_idx,
        det_idx,
        win_w=win_w,
        win_h=win_h,
        shape=shape,
        logical_names=logical_names,
        min_object_px=min_object_px,
        max_object_px=max_object_px,
    )


def _fit_from_windows(
    args: argparse.Namespace,
    rng: np.random.Generator,
    dataset: LISACocoDataset,
    x: np.ndarray,
    y: np.ndarray,
    train_idx: np.ndarray,
    det_idx: np.ndarray,
    win_w: int,
    win_h: int,
    shape: str = "all",
    logical_names: Optional[Sequence[str]] = None,
    min_object_px: float = 12.0,
    max_object_px: float = 220.0,
) -> BoostedAcf:
    allowed = None if logical_names is None else {str(n) for n in logical_names}
    det_frames: List[Tuple[np.ndarray, list, int]] = []
    for index in det_idx:
        bgr, anns, image_id = dataset.get(int(index))
        if allowed is not None:
            anns = [a for a in anns if str(a.get("logical_name")) in allowed]
        if not anns or colorfulness(bgr) < 8.0:
            continue
        det_frames.append((bgr, anns, image_id))
    split_rng = np.random.default_rng(int(args.seed) + 1)
    x_tr, y_tr, x_val, y_val = split_windows(x, y, split_rng, args.val_frac)
    print(f"  train {len(y_tr)}  val {len(y_val)}", flush=True)

    def det_eval(model: BoostedAcf) -> dict:
        if not det_frames:
            return {}
        print(f"  detection eval on {len(det_frames)} frames...", flush=True)
        return evaluate_detection(
            model,
            det_frames,
            stride=2,
            score_threshold=0.0,
            score_ratio=0.0,
            max_boxes=40,
            refine=False,
            fast_pyramid=True,
        )

    print(f"fitting RealBoost ({shape})...", flush=True)
    boost_kw = dict(
        depth=args.depth,
        win_h=int(win_h),
        win_w=int(win_w),
        shrink=args.shrink,
        x_val=x_val,
        y_val=y_val,
        max_features=args.max_features,
        row_frac=args.row_frac,
        n_bins=args.n_bins,
        seed=args.seed,
        eval_fn=det_eval,
        eval_every=args.eval_every,
    )
    model = fit_realboost(x_tr, y_tr, n_trees=args.trees, **boost_kw)
    model.shape = shape
    model.min_object_px = float(min_object_px)
    model.max_object_px = float(max_object_px)
    model.pyramid_step = 1.09

    for round_i in range(args.bootstrap):
        print(f"hard-negative mining round {round_i + 1} ({shape})...", flush=True)
        mine_idx = train_idx[: min(32, len(train_idx))]
        x_h, y_h = hard_negatives(model, dataset, mine_idx, per_image=10, stride=2)
        print(f"  mined {len(y_h)} hard negatives", flush=True)
        if len(y_h) == 0:
            break
        x_tr = np.concatenate([x_tr, x_h], axis=0)
        y_tr = np.concatenate([y_tr, y_h], axis=0)
        extra = max(10, args.trees // 2)
        print(f"continuing boost (+{extra} trees) on {x_tr.shape[0]} windows...", flush=True)
        model = fit_realboost(x_tr, y_tr, n_trees=extra, model=model, **boost_kw)
        model.shape = shape
        model.min_object_px = float(min_object_px)
        model.max_object_px = float(max_object_px)
        model.pyramid_step = 1.09
    if not args.no_cascade:
        x_cal, y_cal = (x_val, y_val) if len(y_val) else (x_tr, y_tr)
        n_pos = int(np.sum(y_cal > 0))
        print(
            f"calibrating soft cascade on {len(y_cal)} windows "
            f"({n_pos} positives, recall={args.cascade_recall})...",
            flush=True,
        )
        if n_pos < 2:
            print("  skip cascade: not enough positives", flush=True)
        else:
            calibrate_cascade(
                model, x_cal, y_cal, pos_recall=args.cascade_recall, linear=True
            )
    else:
        model.cascade_tau = None
    return model


def train_model(
    args: argparse.Namespace,
    rng: np.random.Generator,
    windows_path: str,
) -> BoostedAcf:
    train = LISACocoDataset(split="train", color="bgr")
    train_idx, det_idx = _split_train_indices(len(train), args, rng)
    return train_one_model(
        args,
        rng,
        train,
        windows_path,
        train_idx,
        det_idx,
        win_w=int(args.win),
        win_h=int(args.win),
        shape="all",
    )


def train_bank(
    args: argparse.Namespace,
    rng: np.random.Generator,
    out_dir: str,
) -> AcfBank:
    train = LISACocoDataset(split="train", color="bgr")
    train_idx, det_idx = _split_train_indices(len(train), args, rng)
    windows, object_range, counts = fit_shape_cell_windows(
        train, shrink=args.shrink, target_cells=int(args.target_cells)
    )
    print("shape windows (cells w×h) and train GT counts:", flush=True)
    for name in TEMPLATE_NAMES:
        ww, hh = windows[name]
        lo, hi = object_range[name]
        print(
            f"  {name:<12} {ww:2d}x{hh:<2d}  n={counts[name]:4d}  "
            f"height {lo:.0f}–{hi:.0f}px",
            flush=True,
        )
    collected: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    missing: List[str] = []
    for name in TEMPLATE_NAMES:
        if counts.get(name, 0) < 8:
            print(f"  skip {name}: only {counts.get(name, 0)} GT boxes", flush=True)
            continue
        ww, hh = windows[name]
        wpath = default_windows_path(out_dir, args, shape=name, win_w=ww, win_h=hh)
        cached = None if args.recollect else load_windows(
            wpath, args, shape=name, win_w=ww, win_h=hh
        )
        if cached is not None:
            x, y, _tr, _de = cached
            collected[name] = (x, y)
            print(
                f"  {name}: loaded {x.shape[0]} windows dim={x.shape[1]}",
                flush=True,
            )
        else:
            missing.append(name)
    if missing:
        print(
            f"collecting {len(missing)} silhouettes in one ACF pass "
            f"({len(train_idx)} images)...",
            flush=True,
        )
        names_by_shape = {n: SHAPE_TO_CLASSES[n] for n in missing}
        shape_wins = {n: windows[n] for n in missing}
        fresh = collect_windows_for_shapes(
            train,
            train_idx,
            shrink=args.shrink,
            shape_windows=shape_wins,
            names_by_shape=names_by_shape,
            negatives_per_image=args.neg_per_image,
            rng=rng,
        )
        for name, (x, y) in fresh.items():
            collected[name] = (x, y)
            ww, hh = windows[name]
            wpath = default_windows_path(out_dir, args, shape=name, win_w=ww, win_h=hh)
            print(
                f"  {name}: {int(np.sum(y > 0))} pos, {int(np.sum(y < 0))} neg, "
                f"dim={x.shape[1]}",
                flush=True,
            )
            if not args.no_save_windows:
                saved = save_windows(
                    wpath, x, y, train_idx, det_idx, args,
                    shape=name, win_w=ww, win_h=hh,
                )
                print(f"  wrote windows cache {saved}", flush=True)

    models: Dict[str, BoostedAcf] = {}
    for name in TEMPLATE_NAMES:
        if name not in collected:
            continue
        ww, hh = windows[name]
        lo, hi = object_range[name]
        wpath = default_windows_path(out_dir, args, shape=name, win_w=ww, win_h=hh)
        x, y = collected[name]
        model = _fit_from_windows(
            args,
            rng,
            train,
            x,
            y,
            train_idx,
            det_idx,
            win_w=ww,
            win_h=hh,
            shape=name,
            logical_names=SHAPE_TO_CLASSES[name],
            min_object_px=lo,
            max_object_px=hi,
        )
        models[name] = model
        curve_path = os.path.join(out_dir, f"acf_convergence_{name}.png")
        curve_path = save_png(curve_path, lambda p, m=model: save_convergence_curve(m.history, p))
        print(f"wrote {curve_path}", flush=True)
    if not models:
        raise RuntimeError("no silhouette had enough GT boxes to train")
    return AcfBank(models=models, windows=windows)


def _detector_title(stage: Union[AcfBank, BoostedAcf]) -> str:
    if isinstance(stage, AcfBank):
        n_trees = sum(m.n_trees for m in stage.models.values())
        shapes = ",".join(stage.models.keys())
        return f"ACF bank  ·  {len(stage.models)} shapes ({shapes})  ·  {n_trees} trees total"
    return (
        f"Boosted ACF ROIs  ·  {stage.n_trees} trees, depth {stage.depth}, "
        f"window {stage.win_w}×{stage.win_h} cells + scale pyramid"
    )


def main() -> None:
    args = parse_args()
    out_dir = args.out_dir or os.path.dirname(os.path.abspath(__file__))
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    bank_path = os.path.join(out_dir, "acf_bank.pkl")
    single_path = os.path.join(out_dir, "acf_boosted.pkl")
    windows_path = args.windows_cache or default_windows_path(out_dir, args)

    if args.eval_only:
        load_path = bank_path if args.bank and os.path.isfile(bank_path) else single_path
        if not os.path.isfile(load_path):
            raise FileNotFoundError(f"--eval-only needs {bank_path} or {single_path}")
        print(f"loading {load_path}", flush=True)
        stage: Union[AcfBank, BoostedAcf] = load_acf_stage(load_path)
    elif args.bank:
        stage = train_bank(args, rng, out_dir)
        stage.save(bank_path)
        print(f"wrote {bank_path}", flush=True)
    else:
        stage = train_model(args, rng, windows_path)
        stage.save(single_path)
        print(f"wrote {single_path}", flush=True)
        curve_path = os.path.join(out_dir, "acf_convergence.png")
        curve_path = save_png(curve_path, lambda p: save_convergence_curve(stage.history, p))
        print(f"wrote {curve_path}", flush=True)

    test = LISACocoDataset(split="test", color="bgr")
    frames = sample_test_frames(test, args.n_show, rng)
    examples = []
    n_gt = 0
    n_pred = 0
    tp = 0
    matched_all: List[float] = []
    for bgr, anns, image_id in frames:
        boxes, scores = stage.propose(bgr, max_boxes=40)
        hit, matched = greedy_match_iou(boxes, [a["bbox"] for a in anns], iou_thresh=0.5)
        n_gt += len(anns)
        n_pred += len(boxes)
        tp += hit
        matched_all.extend(matched)
        print(
            f"  image_id={image_id}  detections={len(boxes)}  "
            f"hits={hit}/{len(anns)}  "
            f"iou={np.round(matched, 2) if matched else []}  "
            f"scores={np.round(scores, 2)}",
            flush=True,
        )
        examples.append((bgr, anns, boxes, scores, image_id))
    print(
        f"  test snapshot  recall={tp / max(n_gt, 1):.3f}  "
        f"precision={tp / max(n_pred, 1):.3f}  "
        f"mean IoU={float(np.mean(matched_all)) if matched_all else 0.0:.3f}  "
        f"tp={tp}/{n_gt}",
        flush=True,
    )

    fig_path = os.path.join(out_dir, "acf_boosted.png")
    fig_path = save_png(fig_path, lambda p: save_roi_grid(examples, p, _detector_title(stage)))
    print(f"wrote {fig_path}", flush=True)


if __name__ == "__main__":
    main()
