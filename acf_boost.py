"""Boosted trees on ACF windows (Nam / Dollár RealBoost, simplified).

This is the classifier that sits on top of ``acf.py``. A window is a fixed
``win_h × win_w`` block of aggregated cells, flattened to
``win_h * win_w * 10`` numbers. Each boosting round fits an *orthogonal* histogram tree (one ACF cell
per split, thresholds from quantile bins) and writes a RealBoost value
at every leaf. Splits subsample features; trees subsample negatives.

    score(x) = sum_t  f_t(x)

Detection slides that fixed window over a **scale pyramid**, then NMS.
A 16×16-cell model is only ~32×32 px at shrink=2; without the pyramid
every predicted box is that size (~¼ of a typical LISA sign) and the
true sign is never the right template, so high scores land on random
32×32 patches. After boosting, ``calibrate_cascade`` fills ``cascade_tau``
so ``score_windows`` can drop dead background before later trees.

How the two extra pieces plug in
--------------------------------
**Soft cascade** (Bourdev–Brandt / Dollár "constant soft cascade")
    After boosting, ``calibrate_cascade`` sets ``τ[t]`` from the
    ``pos_recall``-quantile of ``F_t`` on held-out positives, then (by
    default) a linear envelope ``a + b t`` that never exceeds that quantile.
    ``score_windows`` drops a window once ``F_t(x) < τ[t]`` and does not
    revive it. Tree training is unchanged.

**Scale pyramid** (Dollár "fast feature pyramids", the rest of ACF)
    The model window is a *canonical size* (here 16×16 cells ≈ 32×32 px at
    shrink=2). A far-away sign is smaller than that; a near sign is larger.
    Test-time fix: run the *same* forest on a pyramid of aggregated maps::

        for scale in 1, 1/1.07, 1/1.07**2, ...   # ~8–10 scales per octave
            agg_s = channels of image resized by ``scale``, then aggregated
            heat_s = slide the fixed window on agg_s
            box_full = (x, y, win_w, win_h) * shrink / scale

    Computing 10 channels at every scale is expensive. ACF's trick is to
    compute channels once (or at a few octaves) and *approximate* the other
    scales by resampling the aggregated maps. The quality path used by
    ``AcfBank`` recomputes ACF at every scale (``fast_pyramid=False``).

Neither piece changes the tree learner. Cascade is an early-exit on the
score; the pyramid is the same score on resampled channels.
"""

from __future__ import annotations

import pickle
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np

from acf import (
    clahe_bgr,
    crop_aggregated_box,
    extract_acf,
    heatmap_on_image,
    resize_aggregated,
    resize_patch,
)
from lisa_coco import image_colorfulness, iou

EPS = 1e-6
MaxFeatures = Union[int, float, str, None]


def normalize_acf_window(patch: np.ndarray) -> np.ndarray:
    """Color-preserving window norm (see ``windows_to_features``)."""
    return windows_to_features(patch).reshape(patch.shape)


def windows_to_features(patches: np.ndarray) -> np.ndarray:
    """Flatten a window without destroying sign color.

    Per-channel z-score made a red plate and a tree look similar: U/V on a
    tight sign crop are almost constant, so dividing by std amplified noise,
    while busy foliage kept layout. Keep LUV (scaled), L2-normalize only the
    7 gradient channels so Sobel energy cannot drown color.
    """
    if patches.ndim == 3:
        patches = patches[None, ...]
    patches = np.asarray(patches, dtype=np.float32)
    n, h, w, _c = patches.shape
    luv = patches[..., :3] / 100.0
    grad = patches[..., 3:].reshape(n, -1)
    gn = np.linalg.norm(grad, axis=1, keepdims=True) + 1e-4
    grad = (grad / gn).reshape(n, h, w, -1)
    return np.concatenate([luv, grad], axis=3).reshape(n, -1)


def pyramid_scales(
    model_px: int,
    min_object_px: float = 12.0,
    max_object_px: float = 220.0,
    step: float = 1.09,
) -> Tuple[float, ...]:
    """Image scales so a sign of ``object_px`` fills the model window.

    ``scale = model_px / object_px``. A 120 px sign vs a 32 px window needs
    ``scale ≈ 0.27`` (shrink the frame). Using only ``scale=1`` emits 32 px
    boxes no matter how large the GT is. Default ``step=1.09`` is about
    eight scales per octave (Dollár / Nam).
    """
    scales = []
    obj = float(min_object_px)
    while obj <= max_object_px + 1e-6:
        scales.append(model_px / obj)
        obj *= step
    return tuple(sorted({round(s, 4) for s in scales}, reverse=True))


def _xywh_from_cell(
    cell_x: int,
    cell_y: int,
    win_w: int,
    win_h: int,
    shrink: int,
    scale: float = 1.0,
) -> Tuple[float, float, float, float]:
    """Aggregated top-left cell → full-image COCO box at this pyramid scale."""
    x = cell_x * shrink / scale
    y = cell_y * shrink / scale
    w = win_w * shrink / scale
    h = win_h * shrink / scale
    return (x, y, w, h)


def _local_max_mask(heat: np.ndarray) -> np.ndarray:
    """True at cells ≥ all 8 neighbors (ties keep both)."""
    if heat.size == 0:
        return np.zeros(heat.shape, dtype=bool)
    pad = np.pad(heat, 1, mode="constant", constant_values=-np.inf)
    center = pad[1:-1, 1:-1]
    mask = np.ones(heat.shape, dtype=bool)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy == 0 and dx == 0:
                continue
            mask &= center >= pad[1 + dy : heat.shape[0] + 1 + dy, 1 + dx : heat.shape[1] + 1 + dx]
    return mask


def refine_box(bgr: np.ndarray, box: Sequence[float]) -> Tuple[float, float, float, float]:
    """Shrink a square window onto saturated / bright pixels (sign plate).

    GT boxes are the painted plate, often not square. The detector window is
    always square, so IoU against a wide speed-limit sign tops out near 0.5
    even when the sign is found. A tight chroma box raises IoU without
    changing the classifier.
    """
    x, y, w, h = (float(v) for v in box[:4])
    ih, iw = bgr.shape[:2]
    x0 = max(0, int(np.floor(x)))
    y0 = max(0, int(np.floor(y)))
    x1 = min(iw, int(np.ceil(x + w)))
    y1 = min(ih, int(np.ceil(y + h)))
    if x1 - x0 < 4 or y1 - y0 < 4:
        return (x, y, w, h)
    roi = bgr[y0:y1, x0:x1]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]
    mask = ((sat >= 50) & (val >= 40)) | ((sat < 40) & (val >= 180))
    frac = float(mask.mean())
    if frac < 0.04 or frac > 0.92:
        return (x, y, w, h)
    ys, xs = np.nonzero(mask)
    rx0, rx1 = int(xs.min()), int(xs.max()) + 1
    ry0, ry1 = int(ys.min()), int(ys.max()) + 1
    pad_x = max(1, int(round(0.06 * (rx1 - rx0))))
    pad_y = max(1, int(round(0.06 * (ry1 - ry0))))
    rx0 = max(0, rx0 - pad_x)
    ry0 = max(0, ry0 - pad_y)
    rx1 = min(roi.shape[1], rx1 + pad_x)
    ry1 = min(roi.shape[0], ry1 + pad_y)
    return (float(x0 + rx0), float(y0 + ry0), float(rx1 - rx0), float(ry1 - ry0))


def nms(
    boxes: Sequence[Sequence[float]],
    scores: Sequence[float],
    iou_threshold: float = 0.3,
) -> List[int]:
    """Greedy non-max suppression; returns kept indices in score order."""
    if not boxes:
        return []
    order = np.argsort(-np.asarray(scores, dtype=np.float64))
    keep: List[int] = []
    for idx in order:
        box = boxes[int(idx)]
        if all(iou(box, boxes[k]) <= iou_threshold for k in keep):
            keep.append(int(idx))
    return keep


@dataclass
class QuantTree:
    """Axis-aligned tree: one ACF cell lookup per split, quantized thresholds.

    Splits are grown by scanning histogram bins (not every unique float).
    ``apply`` walks ``x[feature] <= threshold`` and returns leaf node ids.
    """

    feature: np.ndarray
    threshold: np.ndarray
    left: np.ndarray
    right: np.ndarray
    n_nodes: int
    depth: int

    def apply(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32)
        if x.ndim == 1:
            x = x.reshape(1, -1)
        n = x.shape[0]
        node = np.zeros(n, dtype=np.int32)
        rows = np.arange(n)
        for _ in range(self.depth + 1):
            feat = self.feature[node]
            leaf = feat < 0
            if np.all(leaf):
                break
            safe = np.where(leaf, 0, feat)
            go_left = x[rows, safe] <= self.threshold[node]
            nxt = np.where(go_left, self.left[node], self.right[node])
            node = np.where(leaf, node, nxt)
        return node


def _feature_count(n_feat: int, max_features: MaxFeatures) -> int:
    if max_features is None or max_features == "all":
        return n_feat
    if isinstance(max_features, str):
        key = max_features.lower()
        if key == "sqrt":
            return max(1, int(np.sqrt(n_feat)))
        if key == "log2":
            return max(1, int(np.log2(n_feat)))
        raise ValueError(f"unknown max_features {max_features!r}")
    if isinstance(max_features, float) and max_features <= 1.0:
        return max(1, int(round(max_features * n_feat)))
    return max(1, min(n_feat, int(max_features)))


def _quantize_features(
    x: np.ndarray,
    n_bins: int,
    rng: np.random.Generator,
    max_rows: int = 4096,
) -> Tuple[np.ndarray, np.ndarray]:
    """Equal-width bins between the 1st/99th percentiles of each cell.

    Returns ``(bins, edges)`` with ``edges`` shape ``(n_bins - 1, D)`` so a
    split ``bins <= t`` is ``x <= edges[t]`` at apply time. Quantile *range*
    (not a Python ``searchsorted`` per column) keeps this vectorized.
    """
    n, _dim = x.shape
    n_bins = max(4, int(n_bins))
    take = min(n, max_rows)
    sample = x if take == n else x[rng.choice(n, size=take, replace=False)]
    lo = np.quantile(sample, 0.01, axis=0).astype(np.float32)
    hi = np.quantile(sample, 0.99, axis=0).astype(np.float32)
    span = np.maximum(hi - lo, 1e-4)
    step = (np.arange(1, n_bins, dtype=np.float32) / float(n_bins))[:, None]
    edges = (lo[None, :] + span[None, :] * step).astype(np.float32)
    raw = (x - lo) / span * n_bins
    dtype = np.uint8 if n_bins <= 256 else np.uint16
    bins = np.clip(np.floor(raw), 0, n_bins - 1).astype(dtype)
    return bins, edges


def _subsample_rows(
    y: np.ndarray,
    frac: float,
    rng: np.random.Generator,
    keep_all_positives: bool = True,
) -> np.ndarray:
    n = int(y.shape[0])
    if frac >= 1.0 - 1e-9:
        return np.arange(n, dtype=np.int32)
    pos = np.flatnonzero(y > 0)
    neg = np.flatnonzero(y < 0)
    if keep_all_positives:
        n_neg = min(neg.size, max(1, int(round(frac * neg.size)))) if neg.size else 0
        take_neg = (
            rng.choice(neg, size=n_neg, replace=False) if n_neg < neg.size else neg
        )
        return np.concatenate([pos, take_neg]).astype(np.int32)
    n_pos = min(pos.size, max(1, int(round(frac * pos.size)))) if pos.size else 0
    n_neg = min(neg.size, max(1, int(round(frac * neg.size)))) if neg.size else 0
    take_pos = rng.choice(pos, size=n_pos, replace=False) if n_pos < pos.size else pos
    take_neg = rng.choice(neg, size=n_neg, replace=False) if n_neg < neg.size else neg
    return np.concatenate([take_pos, take_neg]).astype(np.int32)


def _best_split(
    bins: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
    samples: np.ndarray,
    feat_ids: np.ndarray,
    min_leaf: int,
    edges: np.ndarray,
    n_bins: int,
) -> Optional[Tuple[int, int, float]]:
    """RealBoost split: minimize ``2 (sqrt(W+L W−L) + sqrt(W+R W−R))``."""
    w = weights[samples]
    pos = y[samples] > 0
    w_pos = np.where(pos, w, 0.0)
    w_neg = np.where(~pos, w, 0.0)
    wp_all = float(w_pos.sum())
    wn_all = float(w_neg.sum())
    parent_z = 2.0 * np.sqrt((wp_all + EPS) * (wn_all + EPS))
    n_tot = int(samples.size)
    best_gain = 0.0
    best: Optional[Tuple[int, int, float]] = None
    for f in feat_ids:
        f = int(f)
        b = bins[samples, f]
        wp = np.bincount(b, weights=w_pos, minlength=n_bins).astype(np.float64)
        wn = np.bincount(b, weights=w_neg, minlength=n_bins).astype(np.float64)
        cnt = np.bincount(b, minlength=n_bins)
        c_left = np.cumsum(cnt)
        c_right = n_tot - c_left
        wp_left = np.cumsum(wp)
        wn_left = np.cumsum(wn)
        z = 2.0 * (
            np.sqrt((wp_left + EPS) * (wn_left + EPS))
            + np.sqrt((wp_all - wp_left + EPS) * (wn_all - wn_left + EPS))
        )
        gain = parent_z - z
        valid = (c_left >= min_leaf) & (c_right >= min_leaf)
        valid[-1] = False
        gain[~valid] = -np.inf
        t = int(np.argmax(gain))
        g = float(gain[t])
        if g > best_gain:
            best_gain = g
            best = (f, t, float(edges[t, f]))
    return best


def _fit_quant_tree(
    bins: np.ndarray,
    edges: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
    idx: np.ndarray,
    depth: int,
    min_leaf: int,
    max_features: MaxFeatures,
    rng: np.random.Generator,
) -> QuantTree:
    n_feat = bins.shape[1]
    n_bins = edges.shape[0] + 1
    k_feat = _feature_count(n_feat, max_features)
    cap = (1 << (depth + 1)) - 1
    feature = np.full(cap, -1, dtype=np.int32)
    threshold = np.zeros(cap, dtype=np.float32)
    left = np.full(cap, -1, dtype=np.int32)
    right = np.full(cap, -1, dtype=np.int32)
    members: List[Optional[np.ndarray]] = [None] * cap
    members[0] = np.asarray(idx, dtype=np.int32)
    depths = np.zeros(cap, dtype=np.int32)
    n_used = 1
    queue = [0]
    while queue:
        node = queue.pop(0)
        samples = members[node]
        if samples is None or depths[node] >= depth or samples.size < 2 * min_leaf:
            continue
        feat_ids = rng.choice(n_feat, size=k_feat, replace=False)
        split = _best_split(
            bins, y, weights, samples, feat_ids, min_leaf, edges, n_bins
        )
        if split is None or n_used + 2 > cap:
            continue
        feat, t_bin, thr = split
        left_id = n_used
        right_id = n_used + 1
        n_used += 2
        feature[node] = feat
        threshold[node] = thr
        left[node] = left_id
        right[node] = right_id
        go_left = bins[samples, feat] <= t_bin
        members[left_id] = samples[go_left]
        members[right_id] = samples[~go_left]
        depths[left_id] = depths[node] + 1
        depths[right_id] = depths[node] + 1
        queue.append(left_id)
        queue.append(right_id)
    return QuantTree(
        feature=feature[:n_used],
        threshold=threshold[:n_used],
        left=left[:n_used],
        right=right[:n_used],
        n_nodes=n_used,
        depth=depth,
    )


def _window_grad_l2(
    aggregated: np.ndarray,
    win_h: int,
    win_w: int,
    origin_y: np.ndarray,
    origin_x: np.ndarray,
) -> np.ndarray:
    """Per-window L2 of the 7 gradient channels, matching ``windows_to_features``."""
    energy = np.sum(aggregated[..., 3:] ** 2, axis=2)
    integ = np.pad(energy, ((1, 0), (1, 0)), mode="constant")
    integ = np.cumsum(np.cumsum(integ, axis=0), axis=1)
    y1 = origin_y + int(win_h)
    x1 = origin_x + int(win_w)
    ss = integ[y1, x1] - integ[origin_y, x1] - integ[y1, origin_x] + integ[origin_y, origin_x]
    return np.sqrt(np.maximum(ss, 0.0)).astype(np.float32) + 1e-4


@dataclass
class BoostedAcf:
    """RealBoost forest over a fixed-size ACF window.

    ``stages[t]`` is (QuantTree, leaf-score array).
    ``score(x) = sum of leaf values``; sign > 0 means "sign-like window".
    """

    win_h: int
    win_w: int
    shrink: int
    depth: int
    stages: List[Tuple[Any, np.ndarray]] = field(default_factory=list)
    # Soft cascade. None → evaluate every tree on every window.
    cascade_tau: Optional[np.ndarray] = None
    history: List[Dict[str, float]] = field(default_factory=list)
    max_features: MaxFeatures = 256
    row_frac: float = 0.5
    n_bins: int = 64
    shape: Optional[str] = None
    min_object_px: float = 12.0
    max_object_px: float = 220.0
    pyramid_step: float = 1.09

    @property
    def n_trees(self) -> int:
        return len(self.stages)

    @property
    def n_features(self) -> int:
        return int(self.win_h * self.win_w * 10)

    def score_windows(self, x: np.ndarray) -> np.ndarray:
        """Score a batch of flattened windows, shape (N, n_features).

        With ``cascade_tau``, a window that falls below ``τ[t]`` after tree
        ``t`` is dropped and never revived (partial sum is the returned score).
        Without ``tau``, every window gets the full sum.
        """
        x = np.asarray(x, dtype=np.float32)
        if x.ndim == 1:
            x = x.reshape(1, -1)
        scores = np.zeros(x.shape[0], dtype=np.float64)
        if self.cascade_tau is None:
            for tree, values in self.stages:
                scores += values[tree.apply(x)]
            return scores.astype(np.float32)
        alive = np.ones(x.shape[0], dtype=bool)
        for t, (tree, values) in enumerate(self.stages):
            if not np.any(alive):
                break
            idx = np.flatnonzero(alive)
            leaves = tree.apply(x[idx])
            scores[idx] += values[leaves]
            alive[idx] = scores[idx] >= float(self.cascade_tau[t])
        return scores.astype(np.float32)

    def _apply_tree_on_map(
        self,
        tree: QuantTree,
        aggregated: np.ndarray,
        origin_y: np.ndarray,
        origin_x: np.ndarray,
        grad_norm: np.ndarray,
    ) -> np.ndarray:
        """Walk ``tree`` on sliding windows without copying patches."""
        n = int(origin_y.shape[0])
        node = np.zeros(n, dtype=np.int32)
        tw = int(self.win_w)
        n_ch = 10
        for _ in range(tree.depth + 1):
            feat = tree.feature[node]
            leaf = feat < 0
            if np.all(leaf):
                break
            safe = np.where(leaf, 0, feat)
            fy = safe // (tw * n_ch)
            fx = (safe // n_ch) % tw
            fc = safe % n_ch
            vals = aggregated[origin_y + fy, origin_x + fx, fc]
            color = fc < 3
            feat_val = np.where(color, vals / 100.0, vals / grad_norm)
            go_left = feat_val <= tree.threshold[node]
            nxt = np.where(go_left, tree.left[node], tree.right[node])
            node = np.where(leaf, node, nxt)
        return node

    def score_heatmap(
        self,
        aggregated: np.ndarray,
        stride: int = 4,
        scale: float = 1.0,
    ) -> np.ndarray:
        """Slide the model window over one aggregated map (one pyramid level).

        Looks up ACF cells on the map (same layout as ``windows_to_features``)
        instead of allocating a (win, win, 10) crop per location.
        """
        del scale
        h, w, _ = aggregated.shape
        th, tw = int(self.win_h), int(self.win_w)
        if h < th or w < tw:
            return np.zeros((0, 0), dtype=np.float32)
        ys = np.arange(0, h - th + 1, stride, dtype=np.int32)
        xs = np.arange(0, w - tw + 1, stride, dtype=np.int32)
        out_h, out_w = int(ys.size), int(xs.size)
        origin_y = np.repeat(ys, out_w)
        origin_x = np.tile(xs, out_h)
        grad_norm = _window_grad_l2(aggregated, th, tw, origin_y, origin_x)
        n = int(origin_y.shape[0])
        scores = np.zeros(n, dtype=np.float64)
        if self.cascade_tau is None:
            for tree, values in self.stages:
                node = self._apply_tree_on_map(
                    tree, aggregated, origin_y, origin_x, grad_norm
                )
                scores += values[node]
        else:
            alive = np.ones(n, dtype=bool)
            for t, (tree, values) in enumerate(self.stages):
                if not np.any(alive):
                    break
                idx = np.flatnonzero(alive)
                node = self._apply_tree_on_map(
                    tree,
                    aggregated,
                    origin_y[idx],
                    origin_x[idx],
                    grad_norm[idx],
                )
                scores[idx] += values[node]
                alive[idx] = scores[idx] >= float(self.cascade_tau[t])
        return scores.astype(np.float32).reshape(out_h, out_w)

    def detect_image(
        self,
        bgr: np.ndarray,
        stride: int = 2,
        score_threshold: float = 0.0,
        nms_iou: float = 0.5,
        scales: Optional[Sequence[float]] = None,
        score_ratio: float = 0.0,
        max_boxes: int = 40,
        peaks_per_scale: int = 80,
        return_heat: bool = True,
        refine: bool = False,
        preprocess: bool = True,
        fast_pyramid: bool = False,
    ) -> Tuple[List[Tuple[float, float, float, float]], List[float], np.ndarray]:
        """Run the forest as a sliding-window detector over a scale pyramid.

        ``scale`` resizes the *image* so a sign of size ``model_px / scale``
        matches the trained window. Boxes are mapped back with
        ``(cell * shrink) / scale``. ``scales=None`` uses ``pyramid_scales``
        from this model's height and object-size range. Quality path
        recomputes ACF at every scale (``fast_pyramid=False``). Peaks are
        local maxima. After NMS, keep scores ≥ ``score_threshold``.
        ``refine`` is off: native-aspect windows already match the plate.
        """
        if preprocess:
            bgr = clahe_bgr(bgr)
        if scales is None:
            scales = self.default_scales()

        all_boxes: List[Tuple[float, float, float, float]] = []
        all_scores: List[float] = []
        heat_display = np.zeros((bgr.shape[0], bgr.shape[1]), dtype=np.float32)
        min_side = max(self.win_h, self.win_w) * self.shrink
        base_agg = None
        if fast_pyramid:
            base_agg = extract_acf(bgr, shrink=self.shrink).aggregated

        for scale in scales:
            new_w = max(1, int(round(bgr.shape[1] * scale)))
            new_h = max(1, int(round(bgr.shape[0] * scale)))
            if new_w < min_side or new_h < min_side:
                continue
            agg_h = max(1, new_h // self.shrink)
            agg_w = max(1, new_w // self.shrink)
            if agg_h < self.win_h or agg_w < self.win_w:
                continue
            if fast_pyramid:
                if abs(scale - 1.0) < 1e-6:
                    aggregated = base_agg
                else:
                    aggregated = resize_aggregated(base_agg, agg_h, agg_w)
                image_shape = (new_h, new_w)
            else:
                if abs(scale - 1.0) < 1e-6:
                    image = bgr
                else:
                    image = cv2.resize(
                        bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR
                    )
                aggregated = extract_acf(image, shrink=self.shrink).aggregated
                image_shape = image.shape[:2]
            if return_heat:
                heat = self.score_heatmap(aggregated, stride=stride, scale=scale)
                if heat.size:
                    mapped = heatmap_on_image(
                        heat,
                        image_shape,
                        self.win_h,
                        self.win_w,
                        self.shrink,
                        stride=stride,
                    )
                    if mapped.shape[:2] != (bgr.shape[0], bgr.shape[1]):
                        mapped = cv2.resize(
                            mapped,
                            (bgr.shape[1], bgr.shape[0]),
                            interpolation=cv2.INTER_LINEAR,
                        )
                    heat_display = np.maximum(heat_display, mapped)
            boxes_s, scores_s = self.peaks_on_aggregated(
                aggregated,
                scale=float(scale),
                stride=stride,
                score_threshold=score_threshold,
                peaks_per_scale=peaks_per_scale,
            )
            all_boxes.extend(boxes_s)
            all_scores.extend(scores_s)

        keep = nms(all_boxes, all_scores, iou_threshold=nms_iou)
        boxes = [all_boxes[i] for i in keep]
        scores = [all_scores[i] for i in keep]
        if scores:
            cutoff = float(score_threshold)
            if score_ratio > 0.0:
                cutoff = max(cutoff, float(score_ratio) * max(scores))
            filtered = [(b, s) for b, s in zip(boxes, scores) if s >= cutoff]
            filtered = filtered[: max(1, int(max_boxes))]
            if refine:
                filtered = [(refine_box(bgr, b), s) for b, s in filtered]
            boxes = [b for b, _s in filtered]
            scores = [s for _b, s in filtered]
        return boxes, scores, heat_display

    def default_scales(self) -> Tuple[float, ...]:
        return pyramid_scales(
            int(self.win_h * self.shrink),
            min_object_px=float(getattr(self, "min_object_px", 12.0)),
            max_object_px=float(getattr(self, "max_object_px", 220.0)),
            step=float(getattr(self, "pyramid_step", 1.09)),
        )

    def propose(
        self,
        bgr: np.ndarray,
        max_boxes: int = 40,
        **detect_kw: Any,
    ) -> Tuple[List[Tuple[float, float, float, float]], List[float]]:
        kw = dict(
            stride=2,
            score_threshold=0.0,
            score_ratio=0.0,
            max_boxes=int(max_boxes),
            refine=False,
            return_heat=False,
            fast_pyramid=False,
        )
        kw.update(detect_kw)
        boxes, scores, _heat = self.detect_image(bgr, **kw)
        return boxes, scores

    def peaks_on_aggregated(
        self,
        aggregated: np.ndarray,
        scale: float,
        stride: int = 2,
        score_threshold: float = 0.0,
        peaks_per_scale: int = 80,
    ) -> Tuple[List[Tuple[float, float, float, float]], List[float]]:
        """Local-max windows on one already-built aggregated map."""
        heat = self.score_heatmap(aggregated, stride=stride, scale=scale)
        if heat.size == 0:
            return [], []
        peaks = heat.copy()
        peaks[~_local_max_mask(heat)] = -np.inf
        flat = peaks.reshape(-1)
        finite = np.isfinite(flat)
        if not np.any(finite):
            return [], []
        k = min(int(peaks_per_scale), int(np.count_nonzero(finite)))
        peak_idx = np.argpartition(flat, -k)[-k:]
        boxes: List[Tuple[float, float, float, float]] = []
        scores: List[float] = []
        for linear in peak_idx:
            s = float(flat[int(linear)])
            if not np.isfinite(s) or s < score_threshold:
                continue
            iy, ix = np.unravel_index(int(linear), heat.shape)
            boxes.append(
                _xywh_from_cell(
                    int(ix) * stride,
                    int(iy) * stride,
                    self.win_w,
                    self.win_h,
                    self.shrink,
                    scale=scale,
                )
            )
            scores.append(s)
        return boxes, scores

    def save(self, path: str) -> None:
        with open(path, "wb") as handle:
            pickle.dump(self, handle)

    @staticmethod
    def load(path: str) -> "BoostedAcf":
        with open(path, "rb") as handle:
            model = pickle.load(handle)
        if not isinstance(model, BoostedAcf):
            raise TypeError(f"expected BoostedAcf, got {type(model)}")
        if not hasattr(model, "shape"):
            model.shape = None
        if not hasattr(model, "min_object_px"):
            model.min_object_px = 12.0
        if not hasattr(model, "max_object_px"):
            model.max_object_px = 220.0
        if not hasattr(model, "pyramid_step"):
            model.pyramid_step = 1.09
        return model


def colorfulness(bgr: np.ndarray) -> float:
    return image_colorfulness(bgr)


def _window_metrics(scores: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    """Threshold-free window metrics for the convergence plot."""
    y = np.asarray(y, dtype=np.float64)
    scores = np.asarray(scores, dtype=np.float64)
    pred = np.where(scores >= 0.0, 1.0, -1.0)
    pos = y > 0
    neg = y < 0
    exp_loss = float(np.mean(np.exp(-y * scores)))
    err = float(np.mean(pred != y))
    pos_recall = float(np.mean(scores[pos] >= 0.0)) if np.any(pos) else float("nan")
    neg_reject = float(np.mean(scores[neg] < 0.0)) if np.any(neg) else float("nan")
    return {
        "exp_loss": exp_loss,
        "error": err,
        "pos_recall": pos_recall,
        "neg_reject": neg_reject,
    }


def _leaf_values(
    leaves: np.ndarray,
    n_nodes: int,
    y: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    """RealBoost leaf: 0.5 log(W+ / W−) on the training samples in that leaf."""
    wp = np.zeros(n_nodes, dtype=np.float64)
    wn = np.zeros(n_nodes, dtype=np.float64)
    pos = y > 0
    np.add.at(wp, leaves[pos], weights[pos])
    np.add.at(wn, leaves[~pos], weights[~pos])
    values = 0.5 * np.log((wp + EPS) / (wn + EPS))
    return np.clip(values, -4.0, 4.0)


def fit_realboost(
    x: np.ndarray,
    y: np.ndarray,
    n_trees: int,
    depth: int,
    win_h: int,
    win_w: int,
    shrink: int,
    x_val: Optional[np.ndarray] = None,
    y_val: Optional[np.ndarray] = None,
    model: Optional[BoostedAcf] = None,
    max_features: MaxFeatures = 256,
    row_frac: float = 0.5,
    n_bins: int = 64,
    keep_all_positives: bool = True,
    seed: int = 0,
    eval_fn: Optional[Callable[["BoostedAcf"], Dict[str, float]]] = None,
    eval_every: int = 10,
) -> BoostedAcf:
    """Train orthogonal histogram trees with RealBoost leaf weights.

    ``y`` is +1 (sign window) / −1 (background). Each split is
    ``aggregated cell (u, v) on channel c ≷ quantile threshold``.

    Speed knobs (Dollar-style, not sklearn CART):
      * ``n_bins`` — candidate thresholds per cell (histogram scan).
      * ``max_features`` — random cells tried at each node (``sqrt`` / int / fraction).
      * ``row_frac`` — fraction of *negatives* used to grow each tree
        (positives are kept unless ``keep_all_positives`` is False).
      * Validation *window* scores are updated with the new tree only.
      * ``eval_fn`` (optional) runs sliding-window detection on held-out
        frames every ``eval_every`` trees and logs recall / precision / IoU.

    Pass an existing ``model`` to *continue* boosting. After every tree the
    ensemble train/val error and exponential loss go into ``model.history``.
    """
    x = np.asarray(x, dtype=np.float32)
    y = np.asarray(y, dtype=np.float64)
    n = x.shape[0]
    has_val = x_val is not None and y_val is not None and len(y_val) > 0
    if has_val:
        x_val = np.asarray(x_val, dtype=np.float32)
        y_val = np.asarray(y_val, dtype=np.float64)
    if model is None:
        model = BoostedAcf(
            win_h=win_h,
            win_w=win_w,
            shrink=shrink,
            depth=depth,
            max_features=max_features,
            row_frac=row_frac,
            n_bins=n_bins,
        )
        scores = np.zeros(n, dtype=np.float64)
        val_scores = (
            np.zeros(len(y_val), dtype=np.float64) if has_val else None
        )
    else:
        model.max_features = max_features
        model.row_frac = row_frac
        model.n_bins = n_bins
        model.cascade_tau = None
        scores = model.score_windows(x).astype(np.float64)
        val_scores = (
            model.score_windows(x_val).astype(np.float64) if has_val else None
        )
    weights = np.exp(-y * scores)
    weights = np.clip(weights, 1e-12, None)
    weights /= weights.sum()
    start = model.n_trees
    rng = np.random.default_rng(seed + start)

    t_bin0 = time.perf_counter()
    bins, edges = _quantize_features(x, n_bins, rng)
    n_feat_try = _feature_count(x.shape[1], max_features)
    print(
        f"  histogram trees  bins={edges.shape[0] + 1}  "
        f"max_features={n_feat_try}/{x.shape[1]}  "
        f"row_frac={row_frac}  keep_pos={keep_all_positives}  "
        f"binning={time.perf_counter() - t_bin0:.2f}s",
        flush=True,
    )

    for t in range(n_trees):
        t0 = time.perf_counter()
        fit_idx = _subsample_rows(y, row_frac, rng, keep_all_positives)
        min_leaf = max(8, fit_idx.size // 250)
        tree = _fit_quant_tree(
            bins,
            edges,
            y,
            weights,
            fit_idx,
            depth=depth,
            min_leaf=min_leaf,
            max_features=max_features,
            rng=rng,
        )
        leaves = tree.apply(x)
        values = _leaf_values(leaves, tree.n_nodes, y, weights)
        pred = values[leaves]
        scores += pred
        weights *= np.exp(-y * pred)
        weights = np.clip(weights, 1e-12, None)
        weights /= weights.sum()
        model.stages.append((tree, values))

        row = {
            "tree": float(model.n_trees),
            "fit_start": float(start),
            "fit_seconds": time.perf_counter() - t0,
            "n_fit": float(fit_idx.size),
        }
        row.update({f"train_{k}": v for k, v in _window_metrics(scores, y).items()})
        if has_val:
            val_scores = val_scores + values[tree.apply(x_val)]
            row.update(
                {f"val_{k}": v for k, v in _window_metrics(val_scores, y_val).items()}
            )
        do_eval = eval_fn is not None and (
            t == 0 or (t + 1) % max(1, eval_every) == 0 or t == n_trees - 1
        )
        if do_eval:
            det = eval_fn(model)
            row.update(det)
        model.history.append(row)
        if (t + 1) % 10 == 0 or t == 0 or do_eval:
            msg = (
                f"  tree {model.n_trees:3d}  train err={row['train_error']:.3f}  "
                f"exp-loss={row['train_exp_loss']:.3f}  "
                f"pos-recall={row['train_pos_recall']:.3f}  "
                f"neg-reject={row['train_neg_reject']:.3f}  "
                f"{row['fit_seconds']:.2f}s"
            )
            if "val_error" in row:
                msg += f"  | val err={row['val_error']:.3f}  exp-loss={row['val_exp_loss']:.3f}"
            if "det_recall" in row:
                msg += (
                    f"  | det rec={row['det_recall']:.3f}  "
                    f"prec={row['det_precision']:.3f}  "
                    f"iou={row['det_mean_iou']:.3f}"
                )
            print(msg, flush=True)
    return model


def calibrate_cascade(
    model: BoostedAcf,
    x: np.ndarray,
    y: np.ndarray,
    pos_recall: float = 0.995,
    linear: bool = True,
    slack: float = 1e-3,
) -> np.ndarray:
    """Fill ``model.cascade_tau`` from running scores on labeled windows.

    After tree ``t``, ``τ[t]`` is the ``(1 - pos_recall)`` quantile of
    ``F_t`` on positives (default: keep 99.5%). If ``linear``, those
    quantiles are replaced by a least-squares line ``a + b t`` clipped
    *above* so ``τ`` never exceeds the quantile (Dollár's constant soft
    cascade, without extra positive kills).

    Call this *after* boosting. ``fit_realboost`` clears ``cascade_tau``
    so continuation rounds see full residuals.
    """
    if model.n_trees < 1:
        model.cascade_tau = None
        return np.zeros((0,), dtype=np.float32)
    x = np.asarray(x, dtype=np.float32)
    y = np.asarray(y, dtype=np.float64)
    if x.ndim == 1:
        x = x.reshape(1, -1)
    pos = y > 0
    neg = y < 0
    n_pos = int(np.count_nonzero(pos))
    if n_pos < 2:
        raise ValueError("calibrate_cascade needs at least 2 positive windows")
    pos_recall = float(np.clip(pos_recall, 0.5, 0.9999))
    q = 100.0 * (1.0 - pos_recall)
    x_pos = x[pos]
    scores = np.zeros(n_pos, dtype=np.float64)
    tau_raw = np.zeros(model.n_trees, dtype=np.float64)

    for t, (tree, values) in enumerate(model.stages):
        scores += values[tree.apply(x_pos)]
        tau_raw[t] = float(np.percentile(scores, q)) - slack

    tau = tau_raw.copy()
    if linear and model.n_trees >= 2:
        t_idx = np.arange(model.n_trees, dtype=np.float64)
        design = np.stack([np.ones(model.n_trees), t_idx], axis=1)
        coef, *_ = np.linalg.lstsq(design, tau_raw, rcond=None)
        tau = np.minimum(tau_raw, coef[0] + coef[1] * t_idx)

    model.cascade_tau = tau.astype(np.float32)

    scores = np.zeros(n_pos, dtype=np.float64)
    alive_pos = np.ones(n_pos, dtype=bool)
    alive_neg_frac: List[float] = []
    scores_neg = None
    alive_neg = None
    if np.any(neg):
        x_neg = x[neg]
        scores_neg = np.zeros(int(np.count_nonzero(neg)), dtype=np.float64)
        alive_neg = np.ones(scores_neg.shape[0], dtype=bool)
    for t, (tree, values) in enumerate(model.stages):
        scores += values[tree.apply(x_pos)]
        alive_pos &= scores >= tau[t]
        if scores_neg is not None and alive_neg is not None:
            idx = np.flatnonzero(alive_neg)
            scores_neg[idx] += values[tree.apply(x_neg[idx])]
            alive_neg[idx] = scores_neg[idx] >= tau[t]
            alive_neg_frac.append(float(np.mean(alive_neg)))

    print(
        f"  cascade  stages={model.n_trees}  pos_recall_target={pos_recall:.4f}  "
        f"linear={linear}  tau[0]={tau[0]:.3f}  "
        f"tau[{model.n_trees // 2}]={tau[model.n_trees // 2]:.3f}  "
        f"tau[-1]={tau[-1]:.3f}  "
        f"calib-pos surviving all stages={float(np.mean(alive_pos)):.3f}",
        flush=True,
    )
    if alive_neg_frac:
        checkpoints = sorted(
            {0, min(9, len(alive_neg_frac) - 1), len(alive_neg_frac) - 1}
        )
        bits = ", ".join(
            f"t{i + 1}:{alive_neg_frac[i]:.3f}" for i in checkpoints
        )
        print(f"  cascade  val-neg still alive  {bits}", flush=True)
    return model.cascade_tau


def window_from_box(
    aggregated: np.ndarray,
    box: Sequence[float],
    shrink: int,
    win_h: int,
    win_w: int,
) -> Optional[np.ndarray]:
    """Legacy stretch-to-window crop. Prefer ``window_from_square`` / cell slices."""
    patch = crop_aggregated_box(aggregated, box, shrink)
    if patch.size == 0:
        return None
    return windows_to_features(resize_patch(patch, win_h, win_w)).reshape(-1)


def _scale_box(box: Sequence[float], scale: float) -> Tuple[float, float, float, float]:
    x, y, w, h = (float(v) for v in box[:4])
    return (x * scale, y * scale, w * scale, h * scale)


def _crop_bgr_padded(
    bgr: np.ndarray, x0: int, y0: int, x1: int, y1: int
) -> np.ndarray:
    h, w = bgr.shape[:2]
    pad_left = max(0, -x0)
    pad_top = max(0, -y0)
    pad_right = max(0, x1 - w)
    pad_bottom = max(0, y1 - h)
    crop = bgr[max(0, y0) : min(h, y1), max(0, x0) : min(w, x1)]
    if crop.size == 0:
        return np.zeros((max(1, y1 - y0), max(1, x1 - x0), 3), dtype=bgr.dtype)
    if pad_left or pad_top or pad_right or pad_bottom:
        crop = cv2.copyMakeBorder(
            crop, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_REPLICATE
        )
    return crop


def window_from_model(
    bgr: np.ndarray,
    box: Sequence[float],
    shrink: int,
    win_h: int,
    win_w: int,
    rng: Optional[np.random.Generator] = None,
    jitter: bool = False,
    margin: float = 0.10,
    margin_cells: int = 1,
) -> Optional[np.ndarray]:
    """Native-aspect crop matching the detector window (no square warp).

    Take a rectangle with aspect ``win_w / win_h`` that covers the GT plate
    plus ``margin``, resize it to the model pixel size, extract ACF, and
    slice ``win_h × win_w`` cells from the center.
    """
    _x, _y, w, h = (float(v) for v in box[:4])
    if w < 1.0 or h < 1.0:
        return None
    cx = _x + 0.5 * w
    cy = _y + 0.5 * h
    model_aspect = float(win_w) / max(float(win_h), 1e-6)
    pad = 1.0 + 2.0 * float(margin)
    pw, ph = w * pad, h * pad
    if pw / max(ph, 1e-6) > model_aspect:
        crop_w = pw
        crop_h = pw / model_aspect
    else:
        crop_h = ph
        crop_w = ph * model_aspect
    if jitter and rng is not None:
        scale = float(rng.uniform(0.85, 1.15))
        crop_w *= scale
        crop_h *= scale
        cx += float(rng.uniform(-0.12, 0.12)) * crop_w
        cy += float(rng.uniform(-0.12, 0.12)) * crop_h
    outer_w = int(win_w * shrink + 2 * margin_cells * shrink)
    outer_h = int(win_h * shrink + 2 * margin_cells * shrink)
    x0 = int(round(cx - 0.5 * crop_w))
    y0 = int(round(cy - 0.5 * crop_h))
    x1 = x0 + max(1, int(round(crop_w)))
    y1 = y0 + max(1, int(round(crop_h)))
    patch = _crop_bgr_padded(bgr, x0, y0, x1, y1)
    if patch.size == 0:
        return None
    resized = cv2.resize(patch, (outer_w, outer_h), interpolation=cv2.INTER_LINEAR)
    agg = extract_acf(resized, shrink=shrink).aggregated
    y_off = max(0, (agg.shape[0] - win_h) // 2)
    x_off = max(0, (agg.shape[1] - win_w) // 2)
    window = agg[y_off : y_off + win_h, x_off : x_off + win_w]
    if window.shape[0] != win_h or window.shape[1] != win_w:
        if window.size == 0:
            return None
        window = resize_patch(window, win_h, win_w)
    return windows_to_features(window).reshape(-1)


def window_from_square(
    bgr: np.ndarray,
    box: Sequence[float],
    shrink: int,
    win_h: int,
    win_w: int,
    rng: Optional[np.random.Generator] = None,
    jitter: bool = False,
    margin_cells: int = 1,
) -> Optional[np.ndarray]:
    """Alias for ``window_from_model`` (kept for older call sites)."""
    return window_from_model(
        bgr,
        box,
        shrink,
        win_h,
        win_w,
        rng=rng,
        jitter=jitter,
        margin_cells=margin_cells,
    )


def cell_window_vec(
    aggregated: np.ndarray, cell_x: int, cell_y: int, win_h: int, win_w: int
) -> Optional[np.ndarray]:
    """Exact ``win_h×win_w`` slice — the same crop ``score_heatmap`` scores."""
    patch = aggregated[cell_y : cell_y + win_h, cell_x : cell_x + win_w]
    if patch.shape[0] != win_h or patch.shape[1] != win_w:
        return None
    return windows_to_features(patch).reshape(-1)


def sample_random_cells(
    agg_hw: Tuple[int, int],
    gt_cell: Sequence[Sequence[float]],
    win_h: int,
    win_w: int,
    rng: np.random.Generator,
    n: int,
) -> List[Tuple[int, int]]:
    h, w = agg_hw
    if n < 1 or h <= win_h or w <= win_w:
        return []
    xs = rng.integers(0, w - win_w + 1, size=n * 8)
    ys = rng.integers(0, h - win_h + 1, size=n * 8)
    out: List[Tuple[int, int]] = []
    for x, y in zip(xs, ys):
        box = (float(x), float(y), float(win_w), float(win_h))
        if all(iou(box, gt) < 0.3 for gt in gt_cell):
            out.append((int(x), int(y)))
            if len(out) >= n:
                break
    return out


def sample_textured_cells(
    aggregated: np.ndarray,
    gt_cell: Sequence[Sequence[float]],
    win_h: int,
    win_w: int,
    rng: np.random.Generator,
    n: int,
    n_candidates: int = 48,
) -> List[Tuple[int, int]]:
    """Background windows with high gradient energy, using an integral image."""
    h, w, _ = aggregated.shape
    if n < 1 or h <= win_h or w <= win_w:
        return []
    integral = cv2.integral(aggregated[:, :, 3])
    cx = rng.integers(0, w - win_w + 1, size=n_candidates)
    cy = rng.integers(0, h - win_h + 1, size=n_candidates)
    energy = (
        integral[cy + win_h, cx + win_w]
        - integral[cy, cx + win_w]
        - integral[cy + win_h, cx]
        + integral[cy, cx]
    )
    out: List[Tuple[int, int]] = []
    for idx in np.argsort(-energy):
        x = int(cx[idx])
        y = int(cy[idx])
        box = (float(x), float(y), float(win_w), float(win_h))
        if any(iou(box, gt) >= 0.3 for gt in gt_cell):
            continue
        out.append((x, y))
        if len(out) >= n:
            break
    return out


def greedy_match_iou(
    pred_boxes: Sequence[Sequence[float]],
    gt_boxes: Sequence[Sequence[float]],
    iou_thresh: float = 0.5,
) -> Tuple[int, List[float]]:
    """Greedy one-to-one matches; ``pred_boxes`` should be score-sorted."""
    assigned = set()
    matched: List[float] = []
    for pred in pred_boxes:
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
            matched.append(best_iou)
    return len(matched), matched


def evaluate_detection(
    model: BoostedAcf,
    frames: Sequence[Tuple[np.ndarray, list, int]],
    iou_thresh: float = 0.5,
    **detect_kw: Any,
) -> Dict[str, float]:
    """Image-level precision / recall / mean IoU of matched boxes."""
    n_gt = 0
    n_pred = 0
    tp = 0
    ious: List[float] = []
    kw = dict(return_heat=False, stride=2, score_threshold=0.0, refine=False)
    kw.update(detect_kw)
    for bgr, anns, _image_id in frames:
        # Class-agnostic: ignore category_id / logical_* from the loader.
        gt = [ann["bbox"] for ann in anns]
        boxes, _scores, _heat = model.detect_image(bgr, **kw)
        n_gt += len(gt)
        n_pred += len(boxes)
        hit, matched = greedy_match_iou(boxes, gt, iou_thresh=iou_thresh)
        tp += hit
        ious.extend(matched)
    return {
        "det_recall": float(tp / max(n_gt, 1)),
        "det_precision": float(tp / max(n_pred, 1)),
        "det_mean_iou": float(np.mean(ious)) if ious else 0.0,
        "det_tp": float(tp),
        "det_n_gt": float(n_gt),
        "det_n_pred": float(n_pred),
    }


def collect_windows(
    dataset,
    indices: Iterable[int],
    shrink: int,
    win_h: int,
    win_w: int,
    negatives_per_image: int,
    rng: np.random.Generator,
    min_box_px: int = 8,
    min_color: float = 8.0,
    textured_fraction: float = 0.7,
    jitter_per_gt: int = 1,
    logical_names: Optional[Sequence[str]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Positives = native-aspect model crops; negatives = cell slices.

    ``logical_names`` restricts positives to those silhouettes (stage-1
    still does not *score* class labels). Negatives are exact
    ``win_h×win_w`` slices on ``extract_acf(bgr, shrink)``. CLAHE runs
    once on the full frame, matching ``detect_image``.
    """
    xs: List[np.ndarray] = []
    ys: List[int] = []
    allowed = None if logical_names is None else {str(n) for n in logical_names}
    n_textured = int(round(negatives_per_image * textured_fraction))
    n_random = max(0, negatives_per_image - n_textured)
    index_list = [int(i) for i in indices]
    t0 = time.perf_counter()
    used = 0
    for step, index in enumerate(index_list, start=1):
        bgr, anns, _image_id = dataset.get(index)
        if colorfulness(bgr) < min_color:
            if step % 100 == 0 or step == len(index_list):
                elapsed = time.perf_counter() - t0
                print(
                    f"  collect {step}/{len(index_list)}  windows={len(ys)}  "
                    f"{elapsed:.1f}s",
                    flush=True,
                )
            continue
        bgr = clahe_bgr(bgr)
        acf = extract_acf(bgr, shrink=shrink)
        inv = 1.0 / float(shrink)
        pos_anns = [
            ann
            for ann in anns
            if allowed is None or str(ann.get("logical_name")) in allowed
        ]
        gt_cell = [_scale_box(ann["bbox"], inv) for ann in anns]
        for ann in pos_anns:
            _x, _y, bw, bh = ann["bbox"]
            if bw < min_box_px or bh < min_box_px:
                continue
            vec = window_from_model(
                bgr, ann["bbox"], shrink, win_h, win_w, rng=rng, jitter=False
            )
            if vec is None:
                continue
            xs.append(vec)
            ys.append(1)
            for _ in range(int(jitter_per_gt)):
                jvec = window_from_model(
                    bgr, ann["bbox"], shrink, win_h, win_w, rng=rng, jitter=True
                )
                if jvec is None:
                    continue
                xs.append(jvec)
                ys.append(1)
        for cx, cy in sample_random_cells(
            acf.aggregated.shape[:2], gt_cell, win_h, win_w, rng, n_random
        ):
            vec = cell_window_vec(acf.aggregated, cx, cy, win_h, win_w)
            if vec is None:
                continue
            xs.append(vec)
            ys.append(-1)
        for cx, cy in sample_textured_cells(
            acf.aggregated, gt_cell, win_h, win_w, rng, n_textured
        ):
            vec = cell_window_vec(acf.aggregated, cx, cy, win_h, win_w)
            if vec is None:
                continue
            xs.append(vec)
            ys.append(-1)
        used += 1
        if step % 50 == 0 or step == len(index_list):
            elapsed = time.perf_counter() - t0
            rate = step / max(elapsed, 1e-6)
            eta = (len(index_list) - step) / max(rate, 1e-6)
            print(
                f"  collect {step}/{len(index_list)}  kept={used}  "
                f"windows={len(ys)}  {rate:.1f} img/s  eta={eta:.0f}s",
                flush=True,
            )
    if not xs:
        raise RuntimeError("no training windows (check split / color filter / shape)")
    x = np.stack(xs, axis=0).astype(np.float32)
    y = np.asarray(ys, dtype=np.float64)
    return x, y


def collect_windows_for_shapes(
    dataset,
    indices: Iterable[int],
    shrink: int,
    shape_windows: Dict[str, Tuple[int, int]],
    names_by_shape: Dict[str, Sequence[str]],
    negatives_per_image: int,
    rng: np.random.Generator,
    min_box_px: int = 8,
    min_color: float = 8.0,
    textured_fraction: float = 0.7,
    jitter_per_gt: int = 1,
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """One full-image ACF pass; positives/negatives are sliced per silhouette."""
    buckets: Dict[str, Tuple[List[np.ndarray], List[int]]] = {
        name: ([], []) for name in shape_windows
    }
    allowed = {name: {str(n) for n in names} for name, names in names_by_shape.items()}
    n_textured = int(round(negatives_per_image * textured_fraction))
    n_random = max(0, negatives_per_image - n_textured)
    index_list = [int(i) for i in indices]
    t0 = time.perf_counter()
    used = 0
    for step, index in enumerate(index_list, start=1):
        bgr, anns, _image_id = dataset.get(index)
        if colorfulness(bgr) < min_color:
            if step % 100 == 0 or step == len(index_list):
                print(
                    f"  collect-all {step}/{len(index_list)}  {time.perf_counter() - t0:.1f}s",
                    flush=True,
                )
            continue
        bgr = clahe_bgr(bgr)
        acf = extract_acf(bgr, shrink=shrink)
        inv = 1.0 / float(shrink)
        gt_cell = [_scale_box(ann["bbox"], inv) for ann in anns]
        for shape, (win_w, win_h) in shape_windows.items():
            xs, ys = buckets[shape]
            pos_anns = [
                ann
                for ann in anns
                if str(ann.get("logical_name")) in allowed.get(shape, set())
            ]
            for ann in pos_anns:
                _x, _y, bw, bh = ann["bbox"]
                if bw < min_box_px or bh < min_box_px:
                    continue
                vec = window_from_model(
                    bgr, ann["bbox"], shrink, win_h, win_w, rng=rng, jitter=False
                )
                if vec is None:
                    continue
                xs.append(vec)
                ys.append(1)
                for _ in range(int(jitter_per_gt)):
                    jvec = window_from_model(
                        bgr, ann["bbox"], shrink, win_h, win_w, rng=rng, jitter=True
                    )
                    if jvec is None:
                        continue
                    xs.append(jvec)
                    ys.append(1)
            for cx, cy in sample_random_cells(
                acf.aggregated.shape[:2], gt_cell, win_h, win_w, rng, n_random
            ):
                vec = cell_window_vec(acf.aggregated, cx, cy, win_h, win_w)
                if vec is None:
                    continue
                xs.append(vec)
                ys.append(-1)
            for cx, cy in sample_textured_cells(
                acf.aggregated, gt_cell, win_h, win_w, rng, n_textured
            ):
                vec = cell_window_vec(acf.aggregated, cx, cy, win_h, win_w)
                if vec is None:
                    continue
                xs.append(vec)
                ys.append(-1)
        used += 1
        if step % 50 == 0 or step == len(index_list):
            elapsed = time.perf_counter() - t0
            rate = step / max(elapsed, 1e-6)
            n_win = sum(len(ys) for _xs, ys in buckets.values())
            print(
                f"  collect-all {step}/{len(index_list)}  kept={used}  "
                f"windows={n_win}  {rate:.1f} img/s",
                flush=True,
            )
    out: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    for shape, (xs, ys) in buckets.items():
        if not xs:
            continue
        out[shape] = (
            np.stack(xs, axis=0).astype(np.float32),
            np.asarray(ys, dtype=np.float64),
        )
    if not out:
        raise RuntimeError("no training windows for any silhouette")
    return out


def hard_negatives(
    model: BoostedAcf,
    dataset,
    indices: Iterable[int],
    per_image: int,
    min_color: float = 8.0,
    stride: int = 2,
) -> Tuple[np.ndarray, np.ndarray]:
    """Bootstrap FPs using the same square-crop ACF path as detection."""
    xs: List[np.ndarray] = []
    for index in indices:
        bgr, anns, _image_id = dataset.get(int(index))
        if colorfulness(bgr) < min_color:
            continue
        # False positives vs boxes; logical class labels are unused.
        bgr = clahe_bgr(bgr)
        gt = [ann["bbox"] for ann in anns]
        boxes, _scores, _heat = model.detect_image(
            bgr,
            stride=stride,
            score_threshold=0.0,
            score_ratio=0.0,
            max_boxes=max(per_image * 3, 40),
            return_heat=False,
            refine=False,
            preprocess=False,
            fast_pyramid=True,
        )
        taken = 0
        for box in boxes:
            if taken >= per_image:
                break
            if any(iou(box, g) >= 0.3 for g in gt):
                continue
            vec = window_from_model(
                bgr, box, model.shrink, model.win_h, model.win_w
            )
            if vec is None:
                continue
            xs.append(vec)
            taken += 1
    if not xs:
        return np.zeros((0, model.n_features), dtype=np.float32), np.zeros((0,), dtype=np.float64)
    x = np.stack(xs, axis=0).astype(np.float32)
    y = -np.ones(x.shape[0], dtype=np.float64)
    return x, y


def _round_even(value: float, lo: int, hi: int) -> int:
    rounded = int(round(float(value) / 2.0) * 2)
    return int(np.clip(rounded, lo, hi))


def fit_shape_cell_windows(
    dataset,
    shrink: int = 2,
    target_cells: int = 16,
    min_cells: int = 10,
    max_cells: int = 24,
) -> Tuple[Dict[str, Tuple[int, int]], Dict[str, Tuple[float, float]], Dict[str, int]]:
    """Per-silhouette ``(win_w, win_h)`` in aggregated cells, plus height range.

    Window aspect is the median GT ``w/h``. Cell counts keep area near
    ``target_cells²``. Object-size range is the 5th/95th percentile of GT
    height (the pyramid matches ``win_h * shrink`` to object height).
    """
    from shape_detect import CLASS_TO_SHAPE, TEMPLATE_NAMES

    widths: Dict[str, List[float]] = {name: [] for name in TEMPLATE_NAMES}
    heights: Dict[str, List[float]] = {name: [] for name in TEMPLATE_NAMES}
    for record in dataset.image_records():
        for ann in dataset.annotations_for(int(record["id"])):
            shape = CLASS_TO_SHAPE.get(str(ann["logical_name"]))
            if shape is None:
                continue
            _x, _y, w, h = (float(v) for v in ann["bbox"][:4])
            widths[shape].append(w)
            heights[shape].append(h)
    windows: Dict[str, Tuple[int, int]] = {}
    object_range: Dict[str, Tuple[float, float]] = {}
    counts: Dict[str, int] = {}
    del shrink
    for name in TEMPLATE_NAMES:
        ws = np.asarray(widths[name], dtype=np.float64)
        hs = np.asarray(heights[name], dtype=np.float64)
        counts[name] = int(hs.size)
        if hs.size == 0:
            windows[name] = (int(target_cells), int(target_cells))
            object_range[name] = (12.0, 220.0)
            continue
        aspect = float(np.median(ws / np.maximum(hs, 1e-6)))
        aspect = float(np.clip(aspect, 0.45, 2.4))
        win_h = _round_even(float(target_cells) / np.sqrt(aspect), min_cells, max_cells)
        win_w = _round_even(float(win_h) * aspect, min_cells, max_cells)
        windows[name] = (int(win_w), int(win_h))
        lo = max(8.0, float(np.percentile(hs, 5)))
        hi = max(lo + 8.0, float(np.percentile(hs, 95)))
        object_range[name] = (lo, hi)
    return windows, object_range, counts


@dataclass
class AcfBank:
    """Union of per-silhouette ``BoostedAcf`` detectors, one NMS at the end."""

    models: Dict[str, BoostedAcf] = field(default_factory=dict)
    windows: Dict[str, Tuple[int, int]] = field(default_factory=dict)

    def detect(
        self,
        bgr: np.ndarray,
        max_boxes: int = 40,
        stride: int = 2,
        score_threshold: float = 0.0,
        nms_iou: float = 0.5,
        score_ratio: float = 0.0,
        peaks_per_scale: int = 80,
        preprocess: bool = True,
        fast_pyramid: bool = True,
    ) -> Tuple[List[Tuple[float, float, float, float]], List[float]]:
        if not self.models:
            return [], []
        img = clahe_bgr(bgr) if preprocess else np.asarray(bgr)
        shrink = int(next(iter(self.models.values())).shrink)
        scale_keys = sorted(
            {round(float(s), 4) for model in self.models.values() for s in model.default_scales()},
            reverse=True,
        )
        h0, w0 = img.shape[:2]
        max_win = max(max(m.win_h, m.win_w) for m in self.models.values())
        min_side = int(max_win * shrink)
        base_agg = None
        if fast_pyramid:
            base_agg = extract_acf(img, shrink=shrink).aggregated
        agg_by_scale: Dict[float, np.ndarray] = {}
        for scale in scale_keys:
            new_w = max(1, int(round(w0 * scale)))
            new_h = max(1, int(round(h0 * scale)))
            if new_w < min_side or new_h < min_side:
                continue
            agg_h = max(1, new_h // shrink)
            agg_w = max(1, new_w // shrink)
            if fast_pyramid:
                if abs(scale - 1.0) < 1e-6:
                    agg_by_scale[float(scale)] = base_agg
                else:
                    agg_by_scale[float(scale)] = resize_aggregated(base_agg, agg_h, agg_w)
            else:
                if abs(scale - 1.0) < 1e-6:
                    image = img
                else:
                    image = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
                agg_by_scale[float(scale)] = extract_acf(image, shrink=shrink).aggregated

        all_boxes: List[Tuple[float, float, float, float]] = []
        all_scores: List[float] = []
        for model in self.models.values():
            for scale in model.default_scales():
                aggregated = agg_by_scale.get(round(float(scale), 4))
                if aggregated is None:
                    continue
                if aggregated.shape[0] < model.win_h or aggregated.shape[1] < model.win_w:
                    continue
                boxes_s, scores_s = model.peaks_on_aggregated(
                    aggregated,
                    scale=float(scale),
                    stride=stride,
                    score_threshold=score_threshold,
                    peaks_per_scale=peaks_per_scale,
                )
                all_boxes.extend(boxes_s)
                all_scores.extend(scores_s)

        keep = nms(all_boxes, all_scores, iou_threshold=float(nms_iou))
        boxes = [all_boxes[i] for i in keep]
        scores = [all_scores[i] for i in keep]
        if not scores:
            return [], []
        cutoff = float(score_threshold)
        if score_ratio > 0.0:
            cutoff = max(cutoff, float(score_ratio) * max(scores))
        filtered = [(b, s) for b, s in zip(boxes, scores) if s >= cutoff]
        filtered = filtered[: max(1, int(max_boxes))]
        return [b for b, _s in filtered], [s for _b, s in filtered]

    def propose(
        self,
        bgr: np.ndarray,
        max_boxes: int = 40,
        **detect_kw: Any,
    ) -> Tuple[List[Tuple[float, float, float, float]], List[float]]:
        return self.detect(bgr, max_boxes=int(max_boxes), **detect_kw)

    def detect_image(
        self,
        bgr: np.ndarray,
        max_boxes: int = 40,
        return_heat: bool = False,
        **detect_kw: Any,
    ) -> Tuple[List[Tuple[float, float, float, float]], List[float], np.ndarray]:
        """Same boxes as ``detect``; heat is empty (bank has several maps)."""
        detect_kw.pop("refine", None)
        detect_kw.pop("fast_pyramid", None)
        detect_kw.pop("scales", None)
        boxes, scores = self.detect(bgr, max_boxes=int(max_boxes), **detect_kw)
        heat = np.zeros((bgr.shape[0], bgr.shape[1]), dtype=np.float32)
        del return_heat
        return boxes, scores, heat

    def save(self, path: str) -> None:
        with open(path, "wb") as handle:
            pickle.dump(self, handle)

    @staticmethod
    def load(path: str) -> "AcfBank":
        with open(path, "rb") as handle:
            bank = pickle.load(handle)
        if not isinstance(bank, AcfBank):
            raise TypeError(f"expected AcfBank, got {type(bank)}")
        for model in bank.models.values():
            if not hasattr(model, "min_object_px"):
                model.min_object_px = 12.0
            if not hasattr(model, "max_object_px"):
                model.max_object_px = 220.0
            if not hasattr(model, "pyramid_step"):
                model.pyramid_step = 1.09
        return bank


def load_acf_stage(path: str) -> Union[AcfBank, BoostedAcf]:
    """Load ``AcfBank`` or a legacy single ``BoostedAcf``."""
    with open(path, "rb") as handle:
        obj = pickle.load(handle)
    if isinstance(obj, AcfBank):
        return AcfBank.load(path)
    if isinstance(obj, BoostedAcf):
        return BoostedAcf.load(path)
    raise TypeError(f"expected AcfBank or BoostedAcf, got {type(obj)}")
