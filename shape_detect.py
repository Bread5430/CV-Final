"""Color-channel + silhouette ROI on train-fit window sizes.

ICF-legal ROI: LUV (and Sobel) are computed once per frame. Each of five
US-sign silhouettes is a first-order support (filled mask) on a color map
plus an outline on the edge map. Window sizes come from train ``max(w, h)``
percentiles per silhouette, not from ``pyrDown`` octaves. The window steps
by 20% of its side.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from acf import bgr_to_luv, clahe_bgr
from acf_boost import nms

Box = Tuple[float, float, float, float]
TEMPLATE_NAMES: Tuple[str, ...] = (
    "octagon",
    "triangle",
    "diamond",
    "rectangle",
    "circle",
)
SHAPE_TO_CLASSES: Dict[str, Tuple[str, ...]] = {
    "octagon": ("stop",),
    "triangle": ("yield",),
    "diamond": (
        "pedestrianCrossing",
        "signalAhead",
        "warningRoad",
        "school",
        "controlAhead",
        "laneChange",
    ),
    "rectangle": ("speedLimit", "keepRight"),
    "circle": ("prohibitory", "mandatoryTurn"),
}
CLASS_TO_SHAPE: Dict[str, str] = {
    cls: shape for shape, classes in SHAPE_TO_CLASSES.items() for cls in classes
}
SHAPE_COLOR: Dict[str, str] = {
    "octagon": "red",
    "triangle": "red",
    "diamond": "yellow",
    "rectangle": "white",
    "circle": "red",
}


@dataclass
class ShapeConfig:
    windows: Dict[str, Tuple[int, ...]] = field(default_factory=dict)
    score_threshold: float = 0.25
    nms_iou: float = 0.5
    max_boxes: int = 20
    stride_frac: float = 0.2
    shape_weight: float = 0.5
    min_win: int = 16
    max_win: int = 160


def _regular_polygon(
    n_sides: int,
    radius: float,
    cx: float,
    cy: float,
    rotation: float = 0.0,
) -> np.ndarray:
    angles = rotation + np.linspace(0.0, 2.0 * np.pi, n_sides, endpoint=False)
    xs = cx + radius * np.cos(angles)
    ys = cy + radius * np.sin(angles)
    return np.stack([xs, ys], axis=1).astype(np.int32)


def _round_even(value: float, lo: int, hi: int) -> int:
    rounded = int(round(value / 2.0) * 2)
    return int(np.clip(rounded, lo, hi))


def fit_template_windows(
    dataset,
    min_win: int = 16,
    max_win: int = 160,
) -> Dict[str, Tuple[int, ...]]:
    """p25 / p50 / p75 of ``max(w, h)`` on train boxes, per silhouette."""
    sides: Dict[str, List[float]] = {name: [] for name in TEMPLATE_NAMES}
    for record in dataset.image_records():
        for ann in dataset.annotations_for(int(record["id"])):
            shape = CLASS_TO_SHAPE.get(str(ann["logical_name"]))
            if shape is None:
                continue
            _x, _y, w, h = (float(v) for v in ann["bbox"][:4])
            sides[shape].append(max(w, h))
    global_sides = [s for vals in sides.values() for s in vals]
    fallback = (
        np.percentile(global_sides, [25, 50, 75])
        if global_sides
        else np.array([24.0, 36.0, 52.0])
    )
    windows: Dict[str, Tuple[int, ...]] = {}
    for name in TEMPLATE_NAMES:
        vals = np.asarray(sides[name], dtype=np.float64)
        raw = np.percentile(vals, [25, 50, 75]) if vals.size else fallback
        sized = []
        for q in raw:
            sized.append(_round_even(float(q), min_win, max_win))
        windows[name] = tuple(sorted(set(sized)))
    return windows


def _filled_silhouette(name: str, win: int) -> np.ndarray:
    cx = cy = (win - 1) * 0.5
    margin = max(2, win // 10)
    radius = 0.5 * win - margin
    canvas = np.zeros((win, win), dtype=np.uint8)
    if name == "octagon":
        cv2.fillConvexPoly(
            canvas,
            _regular_polygon(8, radius, cx, cy, rotation=np.pi / 8.0),
            255,
        )
    elif name == "triangle":
        tri = np.array(
            [
                [cx, cy + radius],
                [cx - radius * 0.95, cy - radius * 0.75],
                [cx + radius * 0.95, cy - radius * 0.75],
            ],
            dtype=np.int32,
        )
        cv2.fillConvexPoly(canvas, tri, 255)
    elif name == "diamond":
        dia = np.array(
            [
                [cx, cy - radius],
                [cx + radius, cy],
                [cx, cy + radius],
                [cx - radius, cy],
            ],
            dtype=np.int32,
        )
        cv2.fillConvexPoly(canvas, dia, 255)
    elif name == "rectangle":
        rw = int(round(radius * 1.55))
        rh = int(round(radius * 1.15))
        cv2.rectangle(
            canvas,
            (int(round(cx - rw / 2)), int(round(cy - rh / 2))),
            (int(round(cx + rw / 2)), int(round(cy + rh / 2))),
            255,
            thickness=-1,
        )
    elif name == "circle":
        cv2.circle(
            canvas,
            (int(round(cx)), int(round(cy))),
            int(round(radius)),
            255,
            thickness=-1,
        )
    else:
        raise ValueError(f"unknown silhouette {name!r}")
    return canvas


def build_templates(
    windows: Dict[str, Sequence[int]],
) -> Dict[Tuple[str, int], Dict[str, np.ndarray]]:
    """Filled mask + edge ring for each (silhouette, window) pair."""
    out: Dict[Tuple[str, int], Dict[str, np.ndarray]] = {}
    for name, sizes in windows.items():
        for win in sizes:
            win = int(win)
            ring = max(1, win // 16)
            filled = _filled_silhouette(name, win)
            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (ring * 2 + 1, ring * 2 + 1)
            )
            outline = cv2.subtract(cv2.dilate(filled, kernel), cv2.erode(filled, kernel))
            out[(name, win)] = {
                "filled": filled.astype(np.float32) / 255.0,
                "outline": outline.astype(np.float32) / 255.0,
            }
    return out


def _color_maps(bgr: np.ndarray) -> Dict[str, np.ndarray]:
    """Pointwise LUV maps: red / yellow / white in ``[0, 1]``."""
    luv = bgr_to_luv(bgr)
    light = luv[:, :, 0] / 255.0
    u_ch = (luv[:, :, 1] - 128.0) / 128.0
    v_ch = (luv[:, :, 2] - 128.0) / 128.0
    red = np.clip(u_ch, 0.0, 1.0)
    yellow = np.clip(v_ch, 0.0, 1.0) * np.clip(-u_ch, 0.0, 1.0)
    chroma = np.clip(np.abs(u_ch) + np.abs(v_ch), 0.0, 1.0)
    white = np.clip(light, 0.0, 1.0) * (1.0 - chroma)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    dx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    dy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    edge = cv2.magnitude(dx, dy)
    edge /= float(np.max(edge) + 1e-6)
    return {
        "red": red.astype(np.float32),
        "yellow": yellow.astype(np.float32),
        "white": white.astype(np.float32),
        "edge": edge.astype(np.float32),
    }


def _sample_heatmap(heat: np.ndarray, stride: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    ys = np.arange(0, heat.shape[0], stride, dtype=np.int32)
    xs = np.arange(0, heat.shape[1], stride, dtype=np.int32)
    grid = heat[np.ix_(ys, xs)]
    yy, xx = np.meshgrid(ys, xs, indexing="ij")
    return yy.reshape(-1), xx.reshape(-1), grid.reshape(-1).astype(np.float32)


def detect_shapes(
    bgr: np.ndarray,
    config: Optional[ShapeConfig] = None,
    templates: Optional[Dict[Tuple[str, int], Dict[str, np.ndarray]]] = None,
) -> Tuple[List[Box], List[float]]:
    """Class-agnostic boxes in COCO ``(x, y, w, h)`` on the original frame."""
    cfg = config or ShapeConfig()
    windows = cfg.windows
    if not windows:
        raise ValueError("ShapeConfig.windows is empty; call fit_template_windows first")
    img = np.asarray(bgr)
    if img.ndim != 3 or img.shape[2] != 3:
        raise ValueError("expected a BGR image with shape (H, W, 3)")
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    img = clahe_bgr(img)
    tmpl = templates if templates is not None else build_templates(windows)
    maps = _color_maps(img)
    h, w = img.shape[:2]
    shape_w = float(np.clip(cfg.shape_weight, 0.0, 1.0))
    color_w = 1.0 - shape_w

    boxes: List[Box] = []
    scores: List[float] = []
    for name, sizes in windows.items():
        color_map = maps[SHAPE_COLOR[name]]
        for win in sizes:
            win = int(win)
            if h < win or w < win:
                continue
            pair = tmpl[(name, win)]
            shape_heat = cv2.matchTemplate(
                maps["edge"], pair["outline"], cv2.TM_CCOEFF_NORMED
            )
            color_heat = cv2.matchTemplate(
                color_map, pair["filled"], cv2.TM_CCOEFF_NORMED
            )
            heat = shape_w * shape_heat + color_w * color_heat
            stride = max(1, int(round(cfg.stride_frac * win)))
            ys, xs, vals = _sample_heatmap(heat, stride)
            keep = vals >= float(cfg.score_threshold)
            for y, x, score in zip(ys[keep], xs[keep], vals[keep]):
                boxes.append((float(x), float(y), float(win), float(win)))
                scores.append(float(score))

    if not boxes:
        return [], []
    keep_idx = nms(boxes, scores, iou_threshold=float(cfg.nms_iou))
    boxes = [boxes[i] for i in keep_idx]
    scores = [scores[i] for i in keep_idx]
    budget = int(cfg.max_boxes)
    if budget > 0:
        boxes = boxes[:budget]
        scores = scores[:budget]
    return boxes, scores


def format_window_table(
    windows: Dict[str, Sequence[int]],
    counts: Optional[Dict[str, int]] = None,
) -> str:
    lines = [f"{'shape':<12} {'n':>6}  windows"]
    for name in TEMPLATE_NAMES:
        n = counts.get(name, 0) if counts else 0
        sizes = ",".join(str(v) for v in windows.get(name, ()))
        lines.append(f"{name:<12} {n:6d}  {sizes}")
    return "\n".join(lines)


def count_train_sides(dataset) -> Dict[str, int]:
    counts = {name: 0 for name in TEMPLATE_NAMES}
    for record in dataset.image_records():
        for ann in dataset.annotations_for(int(record["id"])):
            shape = CLASS_TO_SHAPE.get(str(ann["logical_name"]))
            if shape is not None:
                counts[shape] += 1
    return counts
