"""Class-agnostic ROI proposals for LISA (MSER + US-sign HSV masks).

This is stage 1 of the two-stage pipeline. It does **not** use ACF, class
labels, or a learned detector. Boxes are scored cheaply and passed to the
crop classifier, which rejects background.

    frame
      -> CLAHE on LAB L
      -> HSV color masks (red / yellow / blue / yellow-green) + CC
      -> MSER on L and on red/yellow/blue enhancement maps
      -> area / aspect / solidity filters
      -> NMS, keep top-K

Detection still ignores ``logical_*``; those exist only for classification.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from acf import clahe_bgr
from lisa_coco import iou

Box = Tuple[float, float, float, float]


@dataclass
class RoiConfig:
    min_side: float = 8.0
    max_side: float = 280.0
    max_side_frac: float = 0.55
    min_aspect: float = 0.35
    max_aspect: float = 2.80
    nms_iou: float = 0.50
    max_boxes: int = 40
    gray_sat: float = 18.0
    mser_delta: int = 5
    mser_min_area: int = 20
    mser_max_area: int = 28000
    mser_max_variation: float = 0.35
    morph_ksize: int = 9
    box_expand: float = 1.15


def propose_rois(
    bgr: np.ndarray,
    config: Optional[RoiConfig] = None,
    max_boxes: Optional[int] = None,
    preprocess: bool = True,
) -> Tuple[List[Box], List[float]]:
    """Return COCO ``(x, y, w, h)`` boxes and scores, highest score first."""
    cfg = config or RoiConfig()
    img = np.asarray(bgr)
    if img.ndim != 3 or img.shape[2] != 3:
        raise ValueError("expected a BGR image with shape (H, W, 3)")
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    if preprocess:
        img = clahe_bgr(img)

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1].astype(np.float32)
    mean_sat = float(np.mean(sat))
    chromatic = mean_sat >= cfg.gray_sat

    color_mask = _sign_color_mask(hsv)
    white_mask = ((hsv[:, :, 1] < 55) & (hsv[:, :, 2] > 145)).astype(np.uint8) * 255
    mser = _mser_detector(cfg)
    candidates: List[Tuple[Box, float]] = []
    if chromatic:
        candidates.extend(_hsv_components(img, hsv, cfg, color_mask))
        candidates.extend(_white_plates(hsv, cfg, white_mask))
        candidates.extend(_mser_color_enhanced(img, cfg, mser))
    else:
        candidates.extend(_white_plates(hsv, cfg, white_mask))
    candidates.extend(_mser_gray(img, cfg, mser))

    filtered: List[Tuple[Box, float]] = []
    seen = set()
    for box, score in candidates:
        expanded = _expand_box(box, img.shape, cfg.box_expand)
        clipped = _clip_box(expanded, img.shape)
        if clipped is None or not _size_ok(clipped, img.shape, cfg):
            continue
        key = tuple(int(round(v)) for v in clipped)
        if key in seen:
            continue
        seen.add(key)
        crop = _crop(img, clipped)
        if crop is None:
            continue
        geom = _soft_geometry(crop, clipped, color_mask, white_mask)
        filtered.append((clipped, float(score) * geom))

    if not filtered:
        return [], []
    boxes = [item[0] for item in filtered]
    scores = [item[1] for item in filtered]
    keep_idx = _suppress_contained(boxes, cover=0.82, max_area_ratio=8.0)
    boxes = [boxes[i] for i in keep_idx]
    scores = [scores[i] for i in keep_idx]
    keep = _nms(boxes, scores, cfg.nms_iou)
    boxes = [boxes[i] for i in keep]
    scores = [scores[i] for i in keep]
    budget = int(cfg.max_boxes if max_boxes is None else max_boxes)
    if budget > 0:
        boxes = boxes[:budget]
        scores = scores[:budget]
    return boxes, scores


def _hsv_components(
    bgr: np.ndarray,
    hsv: np.ndarray,
    cfg: RoiConfig,
    color_mask: Optional[np.ndarray] = None,
) -> List[Tuple[Box, float]]:
    mask = _sign_color_mask(hsv) if color_mask is None else color_mask.copy()
    k = max(3, int(cfg.morph_ksize) | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = _fill_holes(mask)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    n_labels, _labels, stats, _cent = cv2.connectedComponentsWithStats(mask, connectivity=8)
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]
    out: List[Tuple[Box, float]] = []
    for label in range(1, n_labels):
        x, y, w, h, area = (int(v) for v in stats[label])
        box: Box = (float(x), float(y), float(w), float(h))
        if w < cfg.min_side or h < cfg.min_side:
            continue
        region = sat[y : y + h, x : x + w]
        v_region = val[y : y + h, x : x + w]
        if region.size == 0:
            continue
        mean_s = float(np.mean(region)) / 255.0
        mean_v = float(np.mean(v_region)) / 255.0
        compactness = float(area) / max(float(w * h), 1.0)
        score = (0.55 + 0.45 * mean_s) * (0.4 + 0.6 * compactness) * (0.6 + 0.4 * mean_v)
        out.append((box, score))
    return out


def _sign_color_mask(hsv: np.ndarray) -> np.ndarray:
    """US sign paint: red (wrap), yellow/amber, blue, fluorescent yellow-green."""
    ranges = (
        ((0, 70, 50), (12, 255, 255)),
        ((165, 70, 50), (179, 255, 255)),
        ((14, 70, 70), (38, 255, 255)),
        ((38, 40, 50), (95, 255, 255)),
        ((95, 60, 40), (135, 255, 255)),
    )
    mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for lo, hi in ranges:
        mask = cv2.bitwise_or(mask, cv2.inRange(hsv, np.array(lo), np.array(hi)))
    return mask


def _white_plates(
    hsv: np.ndarray,
    cfg: RoiConfig,
    white_mask: Optional[np.ndarray] = None,
) -> List[Tuple[Box, float]]:
    """Bright, low-saturation blobs: speed-limit / keep-right boards."""
    val = hsv[:, :, 2]
    mask = (
        ((hsv[:, :, 1] < 55) & (val > 145)).astype(np.uint8) * 255
        if white_mask is None
        else white_mask.copy()
    )
    k = max(3, int(cfg.morph_ksize) | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = _fill_holes(mask)
    n_labels, _labels, stats, _cent = cv2.connectedComponentsWithStats(mask, connectivity=8)
    out: List[Tuple[Box, float]] = []
    for label in range(1, n_labels):
        x, y, w, h, area = (int(v) for v in stats[label])
        if w < cfg.min_side or h < cfg.min_side:
            continue
        compactness = float(area) / max(float(w * h), 1.0)
        if compactness < 0.35:
            continue
        score = 0.9 * compactness * (0.5 + 0.5 * float(np.mean(val[y : y + h, x : x + w])) / 255.0)
        out.append(((float(x), float(y), float(w), float(h)), score))
    return out


def _fill_holes(mask: np.ndarray) -> np.ndarray:
    contours, _hier = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(mask)
    if contours:
        cv2.drawContours(filled, contours, -1, 255, thickness=cv2.FILLED)
    return filled


def _mser_gray(bgr: np.ndarray, cfg: RoiConfig, detector) -> List[Tuple[Box, float]]:
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    lightness = lab[:, :, 0]
    return _mser_on(lightness, cfg, 0.85, detector)


def _mser_color_enhanced(bgr: np.ndarray, cfg: RoiConfig, detector) -> List[Tuple[Box, float]]:
    b, g, r = cv2.split(bgr)
    bf = b.astype(np.int16)
    gf = g.astype(np.int16)
    rf = r.astype(np.int16)
    red = np.clip(rf - np.maximum(gf, bf), 0, 255).astype(np.uint8)
    blue = np.clip(bf - np.maximum(rf, gf), 0, 255).astype(np.uint8)
    yellow = np.clip(((rf + gf) // 2) - bf, 0, 255).astype(np.uint8)
    out: List[Tuple[Box, float]] = []
    for channel, weight in ((red, 1.05), (yellow, 1.0), (blue, 0.95)):
        out.extend(_mser_on(channel, cfg, weight, detector))
    return out


def _mser_detector(cfg: RoiConfig):
    args = (
        int(cfg.mser_delta),
        int(cfg.mser_min_area),
        int(cfg.mser_max_area),
        float(cfg.mser_max_variation),
        0.2,
        200,
        1.01,
        0.003,
        5,
    )
    try:
        return cv2.MSER_create(*args)
    except TypeError:
        return cv2.MSER_create()


def _mser_on(
    channel: np.ndarray,
    cfg: RoiConfig,
    base_score: float,
    detector,
) -> List[Tuple[Box, float]]:
    detected = detector.detectRegions(channel)
    regions = detected[0] if isinstance(detected, tuple) else detected
    out: List[Tuple[Box, float]] = []
    h, w = channel.shape[:2]
    for pts in regions:
        if pts is None or len(pts) < 8:
            continue
        x, y, bw, bh = cv2.boundingRect(pts)
        if bw < cfg.min_side or bh < cfg.min_side:
            continue
        if x <= 1 and y <= 1 and bw >= w - 2 and bh >= h - 2:
            continue
        hull = cv2.convexHull(pts)
        hull_area = float(cv2.contourArea(hull))
        if hull_area < 1.0:
            continue
        fill = float(len(pts)) / hull_area
        compactness = hull_area / max(float(bw * bh), 1.0)
        score = float(base_score) * (0.35 + 0.65 * compactness) * (0.4 + 0.6 * min(fill, 1.0))
        out.append(((float(x), float(y), float(bw), float(bh)), score))
    return out


def _size_ok(box: Box, image_shape: Sequence[int], cfg: RoiConfig) -> bool:
    _x, _y, w, h = box
    ih, iw = int(image_shape[0]), int(image_shape[1])
    max_side = min(float(cfg.max_side), cfg.max_side_frac * float(min(ih, iw)))
    if w < cfg.min_side or h < cfg.min_side:
        return False
    if w > max_side or h > max_side:
        return False
    aspect = w / max(h, 1e-6)
    return cfg.min_aspect <= aspect <= cfg.max_aspect


def _soft_geometry(
    crop: np.ndarray,
    box: Box,
    color_mask: np.ndarray,
    white_mask: np.ndarray,
) -> float:
    """Prefer sign-colored / high-contrast plates over textured foliage."""
    w, h = box[2], box[3]
    diag = max(float(np.hypot(w, h)), 1.0)
    size_w = float(np.exp(-0.5 * ((np.log(diag) - np.log(48.0)) / 1.05) ** 2))
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY).astype(np.float32)
    contrast = min(float(np.std(gray)) / 50.0, 1.5) / 1.5
    x, y, bw, bh = (int(round(v)) for v in box)
    y1, y2 = max(0, y), min(color_mask.shape[0], y + max(bh, 1))
    x1, x2 = max(0, x), min(color_mask.shape[1], x + max(bw, 1))
    if y2 <= y1 or x2 <= x1:
        color_frac = white_frac = 0.0
    else:
        color_frac = float(np.mean(color_mask[y1:y2, x1:x2])) / 255.0
        white_frac = float(np.mean(white_mask[y1:y2, x1:x2])) / 255.0
    paint = max(color_frac, 0.65 * white_frac)
    return (0.2 + 0.8 * size_w) * (0.15 + 1.7 * paint + 0.45 * contrast)


def _expand_box(box: Sequence[float], image_shape: Sequence[int], scale: float) -> Box:
    x, y, w, h = (float(v) for v in box[:4])
    cx, cy = x + 0.5 * w, y + 0.5 * h
    nw, nh = w * float(scale), h * float(scale)
    return (cx - 0.5 * nw, cy - 0.5 * nh, nw, nh)


def _suppress_contained(
    boxes: Sequence[Box],
    cover: float = 0.82,
    max_area_ratio: float = 8.0,
) -> List[int]:
    """Drop letter-sized MSER boxes inside a plate, not signs inside foliage."""
    if not boxes:
        return []
    areas = np.array([b[2] * b[3] for b in boxes], dtype=np.float64)
    order = np.argsort(-areas)
    keep: List[int] = []
    for idx in order:
        box = boxes[int(idx)]
        area = max(float(areas[int(idx)]), 1.0)
        contained = False
        for k in keep:
            parent_area = max(float(areas[k]), 1.0)
            if parent_area / area > max_area_ratio:
                continue
            if _intersection_area(box, boxes[k]) / area >= cover:
                contained = True
                break
        if not contained:
            keep.append(int(idx))
    return keep


def _intersection_area(a: Box, b: Box) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix1, iy1 = max(ax, bx), max(ay, by)
    ix2, iy2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    return max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)


def _clip_box(box: Sequence[float], image_shape: Sequence[int]) -> Optional[Box]:
    ih, iw = int(image_shape[0]), int(image_shape[1])
    x, y, w, h = (float(v) for v in box[:4])
    x1 = max(0.0, x)
    y1 = max(0.0, y)
    x2 = min(float(iw), x + w)
    y2 = min(float(ih), y + h)
    nw = x2 - x1
    nh = y2 - y1
    if nw < 4.0 or nh < 4.0:
        return None
    return (x1, y1, nw, nh)


def _crop(bgr: np.ndarray, box: Box) -> Optional[np.ndarray]:
    x, y, w, h = box
    x0 = int(np.floor(x))
    y0 = int(np.floor(y))
    x1 = int(np.ceil(x + w))
    y1 = int(np.ceil(y + h))
    if x1 - x0 < 4 or y1 - y0 < 4:
        return None
    return bgr[y0:y1, x0:x1]


def _nms(
    boxes: Sequence[Box],
    scores: Sequence[float],
    iou_threshold: float,
) -> List[int]:
    if not boxes:
        return []
    order = np.argsort(-np.asarray(scores, dtype=np.float64))
    keep: List[int] = []
    for idx in order:
        box = boxes[int(idx)]
        if all(iou(box, boxes[k]) <= iou_threshold for k in keep):
            keep.append(int(idx))
    return keep


def visualize_proposals(
    bgr: np.ndarray,
    boxes: Iterable[Sequence[float]],
    scores: Optional[Sequence[float]] = None,
    max_draw: int = 40,
) -> np.ndarray:
    """BGR overlay for debugging; does not change detection."""
    canvas = np.asarray(bgr).copy()
    if canvas.dtype != np.uint8:
        canvas = np.clip(canvas, 0, 255).astype(np.uint8)
    score_list = None if scores is None else list(scores)
    for i, box in enumerate(list(boxes)[:max_draw]):
        x, y, w, h = (int(round(v)) for v in box[:4])
        cv2.rectangle(canvas, (x, y), (x + w, y + h), (0, 220, 80), 2)
        if score_list is not None and i < len(score_list):
            cv2.putText(
                canvas,
                f"{score_list[i]:.2f}",
                (x, max(12, y - 4)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (0, 220, 80),
                1,
                cv2.LINE_AA,
            )
    return canvas
