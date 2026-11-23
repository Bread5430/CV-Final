"""Simple Aggregate Channel Features (ACF), matching Nam / Dollár 2014.

This is the *feature* half of the ACF detector used as the LDCF baseline.
The published detector then trains RealBoost over these lookups
(see ``acf_boost.py``). ``visualize_acf.py`` draws the channels.

Pipeline
--------
    BGR frame
      -> CLAHE on LAB lightness (once, full frame; see ``clahe_bgr``)
      -> 10 registered channels (same H x W as the image)
           3 LUV color
           1 gradient magnitude
           6 unsigned orientation maps  (0, 30, ..., 150 deg)
      -> shrink by ``shrink`` (Nam used 2; classic ACF often uses 4)
           each output pixel is the *mean* of a shrink x shrink block
      -> a sliding window is just those aggregated pixels, flattened
           feature dim = (h/shrink) * (w/shrink) * 10

ICF vs ACF (why aggregation)
----------------------------
ICF scores Haar rectangles on *integral images* of the 10 channels.
ACF skips Haar: after pooling, a "feature" is a single cell lookup.
That is cheaper, and nearby pyramid scales can be approximated instead
of recomputing channels at every scale (fast feature pyramids).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import cv2
import numpy as np

# 10 channel names, in the order Dollár / Nam use: L, U, V, |G|, 6 HOG bins.
NUM_ORIENTATIONS = 6
CHANNEL_NAMES: List[str] = [
    "L  lightness",
    "U  green–red",
    "V  blue–yellow",
    "|∇|  magnitude",
    "orient  0°",
    "orient  30°",
    "orient  60°",
    "orient  90°",
    "orient  120°",
    "orient  150°",
]


@dataclass
class AcfChannels:
    """Full-resolution channels plus the pooled (aggregated) tensor.

    ``full``     float32  (H, W, 10)     — one value per pixel per channel
    ``aggregated`` float32 (H/s, W/s, 10) — ACF features live here
    ``shrink``   int
    """

    full: np.ndarray
    aggregated: np.ndarray
    shrink: int

    @property
    def names(self) -> List[str]:
        return list(CHANNEL_NAMES)


def clahe_bgr(
    bgr: np.ndarray,
    clip_limit: float = 2.0,
    tile_size: int = 8,
) -> np.ndarray:
    """Local contrast equalization on lightness only (LAB L channel).

    LISA frames vary a lot in sun/shadow; there is no other color
    normalization before ACF. CLAHE on L leaves a/b (opponent color) so
    red/yellow plates keep their hue. Apply **once** on the full camera
    frame, then crop / pyramid / ACF, so train windows and ``detect_image``
    share the same tiles.
    """
    img = np.asarray(bgr)
    if img.ndim != 3 or img.shape[2] != 3:
        raise ValueError("expected a BGR image with shape (H, W, 3)")
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    lightness, a_ch, b_ch = cv2.split(lab)
    equalizer = cv2.createCLAHE(
        clipLimit=float(clip_limit),
        tileGridSize=(int(tile_size), int(tile_size)),
    )
    merged = cv2.merge([equalizer.apply(lightness), a_ch, b_ch])
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)


def bgr_to_luv(bgr: np.ndarray) -> np.ndarray:
    """LUV color: the 3 channels ACF uses instead of RGB or grayscale.

    L  = brightness (a stop sign's white letters light up here)
    U,V = opponent color (red of a stop / yellow of a warning diamond)

    OpenCV's LUV is an approximation of CIE L*u*v*. Piotr's MATLAB toolbox
    uses a slightly different conversion; both give 3 registered color maps.
    """
    if bgr.ndim != 3 or bgr.shape[2] != 3:
        raise ValueError("expected a BGR image with shape (H, W, 3)")
    luv = cv2.cvtColor(bgr, cv2.COLOR_BGR2LUV).astype(np.float32)
    return luv


def color_gradient(bgr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Per-pixel gradient, taking the RGB channel with the strongest edge.

    Dollár's channel detectors do *not* convert to gray first. At each pixel
    they compute Sobel on R, G, and B and keep the channel whose magnitude
    is largest. That way a red/green sign edge is not washed out by averaging.

    Returns
    -------
    magnitude : (H, W)  L2 length of the winning gradient
    angle     : (H, W)  orientation in [0, pi)  (unsigned; 0 == 180)
    """
    img = bgr.astype(np.float32)
    # One 3-channel Sobel is much cheaper than three Python-looped calls.
    dx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
    dy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(dx * dx + dy * dy)
    h, w = mag.shape[:2]
    ch = np.argmax(mag, axis=2).reshape(-1)
    pix = np.arange(h * w)
    mag_best = mag.reshape(-1, 3)[pix, ch].reshape(h, w)
    angle = np.mod(
        np.arctan2(dy.reshape(-1, 3)[pix, ch], dx.reshape(-1, 3)[pix, ch]),
        np.pi,
    ).reshape(h, w)
    return mag_best.astype(np.float32), angle.astype(np.float32)


def orientation_channels(
    magnitude: np.ndarray,
    angle: np.ndarray,
    n_bins: int = NUM_ORIENTATIONS,
) -> np.ndarray:
    """Soft-bin magnitude into ``n_bins`` unsigned orientation maps.

    Pixel (y, x) with edge strength m and angle θ contributes
        (1 - a) * m  to bin floor(θ / Δ)
            a  * m  to the next bin
    with Δ = π / n_bins. That is a HOG histogram *per pixel*, not per cell.
    Aggregation (the next step) is what turns these maps into cell sums.

    Shape: (H, W, n_bins)
    """
    h, w = magnitude.shape
    delta = np.pi / n_bins
    # bin coordinate in [0, n_bins)
    coord = angle / delta
    bin0 = np.floor(coord).astype(np.int32) % n_bins
    bin1 = (bin0 + 1) % n_bins
    alpha = coord - np.floor(coord)

    mag = magnitude.astype(np.float32)
    a = alpha.astype(np.float32)
    w0 = mag * (1.0 - a)
    w1 = mag * a
    out = np.zeros((n_bins, h, w), dtype=np.float32)
    for k in range(n_bins):
        m0 = bin0 == k
        m1 = bin1 == k
        if np.any(m0):
            out[k][m0] = w0[m0]
        if np.any(m1):
            out[k][m1] += w1[m1]
    return np.transpose(out, (1, 2, 0))


def compute_full_channels(bgr: np.ndarray) -> np.ndarray:
    """Stack the 10 ACF channels at full image resolution.

    Layout along axis=2:
        0:3   L, U, V
        3     gradient magnitude
        4:10  six orientation channels
    """
    luv = bgr_to_luv(bgr)
    mag, ang = color_gradient(bgr)
    orients = orientation_channels(mag, ang)
    mag = mag[:, :, None]
    return np.concatenate([luv, mag, orients], axis=2).astype(np.float32)


def aggregate_channels(channels: np.ndarray, shrink: int = 2) -> np.ndarray:
    """Mean-pool each channel by ``shrink`` (the 'A' in ACF).

    shrink=2 is Nam et al. (LDCF baseline). Classic ACF often uses 4.
    ``INTER_AREA`` with an exact integer scale is a box average, i.e. the
    same thing as summing a shrink×shrink block and dividing by shrink².

    After this, an ACF feature is literally ``aggregated[y, x, c]``.
    """
    if shrink < 1:
        raise ValueError("shrink must be >= 1")
    if shrink == 1:
        return channels.copy()
    h, w, n_ch = channels.shape
    out_h, out_w = h // shrink, w // shrink
    if out_h < 1 or out_w < 1:
        raise ValueError(f"image {h}x{w} is smaller than shrink={shrink}")
    # Crop so H, W are multiples of shrink; INTER_AREA then averages each block.
    cropped = channels[: out_h * shrink, : out_w * shrink]
    return cropped.reshape(out_h, shrink, out_w, shrink, n_ch).mean(axis=(1, 3)).astype(np.float32)


def resize_patch(patch: np.ndarray, win_h: int, win_w: int) -> np.ndarray:
    """Resize an aggregated (h, w, C) crop to the model's fixed cell window."""
    if patch.ndim != 3:
        raise ValueError("patch must be (h, w, C)")
    if patch.shape[0] < 1 or patch.shape[1] < 1:
        raise ValueError("empty patch")
    channels = [
        cv2.resize(patch[:, :, c], (win_w, win_h), interpolation=cv2.INTER_AREA)
        for c in range(patch.shape[2])
    ]
    return np.stack(channels, axis=2).astype(np.float32)


def crop_aggregated_box(
    aggregated: np.ndarray,
    box_xywh: Sequence[float],
    shrink: int,
) -> np.ndarray:
    """Slice the aggregated tensor that falls inside a full-image COCO box."""
    x, y, width, height = (float(v) for v in box_xywh[:4])
    h, w, _ = aggregated.shape
    x0 = max(0, int(np.floor(x / shrink)))
    y0 = max(0, int(np.floor(y / shrink)))
    x1 = min(w, int(np.ceil((x + width) / shrink)))
    y1 = min(h, int(np.ceil((y + height) / shrink)))
    x1 = max(x0 + 1, x1)
    y1 = max(y0 + 1, y1)
    return aggregated[y0:y1, x0:x1, :]


def extract_acf(bgr: np.ndarray, shrink: int = 2) -> AcfChannels:
    """Full ACF transform used by the Nam / Dollár detector (minus boosting)."""
    full = compute_full_channels(bgr)
    aggregated = aggregate_channels(full, shrink=shrink)
    return AcfChannels(full=full, aggregated=aggregated, shrink=shrink)


def resize_aggregated(
    aggregated: np.ndarray,
    out_h: int,
    out_w: int,
) -> np.ndarray:
    """Resize an aggregated (h, w, C) map. Used for fast feature pyramids."""
    h, w, n_ch = aggregated.shape
    if out_h < 1 or out_w < 1:
        raise ValueError("aggregated resize must be at least 1x1")
    if h == out_h and w == out_w:
        return aggregated
    interp = cv2.INTER_AREA if (out_h < h or out_w < w) else cv2.INTER_LINEAR
    channels = [
        cv2.resize(aggregated[:, :, c], (out_w, out_h), interpolation=interp)
        for c in range(n_ch)
    ]
    return np.stack(channels, axis=2).astype(np.float32)


def window_vector(
    aggregated: np.ndarray,
    x: int,
    y: int,
    width: int,
    height: int,
    shrink: int,
) -> np.ndarray:
    """Flatten the aggregated cells that fall inside a full-image box.

    ``(x, y, width, height)`` is a COCO box on the original image.
    Dividing by ``shrink`` maps it onto the pooled grid. This vector is
    exactly what a boosted tree would threshold one coordinate of.

    Dim = ceil(height/shrink) * ceil(width/shrink) * 10, after clipping.
    """
    h, w, _ = aggregated.shape
    x0 = int(np.floor(x / shrink))
    y0 = int(np.floor(y / shrink))
    x1 = int(np.ceil((x + width) / shrink))
    y1 = int(np.ceil((y + height) / shrink))
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(w, x1), min(h, y1)
    patch = aggregated[y0:y1, x0:x1, :]
    if patch.size == 0:
        return np.zeros((0,), dtype=np.float32)
    return patch.reshape(-1).astype(np.float32)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na < 1e-8 or nb < 1e-8:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def template_heatmap(
    aggregated: np.ndarray,
    template: np.ndarray,
    win_h: int,
    win_w: int,
    stride: int = 1,
) -> np.ndarray:
    """Naive sliding-window score: cosine sim of each cell-window to a template.

    The real ACF detector replaces this cosine with a RealBoost forest.
    The heatmap still shows the *geometry* of ACF: one score per pooled
    cell, cheap because the channels were computed once for the image.
    ``win_h``, ``win_w`` are in *aggregated* pixels.
    """
    h, w, n_ch = aggregated.shape
    th, tw = int(win_h), int(win_w)
    if th < 1 or tw < 1 or th > h or tw > w:
        raise ValueError("window does not fit on the aggregated map")
    tmpl = template.reshape(th, tw, n_ch).astype(np.float32)
    tmpl_flat = tmpl.reshape(-1)
    tmpl_norm = float(np.linalg.norm(tmpl_flat)) + 1e-8

    out_h = (h - th) // stride + 1
    out_w = (w - tw) // stride + 1
    heat = np.zeros((out_h, out_w), dtype=np.float32)
    # One row of windows at a time so we do not materialize a 5-D sliding view.
    xs = np.arange(0, w - tw + 1, stride)
    for iy, y in enumerate(range(0, h - th + 1, stride)):
        band = aggregated[y : y + th]
        patches = np.stack(
            [band[:, x : x + tw, :].reshape(-1) for x in xs],
            axis=0,
        )
        denom = (np.linalg.norm(patches, axis=1) + 1e-8) * tmpl_norm
        heat[iy] = patches @ tmpl_flat / denom
    return heat


def heatmap_on_image(
    heat: np.ndarray,
    image_hw: Tuple[int, int],
    win_h: int,
    win_w: int,
    shrink: int,
    stride: int = 1,
) -> np.ndarray:
    """Paint each window's score at its *center* in full-image pixels.

    ``heat[iy, ix]`` is the score of the window whose top-left aggregated
    cell is ``(iy * stride, ix * stride)``. Stretching ``heat`` to the
    image with ``cv2.resize`` is wrong: the last valid origin is
    ``(H - win_h, W - win_w)``, not the bottom-right pixel, so the blob
    slides down and right of the true box.
    """
    img_h, img_w = int(image_hw[0]), int(image_hw[1])
    out_h, out_w = heat.shape
    y0 = (win_h / 2.0) * shrink
    x0 = (win_w / 2.0) * shrink
    y1 = (((out_h - 1) * stride) + win_h / 2.0) * shrink
    x1 = (((out_w - 1) * stride) + win_w / 2.0) * shrink
    dest_h = max(1, int(round(y1 - y0)) + 1)
    dest_w = max(1, int(round(x1 - x0)) + 1)
    blob = cv2.resize(heat, (dest_w, dest_h), interpolation=cv2.INTER_LINEAR)
    canvas = np.zeros((img_h, img_w), dtype=np.float32)
    r0 = int(round(y0))
    c0 = int(round(x0))
    r1 = min(img_h, r0 + blob.shape[0])
    c1 = min(img_w, c0 + blob.shape[1])
    if r0 < img_h and c0 < img_w and r1 > max(0, r0) and c1 > max(0, c0):
        br0, bc0 = max(0, r0), max(0, c0)
        canvas[br0:r1, bc0:c1] = blob[br0 - r0 : r1 - r0, bc0 - c0 : c1 - c0]
    return canvas


def boxes_to_template(
    aggregated: np.ndarray,
    boxes: Sequence[Sequence[float]],
    shrink: int,
    win_h: int,
    win_w: int,
) -> np.ndarray:
    """Average ACF windows from ground-truth boxes, resized to a fixed cell size."""
    acc = np.zeros((win_h, win_w, aggregated.shape[2]), dtype=np.float32)
    n = 0
    h, w, _ = aggregated.shape
    for box in boxes:
        x, y, bw, bh = (float(v) for v in box[:4])
        x0 = int(np.floor(x / shrink))
        y0 = int(np.floor(y / shrink))
        x1 = int(np.ceil((x + bw) / shrink))
        y1 = int(np.ceil((y + bh) / shrink))
        x0, y0 = max(0, x0), max(0, y0)
        x1, y1 = min(w, max(x0 + 1, x1)), min(h, max(y0 + 1, y1))
        patch = aggregated[y0:y1, x0:x1, :]
        if patch.size == 0:
            continue
        resized = resize_patch(patch, win_h, win_w)
        acc += resized
        n += 1
    if n == 0:
        raise ValueError("no valid boxes for template")
    return acc / n
