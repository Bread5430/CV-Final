"""Draw the ACF stages from ``acf.py`` on a LISA frame.

Run from FinalProj::

    python visualize_acf.py
    python visualize_acf.py --image-id 12 --shrink 2

Writes ``acf_channels.png`` (the 10 maps) and ``acf_window.png``
(one GT window, cell grid, and a toy heatmap).
"""

from __future__ import annotations

import argparse
import os
import textwrap
from typing import List, Optional, Sequence, Tuple

import cv2
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

from acf import (
    CHANNEL_NAMES,
    AcfChannels,
    clahe_bgr,
    extract_acf,
    heatmap_on_image,
    template_heatmap,
    window_vector,
)
from lisa_coco import LISACocoDataset

_CAPTION = (
    "ACF = 10 registered channels, then mean-pool by shrink. A sliding-window "
    "feature is one cell lookup (not a Haar rectangle). Nam / Dollár train "
    "RealBoost trees on that vector; panel 5 uses cosine similarity only so "
    "you can see the window geometry without fitting a cascade."
)


def _bgr_to_rgb(bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _stretch(ch: np.ndarray, p_lo: float = 2.0, p_hi: float = 98.0, gamma: float = 0.7) -> np.ndarray:
    """Percentile stretch so sparse orientation energy is visible."""
    lo, hi = np.percentile(ch, (p_lo, p_hi))
    if hi - lo < 1e-6:
        return np.zeros_like(ch, dtype=np.float32)
    x = np.clip((ch - lo) / (hi - lo), 0, 1)
    return np.power(x, gamma).astype(np.float32)


def _colorfulness(bgr: np.ndarray) -> float:
    """0 for grayscale letterboxed frames; higher for red/yellow signs."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    sat = hsv[:, :, 1]
    # Ignore near-black letterbox.
    mask = np.max(bgr, axis=2) > 20
    if not np.any(mask):
        return 0.0
    return float(np.mean(sat[mask]))


def _pick_example(
    dataset: LISACocoDataset,
    image_id: Optional[int],
    min_box: int,
) -> Tuple[np.ndarray, List[dict], int]:
    if image_id is not None:
        return dataset.get_by_id(image_id)

    best_id = None
    best_score = -1.0
    prefer = ("stop", "yield", "pedestrian", "signal", "school", "warning")
    # Colorful + reasonably large box; skip Roboflow grayscale dumps.
    n = min(len(dataset), 400)
    step = max(1, len(dataset) // n)
    for i in range(0, len(dataset), step):
        record = dataset.image_records()[i]
        anns = dataset.annotations_for(int(record["id"]))
        if not anns:
            continue
        bgr, _, image_id = dataset.get(i)
        color = _colorfulness(bgr)
        if color < 18:
            continue
        area = max(a["bbox"][2] * a["bbox"][3] for a in anns)
        names = " ".join(str(a.get("category_name", "")) for a in anns).lower()
        bonus = 1.5 if any(p in names for p in prefer) else 1.0
        w_max = max(a["bbox"][2] for a in anns)
        h_max = max(a["bbox"][3] for a in anns)
        if w_max < min_box or h_max < min_box:
            continue
        score = area * bonus * (1.0 + color / 50.0)
        if score > best_score:
            best_score = score
            best_id = image_id
    if best_id is None:
        return dataset.get(0)
    return dataset.get_by_id(best_id)


def _draw_boxes(ax, anns: Sequence[dict], color: str = "#00e5ff", lw: float = 1.8) -> None:
    for ann in anns:
        x, y, w, h = ann["bbox"]
        ax.add_patch(Rectangle((x, y), w, h, fill=False, edgecolor=color, linewidth=lw))
        ax.text(
            x,
            max(0, y - 3),
            str(ann.get("category_name", "")),
            color=color,
            fontsize=8,
            va="bottom",
            fontweight="bold",
        )


def _plain(ax) -> None:
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color("#333333")


def save_channel_figure(
    rgb: np.ndarray,
    anns: Sequence[dict],
    acf: AcfChannels,
    path: str,
) -> None:
    """RGB plus the 10 full-resolution channels (what aggregation pools)."""
    fig = plt.figure(figsize=(16.0, 9.2), facecolor="#111111")
    gs = gridspec.GridSpec(3, 4, figure=fig, wspace=0.08, hspace=0.28,
                           left=0.04, right=0.99, top=0.88, bottom=0.08)

    fig.suptitle(
        "Simple ACF  —  10 channels used by Nam / Dollár (before pooling)",
        color="white",
        fontsize=15,
        fontweight="bold",
    )
    fig.text(
        0.04,
        0.905,
        textwrap.fill(_CAPTION, width=130),
        color="#cfd8dc",
        fontsize=8.5,
    )

    ax0 = fig.add_subplot(gs[0:2, 0])
    ax0.imshow(rgb)
    _draw_boxes(ax0, anns)
    ax0.set_title("input + GT  (ROI is a window on this)", color="#ffd54f", fontsize=10)
    _plain(ax0)

    # Channel order: L U V |G| then 6 orientations in the remaining slots.
    # Row0: L U V; row1: mag o0 o1; row2: o2 o3 o4 o5  — wait 4 cols.
    # Layout:
    #   [RGB span 2 rows] [L] [U] [V]
    #   [RGB            ] [|G|] [0°] [30°]
    #   [60°] [90°] [120°] [150°]
    positions = [
        (0, 1, 0), (0, 2, 1), (0, 3, 2),
        (1, 1, 3), (1, 2, 4), (1, 3, 5),
        (2, 0, 6), (2, 1, 7), (2, 2, 8), (2, 3, 9),
    ]
    hints = [
        "brightness (letters, sky)",
        "red/green opponent — stop signs light up",
        "blue/yellow opponent — warning diamonds",
        "edge strength, any orientation",
        "near-horizontal edges",
        "shallow diagonals",
        "steep diagonals",
        "near-vertical edges (posts, sides of signs)",
        "other diagonal",
        "other diagonal",
    ]
    for r, c, ch_i in positions:
        ax = fig.add_subplot(gs[r, c])
        ax.imshow(_stretch(acf.full[:, :, ch_i]), cmap="inferno", vmin=0, vmax=1)
        ax.set_title(CHANNEL_NAMES[ch_i], color="white", fontsize=9, pad=3)
        ax.set_xlabel(hints[ch_i], color="#90a4ae", fontsize=7)
        _plain(ax)

    fig.text(
        0.04,
        0.02,
        f"Next step (see acf_window.png): mean-pool every {acf.shrink}×{acf.shrink} block. "
        f"Full {acf.full.shape[1]}×{acf.full.shape[0]}  →  "
        f"{acf.aggregated.shape[1]}×{acf.aggregated.shape[0]} cells × 10 channels. "
        "ICF would Haar-filter these maps; ACF just looks up a cell.",
        color="#80cbc4",
        fontsize=8.5,
    )
    fig.savefig(path, dpi=130, facecolor=fig.get_facecolor())
    plt.close(fig)


def _largest_ann(anns: Sequence[dict]) -> dict:
    return max(anns, key=lambda a: a["bbox"][2] * a["bbox"][3])


def save_window_figure(
    bgr: np.ndarray,
    rgb: np.ndarray,
    anns: Sequence[dict],
    acf: AcfChannels,
    path: str,
) -> None:
    if not anns:
        raise RuntimeError("image has no boxes to build a window from")
    ann = _largest_ann(anns)
    x, y, bw, bh = (float(v) for v in ann["bbox"])
    shrink = acf.shrink
    feat = window_vector(acf.aggregated, x, y, bw, bh, shrink)

    x0 = max(0, int(np.floor(x / shrink)))
    y0 = max(0, int(np.floor(y / shrink)))
    x1 = min(acf.aggregated.shape[1], int(np.ceil((x + bw) / shrink)))
    y1 = min(acf.aggregated.shape[0], int(np.ceil((y + bh) / shrink)))
    win_w = max(1, x1 - x0)
    win_h = max(1, y1 - y0)
    template = acf.aggregated[y0:y1, x0:x1, :].copy()

    heat = template_heatmap(acf.aggregated, template, win_h, win_w, stride=2)
    heat_full = heatmap_on_image(
        heat,
        (bgr.shape[0], bgr.shape[1]),
        win_h,
        win_w,
        shrink,
        stride=2,
    )

    pad = max(8, int(0.15 * max(bw, bh)))
    x_i, y_i = int(round(x)), int(round(y))
    w_i, h_i = int(round(bw)), int(round(bh))
    x0c = max(0, x_i - pad)
    y0c = max(0, y_i - pad)
    x1c = min(rgb.shape[1], x_i + w_i + pad)
    y1c = min(rgb.shape[0], y_i + h_i + pad)
    crop = rgb[y0c:y1c, x0c:x1c].copy()

    fig = plt.figure(figsize=(15.2, 9.4), facecolor="#111111")
    gs = gridspec.GridSpec(
        3, 5,
        figure=fig,
        wspace=0.12,
        hspace=0.32,
        left=0.04,
        right=0.99,
        top=0.88,
        bottom=0.07,
        height_ratios=[1.15, 1.0, 0.95],
    )
    fig.suptitle(
        f"ACF window  ·  {ann.get('category_name', '?')}  ·  "
        f"{w_i}×{h_i} px  →  {win_w}×{win_h} cells × 10  =  {feat.size} features",
        color="white",
        fontsize=14,
        fontweight="bold",
    )

    ax = fig.add_subplot(gs[0, 0:2])
    ax.imshow(rgb)
    _draw_boxes(ax, anns, color="#00e5ff", lw=1.4)
    ax.add_patch(Rectangle((x, y), bw, bh, fill=False, edgecolor="#ffeb3b", linewidth=2.5))
    ax.set_title("1. Detection window on the frame", color="#ffd54f", fontsize=10)
    _plain(ax)

    ax = fig.add_subplot(gs[0, 2])
    ax.imshow(crop)
    for gx in range(0, crop.shape[1] + 1, shrink):
        ax.axvline(gx, color="white", linewidth=0.45, alpha=0.65)
    for gy in range(0, crop.shape[0] + 1, shrink):
        ax.axhline(gy, color="white", linewidth=0.45, alpha=0.65)
    ax.add_patch(
        Rectangle((x_i - x0c, y_i - y0c), w_i, h_i, fill=False, edgecolor="#ffeb3b", linewidth=1.8)
    )
    ax.set_title(f"2. Aggregate {shrink}×{shrink} (ACF cells)", color="#80cbc4", fontsize=10)
    _plain(ax)

    ax = fig.add_subplot(gs[0, 3:5])
    ax.imshow(rgb)
    ax.imshow(heat_full, cmap="jet", alpha=0.5)
    ax.add_patch(Rectangle((x, y), bw, bh, fill=False, edgecolor="white", linewidth=2.0))
    ax.set_title("5. Toy heatmap (score at window center)", color="#ffe082", fontsize=10)
    _plain(ax)

    # 10 aggregated channel tiles of the window itself.
    for i in range(10):
        r, c = divmod(i, 5)
        ax = fig.add_subplot(gs[1 + r, c])
        ax.imshow(_stretch(template[:, :, i], 1, 99, 0.65), cmap="inferno", vmin=0, vmax=1)
        ax.set_title(CHANNEL_NAMES[i], color="white", fontsize=8, pad=2)
        _plain(ax)

    fig.text(
        0.04,
        0.015,
        textwrap.dedent(
            f"""
            Yellow box = the ROI. White grid = ACF cells (mean of each {shrink}×{shrink} block).
            The 10 tiles are exactly the numbers a boosted tree would threshold:  cell (u,v) on channel c.
            Heatmap is NOT AdaBoost — cosine(this crop, every same-size window), drawn at the window
            center. A peak should sit inside the yellow box. Real ACF still needs RealBoost + cascade
            + a scale pyramid; cosine is a weak stand-in on both pedestrians and signs.
            """
        ).strip(),
        color="#b0bec5",
        fontsize=8.5,
    )
    fig.savefig(path, dpi=130, facecolor=fig.get_facecolor())
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Visualize a simple ACF pipeline on LISA.")
    p.add_argument("--split", default="test", choices=("train", "valid", "test"))
    p.add_argument("--image-id", type=int, default=None)
    p.add_argument("--shrink", type=int, default=2, help="Nam et al. used 2; classic ACF often 4.")
    p.add_argument("--min-box", type=int, default=36)
    p.add_argument("--out-dir", default=None)
    p.add_argument("--model", default=None, help="Optional acf_boosted.pkl to draw a boosted heatmap.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = args.out_dir or os.path.dirname(os.path.abspath(__file__))
    os.makedirs(out_dir, exist_ok=True)

    data = LISACocoDataset(split=args.split, color="bgr")
    bgr, anns, image_id = _pick_example(data, args.image_id, args.min_box)
    bgr = clahe_bgr(bgr)
    rgb = _bgr_to_rgb(bgr)
    acf = extract_acf(bgr, shrink=args.shrink)

    ch_path = os.path.join(out_dir, "acf_channels.png")
    win_path = os.path.join(out_dir, "acf_window.png")
    save_channel_figure(rgb, anns, acf, ch_path)
    save_window_figure(bgr, rgb, anns, acf, win_path)

    print(f"image_id={image_id}  shape={bgr.shape}  colorfulness={_colorfulness(bgr):.1f}")
    print(f"full channels    {acf.full.shape}")
    print(f"aggregated cells {acf.aggregated.shape}  shrink={acf.shrink}")
    print(f"wrote {ch_path}")
    print(f"wrote {win_path}")

    if args.model:
        from acf_boost import BoostedAcf

        booster = BoostedAcf.load(args.model)
        boxes, scores, heat = booster.detect_image(bgr, stride=2)
        print(f"boosted detections: {len(boxes)}  scores={np.round(scores[:8], 3)}")
        # Overlay is already written by train_acf.py; keep this as a sanity print.


if __name__ == "__main__":
    main()
