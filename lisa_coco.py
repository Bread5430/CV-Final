"""COCO-style loader for the Roboflow LISA traffic-sign export."""

from __future__ import annotations

import json
import os
from collections import defaultdict
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np

Prediction = Union[
    Sequence[float],
    Dict[str, Any],
]

_SPLITS = frozenset({"train", "valid", "test"})
_DEFAULT_DATASET_DIR = "LISA.v2i.coco"
_ANNOTATION_FILE = "_annotations.coco.json"
# Mean HSV saturation; Roboflow grayscale dumps sit near 0.
COLOR_SAT_MIN = 16.0


def image_colorfulness(bgr: np.ndarray) -> float:
    """Mean saturation of non-black pixels. ~0 for letterboxed grayscale frames."""
    img = np.asarray(bgr)
    if img.ndim != 3 or img.shape[2] != 3:
        return 0.0
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = np.max(img, axis=2) > 20
    if not np.any(mask):
        return 0.0
    return float(np.mean(hsv[:, :, 1][mask]))


def is_color_photo(bgr: np.ndarray, min_sat: float = COLOR_SAT_MIN) -> bool:
    return image_colorfulness(bgr) >= float(min_sat)

# Sign classes used by the crop classifier. Detection ignores these and
# scores boxes only. ``background`` is sampled at train time, not a COCO type.
LOGICAL_CLASSES: List[str] = [
    "stop",
    "speedLimit",
    "pedestrianCrossing",
    "signalAhead",
    "laneChange",
    "keepRight",
    "warningRoad",
    "school",
    "yield",
    "controlAhead",
    "mandatoryTurn",
    "prohibitory",
]
BACKGROUND_CLASS = "background"
CLASSIFY_CLASSES: List[str] = [BACKGROUND_CLASS] + LOGICAL_CLASSES
LOGICAL_NAME_TO_ID: Dict[str, int] = {name: i for i, name in enumerate(LOGICAL_CLASSES)}
CLASSIFY_NAME_TO_ID: Dict[str, int] = {name: i for i, name in enumerate(CLASSIFY_CLASSES)}

LOGICAL_CLASS_MAP: Dict[str, str] = {
    "stop": "stop",
    "pedestrianCrossing": "pedestrianCrossing",
    "signalAhead": "signalAhead",
    "keepRight": "keepRight",
    "yield": "yield",
    "speedLimit15": "speedLimit",
    "speedLimit25": "speedLimit",
    "speedLimit30": "speedLimit",
    "speedLimit35": "speedLimit",
    "speedLimit40": "speedLimit",
    "speedLimit45": "speedLimit",
    "speedLimit50": "speedLimit",
    "speedLimit55": "speedLimit",
    "speedLimit65": "speedLimit",
    "speedLimitUrdbl": "speedLimit",
    "truckSpeedLimit55": "speedLimit",
    "school": "school",
    "schoolSpeedLimit25": "school",
    "stopAhead": "controlAhead",
    "yieldAhead": "controlAhead",
    "addedLane": "laneChange",
    "merge": "laneChange",
    "laneEnds": "laneChange",
    "thruTrafficMergeLeft": "laneChange",
    "thruMergeLeft": "laneChange",
    "thruMergeRight": "laneChange",
    "turnRight": "mandatoryTurn",
    "turnLeft": "mandatoryTurn",
    "rightLaneMustTurn": "mandatoryTurn",
    "noLeftTurn": "prohibitory",
    "noRightTurn": "prohibitory",
    "doNotEnter": "prohibitory",
    "doNotPass": "prohibitory",
    "curveLeft": "warningRoad",
    "curveRight": "warningRoad",
    "dip": "warningRoad",
    "slow": "warningRoad",
    "roundabout": "warningRoad",
    "intersection": "warningRoad",
    "rampSpeedAdvisory20": "warningRoad",
    "rampSpeedAdvisory35": "warningRoad",
    "rampSpeedAdvisory40": "warningRoad",
    "rampSpeedAdvisory45": "warningRoad",
    "rampSpeedAdvisory50": "warningRoad",
    "rampSpeedAdvisoryUrdbl": "warningRoad",
    "zoneAhead25": "warningRoad",
    "zoneAhead45": "warningRoad",
}


def logical_name_for(category_name: str) -> str:
    """Map a fine LISA / Roboflow name to a logical sign class."""
    if category_name not in LOGICAL_CLASS_MAP:
        raise KeyError(f"unmapped LISA category {category_name!r}")
    return LOGICAL_CLASS_MAP[category_name]


def logical_id_for(category_name: str) -> int:
    return LOGICAL_NAME_TO_ID[logical_name_for(category_name)]


def classify_id_for_logical(logical_id: int) -> int:
    """Shift a sign ``logical_id`` so ``background`` can occupy class 0."""
    return int(logical_id) + 1


def letterbox_bgr(crop: np.ndarray, size: int = 64) -> np.ndarray:
    """Resize a crop into a square canvas without stretching."""
    if size < 1:
        raise ValueError("size must be >= 1")
    img = np.asarray(crop)
    if img.ndim != 3 or img.shape[2] != 3:
        raise ValueError("expected a BGR crop with shape (H, W, 3)")
    h, w = img.shape[:2]
    canvas = np.zeros((size, size, 3), dtype=np.uint8)
    if h < 1 or w < 1:
        return canvas
    scale = size / float(max(h, w))
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    y0 = (size - new_h) // 2
    x0 = (size - new_w) // 2
    canvas[y0 : y0 + new_h, x0 : x0 + new_w] = resized
    return canvas


def letterbox_crop(
    bgr: np.ndarray,
    bbox: Sequence[float],
    pad: float = 1.4,
    size: int = 64,
) -> np.ndarray:
    """Padded, letterboxed BGR crop from a COCO ``[x, y, w, h]`` box."""
    x1, y1, x2, y2 = padded_box_xyxy(bbox, bgr.shape, pad=pad)
    return letterbox_bgr(bgr[y1:y2, x1:x2], size=size)


def padded_box_xyxy(
    bbox: Sequence[float],
    image_shape: Sequence[int],
    pad: float = 1.4,
) -> Tuple[int, int, int, int]:
    """Expand a COCO ``[x, y, w, h]`` box about its center, then clip."""
    height, width = int(image_shape[0]), int(image_shape[1])
    x, y, w, h = (float(v) for v in bbox[:4])
    cx = x + 0.5 * w
    cy = y + 0.5 * h
    nw = max(1.0, w * float(pad))
    nh = max(1.0, h * float(pad))
    x1 = int(np.floor(cx - 0.5 * nw))
    y1 = int(np.floor(cy - 0.5 * nh))
    x2 = int(np.ceil(cx + 0.5 * nw))
    y2 = int(np.ceil(cy + 0.5 * nh))
    x1 = max(0, min(x1, width - 1))
    y1 = max(0, min(y1, height - 1))
    x2 = max(x1 + 1, min(x2, width))
    y2 = max(y1 + 1, min(y2, height))
    return x1, y1, x2, y2


def iou(box_a: Sequence[float], box_b: Sequence[float]) -> float:
    """Intersection-over-union of two COCO boxes ``[x, y, w, h]``."""
    ax, ay, aw, ah = (float(v) for v in box_a[:4])
    bx, by, bw, bh = (float(v) for v in box_b[:4])
    if aw <= 0 or ah <= 0 or bw <= 0 or bh <= 0:
        return 0.0

    ax2, ay2 = ax + aw, ay + ah
    bx2, by2 = bx + bw, by + bh
    ix1, iy1 = max(ax, bx), max(ay, by)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter_w, inter_h = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = inter_w * inter_h
    if inter <= 0:
        return 0.0
    union = aw * ah + bw * bh - inter
    if union <= 0:
        return 0.0
    return inter / union


def _precision_recall(tp: int, fp: int, fn: int) -> Tuple[float, float]:
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    return precision, recall


def _parse_prediction(pred: Prediction) -> Tuple[Tuple[float, float, float, float], int, float]:
    """Return ``(bbox_xywh, category_id, score)`` from a tuple or dict."""
    if isinstance(pred, dict):
        bbox = pred.get("bbox")
        if bbox is None:
            bbox = (
                pred["x"],
                pred["y"],
                pred["w"],
                pred["h"],
            )
        category_id = int(pred.get("category_id", pred.get("label")))
        score = float(pred.get("score", 0.0))
        x, y, w, h = (float(v) for v in bbox[:4])
        return (x, y, w, h), category_id, score

    values = list(pred)
    if len(values) < 5:
        raise ValueError("Prediction must be (x, y, w, h, category_id[, score])")
    x, y, w, h = (float(v) for v in values[:4])
    category_id = int(values[4])
    score = float(values[5]) if len(values) > 5 else 0.0
    return (x, y, w, h), category_id, score


class LISACocoDataset:
    """Indexed access and batching over one LISA COCO split.

    Image ids restart at 0 in every split, so they are unique only within
    a single ``LISACocoDataset`` instance.

    Each annotation keeps the original COCO ``category_id`` / ``category_name``
    and also ``logical_id`` / ``logical_name`` for classification. Detection
    uses ``bbox`` only and ignores the logical labels.

    Grayscale Roboflow frames are dropped by default (``color_only=True``):
    they are not a realistic camera input for a color channel detector.
    """

    def __init__(
        self,
        split: str = "train",
        root: Optional[str] = None,
        color: str = "bgr",
        color_only: bool = True,
        min_colorfulness: float = COLOR_SAT_MIN,
    ) -> None:
        split = split.lower()
        if split not in _SPLITS:
            raise ValueError(f"split must be one of {sorted(_SPLITS)}, got {split!r}")
        color = color.lower()
        if color not in {"bgr", "rgb"}:
            raise ValueError("color must be 'bgr' or 'rgb'")

        if root is None:
            root = os.path.join(os.path.dirname(os.path.abspath(__file__)), _DEFAULT_DATASET_DIR)
        self.root = os.path.abspath(root)
        self.split = split
        self.color = color
        self.color_only = bool(color_only)
        self.min_colorfulness = float(min_colorfulness)
        self.split_dir = os.path.join(self.root, split)
        annotation_path = os.path.join(self.split_dir, _ANNOTATION_FILE)
        if not os.path.isfile(annotation_path):
            raise FileNotFoundError(f"Missing COCO annotations: {annotation_path}")

        with open(annotation_path, encoding="utf-8") as handle:
            coco = json.load(handle)

        self._images: List[Dict[str, Any]] = list(coco["images"])
        if self.color_only:
            before = len(self._images)
            self._images = self._keep_color_images(self._images)
            dropped = before - len(self._images)
            print(
                f"LISA {self.split}: kept {len(self._images)} color / {before} "
                f"images (dropped {dropped} grayscale, sat<{self.min_colorfulness:.0f})",
                flush=True,
            )
        self._id_to_index = {int(record["id"]): i for i, record in enumerate(self._images)}

        used_category_ids = {int(ann["category_id"]) for ann in coco["annotations"]}
        # Drop unused classes, including the Roboflow parent id 0 when it has no boxes.
        self._categories: List[Dict[str, Any]] = [
            cat for cat in coco["categories"] if int(cat["id"]) in used_category_ids
        ]
        self._categories.sort(key=lambda cat: int(cat["id"]))

        self.id_to_name = {int(cat["id"]): cat["name"] for cat in self._categories}
        self.name_to_id = {name: cid for cid, name in self.id_to_name.items()}
        self.categories = list(self._categories)

        anns_by_image: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
        for raw in coco["annotations"]:
            image_id = int(raw["image_id"])
            category_id = int(raw["category_id"])
            bbox = tuple(float(v) for v in raw["bbox"][:4])
            category_name = self.id_to_name.get(category_id, str(category_id))
            logical_name = logical_name_for(category_name)
            anns_by_image[image_id].append(
                {
                    "id": int(raw["id"]),
                    "image_id": image_id,
                    "bbox": bbox,
                    "category_id": category_id,
                    "category_name": category_name,
                    "logical_name": logical_name,
                    "logical_id": LOGICAL_NAME_TO_ID[logical_name],
                    "area": float(raw.get("area", bbox[2] * bbox[3])),
                    "iscrowd": int(raw.get("iscrowd", 0)),
                }
            )
        self._anns_by_image = dict(anns_by_image)

    def _color_index_path(self) -> str:
        sat = int(round(self.min_colorfulness))
        return os.path.join(self.split_dir, f"_color_only_sat{sat}.json")

    def _keep_color_images(self, images: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Drop near-grayscale frames. Caches file names in the split folder."""
        cache_path = self._color_index_path()
        names: Optional[set] = None
        if os.path.isfile(cache_path):
            with open(cache_path, encoding="utf-8") as handle:
                payload = json.load(handle)
            if float(payload.get("min_colorfulness", -1)) == self.min_colorfulness:
                names = set(payload.get("file_names", []))
        if names is None:
            names = set()
            for i, record in enumerate(images):
                path = os.path.join(self.split_dir, record["file_name"])
                bgr = cv2.imread(path, cv2.IMREAD_COLOR)
                if bgr is not None and is_color_photo(bgr, self.min_colorfulness):
                    names.add(record["file_name"])
                if (i + 1) % 200 == 0 or i + 1 == len(images):
                    print(
                        f"  scanning {self.split} color {i + 1}/{len(images)}",
                        flush=True,
                    )
            with open(cache_path, "w", encoding="utf-8") as handle:
                json.dump(
                    {
                        "min_colorfulness": self.min_colorfulness,
                        "file_names": sorted(names),
                    },
                    handle,
                )
        return [record for record in images if record["file_name"] in names]

    def _all_annotations(self) -> List[Dict[str, Any]]:
        anns: List[Dict[str, Any]] = []
        for image_id in (int(record["id"]) for record in self._images):
            anns.extend(self._anns_by_image.get(image_id, []))
        return anns

    def class_distribution(self) -> Dict[str, Any]:
        """Per-class box counts, including classes present in this split only."""
        counts: Dict[int, int] = defaultdict(int)
        for ann in self._all_annotations():
            counts[int(ann["category_id"])] += 1
        rows = []
        for cid in sorted(self.id_to_name, key=lambda c: (-counts[c], c)):
            rows.append(
                {
                    "category_id": cid,
                    "name": self.id_to_name[cid],
                    "count": int(counts[cid]),
                }
            )
        total = sum(row["count"] for row in rows)
        return {
            "split": self.split,
            "n_images": len(self._images),
            "n_annotations": total,
            "n_classes": len(rows),
            "classes": rows,
        }

    def logical_class_distribution(self) -> Dict[str, Any]:
        """Box counts after merging rare LISA types into logical sign classes."""
        counts: Dict[int, int] = defaultdict(int)
        for ann in self._all_annotations():
            counts[int(ann["logical_id"])] += 1
        rows = []
        for lid, name in enumerate(LOGICAL_CLASSES):
            rows.append(
                {
                    "logical_id": lid,
                    "name": name,
                    "count": int(counts[lid]),
                }
            )
        rows.sort(key=lambda row: (-row["count"], row["logical_id"]))
        total = sum(row["count"] for row in rows)
        return {
            "split": self.split,
            "n_images": len(self._images),
            "n_annotations": total,
            "n_classes": len(LOGICAL_CLASSES),
            "classes": rows,
        }

    def size_distribution(self) -> Dict[str, Any]:
        """Width, height, area, and aspect-ratio stats for GT boxes."""
        anns = self._all_annotations()
        if not anns:
            empty = np.zeros(0, dtype=np.float64)
            return {
                "split": self.split,
                "widths": empty,
                "heights": empty,
                "areas": empty,
                "aspect_ratios": empty,
                "summary": {},
            }
        widths = np.array([ann["bbox"][2] for ann in anns], dtype=np.float64)
        heights = np.array([ann["bbox"][3] for ann in anns], dtype=np.float64)
        areas = np.array([ann["area"] for ann in anns], dtype=np.float64)
        aspect_ratios = np.divide(widths, heights, out=np.zeros_like(widths), where=heights > 0)

        def _stats(values: np.ndarray) -> Dict[str, float]:
            return {
                "min": float(np.min(values)),
                "median": float(np.median(values)),
                "mean": float(np.mean(values)),
                "max": float(np.max(values)),
                "p25": float(np.percentile(values, 25)),
                "p75": float(np.percentile(values, 75)),
            }

        return {
            "split": self.split,
            "widths": widths,
            "heights": heights,
            "areas": areas,
            "aspect_ratios": aspect_ratios,
            "summary": {
                "width": _stats(widths),
                "height": _stats(heights),
                "area": _stats(areas),
                "aspect_ratio": _stats(aspect_ratios),
            },
        }

    def plot_analysis(self, out_dir: Optional[str] = None) -> Dict[str, str]:
        """Write class-count and target-size figures for this split.

        Returns a mapping of figure name to saved path.
        """
        if out_dir is None:
            out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset_analysis")
        os.makedirs(out_dir, exist_ok=True)

        class_path = os.path.join(out_dir, f"class_distribution_{self.split}.png")
        logical_path = os.path.join(out_dir, f"logical_class_distribution_{self.split}.png")
        size_path = os.path.join(out_dir, f"target_size_distribution_{self.split}.png")
        _plot_class_distribution(self.class_distribution(), class_path)
        _plot_class_distribution(
            self.logical_class_distribution(),
            logical_path,
            title_prefix="Logical class distribution",
        )
        _plot_size_distribution(self.size_distribution(), size_path)
        return {
            "class_distribution": class_path,
            "logical_class_distribution": logical_path,
            "target_size_distribution": size_path,
        }

    def __len__(self) -> int:
        return len(self._images)

    def image_records(self) -> List[Dict[str, Any]]:
        """Image metadata without decoding JPEG pixels."""
        return list(self._images)

    def annotations_for(self, image_id: int) -> List[Dict[str, Any]]:
        return list(self._anns_by_image.get(int(image_id), []))

    def iter_sign_crops(
        self,
        pad: float = 1.4,
        size: int = 64,
        clahe: bool = True,
        min_box_px: float = 4.0,
        indices: Optional[Iterable[int]] = None,
    ) -> Iterator[Tuple[np.ndarray, int, Dict[str, Any]]]:
        """Letterboxed GT crops labeled with ``logical_id`` (sign classes only).

        Detection never uses these labels. CLAHE runs on the full frame so
        tiles match ``acf.clahe_bgr`` before the crop is taken.
        """
        from acf import clahe_bgr

        index_list = (
            list(range(len(self._images)))
            if indices is None
            else [int(i) for i in indices]
        )
        for index in index_list:
            image, anns, image_id = self.get(index)
            if clahe:
                image = clahe_bgr(image)
            height, width = image.shape[:2]
            for ann in anns:
                x, y, w, h = (float(v) for v in ann["bbox"][:4])
                if w < min_box_px or h < min_box_px:
                    continue
                x1, y1, x2, y2 = padded_box_xyxy(
                    ann["bbox"], (height, width), pad=pad
                )
                crop = letterbox_bgr(image[y1:y2, x1:x2], size=size)
                meta = {
                    "image_id": image_id,
                    "index": int(index),
                    "bbox": ann["bbox"],
                    "category_id": ann["category_id"],
                    "category_name": ann["category_name"],
                    "logical_name": ann["logical_name"],
                }
                yield crop, int(ann["logical_id"]), meta

    def sample_background_crops(
        self,
        n: int,
        size: int = 64,
        max_iou: float = 0.3,
        rng: Optional[np.random.Generator] = None,
        clahe: bool = True,
        min_side: float = 24.0,
        max_side: float = 220.0,
        tries_per_image: int = 24,
        indices: Optional[Iterable[int]] = None,
    ) -> List[np.ndarray]:
        """Random patches with IoU < ``max_iou`` against every GT box.

        Easy negatives for the crop classifier. ACF hard-negatives are not
        sampled here.
        """
        from acf import clahe_bgr

        if n < 1:
            return []
        if rng is None:
            rng = np.random.default_rng()
        index_list = (
            list(range(len(self._images)))
            if indices is None
            else [int(i) for i in indices]
        )
        if not index_list:
            raise RuntimeError("no images to sample background crops from")

        crops: List[np.ndarray] = []
        guard = 0
        max_rounds = max(8, int(np.ceil(n / max(len(index_list), 1))) + 4)
        while len(crops) < n and guard < max_rounds:
            guard += 1
            order = np.array(index_list, dtype=np.int64)
            rng.shuffle(order)
            for index in order:
                image, anns, _image_id = self.get(int(index))
                height, width = image.shape[:2]
                if height < 8 or width < 8:
                    continue
                if clahe:
                    image = clahe_bgr(image)
                gt_boxes = [ann["bbox"] for ann in anns]
                side_hi = min(float(min(height, width)), float(max_side))
                side_lo = min(float(min_side), side_hi)
                for _ in range(tries_per_image):
                    if len(crops) >= n:
                        return crops
                    side = float(rng.uniform(side_lo, side_hi))
                    max_x = width - side
                    max_y = height - side
                    if max_x <= 0 or max_y <= 0:
                        box = (0.0, 0.0, float(width), float(height))
                    else:
                        x = float(rng.uniform(0.0, max_x))
                        y = float(rng.uniform(0.0, max_y))
                        box = (x, y, side, side)
                    if any(iou(box, gt) >= max_iou for gt in gt_boxes):
                        continue
                    x1, y1, x2, y2 = padded_box_xyxy(box, (height, width), pad=1.0)
                    crops.append(letterbox_bgr(image[y1:y2, x1:x2], size=size))
        if len(crops) < n:
            raise RuntimeError(
                f"only sampled {len(crops)} background crops (wanted {n})"
            )
        return crops[:n]

    def _load_image(self, record: Dict[str, Any]) -> np.ndarray:
        path = os.path.join(self.split_dir, record["file_name"])
        image = cv2.imread(path, cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Failed to read image: {path}")
        if self.color == "rgb":
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image

    def get(self, index: int) -> Tuple[np.ndarray, List[Dict[str, Any]], int]:
        record = self._images[index]
        image_id = int(record["id"])
        return self._load_image(record), self.annotations_for(image_id), image_id

    def __getitem__(self, index: int) -> Tuple[np.ndarray, List[Dict[str, Any]], int]:
        return self.get(index)

    def get_by_id(self, image_id: int) -> Tuple[np.ndarray, List[Dict[str, Any]], int]:
        index = self._id_to_index[int(image_id)]
        return self.get(index)

    def iter_batches(
        self,
        batch_size: int,
        shuffle: bool = False,
        drop_last: bool = False,
        rng: Optional[np.random.Generator] = None,
    ) -> Iterator[Tuple[List[np.ndarray], List[List[Dict[str, Any]]], List[int]]]:
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        indices = np.arange(len(self._images))
        if shuffle:
            if rng is None:
                rng = np.random.default_rng()
            rng.shuffle(indices)

        n = len(indices)
        end = n - (n % batch_size) if drop_last else n
        for start in range(0, end, batch_size):
            batch_idx = indices[start : start + batch_size]
            images: List[np.ndarray] = []
            anns_list: List[List[Dict[str, Any]]] = []
            image_ids: List[int] = []
            for i in batch_idx:
                image, anns, image_id = self.get(int(i))
                images.append(image)
                anns_list.append(anns)
                image_ids.append(image_id)
            yield images, anns_list, image_ids

    def verify_predictions(
        self,
        image_id: int,
        predictions: Sequence[Prediction],
        iou_threshold: float = 0.5,
    ) -> Dict[str, Any]:
        """Class-aware greedy matching of predicted COCO boxes to ground truth.

        A prediction is a true positive if IoU with an unmatched GT box of the
        same ``category_id`` is strictly greater than ``iou_threshold``.
        Duplicate detections on one GT count as false positives.
        """
        gt = self.annotations_for(image_id)
        parsed = [_parse_prediction(pred) for pred in predictions]
        order = sorted(
            range(len(parsed)),
            key=lambda i: (-parsed[i][2], i),
        )

        used_gt = set()
        matches: List[Dict[str, Any]] = []
        unmatched_preds: List[int] = []

        for pred_i in order:
            bbox, category_id, _score = parsed[pred_i]
            best_gt = None
            best_iou = iou_threshold
            for gt_i, gt_ann in enumerate(gt):
                if gt_i in used_gt:
                    continue
                if int(gt_ann["category_id"]) != category_id:
                    continue
                overlap = iou(bbox, gt_ann["bbox"])
                if overlap > best_iou:
                    best_iou = overlap
                    best_gt = gt_i
            if best_gt is None:
                unmatched_preds.append(pred_i)
            else:
                used_gt.add(best_gt)
                matches.append(
                    {
                        "pred_index": pred_i,
                        "gt_index": best_gt,
                        "iou": best_iou,
                        "category_id": category_id,
                    }
                )

        unmatched_gts = [i for i in range(len(gt)) if i not in used_gt]
        tp = len(matches)
        fp = len(unmatched_preds)
        fn = len(unmatched_gts)
        precision, recall = _precision_recall(tp, fp, fn)
        return {
            "image_id": int(image_id),
            "matches": matches,
            "unmatched_predictions": unmatched_preds,
            "unmatched_ground_truth": unmatched_gts,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": precision,
            "recall": recall,
        }

    def verify_batch(
        self,
        image_ids: Sequence[int],
        predictions_per_image: Sequence[Sequence[Prediction]],
        iou_threshold: float = 0.5,
    ) -> Dict[str, Any]:
        if len(image_ids) != len(predictions_per_image):
            raise ValueError("image_ids and predictions_per_image must be the same length")

        per_image = [
            self.verify_predictions(image_id, preds, iou_threshold=iou_threshold)
            for image_id, preds in zip(image_ids, predictions_per_image)
        ]
        tp = sum(item["tp"] for item in per_image)
        fp = sum(item["fp"] for item in per_image)
        fn = sum(item["fn"] for item in per_image)
        precision, recall = _precision_recall(tp, fp, fn)
        return {
            "per_image": per_image,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": precision,
            "recall": recall,
        }


def load_splits(
    root: Optional[str] = None,
    color: str = "bgr",
    splits: Iterable[str] = ("train", "valid", "test"),
    color_only: bool = True,
    min_colorfulness: float = COLOR_SAT_MIN,
) -> Dict[str, LISACocoDataset]:
    """Convenience constructor for train / valid / test loaders."""
    return {
        split: LISACocoDataset(
            split=split,
            root=root,
            color=color,
            color_only=color_only,
            min_colorfulness=min_colorfulness,
        )
        for split in splits
    }


def plot_dataset_analysis(
    root: Optional[str] = None,
    out_dir: Optional[str] = None,
    splits: Iterable[str] = ("train", "valid", "test"),
) -> Dict[str, Any]:
    """Analyze every split and write per-split plus combined graphs."""
    datasets = load_splits(root=root, splits=splits)
    if out_dir is None:
        out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset_analysis")
    os.makedirs(out_dir, exist_ok=True)

    paths: Dict[str, Any] = {}
    distributions = []
    logical_distributions = []
    size_stats = []
    for split, dataset in datasets.items():
        paths[split] = dataset.plot_analysis(out_dir=out_dir)
        distributions.append(dataset.class_distribution())
        logical_distributions.append(dataset.logical_class_distribution())
        size_stats.append(dataset.size_distribution())

    combined_class = os.path.join(out_dir, "class_distribution_all_splits.png")
    combined_logical = os.path.join(out_dir, "logical_class_distribution_all_splits.png")
    combined_size = os.path.join(out_dir, "target_size_distribution_all_splits.png")
    _plot_class_distribution_by_split(distributions, combined_class)
    _plot_class_distribution_by_split(
        logical_distributions,
        combined_logical,
        title="Logical class distribution by split",
    )
    _plot_size_distribution_by_split(size_stats, combined_size)
    paths["all_splits"] = {
        "class_distribution": combined_class,
        "logical_class_distribution": combined_logical,
        "target_size_distribution": combined_size,
    }
    return paths


_plt = None


def _use_agg() -> Any:
    global _plt
    if _plt is None:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        _plt = plt
    return _plt


def _plot_class_distribution(
    dist: Dict[str, Any],
    path: str,
    title_prefix: str = "Class distribution",
) -> None:
    plt = _use_agg()
    rows = list(dist["classes"])
    names = [row["name"] for row in rows]
    counts = [row["count"] for row in rows]
    fig_h = max(4.5, 0.22 * len(rows) + 1.6)
    fig, ax = plt.subplots(figsize=(10.5, fig_h))
    y = np.arange(len(rows))
    ax.barh(y, counts, color="#3b6ea5")
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Number of boxes")
    ax.set_title(
        f"{title_prefix} ({dist['split']}): "
        f"{dist['n_annotations']} boxes, {dist['n_images']} images, {dist['n_classes']} classes"
    )
    for yi, count in zip(y, counts):
        ax.text(count, yi, f" {count}", va="center", ha="left", fontsize=7, color="#333333")
    xmax = max(counts) if counts else 1
    ax.set_xlim(0, xmax * 1.12)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _plot_size_distribution(sizes: Dict[str, Any], path: str) -> None:
    plt = _use_agg()
    widths = sizes["widths"]
    heights = sizes["heights"]
    areas = sizes["areas"]
    split = sizes["split"]
    summary = sizes.get("summary") or {}

    fig, axes = plt.subplots(2, 2, figsize=(10.5, 8.2))
    fig.suptitle(f"Target box size distribution ({split})", fontsize=13)

    def _hist(ax, values, xlabel, median=None, title_stats=None):
        if values.size == 0:
            ax.set_title("no boxes")
            return
        ax.hist(values, bins=40, color="#3b6ea5", edgecolor="white", linewidth=0.4)
        if median is not None:
            ax.axvline(median, color="#c0392b", linestyle="--", linewidth=1.2, label="median")
            ax.legend(frameon=False, fontsize=8)
        if title_stats:
            ax.set_title(
                f"{xlabel}: min {title_stats['min']:.1f}  "
                f"med {title_stats['median']:.1f}  max {title_stats['max']:.1f}"
            )
        else:
            ax.set_title(xlabel)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Count")

    w_stats = summary.get("width")
    h_stats = summary.get("height")
    _hist(axes[0, 0], widths, "Width (px)", median=None if not w_stats else w_stats["median"], title_stats=w_stats)
    _hist(axes[0, 1], heights, "Height (px)", median=None if not h_stats else h_stats["median"], title_stats=h_stats)

    if areas.size:
        sqrt_area = np.sqrt(areas)
        sqrt_stats = {
            "min": float(np.min(sqrt_area)),
            "median": float(np.median(sqrt_area)),
            "max": float(np.max(sqrt_area)),
        }
        _hist(axes[1, 0], sqrt_area, "sqrt(area) (px)", median=sqrt_stats["median"], title_stats=sqrt_stats)
    else:
        axes[1, 0].set_title("no boxes")

    ax = axes[1, 1]
    if widths.size:
        ax.scatter(widths, heights, s=8, alpha=0.25, c="#3b6ea5", linewidths=0)
        ax.set_xlabel("Width (px)")
        ax.set_ylabel("Height (px)")
        ar = summary.get("aspect_ratio", {})
        extra = f"  aspect w/h med {ar['median']:.2f}" if ar else ""
        ax.set_title(f"Width vs height{extra}")
        ax.set_aspect("equal", adjustable="box")
    else:
        ax.set_title("no boxes")

    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _plot_class_distribution_by_split(
    dists: Sequence[Dict[str, Any]],
    path: str,
    title: str = "Class distribution by split",
) -> None:
    plt = _use_agg()
    names = []
    seen = set()
    for dist in dists:
        for row in sorted(dist["classes"], key=lambda r: (-r["count"], r["name"])):
            if row["name"] not in seen:
                seen.add(row["name"])
                names.append(row["name"])
    # Order by the first split's counts (usually train).
    if dists:
        order_map = {row["name"]: row["count"] for row in dists[0]["classes"]}
        names.sort(key=lambda n: (-order_map.get(n, 0), n))

    splits = [dist["split"] for dist in dists]
    counts = {
        dist["split"]: {row["name"]: row["count"] for row in dist["classes"]}
        for dist in dists
    }
    n_splits = len(splits)
    y = np.arange(len(names))
    height = 0.8 / max(n_splits, 1)
    colors = ["#3b6ea5", "#e67e22", "#27ae60", "#8e44ad"]
    fig_h = max(5.0, 0.24 * len(names) + 1.8)
    fig, ax = plt.subplots(figsize=(11, fig_h))
    for i, split in enumerate(splits):
        values = [counts[split].get(name, 0) for name in names]
        ax.barh(y + (i - (n_splits - 1) / 2) * height, values, height=height * 0.92, label=split, color=colors[i % len(colors)])
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Number of boxes")
    ax.set_title(title)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _plot_size_distribution_by_split(size_stats: Sequence[Dict[str, Any]], path: str) -> None:
    plt = _use_agg()
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.2))
    fig.suptitle("Target box size by split", fontsize=13)
    colors = ["#3b6ea5", "#e67e22", "#27ae60"]
    panels = [
        ("widths", "Width (px)"),
        ("heights", "Height (px)"),
        ("areas", "sqrt(area) (px)"),
    ]
    for ax, (key, xlabel) in zip(axes, panels):
        for i, sizes in enumerate(size_stats):
            values = sizes[key]
            if key == "areas":
                values = np.sqrt(values) if values.size else values
            if values.size == 0:
                continue
            ax.hist(
                values,
                bins=40,
                density=True,
                histtype="step",
                linewidth=1.6,
                label=sizes["split"],
                color=colors[i % len(colors)],
            )
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Density")
    axes[0].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    saved = plot_dataset_analysis()
    for split, files in saved.items():
        print(split)
        if isinstance(files, dict):
            for name, path in files.items():
                print(f"  {name}: {path}")
        else:
            print(f"  {files}")
