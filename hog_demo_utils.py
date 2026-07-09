from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import matplotlib.pyplot as plt
import numpy as np
from skimage.feature import hog


DEFAULT_HOG_CONFIG = {
    "image_size": (64, 64),
    "orientations": 9,
    "pixels_per_cell": (8, 8),
    "cells_per_block": (2, 2),
    "block_norm": "L2-Hys",
    "preprocessing": "gray_clahe",
}


def normalize_tuple(value: Any) -> tuple[int, int]:
    if isinstance(value, tuple):
        return tuple(int(v) for v in value)
    if isinstance(value, list):
        return tuple(int(v) for v in value)
    if isinstance(value, str):
        cleaned = value.strip().strip("()[]")
        parts = [p.strip() for p in cleaned.split(",") if p.strip()]
        if len(parts) == 1 and "x" in parts[0].lower():
            parts = parts[0].lower().split("x")
        return tuple(int(p) for p in parts)
    raise ValueError(f"Không chuyển được về tuple: {value}")


def normalize_config(config: dict[str, Any]) -> dict[str, Any]:
    merged = DEFAULT_HOG_CONFIG.copy()
    merged.update(config)
    merged["image_size"] = normalize_tuple(merged["image_size"])
    merged["pixels_per_cell"] = normalize_tuple(merged["pixels_per_cell"])
    merged["cells_per_block"] = normalize_tuple(merged["cells_per_block"])
    merged["orientations"] = int(merged["orientations"])
    return merged


def json_ready_config(config: dict[str, Any]) -> dict[str, Any]:
    config = normalize_config(config)
    return {
        "image_size": list(config["image_size"]),
        "orientations": int(config["orientations"]),
        "pixels_per_cell": list(config["pixels_per_cell"]),
        "cells_per_block": list(config["cells_per_block"]),
        "block_norm": config["block_norm"],
        "preprocessing": config["preprocessing"],
    }


def read_image(path: str | Path) -> np.ndarray:
    image_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise FileNotFoundError(f"Không đọc được ảnh: {path}")
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def _clip_box(box: tuple[int, int, int, int] | list[int], width: int, height: int) -> tuple[int, int, int, int]:
    xmin, ymin, xmax, ymax = [int(round(v)) for v in box]
    xmin = max(0, min(xmin, width - 1))
    ymin = max(0, min(ymin, height - 1))
    xmax = max(0, min(xmax, width - 1))
    ymax = max(0, min(ymax, height - 1))
    if xmax < xmin:
        xmin, xmax = xmax, xmin
    if ymax < ymin:
        ymin, ymax = ymax, ymin
    return xmin, ymin, xmax, ymax


def crop_box(image: np.ndarray, box: tuple[int, int, int, int] | list[int]) -> np.ndarray:
    h, w = image.shape[:2]
    xmin, ymin, xmax, ymax = _clip_box(box, w, h)
    return image[ymin : ymax + 1, xmin : xmax + 1]


def preprocess_for_hog(
    image_rgb: np.ndarray,
    image_size: tuple[int, int] = (64, 64),
    preprocessing: str = "gray_clahe",
) -> np.ndarray:
    image_size = normalize_tuple(image_size)
    resized = cv2.resize(image_rgb, image_size, interpolation=cv2.INTER_AREA)

    if preprocessing == "gray":
        return cv2.cvtColor(resized, cv2.COLOR_RGB2GRAY)

    if preprocessing == "gray_clahe":
        gray = cv2.cvtColor(resized, cv2.COLOR_RGB2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        return clahe.apply(gray)

    if preprocessing == "gray_clahe_blur":
        gray = cv2.cvtColor(resized, cv2.COLOR_RGB2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        return cv2.GaussianBlur(enhanced, (3, 3), 0)

    if preprocessing == "yuv_y":
        return cv2.cvtColor(resized, cv2.COLOR_RGB2YUV)[:, :, 0]

    if preprocessing == "hsv_v":
        return cv2.cvtColor(resized, cv2.COLOR_RGB2HSV)[:, :, 2]

    raise ValueError(f"Unsupported preprocessing mode: {preprocessing}")


def extract_hog_feature(image_rgb: np.ndarray, config: dict[str, Any]) -> np.ndarray:
    config = normalize_config(config)
    hog_input = preprocess_for_hog(
        image_rgb,
        image_size=config["image_size"],
        preprocessing=config["preprocessing"],
    )
    return hog(
        hog_input,
        orientations=config["orientations"],
        pixels_per_cell=config["pixels_per_cell"],
        cells_per_block=config["cells_per_block"],
        block_norm=config["block_norm"],
        visualize=False,
        feature_vector=True,
    ).astype(np.float32)


def _normalize_for_display(image: np.ndarray) -> np.ndarray:
    image = image.astype(np.float32)
    lo, hi = float(image.min()), float(image.max())
    if hi - lo < 1e-8:
        return np.zeros_like(image)
    return (image - lo) / (hi - lo)


def compute_hog_visualization(image_rgb: np.ndarray, config: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    config = normalize_config(config)
    hog_input = preprocess_for_hog(
        image_rgb,
        image_size=config["image_size"],
        preprocessing=config["preprocessing"],
    )
    _, hog_image = hog(
        hog_input,
        orientations=config["orientations"],
        pixels_per_cell=config["pixels_per_cell"],
        cells_per_block=config["cells_per_block"],
        block_norm=config["block_norm"],
        visualize=True,
        feature_vector=True,
    )
    return hog_input, _normalize_for_display(hog_image)


def compute_gradients(preprocessed_gray: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    gray = preprocessed_gray.astype(np.float32)
    gradient_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gradient_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = cv2.magnitude(gradient_x, gradient_y)
    orientation = (np.degrees(np.arctan2(gradient_y, gradient_x)) + 180.0) % 180.0
    return gradient_x, gradient_y, magnitude, orientation


def make_gradient_orientation_display(
    orientation_deg: np.ndarray,
    magnitude: np.ndarray | None = None,
) -> np.ndarray:
    hue = np.clip(orientation_deg / 180.0 * 179.0, 0, 179).astype(np.uint8)
    saturation = np.full_like(hue, 255, dtype=np.uint8)
    if magnitude is None:
        value = np.full_like(hue, 255, dtype=np.uint8)
    else:
        value = (_normalize_for_display(magnitude) * 255).astype(np.uint8)
    hsv = np.dstack([hue, saturation, value])
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)


def cell_histogram(
    preprocessed_gray: np.ndarray,
    cell_x: int,
    cell_y: int,
    pixels_per_cell: tuple[int, int],
    orientations: int,
) -> tuple[np.ndarray, np.ndarray]:
    _, _, magnitude, orientation = compute_gradients(preprocessed_gray)
    cell_w, cell_h = normalize_tuple(pixels_per_cell)
    x1 = int(cell_x) * cell_w
    y1 = int(cell_y) * cell_h
    x2 = min(x1 + cell_w, preprocessed_gray.shape[1])
    y2 = min(y1 + cell_h, preprocessed_gray.shape[0])
    cell_orientation = orientation[y1:y2, x1:x2].ravel()
    cell_magnitude = magnitude[y1:y2, x1:x2].ravel()
    edges = np.linspace(0.0, 180.0, int(orientations) + 1)
    hist, _ = np.histogram(cell_orientation, bins=edges, weights=cell_magnitude)
    centers = (edges[:-1] + edges[1:]) / 2.0
    return centers, hist.astype(np.float32)


def block_hog_vector(
    preprocessed_gray: np.ndarray,
    block_x: int,
    block_y: int,
    pixels_per_cell: tuple[int, int],
    cells_per_block: tuple[int, int],
    orientations: int,
    block_norm: str = "L2-Hys",
) -> np.ndarray:
    hog_blocks = hog(
        preprocessed_gray,
        orientations=int(orientations),
        pixels_per_cell=normalize_tuple(pixels_per_cell),
        cells_per_block=normalize_tuple(cells_per_block),
        block_norm=block_norm,
        visualize=False,
        feature_vector=False,
    )
    if hog_blocks.size == 0:
        return np.array([], dtype=np.float32)

    max_y = max(0, hog_blocks.shape[0] - 1)
    max_x = max(0, hog_blocks.shape[1] - 1)
    y = min(max(0, int(block_y)), max_y)
    x = min(max(0, int(block_x)), max_x)
    return hog_blocks[y, x].ravel().astype(np.float32)


def draw_cell_on_preprocessed(
    preprocessed_gray: np.ndarray,
    cell_x: int,
    cell_y: int,
    pixels_per_cell: tuple[int, int],
) -> np.ndarray:
    cell_w, cell_h = normalize_tuple(pixels_per_cell)
    out = cv2.cvtColor(preprocessed_gray.astype(np.uint8), cv2.COLOR_GRAY2RGB)
    x1 = int(cell_x) * cell_w
    y1 = int(cell_y) * cell_h
    x2 = min(x1 + cell_w, preprocessed_gray.shape[1]) - 1
    y2 = min(y1 + cell_h, preprocessed_gray.shape[0]) - 1
    cv2.rectangle(out, (x1, y1), (x2, y2), (255, 80, 40), 1)
    return out


def compute_feature_dim(config: dict[str, Any]) -> int:
    return feature_dim_breakdown(config)["feature_dim"]


def feature_dim_breakdown(config: dict[str, Any]) -> dict[str, Any]:
    config = normalize_config(config)
    width, height = config["image_size"]
    cell_x, cell_y = config["pixels_per_cell"]
    block_x, block_y = config["cells_per_block"]
    orientations = int(config["orientations"])
    n_cells_x = width // cell_x if cell_x > 0 else 0
    n_cells_y = height // cell_y if cell_y > 0 else 0
    n_blocks_x = n_cells_x - block_x + 1
    n_blocks_y = n_cells_y - block_y + 1
    feature_dim = 0
    if n_blocks_x > 0 and n_blocks_y > 0 and block_x > 0 and block_y > 0 and orientations > 0:
        feature_dim = n_blocks_x * n_blocks_y * block_x * block_y * orientations
    return {
        "image_size": config["image_size"],
        "pixels_per_cell": config["pixels_per_cell"],
        "cells_per_block": config["cells_per_block"],
        "orientations": orientations,
        "n_cells_x": n_cells_x,
        "n_cells_y": n_cells_y,
        "n_blocks_x": max(0, n_blocks_x),
        "n_blocks_y": max(0, n_blocks_y),
        "feature_dim": int(feature_dim),
    }


def apply_condition_transform(image_rgb: np.ndarray, condition: str, seed: int = 42) -> np.ndarray:
    if condition == "clean":
        return image_rgb.copy()
    if condition == "low_brightness":
        return np.clip(image_rgb.astype(np.float32) * 0.65, 0, 255).astype(np.uint8)
    if condition == "high_brightness":
        return np.clip(image_rgb.astype(np.float32) * 1.25 + 10, 0, 255).astype(np.uint8)
    if condition == "low_contrast":
        mean = np.array([127.5], dtype=np.float32)
        return np.clip((image_rgb.astype(np.float32) - mean) * 0.65 + mean, 0, 255).astype(np.uint8)
    if condition == "gaussian_blur":
        return cv2.GaussianBlur(image_rgb, (5, 5), 0)
    if condition == "gaussian_noise":
        rng = np.random.default_rng(seed)
        noise = rng.normal(0, 8.0, image_rgb.shape)
        return np.clip(image_rgb.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    raise ValueError(f"Không hỗ trợ condition: {condition}")


def predict_roi(model: Any, image_rgb: np.ndarray, config: dict[str, Any]) -> str:
    feature = extract_hog_feature(image_rgb, config).reshape(1, -1)
    return str(model.predict(feature)[0])


def draw_bbox(
    image_rgb: np.ndarray,
    box: tuple[int, int, int, int] | list[int],
    label: str | None = None,
    color: tuple[int, int, int] = (46, 204, 113),
    thickness: int = 3,
) -> np.ndarray:
    out = image_rgb.copy()
    h, w = out.shape[:2]
    x1, y1, x2, y2 = _clip_box(box, w, h)
    cv2.rectangle(out, (x1, y1), (x2, y2), color, thickness)
    if label:
        cv2.putText(
            out,
            str(label),
            (x1, max(18, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2,
            cv2.LINE_AA,
        )
    return out


def make_orientation_bin_figure(baseline_bins: int = 9, optimized_bins: int = 9) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.2), subplot_kw={"projection": "polar"})
    baseline_bins = int(baseline_bins)
    optimized_bins = int(optimized_bins)
    specs = [
        (baseline_bins, f"Baseline: {baseline_bins} bins ~= {180 / baseline_bins:.1f} do/bin"),
        (optimized_bins, f"Optimized: {optimized_bins} bins ~= {180 / optimized_bins:.1f} do/bin"),
    ]
    theta = np.linspace(0, np.pi, 200)
    for ax, (bins, title) in zip(axes, specs):
        ax.plot(theta, np.ones_like(theta), color="#2d4059", linewidth=2)
        for i in range(bins + 1):
            angle = i * np.pi / bins
            ax.plot([angle, angle], [0, 1], color="#ea5455", linewidth=1.2)
        centers = (np.arange(bins) + 0.5) * np.pi / bins
        for idx, angle in enumerate(centers, start=1):
            ax.text(angle, 0.72, str(idx), ha="center", va="center", fontsize=8)
        ax.set_thetamin(0)
        ax.set_thetamax(180)
        ax.set_rticks([])
        ax.set_title(title, fontsize=11)
        ax.grid(False)
    fig.tight_layout()
    return fig

