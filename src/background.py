"""Reusable background removal utilities for plant seedling images."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import cv2
import numpy as np


@dataclass
class MaskResult:
    """Encapsulates the outcome of plant segmentation."""

    image: np.ndarray
    mask: np.ndarray
    ratio: float
    used_mask: bool


def create_plant_mask(image_bgr: np.ndarray) -> np.ndarray:
    """Construct a binary mask that highlights green plant regions."""
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)

    lower_green1 = np.array([35, 40, 40])
    upper_green1 = np.array([85, 255, 255])
    lower_green2 = np.array([25, 40, 40])
    upper_green2 = np.array([95, 255, 255])
    lower_green3 = np.array([35, 20, 20])
    upper_green3 = np.array([85, 255, 200])

    mask1 = cv2.inRange(hsv, lower_green1, upper_green1)
    mask2 = cv2.inRange(hsv, lower_green2, upper_green2)
    mask3 = cv2.inRange(hsv, lower_green3, upper_green3)
    mask = cv2.bitwise_or(mask1, mask2)
    mask = cv2.bitwise_or(mask, mask3)

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.dilate(mask, kernel, iterations=1)
    return mask


def segment_plant(image_bgr: np.ndarray, min_ratio: float = 0.05) -> MaskResult:
    """Apply the plant mask and return the masked image along with stats."""
    mask = create_plant_mask(image_bgr)
    plant_pixels = np.count_nonzero(mask)
    total_pixels = mask.size
    ratio = plant_pixels / float(total_pixels)
    used_mask = ratio >= min_ratio
    if used_mask:
        plant_image = cv2.bitwise_and(image_bgr, image_bgr, mask=mask)
    else:
        plant_image = image_bgr
    return MaskResult(image=plant_image, mask=mask, ratio=ratio, used_mask=used_mask)


def ensure_bgr(image: np.ndarray, color_space: str) -> np.ndarray:
    """Convert an RGB image to BGR if needed for OpenCV ops."""
    if color_space.lower() == "bgr":
        return image
    if color_space.lower() == "rgb":
        return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    raise ValueError(f"Unsupported color space: {color_space}")


def remove_background(image: np.ndarray, *, color_space: str = "bgr", min_ratio: float = 0.05) -> MaskResult:
    """High-level helper that segments plants from either RGB or BGR arrays."""
    bgr = ensure_bgr(image, color_space)
    return segment_plant(bgr, min_ratio=min_ratio)
