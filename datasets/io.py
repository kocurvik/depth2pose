import io
import json
import os
from pathlib import Path
from typing import *

import cv2
import numpy as np
from PIL import Image


def read_image(path: Union[str, os.PathLike, IO]) -> np.ndarray:
    """
    Read a image, return uint8 RGB array of shape (H, W, 3).
    """
    if isinstance(path, (str, os.PathLike)):
        data = Path(path).read_bytes()
    else:
        data = path.read()
    image = cv2.cvtColor(
        cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB
    )
    return image


def read_segmentation(
    path: Union[str, os.PathLike, IO],
) -> Tuple[np.ndarray, Dict[str, int]]:
    """
    Read a segmentation mask
    ### Parameters:
    - `path: Union[str, os.PathLike, IO]`
        The file path or file object to read from.
    ### Returns:
    - `Tuple[np.ndarray, Dict[str, int]]`
        A tuple containing:
        - `mask`: uint8 or uint16 numpy.ndarray of shape (H, W).
        - `labels`: Dict[str, int]. The label mapping, a dictionary of {label_name: label_id}.
    """
    if isinstance(path, (str, os.PathLike)):
        data = Path(path).read_bytes()
    else:
        data = path.read()
    pil_image = Image.open(io.BytesIO(data))
    labels = (
        json.loads(pil_image.info["labels"]) if "labels" in pil_image.info else None
    )
    mask = np.array(pil_image)
    return mask, labels


def read_depth(path: Union[str, os.PathLike, IO]) -> np.ndarray:
    """
    Read a depth image, return float32 depth array of shape (H, W).
    """
    if isinstance(path, (str, os.PathLike)):
        data = Path(path).read_bytes()
    else:
        data = path.read()
    pil_image = Image.open(io.BytesIO(data))
    near = float(pil_image.info.get("near"))
    far = float(pil_image.info.get("far"))
    depth = np.array(pil_image)
    mask_nan, mask_inf = depth == 0, depth == 65535
    depth = (depth.astype(np.float32) - 1) / 65533
    depth = near ** (1 - depth) * far**depth
    if "unit" in pil_image.info:  # Legacy support for depth units
        unit = float(pil_image.info.get("unit"))
        depth = depth * unit
    depth[mask_nan] = np.nan
    depth[mask_inf] = np.inf
    return depth


def read_depth_npz(path: str | os.PathLike) -> np.ndarray:
    return np.load(path)["depth"]


def read_mask(path: Union[str, os.PathLike, IO[bytes]]) -> np.ndarray:
    """
    Read a binary mask, return bool array of shape (H, W).
    """
    if isinstance(path, (str, os.PathLike)):
        data = Path(path).read_bytes()
    else:
        data = path.read()
    mask = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_UNCHANGED)
    if len(mask.shape) == 3:
        mask = mask[..., 0]
    return mask > 0


JSON_TYPE = Union[str, int, float, bool, None, Dict[str, "JSON"], List["JSON"]]


def read_json(path: Union[str, os.PathLike, IO[str]]) -> JSON_TYPE:
    if isinstance(path, (str, os.PathLike)):
        text = Path(path).read_text()
    else:
        text = path.read()
    return json.loads(text)
