"""B.2: axis-aligned 40x40 windows sampled into normalized 8x8 descriptors."""

import numpy as np

from corners import _gaussian


def grayscale(image):
    source = np.asarray(image)
    if (source.ndim not in (2, 3) or min(source.shape) < 2
            or (source.ndim == 3 and source.shape[2] != 3)
            or not np.issubdtype(source.dtype, np.number) or np.iscomplexobj(source)
            or not np.isfinite(source).all() or np.any(source < 0) or np.any(source > 255)):
        raise ValueError("Use a finite grayscale/RGB image with values in [0,255].")
    gray = source.astype(np.float64)
    if gray.ndim == 3:
        gray = gray @ np.array([0.2126, 0.7152, 0.0722])
    if gray.max() > 1:
        gray /= 255
    return gray


def extract_descriptors(image, points, *, sigma=2.0, min_std=1e-6):
    """Extract descriptors for integer original-pixel (x,y) centers.

    Smooth the whole grayscale image once to avoid individual patch padding.
    Use [y-20:y+20,x-20:x+20], then take the centers of its 5x5 blocks
    (indices 2,7,...,37). No rotation or Haar transform is applied.
    Reject incomplete windows and near-constant 8x8 samples rather than
    passing zero-variance descriptors into later feature matching.
    """
    gray = grayscale(image)
    points = np.asarray(points, dtype=np.float64)
    if (points.ndim != 2 or points.shape[1] != 2 or not np.isfinite(points).all()
            or not np.array_equal(points, np.rint(points))):
        raise ValueError("Points must be a finite Nx2 array of integer (x,y) centers.")
    if not np.isfinite(min_std) or min_std <= 0:
        raise ValueError("min_std must be positive and finite.")
    blurred = _gaussian(gray, sigma)
    height, width = gray.shape
    vectors, samples, means, stds, indices = [], [], [], [], []
    border_rejected, contrast_rejected = [], []
    for index, (px, py) in enumerate(points):
        if not (20 <= px <= width-20 and 20 <= py <= height-20):
            border_rejected.append(index)
            continue
        x, y = int(px), int(py)
        patch40 = blurred[y-20:y+20, x-20:x+20]
        patch8 = patch40[2::5, 2::5]
        mean, std = patch8.mean(), patch8.std()
        if std < min_std:
            contrast_rejected.append(index)
            continue
        descriptor = ((patch8-mean)/std).reshape(64)
        vectors.append(descriptor)
        samples.append(patch8)
        means.append(mean)
        stds.append(std)
        indices.append(index)
    indices = np.asarray(indices, dtype=np.int32)
    return {
        "descriptors": np.asarray(vectors, dtype=np.float32).reshape(-1, 64),
        "samples8": np.asarray(samples, dtype=np.float32).reshape(-1, 8, 8),
        "means": np.asarray(means), "stds": np.asarray(stds),
        "source_indices": indices, "points": points[indices].astype(np.int32),
        "border_rejected": np.asarray(border_rejected, dtype=np.int32),
        "contrast_rejected": np.asarray(contrast_rejected, dtype=np.int32),
    }
