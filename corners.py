"""B.1: single-scale Harris corners and adaptive non-maximal suppression.

Coordinates are integer (x,y) in the EXIF-oriented original image. Harris
strength uses det(M)/trace(M), as in Brown et al., section 2. No feature or
corner detection libraries are used.
"""

import numpy as np


def _gaussian(image, sigma):
    if not np.isfinite(sigma) or sigma <= 0:
        raise ValueError("Gaussian sigma must be positive and finite.")
    radius = int(np.ceil(3*sigma))
    positions = np.arange(-radius, radius+1, dtype=float)
    kernel = np.exp(-positions**2 / (2*sigma**2))
    kernel /= kernel.sum()
    result = image
    for axis in (0, 1):
        padding = [(0, 0), (0, 0)]
        padding[axis] = (radius, radius)
        padded = np.pad(result, padding, mode="reflect")
        result = np.zeros_like(image, dtype=np.float64)
        for index, coefficient in enumerate(kernel):
            selection = [slice(None), slice(None)]
            selection[axis] = slice(index, index+image.shape[axis])
            result += coefficient * padded[tuple(selection)]
    return result


def detect_harris(image, *, sigma_derivative=1.0, sigma_integration=1.5,
                  threshold_rel=0.005, border=20):
    """Return (response map, Nx2 candidate coordinates, N strengths).

    Detect 3x3 local maxima above a relative positive-response threshold.
    A 20-pixel border is reserved for the later 40x40 descriptor window.
    """
    source = np.asarray(image)
    if (source.ndim not in (2, 3) or min(source.shape) < 2
            or (source.ndim == 3 and source.shape[2] != 3)
            or not np.issubdtype(source.dtype, np.number) or np.iscomplexobj(source)
            or not np.isfinite(source).all() or np.any(source < 0) or np.any(source > 255)):
        raise ValueError("Use a finite grayscale/RGB image with values in [0,255].")
    if not 0 < threshold_rel < 1 or int(border) != border or border < 1:
        raise ValueError("Use 0 < threshold_rel < 1 and a positive integer border.")
    gray = source.astype(np.float64)
    if gray.ndim == 3:
        gray = gray @ np.array([0.2126, 0.7152, 0.0722])
    if gray.max() > 1:
        gray /= 255
    smooth = _gaussian(gray, sigma_derivative)
    iy, ix = np.gradient(smooth)
    a = _gaussian(ix*ix, sigma_integration)
    b = _gaussian(ix*iy, sigma_integration)
    c = _gaussian(iy*iy, sigma_integration)
    trace = a+c
    response = np.divide(np.maximum(a*c-b*b, 0), trace,
                         out=np.zeros_like(trace), where=trace > 1e-15)
    padded = np.pad(response, 1, mode="constant", constant_values=-np.inf)
    maxima = (response > 0) & (response > threshold_rel*response.max())
    height, width = response.shape
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dx == dy == 0:
                continue
            neighbor = padded[1+dy:1+dy+height, 1+dx:1+dx+width]
            # Resolve adjacent equal maxima deterministically toward top-left.
            maxima &= response > neighbor if (dy, dx) < (0, 0) else response >= neighbor
    maxima[:border] = maxima[-border:] = False
    maxima[:, :border] = maxima[:, -border:] = False
    y, x = np.nonzero(maxima)
    return response, np.column_stack((x, y)).astype(np.int32), response[y, x]


def anms(points, strengths, count=500, *, c_robust=0.9, block_size=256):
    """Return selected candidate indices and every candidate's radius.

    r_i = min_j ||x_i-x_j||, subject to f_i < c_robust*f_j.
    With no sufficiently stronger neighbor the radius is infinity. Evaluate
    all candidates in blocks, without allocating the full NxN matrix.
    """
    points = np.asarray(points, dtype=np.float64)
    strengths = np.asarray(strengths, dtype=np.float64)
    if (points.ndim != 2 or points.shape[1] != 2 or strengths.shape != (len(points),)
            or not np.isfinite(points).all() or not np.isfinite(strengths).all()
            or np.any(strengths <= 0)):
        raise ValueError("Use Nx2 finite points and N positive finite strengths.")
    if (int(count) != count or count < 1 or not 0 < c_robust < 1
            or int(block_size) != block_size or block_size < 1):
        raise ValueError("Use positive integer count/block size and 0 < c_robust < 1.")
    radii = np.full(len(points), np.inf)
    for start in range(0, len(points), block_size):
        stop = min(start+block_size, len(points))
        dx = points[start:stop, 0, None] - points[None, :, 0]
        dy = points[start:stop, 1, None] - points[None, :, 1]
        distance_squared = dx*dx + dy*dy
        stronger = strengths[start:stop, None] < c_robust*strengths[None, :]
        distance_squared[~stronger] = np.inf
        radii[start:stop] = np.sqrt(distance_squared.min(axis=1))
    order = np.lexsort((points[:, 0], points[:, 1], -strengths, -radii))
    return order[:int(count)], radii
