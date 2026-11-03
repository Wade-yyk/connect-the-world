"""A.2: estimate an image1 -> image2 homography using linear least squares.

Only NumPy's basic linear algebra is used. Coordinates are (x, y), with the
first pixel center at (0, 0). H acts on homogeneous column vectors.
"""

import numpy as np


def _points(value, name):
    points = np.asarray(value, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError(f"{name} must have shape (n, 2).")
    if not np.isfinite(points).all():
        raise ValueError(f"{name} must contain only finite coordinates.")
    return points


def build_linear_system(im1_pts, im2_pts):
    """Return A (2n x 8), b (2n,) for h = [h11,h12,h13,h21,...,h32].

    For every (x,y) -> (u,v), cross-multiplication of p' ~ H p gives:
        x*h11 + y*h12 + h13 - u*x*h31 - u*y*h32 = u
        x*h21 + y*h22 + h23 - v*x*h31 - v*y*h32 = v
    The projective scale is fixed by setting h33 = 1.
    """
    source = _points(im1_pts, "im1_pts")
    target = _points(im2_pts, "im2_pts")
    if source.shape != target.shape:
        raise ValueError("Both point arrays must have the same shape.")
    if len(source) < 4:
        raise ValueError("At least four point correspondences are required.")
    for points, name in ((source, "im1_pts"), (target, "im2_pts")):
        if np.linalg.matrix_rank(points - points.mean(axis=0)) < 2:
            raise ValueError(f"{name} is degenerate: all points are collinear or coincident.")

    x, y = source.T
    u, v = target.T
    A = np.zeros((2 * len(source), 8), dtype=np.float64)
    A[0::2, 0:3] = np.column_stack((x, y, np.ones(len(source))))
    A[1::2, 3:6] = np.column_stack((x, y, np.ones(len(source))))
    A[0::2, 6:8] = -np.column_stack((u * x, u * y))
    A[1::2, 6:8] = -np.column_stack((v * x, v * y))
    b = target.reshape(-1).copy()
    return A, b


def computeH(im1_pts, im2_pts):
    """Recover a 3x3 H mapping im1_pts to im2_pts, with H[2,2] == 1.

    All supplied correspondences participate in the least-squares fit; no
    RANSAC, outlier removal, or library homography estimator is used.
    Column scaling improves conditioning while preserving the objective
    min ||A h - b||_2. The equivalent scaled problem is (A / s) z = b,
    followed by h = z / s. lstsq avoids forming the normal equations.
    """
    A, b = build_linear_system(im1_pts, im2_pts)
    scales = np.linalg.norm(A, axis=0)
    scales[scales == 0] = 1.0
    z, _, rank, _ = np.linalg.lstsq(A / scales, b, rcond=None)
    if rank < 8:
        raise ValueError("The point configuration cannot uniquely determine all eight unknowns.")
    h = z / scales
    H = np.append(h, 1.0).reshape(3, 3)
    if not np.isfinite(H).all() or np.linalg.matrix_rank(H) < 3:
        raise ValueError("The estimated homography is singular or non-finite.")
    return H


def project_points(points, H):
    """Apply H to points and divide by the homogeneous coordinate."""
    points = _points(points, "points")
    H = np.asarray(H, dtype=np.float64)
    if H.shape != (3, 3) or not np.isfinite(H).all() or not np.any(H):
        raise ValueError("H must be a finite, nonzero 3x3 matrix.")
    # Make the infinity check independent of H's arbitrary projective scale.
    H = H / np.max(np.abs(H))
    homogeneous = np.column_stack((points, np.ones(len(points)))) @ H.T
    scale = np.max(np.abs(homogeneous), axis=1)
    tolerance = 64 * np.finfo(np.float64).eps * np.maximum(scale, np.finfo(np.float64).tiny)
    if np.any(np.abs(homogeneous[:, 2]) <= tolerance):
        raise ValueError("A point maps to infinity (homogeneous denominator is zero).")
    return homogeneous[:, :2] / homogeneous[:, 2, None]
