"""A.3: inverse image warping and two manually implemented interpolators.

H maps source (x,y) pixel centers to destination coordinates. Optional
output_shape is (height,width), and origin is the destination coordinate of
the first output pixel. Without them, the full transformed bounds are used.
"""

import numpy as np


def _homography(H):
    H = np.asarray(H, dtype=np.float64)
    if H.shape != (3, 3) or not np.isfinite(H).all() or not np.any(H):
        raise ValueError("H must be a finite nonzero 3x3 matrix.")
    H = H / np.max(np.abs(H))
    try:
        inverse = np.linalg.inv(H)
    except np.linalg.LinAlgError as error:
        raise ValueError("H must be invertible.") from error
    if not np.isfinite(inverse).all():
        raise ValueError("H inverse must be finite.")
    return H, inverse / np.max(np.abs(inverse))


def warped_bounds(image_shape, H):
    """Return ((height,width),(origin_x,origin_y)) covering warped corners."""
    if len(image_shape) < 2 or min(image_shape[:2]) < 1:
        raise ValueError("Image dimensions must be positive.")
    height, width = image_shape[:2]
    H, _ = _homography(H)
    corners = np.array([[0, 0, 1], [width - 1, 0, 1],
                        [width - 1, height - 1, 1], [0, height - 1, 1]], dtype=float)
    projected = corners @ H.T
    denominators = projected[:, 2]
    tolerance = 64 * np.finfo(float).eps * np.maximum(np.max(np.abs(projected), axis=1), np.finfo(float).tiny)
    if (np.any(np.abs(denominators) <= tolerance)
            or (np.min(denominators) < 0 < np.max(denominators))):
        raise ValueError("The projective horizon intersects the image; specify a finite output canvas.")
    points = projected[:, :2] / denominators[:, None]
    # Remove roundoff around exact integer coordinates before floor/ceil.
    rounded = np.rint(points)
    points = np.where(np.abs(points - rounded) < 1e-8, rounded, points)
    low = np.floor(points.min(axis=0)).astype(np.int64)
    high = np.ceil(points.max(axis=0)).astype(np.int64)
    return (int(high[1] - low[1] + 1), int(high[0] - low[0] + 1)), (int(low[0]), int(low[1]))


def _warp(im, H, method, output_shape, origin, return_mask):
    source = np.asarray(im)
    if source.ndim not in (2, 3) or min(source.shape) < 1 or not np.issubdtype(source.dtype, np.number):
        raise ValueError("im must be a nonempty numeric grayscale or multichannel image.")
    if np.iscomplexobj(source) or not np.isfinite(source).all():
        raise ValueError("Image values must be finite and real.")
    H, inverse = _homography(H)
    if output_shape is None:
        if origin is not None:
            raise ValueError("Specify output_shape when supplying origin.")
        output_shape, origin = warped_bounds(source.shape, H)
    else:
        if len(output_shape) != 2 or any(int(v) != v or v <= 0 for v in output_shape):
            raise ValueError("output_shape must be (positive height, positive width).")
        output_shape = tuple(map(int, output_shape))
        origin = (0, 0) if origin is None else origin
    if len(origin) != 2 or not np.isfinite(origin).all():
        raise ValueError("origin must contain two finite coordinates.")
    height, width = output_shape
    if height * width > 40_000_000:
        raise ValueError("Output canvas exceeds 40 million pixels; use a smaller explicit canvas.")
    channels = source.shape[2:]  # Empty for grayscale.
    output = np.zeros((height, width) + channels, dtype=source.dtype)
    mask = np.zeros((height, width), dtype=bool)
    source_height, source_width = source.shape[:2]
    samples = source.astype(np.float64, copy=False) if method == "bilinear" else source
    columns = np.arange(width, dtype=float)[None, :] + origin[0]

    for start in range(0, height, 128):
        stop = min(height, start + 128)
        rows = np.arange(start, stop, dtype=float)[:, None] + origin[1]
        a = inverse[0, 0] * columns + inverse[0, 1] * rows + inverse[0, 2]
        b = inverse[1, 0] * columns + inverse[1, 1] * rows + inverse[1, 2]
        c = inverse[2, 0] * columns + inverse[2, 1] * rows + inverse[2, 2]
        tolerance = 64 * np.finfo(float).eps * np.maximum(np.maximum(np.abs(a), np.abs(b)), np.abs(c))
        finite = np.isfinite(a) & np.isfinite(b) & np.isfinite(c) & (np.abs(c) > tolerance)
        x = np.divide(a, c, out=np.zeros_like(a), where=finite)
        y = np.divide(b, c, out=np.zeros_like(b), where=finite)
        valid = (finite & (x >= -1e-8) & (y >= -1e-8)
                 & (x <= source_width - 1 + 1e-8) & (y <= source_height - 1 + 1e-8))
        mask[start:stop] = valid
        x = np.clip(x[valid], 0, source_width - 1)
        y = np.clip(y[valid], 0, source_height - 1)
        if method == "nearest":
            values = samples[np.floor(y + 0.5).astype(int), np.floor(x + 0.5).astype(int)]
        else:
            x0, y0 = np.floor(x).astype(int), np.floor(y).astype(int)
            x1, y1 = np.minimum(x0 + 1, source_width - 1), np.minimum(y0 + 1, source_height - 1)
            dx, dy = x - x0, y - y0
            if channels:
                dx, dy = dx[:, None], dy[:, None]
            values = ((1 - dx) * (1 - dy) * samples[y0, x0]
                      + dx * (1 - dy) * samples[y0, x1]
                      + (1 - dx) * dy * samples[y1, x0]
                      + dx * dy * samples[y1, x1])
            if np.issubdtype(source.dtype, np.integer):
                limits = np.iinfo(source.dtype)
                values = np.clip(np.rint(values), limits.min, limits.max)
        output[start:stop][valid] = values
    return (output, mask) if return_mask else output


def warpImageNearestNeighbor(im, H, output_shape=None, *, origin=None, return_mask=False):
    """Inverse-map destination pixels and choose the nearest source pixel."""
    return _warp(im, H, "nearest", output_shape, origin, return_mask)


def warpImageBilinear(im, H, output_shape=None, *, origin=None, return_mask=False):
    """Inverse-map destination pixels and weight four neighboring samples."""
    return _warp(im, H, "bilinear", output_shape, origin, return_mask)
