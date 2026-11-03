"""B.3: exact two-nearest-neighbor descriptor matching and Lowe ratio test."""

import numpy as np


def match_descriptors(descriptors1, descriptors2, *, ratio_threshold=0.7, block_size=256):
    """Match each row in image 1 against all rows in image 2.

    Use Euclidean distances: d1/d2 < ratio_threshold. Equivalently compare
    squared distances with ratio_threshold**2. Tied zero-distance neighbors
    are ambiguous and rejected. No geometry or manual points are used.
    """
    inputs = [np.asarray(value) for value in (descriptors1, descriptors2)]
    for value in inputs:
        if (value.ndim != 2 or value.shape[1] < 1 or not np.issubdtype(value.dtype, np.number)
                or np.iscomplexobj(value) or not np.isfinite(value).all()):
            raise ValueError("Descriptors must be finite real NxD arrays.")
    a, b = [value.astype(np.float64) for value in inputs]
    if a.shape[1] != b.shape[1]:
        raise ValueError("Descriptor dimensions must agree.")
    if not 0 < ratio_threshold < 1 or int(block_size) != block_size or block_size < 1:
        raise ValueError("Use 0 < ratio_threshold < 1 and a positive integer block_size.")
    n = len(a)
    nearest = np.full(n, -1, dtype=np.int32)
    second = np.full(n, -1, dtype=np.int32)
    distance1, distance2 = np.full(n, np.inf), np.full(n, np.inf)
    ratios = np.full(n, np.inf)
    if len(b) >= 2:
        norm_b = np.sum(b*b, axis=1)
        for start in range(0, n, block_size):
            stop = min(start+block_size, n)
            queries = a[start:stop]
            squared = np.maximum(np.sum(queries*queries, axis=1)[:, None]
                                 + norm_b[None, :] - 2*queries @ b.T, 0)
            # A stable sort breaks equal-distance ties by target descriptor ID.
            order = np.argsort(squared, axis=1, kind="stable")[:, :2]
            rows = np.arange(stop-start)
            e1, e2 = squared[rows, order[:, 0]], squared[rows, order[:, 1]]
            nearest[start:stop], second[start:stop] = order[:, 0], order[:, 1]
            distance1[start:stop], distance2[start:stop] = np.sqrt(e1), np.sqrt(e2)
            squared_ratio = np.divide(e1, e2, out=np.full_like(e1, np.inf), where=e2 > 1e-12)
            ratios[start:stop] = np.sqrt(squared_ratio)
    accepted = ratios < ratio_threshold
    matches = np.column_stack((np.flatnonzero(accepted), nearest[accepted])).astype(np.int32)
    return {"matches": matches, "nearest_indices": nearest, "second_indices": second,
            "distance1": distance1, "distance2": distance2, "ratios": ratios,
            "accepted": accepted}
