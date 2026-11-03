"""A.4: register RGB images on a shared canvas and feather their overlap."""

import numpy as np

from warping import warped_bounds, warpImageBilinear


def largest_valid_rectangle(mask):
    """Return (left, top, right, bottom) for the largest all-valid rectangle.

    Right and bottom are exclusive. A histogram and monotonic stack find the
    exact maximum area in O(height * width), including masks with inner holes.
    """
    mask = np.asarray(mask, dtype=bool)
    if mask.ndim != 2 or mask.size == 0 or not mask.any():
        raise ValueError("The mask must contain at least one valid pixel.")
    heights = np.zeros(mask.shape[1], dtype=int)
    best_area, best_box = 0, None
    for row, valid in enumerate(mask):
        heights = np.where(valid, heights + 1, 0)
        stack = []
        for col in range(len(heights) + 1):
            height = int(heights[col]) if col < len(heights) else 0
            start = col
            while stack and stack[-1][1] > height:
                left, previous_height = stack.pop()
                area = (col - left) * previous_height
                if area > best_area:
                    best_area = area
                    best_box = (left, row + 1 - previous_height, col, row + 1)
                start = left
            if height and (not stack or stack[-1][1] < height):
                stack.append((start, height))
    return best_box


def feather_weights(shape):
    """Normalized distance to the source border, with a one-pixel edge floor."""
    height, width = shape[:2]
    if min(height, width) < 1:
        raise ValueError("Source dimensions must be positive.")
    x = np.minimum(np.arange(width) + 1, width - np.arange(width))
    y = np.minimum(np.arange(height) + 1, height - np.arange(height))
    distance = np.minimum(y[:, None], x[None, :]).astype(np.float32)
    return distance / distance.max()


def stitch_mosaic(images, homographies, *, pyramid_levels=0):
    """H[i] maps image i into the reference view, before canvas translation.

    Return RGB weighted/overwrite results, coverage count, and canvas origin.
    With pyramid_levels > 0, append the Laplacian-blended RGB result.
    Coverage is independent of color: black source pixels remain valid.
    """
    if not images or len(images) != len(homographies) or len(images) > 255:
        raise ValueError("Supply 1-255 images with one homography per image.")
    if int(pyramid_levels) != pyramid_levels or not 0 <= pyramid_levels <= 12:
        raise ValueError("Use 0-12 pyramid levels (0 disables the comparison).")
    sources = [np.asarray(image) for image in images]
    for image in sources:
        if (image.ndim != 3 or image.shape[2] != 3 or min(image.shape) < 1
                or not np.issubdtype(image.dtype, np.number)
                or np.iscomplexobj(image) or not np.isfinite(image).all()
                or np.any(image < 0) or np.any(image > 255)):
            raise ValueError("Images must be finite RGB arrays in [0,255].")
    bounds = [warped_bounds(image.shape, H) for image, H in zip(sources, homographies)]
    low = np.min([origin for _, origin in bounds], axis=0)
    high = np.max([(origin[0] + shape[1] - 1, origin[1] + shape[0] - 1)
                   for shape, origin in bounds], axis=0)
    width, height = map(int, high - low + 1)
    if width * height > 40_000_000:
        raise ValueError("Mosaic exceeds 40 million pixels; choose a closer reference view.")
    origin = tuple(map(int, low))
    color_sum = np.zeros((height, width, 3), dtype=np.float32)
    weight_sum = np.zeros((height, width), dtype=np.float32)
    coverage = np.zeros((height, width), dtype=np.uint8)
    overwrite = np.zeros((height, width, 3), dtype=np.uint8)
    if pyramid_levels:
        strongest_weight = np.zeros((height, width), dtype=np.float32)
        owner = np.full((height, width), 255, dtype=np.uint8)

    for index, (source, H) in enumerate(zip(sources, homographies)):
        # Warp color and alpha together with the hand-written A.3 interpolator.
        source_with_alpha = np.dstack((source.astype(np.float32), feather_weights(source.shape)))
        warped, valid = warpImageBilinear(source_with_alpha, H, (height, width),
                                         origin=origin, return_mask=True)
        weight = np.where(valid, warped[..., 3], 0)
        color_sum += warped[..., :3] * weight[..., None]
        weight_sum += weight
        coverage += valid
        overwrite[valid] = np.clip(np.rint(warped[..., :3][valid]), 0, 255).astype(np.uint8)
        if pyramid_levels:
            selected = weight > strongest_weight
            owner[selected] = index
            strongest_weight[selected] = weight[selected]

    np.divide(color_sum, weight_sum[..., None], out=color_sum,
              where=weight_sum[..., None] > 0)
    mosaic = np.clip(np.rint(color_sum), 0, 255).astype(np.uint8)
    if pyramid_levels:
        from pyramid_blending import blend_laplacian
        del color_sum, weight_sum, strongest_weight, warped, source_with_alpha

        def layers():
            for index, (source, H) in enumerate(zip(sources, homographies)):
                registered, valid = warpImageBilinear(source.astype(np.float32), H,
                                                      (height, width), origin=origin,
                                                      return_mask=True)
                # Extend undefined colors using the common feather result to
                # avoid filtering a black border into the pyramid bands.
                registered[~valid] = mosaic[~valid]
                yield registered, (owner == index).astype(np.float32)

        laplacian = blend_laplacian(layers(), levels=pyramid_levels)
        laplacian[coverage == 0] = 0
        laplacian = np.clip(np.rint(laplacian), 0, 255).astype(np.uint8)
        return mosaic, overwrite, coverage, origin, laplacian
    return mosaic, overwrite, coverage, origin
