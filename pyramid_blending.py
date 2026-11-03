"""Laplacian pyramid blending using only NumPy filtering and sampling."""

import numpy as np


def gaussian_blur(image):
    """Separable [1,4,6,4,1]/16 filter with reflected spatial boundaries."""
    result = np.asarray(image, dtype=np.float32)
    for axis in (0, 1):
        padding = [(0, 0)] * result.ndim
        padding[axis] = (2, 2)
        padded = np.pad(result, padding, mode="reflect")
        filtered = np.zeros_like(result)
        for offset, coefficient in enumerate((1, 4, 6, 4, 1)):
            selection = [slice(None)] * result.ndim
            selection[axis] = slice(offset, offset + result.shape[axis])
            filtered += (coefficient / 16) * padded[tuple(selection)]
        result = filtered
    return result


def expand(image, target_shape):
    """Insert zeros, filter, and multiply by four; preserve odd target sizes."""
    expanded = np.zeros(tuple(target_shape[:2]) + image.shape[2:], dtype=np.float32)
    if expanded[::2, ::2].shape != image.shape:
        raise ValueError("Target shape must match a 2x pyramid expansion.")
    expanded[::2, ::2] = image
    return gaussian_blur(expanded) * 4


def gaussian_pyramid(image, levels):
    if int(levels) != levels or not 1 <= levels <= 12:
        raise ValueError("Use 1-12 pyramid levels.")
    image = np.asarray(image, dtype=np.float32)
    if image.ndim not in (2, 3) or min(image.shape) < 1 or not np.isfinite(image).all():
        raise ValueError("Expected a nonempty finite image.")
    pyramid = [image]
    while len(pyramid) < levels and min(pyramid[-1].shape[:2]) > 1:
        pyramid.append(gaussian_blur(pyramid[-1])[::2, ::2].copy())
    return pyramid


def laplacian_pyramid(image, levels):
    gaussian = gaussian_pyramid(image, levels)
    return [gaussian[i] - expand(gaussian[i + 1], gaussian[i].shape)
            for i in range(len(gaussian) - 1)] + [gaussian[-1]]


def reconstruct(pyramid):
    result = pyramid[-1]
    for band in reversed(pyramid[:-1]):
        result = expand(result, band.shape) + band
    return result


def blend_laplacian(layers, levels=6):
    """Blend an iterable of (RGB image, nonnegative weight mask) layers.

    Images must already share a canvas and have sensible colors outside their
    support. Normalize Gaussian mask weights independently at every scale.
    """
    color_sums, weight_sums, shape = None, None, None
    for image, weights in layers:
        image, weights = np.asarray(image), np.asarray(weights, dtype=np.float32)
        if (image.ndim != 3 or image.shape[2] != 3 or weights.shape != image.shape[:2]
                or not np.isfinite(weights).all() or np.any(weights < 0)):
            raise ValueError("Each layer needs RGB colors and a finite nonnegative HxW mask.")
        if shape is not None and image.shape != shape:
            raise ValueError("All layers must share the same canvas.")
        shape = image.shape
        bands = laplacian_pyramid(image, levels)
        masks = gaussian_pyramid(weights, levels)
        if color_sums is None:
            color_sums = [np.zeros_like(band) for band in bands]
            weight_sums = [np.zeros_like(mask) for mask in masks]
        for i, (band, mask) in enumerate(zip(bands, masks)):
            color_sums[i] += band * mask[..., None]
            weight_sums[i] += mask
    if color_sums is None:
        raise ValueError("Supply at least one image layer.")
    for color, weight in zip(color_sums, weight_sums):
        np.divide(color, weight[..., None], out=color, where=weight[..., None] > 0)
    return reconstruct(color_sums)
