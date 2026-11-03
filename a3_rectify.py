"""Generate two A.3 rectification examples and interpolation timings."""

import argparse
import csv
import json
from pathlib import Path
import platform
from statistics import median
from time import perf_counter

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

from homography import computeH, project_points
from warping import warpImageBilinear, warpImageNearestNeighbor


ROOT = Path(__file__).resolve().parent


def annotated_source(image, points, directory):
    points = np.asarray(points)
    low, high = points.min(axis=0), points.max(axis=0)
    padding = np.maximum((high - low) * 0.15, 25)
    box = (max(0, int(np.floor(low[0] - padding[0]))),
           max(0, int(np.floor(low[1] - padding[1]))),
           min(image.width, int(np.ceil(high[0] + padding[0]))),
           min(image.height, int(np.ceil(high[1] + padding[1]))))
    annotated = image.copy()
    draw = ImageDraw.Draw(annotated)
    try:
        label_font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 16)
    except OSError:
        label_font = ImageFont.load_default()
    draw.line([tuple(point) for point in points] + [tuple(points[0])], fill="#fbd54b", width=3)
    for index, (x, y) in enumerate(points, 1):
        draw.ellipse((x - 4, y - 4, x + 4, y + 4), fill="#fbd54b", outline="black")
        draw.text((x + 7, y - 20), str(index), font=label_font, fill="#fbd54b", stroke_width=2, stroke_fill="black")
    annotated.save(directory / "source_points.png")
    annotated.crop(box).save(directory / "source_crop.png")
    return list(box)


def benchmark(image, H, shape, repeats):
    methods = {"nearest": warpImageNearestNeighbor, "bilinear": warpImageBilinear}
    for method in methods.values():
        method(image, H, shape)
    timings = {name: [] for name in methods}
    for repetition in range(repeats):
        names = list(methods) if repetition % 2 == 0 else list(reversed(methods))
        for name in names:
            start = perf_counter()
            methods[name](image, H, shape)
            timings[name].append((perf_counter() - start) * 1000)
    return {name + "_ms": median(values) for name, values in timings.items()}, timings


def process_example(example, output, repeats):
    with Image.open(ROOT / example["image"]) as image:
        image = ImageOps.exif_transpose(image).convert("RGB")
    if list(image.size) != example["source_size"]:
        raise ValueError(f"Source image dimensions changed: {example['image']}")
    width, height = map(int, example["target_size"])
    if min(width, height) < 2:
        raise ValueError("Target dimensions must be at least two pixels.")
    points = np.asarray(example["source_points"], dtype=float)
    if points.shape != (4, 2) or np.any(points < 0) or np.any(points[:, 0] > image.width - 1) or np.any(points[:, 1] > image.height - 1):
        raise ValueError("Expected four valid source corners in TL, TR, BR, BL order.")
    target = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype=float)
    H = computeH(points, target)
    source = np.asarray(image)
    shape = (height, width)
    nearest, nearest_mask = warpImageNearestNeighbor(source, H, shape, return_mask=True)
    bilinear, bilinear_mask = warpImageBilinear(source, H, shape, return_mask=True)
    if not np.array_equal(nearest_mask, bilinear_mask) or not nearest_mask.all():
        raise ValueError("The selected rectangle is not fully covered by the source image.")
    directory = output / example["id"]
    directory.mkdir(parents=True, exist_ok=True)
    crop_box = annotated_source(image, points, directory)
    Image.fromarray(nearest).save(directory / "nearest.png")
    Image.fromarray(bilinear).save(directory / "bilinear.png")
    Image.fromarray((nearest_mask * 255).astype(np.uint8)).save(directory / "valid_mask.png")
    left, top, right, bottom = example["zoom_box"]
    if not (0 <= left < right <= width and 0 <= top < bottom <= height):
        raise ValueError("Detail crop must be within the target image.")
    for name, array in (("nearest", nearest), ("bilinear", bilinear)):
        detail = array[top:bottom, left:right]
        enlarged = np.repeat(np.repeat(detail, 4, axis=0), 4, axis=1)
        Image.fromarray(enlarged).save(directory / f"detail_{name}.png")
    timings, raw_timings = benchmark(source, H, shape, repeats)
    error = np.linalg.norm(project_points(points, H) - target, axis=1)
    relative = directory.relative_to(ROOT).as_posix()
    result = {
        **example,
        "target_points": target.tolist(), "H": H.tolist(),
        "corner_fit_max_error_px": float(error.max()),
        "valid_pixel_count": int(nearest_mask.sum()),
        "source_crop_box": crop_box, "zoom_scale": 4,
        "source_crop": f"{relative}/source_crop.png",
        "source_annotated": f"{relative}/source_points.png",
        "nearest_image": f"{relative}/nearest.png", "bilinear_image": f"{relative}/bilinear.png",
        "detail_nearest": f"{relative}/detail_nearest.png", "detail_bilinear": f"{relative}/detail_bilinear.png",
        "record_path": f"{relative}/rectification.json",
        **timings, "raw_timings_ms": raw_timings,
    }
    (directory / "rectification.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    np.savetxt(directory / "H.csv", H, delimiter=",", fmt="%.17g")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=7)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    configuration = json.loads((ROOT / "correspondences" / "rectification" / "points.json").read_text(encoding="utf-8"))
    output = ROOT / "output" / "a3"
    output.mkdir(parents=True, exist_ok=True)
    results = [process_example(example, output, args.repeats) for example in configuration["examples"]]
    summary = {
        "method": "Inverse warping; nearest neighbor and four-sample bilinear interpolation implemented in NumPy",
        "corner_order": configuration["corner_order"], "aspect_ratio_note": configuration["aspect_ratio_note"],
        "timing": {"repeats": args.repeats, "warmups_per_method": 1, "statistic": "median",
                   "includes": "array validation, inverse coordinate mapping, interpolation, output and mask allocation",
                   "excludes": "image I/O, annotations, homography recovery, webpage generation",
                   "order": "alternating nearest-first and bilinear-first"},
        "environment": {"python": platform.python_version(), "numpy": np.__version__},
        "examples": results,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    with (output / "timings.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["example", "width", "height", "nearest_ms", "bilinear_ms", "repeats"])
        for result in results:
            writer.writerow([result["id"], *result["target_size"], result["nearest_ms"], result["bilinear_ms"], args.repeats])
            print(f'{result["title"]}: {result["target_size"]}, NN {result["nearest_ms"]:.3f} ms, bilinear {result["bilinear_ms"]:.3f} ms')
    from build_showcase import export_showcase_data
    export_showcase_data()
    print(f"A.3 results: {output}")


if __name__ == "__main__":
    main()
