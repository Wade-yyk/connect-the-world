"""Generate Harris/ANMS overlays and original-pixel coordinates for all photos."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageOps

from corners import anms, detect_harris


ROOT = Path(__file__).resolve().parent


def occupied_cells(points, size, columns=8, rows=6):
    if not len(points):
        return 0
    cells = np.floor(points / np.array(size) * [columns, rows]).astype(int)
    return len(np.unique(cells, axis=0))


def overlay(image, points, path):
    marked = image.copy()
    draw = ImageDraw.Draw(marked)
    for x, y in points:
        draw.ellipse((int(x)-4, int(y)-4, int(x)+4, int(y)+4), outline="black", width=3)
        draw.ellipse((int(x)-3, int(y)-3, int(x)+3, int(y)+3), outline="#fbd54b", width=2)
    marked.save(path, quality=93)


def process_image(path, count, threshold_rel):
    with Image.open(path) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")
    _, points, strengths = detect_harris(np.asarray(image), threshold_rel=threshold_rel)
    selected, radii = anms(points, strengths, count=count)
    strongest = np.argsort(-strengths, kind="stable")[:count]
    identifier = f"set_{path.parent.name}_{path.stem}"
    output = ROOT / "output" / "b1" / identifier
    output.mkdir(parents=True, exist_ok=True)
    for name, indices in (("harris_all", np.arange(len(points))),
                          ("harris_top", strongest), ("anms", selected)):
        overlay(image, points[indices], output / f"{name}.jpg")
    np.savez_compressed(output / "features.npz", points=points, strengths=strengths,
                        radii=radii, anms_indices=selected, strongest_indices=strongest,
                        image_size=np.array(image.size))
    selected_flags, strongest_flags = np.zeros(len(points), dtype=bool), np.zeros(len(points), dtype=bool)
    selected_flags[selected], strongest_flags[strongest] = True, True
    with (output / "corners.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["candidate_id", "x", "y", "strength", "radius_px", "anms_selected", "strongest_selected"])
        for i, ((x, y), strength, radius) in enumerate(zip(points, strengths, radii)):
            writer.writerow([i, int(x), int(y), float(strength), float(radius),
                             int(selected_flags[i]), int(strongest_flags[i])])
    relative = output.relative_to(ROOT).as_posix()
    result = {
        "id": identifier, "title": f'{"Outdoor" if path.parent.name == "1" else "Kitchen"} {path.stem}',
        "image": path.relative_to(ROOT).as_posix(), "size": list(image.size),
        "candidate_count": len(points), "selected_count": len(selected),
        "strongest_cells": occupied_cells(points[strongest], image.size),
        "anms_cells": occupied_cells(points[selected], image.size),
        "harris_all": f"{relative}/harris_all.jpg", "harris_top": f"{relative}/harris_top.jpg",
        "anms_image": f"{relative}/anms.jpg", "features": f"{relative}/features.npz",
        "csv": f"{relative}/corners.csv", "record_path": f"{relative}/corners.json",
        "selected_points_xy": points[selected].tolist(),
        "selected_strengths": strengths[selected].tolist(),
        "selected_radii_px": [float(r) if np.isfinite(r) else None for r in radii[selected]],
        "infinite_radius_note": "null in JSON / inf in CSV means no sufficiently stronger candidate exists.",
    }
    (output / "corners.json").write_text(json.dumps(result, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    print(f'{result["title"]}: {len(points)} candidates -> {len(selected)} ANMS; grid cells {result["strongest_cells"]} -> {result["anms_cells"]}', flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=500)
    parser.add_argument("--threshold-rel", type=float, default=0.005)
    args = parser.parse_args()
    if args.count < 1 or not 0 < args.threshold_rel < 1:
        parser.error("Use --count >= 1 and 0 < --threshold-rel < 1")
    images = sorted((ROOT / "photos").glob("*/*.jpg"))
    if not images:
        raise ValueError("No source photos found.")
    results = [process_image(path, args.count, args.threshold_rel) for path in images]
    summary = {
        "method": "Single-scale Harris det(M)/trace(M); 3x3 local maxima; exact ANMS over all candidates",
        "parameters": {"sigma_derivative": 1.0, "sigma_integration": 1.5,
                       "threshold_rel": args.threshold_rel, "border_px": 20,
                       "count": args.count, "c_robust": 0.9},
        "coordinates": "EXIF-oriented original pixels, (x,y), integer centers; no subpixel refinement or resizing",
        "spatial_metric": "Number of occupied cells in an 8-column by 6-row grid (48 cells total)",
        "reference": "https://www.microsoft.com/en-us/research/wp-content/uploads/2005/06/cvpr05.pdf",
        "examples": results,
    }
    output = ROOT / "output" / "b1"
    (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    with (output / "summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["image", "candidates", "selected", "strongest_grid_cells", "anms_grid_cells"])
        for result in results:
            writer.writerow([result["title"], result["candidate_count"], result["selected_count"],
                             result["strongest_cells"], result["anms_cells"]])
    from build_showcase import export_showcase_data
    export_showcase_data()
    print(f"B.1 results: {output}")


if __name__ == "__main__":
    main()
