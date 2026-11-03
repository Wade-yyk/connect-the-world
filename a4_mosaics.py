"""Generate three A.4 mosaics using existing manual correspondences."""

import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from homography import computeH
from mosaicing import largest_valid_rectangle, stitch_mosaic


ROOT = Path(__file__).resolve().parent
PYRAMID_LEVELS = 6
DETAIL_REGIONS = {
    "outdoor_1_2": (0.30, 0.35, 0.60, 0.70),
    "outdoor_3_4": (0.43, 0.36, 0.69, 0.75),
    "kitchen_1_2_3": (0.45, 0.48, 0.72, 0.88),
}
EXAMPLES = [
    ("outdoor_1_2", "Outdoor 1 + 2", 1, [1, 2], 2),
    ("outdoor_3_4", "Outdoor 3 + 4", 1, [3, 4], 4),
    ("kitchen_1_2_3", "Kitchen 1 + 2 + 3", 2, [1, 2, 3], 2),
]


def recover_transforms(set_id, frames, reference):
    edges, records = {}, []
    for frame in range(min(frames), max(frames)):
        path = ROOT / "correspondences" / f"set_{set_id}_{frame}_to_{frame+1}.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        expected = [f"photos/{set_id}/{frame}.jpg", f"photos/{set_id}/{frame+1}.jpg"]
        if [data["image1"], data["image2"]] != expected:
            raise ValueError(f"Unexpected image order in {path}")
        for image_key, size_key in (("image1", "image1_size"), ("image2", "image2_size")):
            with Image.open(ROOT / data[image_key]) as image:
                if list(ImageOps.exif_transpose(image).size) != data[size_key]:
                    raise ValueError(f"Source size changed: {data[image_key]}")
        H = computeH(data["im1_pts"], data["im2_pts"])
        edges[frame] = H
        records.append({"file": path.relative_to(ROOT).as_posix(),
                        "direction": f"{frame} -> {frame+1}",
                        "points": len(data["im1_pts"]), "H": H.tolist()})
    transforms = []
    for frame in frames:
        H = np.eye(3)
        if frame < reference:
            for step in range(frame, reference):
                H = edges[step] @ H
        else:
            for step in range(frame - 1, reference - 1, -1):
                H = np.linalg.inv(edges[step]) @ H
        transforms.append(H / np.max(np.abs(H)))
    return transforms, records


def generate_example(example):
    identifier, title, set_id, frames, reference = example
    paths = [f"photos/{set_id}/{frame}.jpg" for frame in frames]
    images = []
    for path in paths:
        with Image.open(ROOT / path) as image:
            images.append(np.array(ImageOps.exif_transpose(image).convert("RGB")))
    transforms, records = recover_transforms(set_id, frames, reference)
    mosaic, overwrite, coverage, origin, laplacian = stitch_mosaic(
        images, transforms, pyramid_levels=PYRAMID_LEVELS)
    output = ROOT / "output" / "a4" / identifier
    output.mkdir(parents=True, exist_ok=True)
    alpha = (coverage > 0).astype(np.uint8) * 255
    for name, pixels in (("mosaic", mosaic), ("overwrite", overwrite), ("laplacian", laplacian)):
        Image.fromarray(np.dstack((pixels, alpha))).save(output / f"{name}.png")
    left, top, right, bottom = largest_valid_rectangle(coverage > 0)
    Image.fromarray(mosaic[top:bottom, left:right]).save(output / "cropped.png")
    Image.fromarray(laplacian[top:bottom, left:right]).save(output / "laplacian_cropped.png")
    crop_width, crop_height = right-left, bottom-top
    fractions = DETAIL_REGIONS[identifier]
    detail_box = [int(fractions[0]*crop_width), int(fractions[1]*crop_height),
                  int(fractions[2]*crop_width), int(fractions[3]*crop_height)]
    dx0, dy0, dx1, dy1 = detail_box
    for name, pixels in (("feather", mosaic), ("laplacian", laplacian)):
        Image.fromarray(pixels[top+dy0:top+dy1, left+dx0:left+dx1]).save(output / f"{name}_detail.png")
    Image.fromarray(alpha).save(output / "valid_mask.png")
    Image.fromarray(coverage).save(output / "coverage.png")
    relative = output.relative_to(ROOT).as_posix()
    translation = np.array([[1., 0., -origin[0]], [0., 1., -origin[1]], [0., 0., 1.]])
    result = {
        "id": identifier, "title": title, "sources": paths,
        "reference_image": f"photos/{set_id}/{reference}.jpg",
        "reference_origin_xy": list(origin), "output_size": list(mosaic.shape[1::-1]),
        "source_to_reference_H": [H.tolist() for H in transforms],
        "source_to_canvas_H": [(translation @ H).tolist() for H in transforms],
        "correspondences": records,
        "valid_pixels": int(np.count_nonzero(coverage)),
        "overlap_pixels": int(np.count_nonzero(coverage >= 2)),
        "mosaic": f"{relative}/mosaic.png", "overwrite": f"{relative}/overwrite.png",
        "cropped": f"{relative}/cropped.png",
        "laplacian": f"{relative}/laplacian.png",
        "laplacian_cropped": f"{relative}/laplacian_cropped.png",
        "pyramid_levels": PYRAMID_LEVELS,
        "detail_box_in_crop_ltrb": detail_box,
        "feather_detail": f"{relative}/feather_detail.png",
        "laplacian_detail": f"{relative}/laplacian_detail.png",
        "crop_box_ltrb": [left, top, right, bottom],
        "cropped_size": [right - left, bottom - top],
        "valid_mask": f"{relative}/valid_mask.png", "coverage": f"{relative}/coverage.png",
        "record_path": f"{relative}/mosaic.json",
    }
    result["overlap_percent"] = 100 * result["overlap_pixels"] / result["valid_pixels"]
    (output / "mosaic.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f'{title}: {result["output_size"]}, overlap {result["overlap_percent"]:.1f}%')
    return result


def main():
    results = [generate_example(example) for example in EXAMPLES]
    summary = {
        "method": "Manual least-squares H, shared full-bounds canvas, inverse bilinear warping, normalized feather blending",
        "weight": "min(x+1, width-x, y+1, height-y), normalized to a peak of 1 in each source image",
        "edge_note": "A one-pixel positive floor retains valid border pixels; invalid samples have zero weight.",
        "comparison": "Same H, bilinear samples and canvas; overwrite uses source order, feather uses weighted averaging.",
        "scene_note": "Three mosaics from two captured scenes: two outdoor pairs and one three-image kitchen mosaic.",
        "crop_method": "Largest axis-aligned rectangle containing only valid mosaic pixels; right/bottom crop bounds are exclusive.",
        "laplacian_method": {
            "levels": PYRAMID_LEVELS,
            "filter": "Separable binomial [1,4,6,4,1]/16; reflected padding; 2x down/up sampling",
            "weights": "Binary ownership by largest source feather weight; Gaussian mask pyramid; normalized per scale",
            "invalid_colors": "Extended from common feather result before filtering; original valid mask retained",
            "controls": "Same input points, H matrices, canvas, bilinear warp, and crop; no exposure compensation or alignment changes",
        },
        "examples": results,
    }
    output = ROOT / "output" / "a4"
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    with (output / "summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["mosaic", "source_count", "width", "height", "valid_pixels", "overlap_pixels", "overlap_percent"])
        for result in results:
            writer.writerow([result["id"], len(result["sources"]), *result["output_size"],
                             result["valid_pixels"], result["overlap_pixels"], result["overlap_percent"]])
    from build_showcase import export_showcase_data
    export_showcase_data()
    print(f"A.4 results: {output}")


if __name__ == "__main__":
    main()
