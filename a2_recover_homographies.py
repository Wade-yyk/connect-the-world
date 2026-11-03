"""Read saved manual correspondences and produce all A.2 deliverables.

Run: python a2_recover_homographies.py
Optional: --input correspondences/set_2_1_to_2.json --output output/a2
The HTML report opens locally; photographs are never uploaded.
"""

import argparse
import colorsys
import csv
from html import escape
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

from homography import build_linear_system, computeH, project_points


ROOT = Path(__file__).resolve().parent


def font(size):
    for path in ("C:/Windows/Fonts/arial.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            pass
    return ImageFont.load_default(size=size)


def load_pair(path):
    data = json.loads(path.read_text(encoding="utf-8"))
    p, q = np.asarray(data["im1_pts"], dtype=float), np.asarray(data["im2_pts"], dtype=float)
    build_linear_system(p, q)  # Validate before indexing or opening output files.
    if data.get("num_pairs", len(p)) != len(p):
        raise ValueError("num_pairs does not agree with the saved point arrays.")
    images = []
    for key, points in (("image1", p), ("image2", q)):
        with Image.open(ROOT / data[key]) as image:
            image = ImageOps.exif_transpose(image).convert("RGB")
        if key + "_size" in data and list(image.size) != data[key + "_size"]:
            raise ValueError(f"{data[key]} changed size after annotation.")
        if np.any(points < 0) or np.any(points[:, 0] > image.width - 1) or np.any(points[:, 1] > image.height - 1):
            raise ValueError(f"Out-of-bounds point in {data[key]}.")
        images.append(image)
    if data.get("homography_direction", "image1 -> image2") != "image1 -> image2":
        raise ValueError("Expected annotations in the image1 -> image2 direction.")
    return data, p, q, images


def color(index, count):
    return tuple(round(v * 255) for v in colorsys.hsv_to_rgb(index / count, 0.70, 1))


def label(draw, position, text, fill):
    draw.text(position, str(text), font=font(19), fill=fill, stroke_width=2, stroke_fill="#121820")


def matches_figure(images, p, q, data, path):
    height, top, gap, margin = 850, 76, 80, 24
    scales = [height / image.height for image in images]
    widths = [round(image.width * scale) for image, scale in zip(images, scales)]
    offsets = [(margin, top), (margin + widths[0] + gap, top)]
    figure = Image.new("RGB", (sum(widths) + gap + 2 * margin, height + top + 48), "#17202b")
    draw = ImageDraw.Draw(figure)
    draw.text((margin, 12), "Manual point correspondences | equal numbers indicate the same scene point", font=font(21), fill="white")
    for image, width, offset, key in zip(images, widths, offsets, ("image1", "image2")):
        figure.paste(image.resize((width, height), Image.Resampling.LANCZOS), offset)
        draw.text((offset[0], 43), data[key], font=font(17), fill="#d2dce6")
    for i, (left, right) in enumerate(zip(p, q)):
        c = color(i, len(p))
        positions = [(offset[0] + (point[0] + 0.5) * scale,
                      offset[1] + (point[1] + 0.5) * scale)
                     for point, offset, scale in zip((left, right), offsets, scales)]
        draw.line((*positions[0], *positions[1]), fill=c, width=2)
    for i, (left, right) in enumerate(zip(p, q)):
        c = color(i, len(p))
        for point, offset, scale in zip((left, right), offsets, scales):
            x = offset[0] + (point[0] + 0.5) * scale
            y = offset[1] + (point[1] + 0.5) * scale
            draw.ellipse((x - 6, y - 6, x + 6, y + 6), fill=c, outline="black", width=2)
            label(draw, (x + 9, y - 23), i + 1, c)
    draw.text((margin, height + top + 13), f"{len(p)} point pairs | stored coordinates are in original image pixels", font=font(18), fill="white")
    figure.save(path)


def reprojection_figure(image, target, predicted, errors, path):
    height, top, margin = 1000, 104, 24
    scale = height / image.height
    width = round(image.width * scale)
    figure = Image.new("RGB", (width + 2 * margin, height + top + 64), "#17202b")
    figure.paste(image.resize((width, height), Image.Resampling.LANCZOS), (margin, top))
    draw = ImageDraw.Draw(figure)
    draw.text((margin, 12), "Reprojection check in image2", font=font(24), fill="white")
    draw.text((margin, 47), "Yellow circle: manual target | cyan cross: H(source)", font=font(18), fill="white")
    draw.text((margin, 75), "Labels: point ID and forward error in original pixels", font=font(17), fill="#d2dce6")
    for i, (q, estimate, error) in enumerate(zip(target, predicted, errors), 1):
        x, y = (q + 0.5) * scale + (margin, top)
        u, v = (estimate + 0.5) * scale + (margin, top)
        draw.line((x, y, u, v), fill="#ff637d", width=3)
        draw.ellipse((x - 6, y - 6, x + 6, y + 6), outline="#ffe04a", width=3)
        draw.line((u - 7, v, u + 7, v), fill="#32f1ff", width=3)
        draw.line((u, v - 7, u, v + 7), fill="#32f1ff", width=3)
        # Keep labels inside the panel even for annotations near the image edge.
        label_x = min(max(margin, x + 11), figure.width - 185)
        label(draw, (label_x, max(top, y - 28)), f"{i}: {error:.2f}px", "#ffe04a")
    rmse = np.sqrt(np.mean(errors ** 2))
    draw.text((margin, height + top + 12), f"Forward RMSE: {rmse:.3f}px | maximum: {errors.max():.3f}px", font=font(20), fill="white")
    figure.save(path)


def matrix_text(matrix):
    return "\n".join("[ " + "  ".join(f"{value: .10g}" for value in row) + " ]" for row in np.atleast_2d(matrix))


def write_csv(path, header, rows):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(header)
        writer.writerows(rows)


def process_pair(input_path, output):
    data, p, q, images = load_pair(input_path)
    A, b = build_linear_system(p, q)
    H = computeH(p, q)
    predicted = project_points(p, H)
    errors = np.linalg.norm(predicted - q, axis=1)
    directory = output / input_path.stem
    directory.mkdir(parents=True, exist_ok=True)
    matches_figure(images, p, q, data, directory / "correspondences.png")
    reprojection_figure(images[1], q, predicted, errors, directory / "reprojection.png")
    scales = np.linalg.norm(A, axis=0)
    scales[scales == 0] = 1
    result = {
        "correspondence_file": input_path.relative_to(ROOT).as_posix() if input_path.is_relative_to(ROOT) else str(input_path),
        "image1": data["image1"], "image2": data["image2"],
        "direction": "image1 -> image2", "coordinate_system": data.get("coordinate_system", "original pixel coordinates, x right / y down"),
        "method": "All-point linear least squares, h33=1, column scaling; no outlier removal",
        "num_pairs": len(p), "linear_system_shape": list(A.shape),
        "rank": int(np.linalg.matrix_rank(A / scales)),
        "condition_number_raw_A": float(np.linalg.cond(A)),
        "condition_number_column_scaled_A": float(np.linalg.cond(A / scales)),
        "H": H.tolist(), "im1_pts": p.tolist(), "im2_pts": q.tolist(),
        "projected_im1_pts": predicted.tolist(), "forward_errors_px": errors.tolist(),
        "mean_forward_error_px": float(errors.mean()),
        "forward_rmse_px": float(np.sqrt(np.mean(errors ** 2))),
        "max_forward_error_px": float(errors.max()),
        "largest_error_point_id": int(np.argmax(errors) + 1),
        "algebraic_residual_l2": float(np.linalg.norm(A @ H.reshape(-1)[:8] - b)),
    }
    (directory / "homography.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    np.save(directory / "H.npy", H)
    np.savetxt(directory / "H.csv", H, delimiter=",", fmt="%.17g")
    write_csv(directory / "linear_system.csv", ["h11", "h12", "h13", "h21", "h22", "h23", "h31", "h32", "b"], np.column_stack((A, b)))
    np.savez(directory / "linear_system.npz", A=A, b=b, h=H.reshape(-1)[:8], column_scales=scales)
    rows = [(i + 1, *left, *right, *estimate, error) for i, (left, right, estimate, error) in enumerate(zip(p, q, predicted, errors))]
    write_csv(directory / "point_errors.csv", ["point_id", "x", "y", "u_marked", "v_marked", "u_projected", "v_projected", "error_px"], rows)
    result["name"] = input_path.stem
    result["A"], result["b"], result["rows"] = A, b, rows
    return result


def write_report(results, output, failures):
    parts = ["""<!doctype html><html lang="en"><meta charset="utf-8">
<title>A.2 — Recover Homographies</title><link rel="icon" href="../../assets/favicon.svg" type="image/svg+xml"><style>
body{font:16px/1.6 Arial,Helvetica,sans-serif;color:#171717;max-width:1120px;margin:40px auto;padding:0 24px;background:white}
h1,h2,h3{line-height:1.2;color:#171717}h1{font-size:36px;letter-spacing:-1px}h2{margin-top:45px;border-top:2px solid #fbd54b;padding-top:24px}
table{border-collapse:collapse;width:100%;margin:18px 0;font-variant-numeric:tabular-nums;background:white;display:block;overflow:auto}th,td{border:1px solid #d8d8d5;padding:8px;text-align:right}th{background:#f3f3ef}th:first-child,td:first-child{text-align:left}
pre{font:14px/1.6 Consolas,monospace;overflow:auto;background:#f3f3ef;padding:18px;border-left:4px solid #fbd54b}img{max-width:100%;height:auto}a{color:#171717;text-underline-offset:3px}
.report-brand{background:#171717;color:white;border-top:5px solid #fbd54b;padding:18px 22px;font-size:11px;font-weight:bold;letter-spacing:1px}.report-home{display:inline-block;margin-top:18px;font-size:12px}
.note{background:#fff2b6;padding:16px;border-left:4px solid #d3ae25}.small{font-size:14px}.reprojection{max-height:900px;width:auto}summary{cursor:pointer;font-weight:600}
@media print{body{margin:0;background:white;font-size:11px}h2{break-before:page}pre{font-size:9px;white-space:pre-wrap}details>pre{display:block}img{max-height:800px}a{color:inherit}}
</style><div class="report-brand">PHOTO OF THE WORLD / SYDE 671 / ASSIGNMENT 03</div><a class="report-home" href="../../index.html#homographies">← Return to the assignment</a><h1>A.2 — Recover Homographies</h1>
<p>Manual correspondences are used to estimate a projective transformation between each adjacent image pair.
All supplied points are used; no points are automatically removed. Each H maps <strong>image1 to image2</strong>.</p>
<h2>Coordinate convention and linear system</h2>
<p>Images are loaded with EXIF orientation applied, matching the annotation tool. Coordinates refer to the original image resolution;
x increases rightward, y downward, and (0,0) is the center of the first pixel. Display resizing is used only for visualization.</p>
<pre>[u, v, 1]^T ~ H [x, y, 1]^T

    [ h11 h12 h13 ]
H = [ h21 h22 h23 ],   h33 = 1
    [ h31 h32  1  ]

u = (h11*x + h12*y + h13) / (h31*x + h32*y + 1)
v = (h21*x + h22*y + h23) / (h31*x + h32*y + 1)

For each correspondence (x,y) -> (u,v):
[ x  y  1  0  0  0  -u*x  -u*y ] h = u
[ 0  0  0  x  y  1  -v*x  -v*y ] h = v

h = [h11, h12, h13, h21, h22, h23, h31, h32]^T
A has shape (2n,8), b = [u1,v1,u2,v2,...]^T

h* = argmin_h ||A h - b||_2^2
s_j = ||A[:,j]||_2; replace a zero scale with 1
z = numpy.linalg.lstsq(A / s, b, rcond=None)[0]
h = z / s</pre>
<p>Every pair contains more than four correspondences, so the system is overdetermined. Column scaling improves numerical conditioning
without changing the original-pixel algebraic least-squares objective. The solver uses NumPy linear algebra, rather than a homography library.
This linear fit minimizes algebraic residuals; it does not directly minimize geometric reprojection distances.</p>
<p>Validation measures forward errors in original target-image pixels:
<code>e_i = ||project(H, p_i) - q_i||_2</code>,
<code>RMSE = sqrt(mean(e_i^2))</code>. These are errors on the same points used to fit H, not held-out accuracy.</p>
<h2>Results overview</h2><table><tr><th>Pair</th><th>Points</th><th>System</th><th>Rank</th><th>Mean (px)</th><th>RMSE (px)</th><th>Maximum (px)</th></tr>"""]
    for result in results:
        r = result
        parts.append(f'<tr><td><a href="#{r["name"]}">{escape(r["name"])}</a></td><td>{r["num_pairs"]}</td><td>{2*r["num_pairs"]} × 8</td><td>{r["rank"]}</td><td>{r["mean_forward_error_px"]:.3f}</td><td>{r["forward_rmse_px"]:.3f}</td><td>{r["max_forward_error_px"]:.3f}</td></tr>')
    parts.append('</table><p class="note">Larger residuals can arise from inaccurate clicks, parallax from camera translation, or lens distortion. '
                 'The residuals alone cannot identify the cause. Inspect the labeled points before proceeding to A.3; do not interpret a full-rank system as proof of accurate alignment.</p>')
    for r in results:
        name = r["name"]
        parts.append(f'<h2 id="{name}">{escape(name)}</h2><p>{escape(r["image1"])} → {escape(r["image2"])}</p>')
        parts.append(f'<h3>Manual correspondences</h3><a href="{name}/correspondences.png"><img src="{name}/correspondences.png" alt="Numbered manual point matches"></a>')
        parts.append(f'<h3>Recovered H (image1 → image2)</h3><pre>{matrix_text(r["H"])}</pre>')
        parts.append(f'<p>Full-precision outputs: <a href="{name}/homography.json">H and diagnostics (JSON)</a> · <a href="{name}/H.npy">H (NumPy)</a> · <a href="{name}/H.csv">H (CSV)</a></p>')
        parts.append(f'<p>Rank: {r["rank"]}/8. Condition number: raw A = {r["condition_number_raw_A"]:.4g}; column-scaled A = {r["condition_number_column_scaled_A"]:.4g}.</p>')
        parts.append(f'<details open><summary>Numerical system A h = b ({2*r["num_pairs"]} equations)</summary><pre>Each row below is [A | b]:\n{matrix_text(np.column_stack((r["A"], r["b"])))}</pre></details>')
        parts.append(f'<p>The CSV preserves the numeric coefficients: <a href="{name}/linear_system.csv">[A | b] (CSV)</a> · <a href="{name}/linear_system.npz">A, b, h and column scales (NumPy)</a>.</p>')
        parts.append('<h3>Reprojection diagnostics</h3><table><tr><th>Point</th><th>x</th><th>y</th><th>u marked</th><th>v marked</th><th>u projected</th><th>v projected</th><th>Error (px)</th></tr>')
        for row in r["rows"]:
            parts.append('<tr><td>' + str(row[0]) + '</td>' + ''.join(f'<td>{value:.3f}</td>' for value in row[1:]) + '</tr>')
        parts.append(f'</table><p>Forward RMSE: <strong>{r["forward_rmse_px"]:.3f} px</strong>. Largest error: point {r["largest_error_point_id"]}, {r["max_forward_error_px"]:.3f} px. <a href="{name}/point_errors.csv">Download point errors</a>.</p>')
        parts.append(f'<a href="{name}/reprojection.png"><img class="reprojection" src="{name}/reprojection.png" alt="Manual targets and homography projections"></a>')
    if failures:
        parts.append('<h2>Inputs that could not be processed</h2><pre>' + escape('\n'.join(failures)) + '</pre>')
    parts.append('<h2>Reproduce and reuse</h2><pre>python a2_recover_homographies.py\npython -m unittest discover -s tests -v\n\nfrom homography import computeH\nH = computeH(im1_pts, im2_pts)</pre>'
                 '<p>For A.3, use H to map image1 into image2 coordinates. Inverse warping will use its inverse to sample image1. '
                 'The current deliverables cover A.2 only; image warping and panorama blending are subsequent steps.</p></html>')
    (output / "report.html").write_text('\n'.join(parts), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "correspondences", help="One annotation JSON file or a directory")
    parser.add_argument("--output", type=Path, default=ROOT / "output" / "a2")
    args = parser.parse_args()
    input_path, output = args.input.resolve(), args.output.resolve()
    files = sorted(input_path.glob("*.json")) if input_path.is_dir() else [input_path]
    if not files:
        parser.error("No correspondence JSON files found.")
    output.mkdir(parents=True, exist_ok=True)
    results, failures = [], []
    for path in files:
        try:
            result = process_pair(path, output)
            results.append(result)
            print(f'{path.stem}: {result["num_pairs"]} points, rank {result["rank"]}, forward RMSE {result["forward_rmse_px"]:.3f} px')
        except (OSError, ValueError, KeyError, np.linalg.LinAlgError) as error:
            failures.append(f"{path.name}: {error}")
            print(f"ERROR: {failures[-1]}")
    write_report(results, output, failures)
    write_csv(output / "summary.csv", ["pair", "num_pairs", "rank", "mean_error_px", "rmse_px", "max_error_px", "largest_error_point_id"],
              [(r["name"], r["num_pairs"], r["rank"], r["mean_forward_error_px"], r["forward_rmse_px"], r["max_forward_error_px"], r["largest_error_point_id"]) for r in results])
    print(f"Report: {output / 'report.html'}")
    if output == ROOT / "output" / "a2" and not failures:
        from build_showcase import export_showcase_data
        print(f"Showcase data: {export_showcase_data()}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
