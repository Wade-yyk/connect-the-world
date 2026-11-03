"""Refresh assets/results.js from original photos and saved A.2 calculations.

The site is buildless and opens from file:// or a static host. A JavaScript data
file avoids fetch restrictions when the user opens index.html directly.
"""

import json
from html import escape
from pathlib import Path
from string import Template

from PIL import Image, ImageOps

from homography import build_linear_system


ROOT = Path(__file__).resolve().parent


def render_a3_section():
    path = ROOT / 'output' / 'a3' / 'summary.json'
    if not path.exists():
        return '<p class="placeholder">To add: two rectification examples and nearest-neighbor / bilinear comparison.</p>'
    summary = json.loads(path.read_text(encoding='utf-8'))
    examples, rows = [], []
    for result in summary['examples']:
        examples.append(f'<h3>{escape(result["title"])}</h3><div class="rectification-grid">')
        for key, label in (('source_crop', 'Original crop · selected corners'), ('nearest_image', 'Nearest neighbor'), ('bilinear_image', 'Bilinear')):
            source = escape(result[key])
            examples.append(f'<figure><a href="{source}" target="_blank" rel="noopener"><img src="{source}" alt="{escape(result["title"])}: {label}" loading="lazy"></a><figcaption>{label}</figcaption></figure>')
        examples.append('</div>')
        examples.append(f'<p class="small">Full photo: <a href="{escape(result["image"])}" target="_blank" rel="noopener">original</a> · <a href="{escape(result["source_annotated"])}" target="_blank" rel="noopener">marked corners</a>. 4× pixel detail: <a href="{result["detail_nearest"]}" target="_blank" rel="noopener">nearest</a> / <a href="{result["detail_bilinear"]}" target="_blank" rel="noopener">bilinear</a>. <a href="{result["record_path"]}">H and coordinates</a>.</p>')
        width, height = result['target_size']
        rows.append(f'<tr><td>{escape(result["title"])}</td><td>{width} × {height}</td><td>{result["nearest_ms"]:.3f}</td><td>{result["bilinear_ms"]:.3f}</td></tr>')
    template = Template((ROOT / 'templates' / 'a3.html').read_text(encoding='utf-8'))
    return template.substitute(examples='\n'.join(examples), timing_rows='\n'.join(rows), repeats=summary['timing']['repeats'])


def render_a4_section():
    path = ROOT / 'output' / 'a4' / 'summary.json'
    if not path.exists():
        return '<p class="placeholder">To add: three mosaics, source images, and a short blending explanation.</p>'
    summary = json.loads(path.read_text(encoding='utf-8'))
    examples, rows = [], []
    for result in summary['examples']:
        examples.append(f'<h3>{escape(result["title"])}</h3><div class="mosaic-sources">')
        for source in result['sources']:
            label = f'{"Outdoor" if Path(source).parent.name == "1" else "Kitchen"} {Path(source).stem}'
            source = escape(source)
            examples.append(f'<figure><a href="{source}" target="_blank" rel="noopener"><img src="{source}" alt="Source: {label}" loading="lazy"></a><figcaption>{label}</figcaption></figure>')
        examples.append('</div><div class="mosaic-results">')
        for key, label in (('overwrite', 'Before · overwrite'), ('mosaic', 'After · weighted blending')):
            source = escape(result[key])
            examples.append(f'<figure><a href="{source}" target="_blank" rel="noopener"><img src="{source}" alt="{escape(result["title"])}: {label}" loading="lazy"></a><figcaption>{label}</figcaption></figure>')
        examples.append('</div>')
        reference = Path(result['reference_image']).stem
        crop_link = f' · <a href="#crop-{escape(result["id"])}">Cropped version ↓</a>' if 'cropped' in result else ''
        examples.append(f'<p class="small">Reference: image {reference}. <a href="{escape(result["mosaic"])}" download>Full-resolution mosaic</a> · <a href="{escape(result["record_path"])}">H matrices and canvas</a>{crop_link}</p>')
        if 'cropped' in result:
            crop_width, crop_height = result['cropped_size']
            if 'laplacian_cropped' in result:
                examples.append(f'<div class="mosaic-results mosaic-cropped pyramid-comparison" id="crop-{escape(result["id"])}">')
                for key, label in (('cropped', 'Feathering · cropped'), ('laplacian_cropped', f'Laplacian pyramid · {result["pyramid_levels"]} levels · cropped')):
                    source = escape(result[key])
                    examples.append(f'<figure><a href="{source}" target="_blank" rel="noopener"><img src="{source}" alt="{escape(result["title"])}: {label}" width="{crop_width}" height="{crop_height}" loading="lazy"></a><figcaption>{label} · <a href="{source}" download>Download</a></figcaption></figure>')
                examples.append('</div>')
                examples.append(f'<p class="small">Same H, canvas, and {crop_width} × {crop_height} crop. <a href="{escape(result["laplacian"])}" target="_blank" rel="noopener">Full Laplacian mosaic</a>.</p>')
                if 'feather_detail' in result:
                    examples.append('<details><summary>Close-up comparison</summary><div class="mosaic-results">')
                    for key, label in (('feather_detail', 'Feathering · detail'), ('laplacian_detail', 'Laplacian pyramid · same detail')):
                        source = escape(result[key])
                        examples.append(f'<figure><a href="{source}" target="_blank" rel="noopener"><img src="{source}" alt="{escape(result["title"])}: {label}" loading="lazy"></a><figcaption>{label}</figcaption></figure>')
                    examples.append('</div></details>')
            else:
                source = escape(result['cropped'])
                examples.append(f'<figure class="mosaic-cropped" id="crop-{escape(result["id"])}"><a href="{source}" target="_blank" rel="noopener"><img src="{source}" alt="{escape(result["title"])}: cropped mosaic without empty borders" width="{crop_width}" height="{crop_height}" loading="lazy"></a><figcaption>Cropped mosaic · {crop_width} × {crop_height} · empty borders removed. <a href="{source}" download>Download</a></figcaption></figure>')
        width, height = result['output_size']
        rows.append(f'<tr><td>{escape(result["title"])}</td><td>{len(result["sources"])}</td><td>{width} × {height}</td><td>{result["overlap_percent"]:.1f}%</td></tr>')
    template = Template((ROOT / 'templates' / 'a4.html').read_text(encoding='utf-8'))
    return template.substitute(examples='\n'.join(examples), result_rows='\n'.join(rows))


def render_b1_section():
    path = ROOT / 'output' / 'b1' / 'summary.json'
    data = json.loads(path.read_text(encoding='utf-8')) if path.exists() else {'examples': []}
    (ROOT / 'assets' / 'corners.js').write_text('window.B1_DATA = ' + json.dumps(data, ensure_ascii=False, allow_nan=False).replace('</', '<\\/') + ';\n', encoding='utf-8')
    if not data['examples']:
        return '<p class="placeholder">To add: Harris corners with and without ANMS.</p>'
    results = data['examples']
    selected = next((item for item in results if item['id'] == 'set_2_2'), results[0])
    options, rows = [], []
    for result in results:
        options.append(f'<option value="{result["id"]}"{" selected" if result is selected else ""}>{escape(result["title"])}</option>')
        rows.append(f'<tr><td>{escape(result["title"])}</td><td>{result["candidate_count"]}</td><td>{result["selected_count"]}</td><td>{result["strongest_cells"]}</td><td>{result["anms_cells"]}</td></tr>')
    template = Template((ROOT / 'templates' / 'b1.html').read_text(encoding='utf-8'))
    return template.substitute(count=data['parameters']['count'], options='\n'.join(options),
                               width=selected['size'][0], height=selected['size'][1],
                               threshold_percent=f"{data['parameters']['threshold_rel']*100:g}",
                               harris_top=selected['harris_top'], anms_image=selected['anms_image'],
                               harris_all=selected['harris_all'], csv=selected['csv'],
                               record_path=selected['record_path'], features=selected['features'],
                               top_caption=f'{selected["title"]} · strongest {selected["selected_count"]} Harris corners · no ANMS',
                               anms_caption=f'{selected["title"]} · ANMS {selected["selected_count"]} corners',
                               all_caption=f'{selected["title"]} · all {selected["candidate_count"]} Harris candidates',
                               result_rows='\n'.join(rows))


def render_b2_section():
    path = ROOT / 'output' / 'b2' / 'summary.json'
    if not path.exists():
        return '<p class="placeholder">To add: normalized 8 × 8 descriptors and example patches.</p>'
    data = json.loads(path.read_text(encoding='utf-8'))
    results = data['examples']
    selected = next((item for item in results if item['id'] == 'set_2_2'), results[0])
    options, panels, rows = [], [], []
    for result in results:
        identifier, title = escape(result['id']), escape(result['title'])
        options.append(f'<option value="{identifier}"{" selected" if result is selected else ""}>{title}</option>')
        panels.append(f'<div class="descriptor-panel" data-descriptor-image="{identifier}"{" hidden" if result is not selected else ""}><div class="descriptor-layout">')
        source = escape(result['annotated'])
        width, height = result['size']
        panels.append(f'<figure class="descriptor-context"><a href="{source}" target="_blank" rel="noopener"><img src="{source}" alt="{title}: numbered descriptor windows" width="{width}" height="{height}" loading="lazy"></a><figcaption>{title} · selected 40 × 40 windows</figcaption></figure><div class="descriptor-grid">')
        for example in result['examples']:
            number = example['number']
            x, y = example['point']
            panels.append(f'<div class="descriptor-card"><p class="patch-label"><strong>{number}</strong> ({x}, {y})</p><div class="patch-pair">')
            for key, size, label in (('raw_patch', 40, '40 × 40'), ('normalized_patch', 8, '8 × 8')):
                patch = escape(example[key])
                panels.append(f'<figure><a href="{patch}" target="_blank" rel="noopener"><img src="{patch}" width="{size}" height="{size}" alt="{title}, point {number}: {label} patch" loading="lazy"></a><figcaption>{label}</figcaption></figure>')
            panels.append('</div></div>')
        panels.append('</div></div>')
        panels.append(f'<p class="small"><a href="{escape(result["features"])}" download>All descriptors (NPZ)</a> · <a href="{escape(result["csv"])}" download>64-value vectors (CSV)</a> · <a href="{escape(result["record_path"])}">Example coordinates</a></p></div>')
        skipped = result['border_rejected']+result['low_contrast_rejected']
        rows.append(f'<tr><td>{title}</td><td>{result["input_count"]}</td><td>{result["descriptor_count"]} × 64</td><td>{skipped}</td></tr>')
    template = Template((ROOT / 'templates' / 'b2.html').read_text(encoding='utf-8'))
    return template.substitute(options='\n'.join(options), panels='\n'.join(panels), result_rows='\n'.join(rows))


def render_b3_section():
    path = ROOT / 'output' / 'b3' / 'summary.json'
    if not path.exists():
        return '<p class="placeholder">To add: feature matching and corresponding-point figures.</p>'
    data = json.loads(path.read_text(encoding='utf-8'))
    results = data['examples']
    selected = next((item for item in results if item['id'] == 'set_2_1_to_2'), results[0])
    options, panels, rows = [], [], []
    for result in results:
        identifier, title = escape(result['id']), escape(result['title'])
        options.append(f'<option value="{identifier}"{" selected" if result is selected else ""}>{title}</option>')
        source = escape(result['matches_preview'])
        width, height = result['figure_size']
        panels.append(f'<div class="match-panel" data-match-pair="{identifier}"{" hidden" if result is not selected else ""}><figure><a href="{source}" target="_blank" rel="noopener"><img src="{source}" width="{width}" height="{height}" alt="{title}: automatically matched features after the ratio test" loading="lazy"></a><figcaption>{title} · {result["preview_count"]} of {result["match_count"]} retained matches · lowest distance ratios</figcaption></figure>')
        panels.append(f'<p class="small"><a href="{escape(result["matches_all"])}" target="_blank" rel="noopener">All {result["match_count"]} matches</a> · <a href="{escape(result["csv"])}" download>Match coordinates (CSV)</a> · <a href="{escape(result["features"])}" download>Matches (NPZ)</a> · <a href="{escape(result["queries_csv"])}" download>All nearest-neighbor tests</a></p></div>')
        median = f'{result["median_ratio"]:.3f}' if result['median_ratio'] is not None else '—'
        rows.append(f'<tr><td>{title}</td><td>{result["match_count"]}</td><td>{result["retained_percent"]:.1f}%</td><td>{median}</td></tr>')
    template = Template((ROOT / 'templates' / 'b3.html').read_text(encoding='utf-8'))
    return template.substitute(options='\n'.join(options), panels='\n'.join(panels), result_rows='\n'.join(rows),
                               ratio_threshold=f'{data["ratio_threshold"]:g}',
                               squared_threshold=f'{data["squared_ratio_threshold"]:g}',
                               preview_limit=max(item['preview_count'] for item in results))


def render_b4_section():
    path = ROOT / 'output' / 'b4' / 'summary.json'
    if not path.exists():
        return '<p class="placeholder">To add: RANSAC and three manual/automatic mosaic comparisons.</p>'
    data = json.loads(path.read_text(encoding='utf-8'))
    mosaics, panels, options, rows = [], [], [], []
    for result in data['mosaics']:
        title = escape(result['title'])
        mosaics.append(f'<h3>{title}</h3><div class="mosaic-sources">')
        for source in result['sources']:
            label = f'{"Outdoor" if Path(source).parent.name == "1" else "Kitchen"} {Path(source).stem}'
            mosaics.append(f'<figure><a href="{escape(source)}" target="_blank" rel="noopener"><img src="{escape(source)}" alt="{label}: original source" loading="lazy"></a><figcaption>{label}</figcaption></figure>')
        mosaics.append(f'</div><p class="small"><strong>Uncropped</strong></p><div class="mosaic-results mosaic-full" id="full-auto-{escape(result["id"])}">')
        for key, label in (('manual_full', 'Manual · uncropped'), ('automatic_full', 'Automatic · uncropped')):
            source = escape(result[key])
            with Image.open(ROOT / result[key]) as image:
                full_width, full_height = image.size
            mosaics.append(f'<figure><a href="{source}" target="_blank" rel="noopener"><img src="{source}" alt="{title}: {label}" width="{full_width}" height="{full_height}" loading="lazy"></a><figcaption>{label} · {full_width} × {full_height}</figcaption></figure>')
        mosaics.append(f'</div><p class="small"><strong>Shared crop</strong></p><div class="mosaic-results" id="auto-{escape(result["id"])}">')
        width, height = result['comparison_size']
        for key, label in (('manual_comparison', 'Manual points · Part A'), ('automatic_comparison', 'Automatic matches · RANSAC')):
            source = escape(result[key])
            mosaics.append(f'<figure><a href="{source}" target="_blank" rel="noopener"><img src="{source}" alt="{title}: {label}" width="{width}" height="{height}" loading="lazy"></a><figcaption>{label} · shared crop</figcaption></figure>')
        mosaics.append('</div>')
        mosaics.append(f'<p class="small">Reference: image {Path(result["reference_image"]).stem}. Full image: <a href="{escape(result["manual_full"])}" target="_blank" rel="noopener">manual</a> / <a href="{escape(result["automatic_full"])}" target="_blank" rel="noopener">automatic</a>. <a href="{escape(result["automatic_feather"])}" target="_blank" rel="noopener">Automatic feather blend</a> · <a href="{escape(result["record_path"])}">H matrices and crop coordinates</a></p>')
        mosaics.append('<details><summary>Close-up: manual vs automatic</summary><div class="mosaic-results">')
        for key, label in (('manual_detail', 'Manual · detail'), ('automatic_detail', 'Automatic · same detail')):
            source = escape(result[key])
            mosaics.append(f'<figure><a href="{source}" target="_blank" rel="noopener"><img src="{source}" alt="{title}: {label}" loading="lazy"></a><figcaption>{label}</figcaption></figure>')
        mosaics.append('</div></details>')
    selected = next((p for p in data['pairs'] if p['id'] == 'set_2_1_to_2'), data['pairs'][0])
    for pair in data['pairs']:
        title, identifier = escape(pair['title']), escape(pair['id'])
        options.append(f'<option value="{identifier}"{" selected" if pair is selected else ""}>{title}</option>')
        source = escape(pair['inliers_image'])
        width, height = pair['figure_size']
        matrix = '\n'.join('[ ' + ' '.join(f'{v:14.6e}' for v in row) + ' ]' for row in pair['H'])
        panels.append(f'<div class="ransac-panel" data-ransac-pair="{identifier}"{" hidden" if pair is not selected else ""}><figure><a href="{source}" target="_blank" rel="noopener"><img src="{source}" alt="{title}: RANSAC inlier correspondences" width="{width}" height="{height}" loading="lazy"></a><figcaption>{title} · {pair["inlier_count"]} / {pair["match_count"]} inliers · H maps left image to right image</figcaption></figure><pre>{matrix}</pre>')
        panels.append(f'<p class="small"><a href="{escape(pair["rejected_image"])}" target="_blank" rel="noopener">Rejected matches</a> · <a href="{escape(pair["errors_csv"])}" download>All point errors (CSV)</a> · <a href="{escape(pair["record_path"])}">H and sampling details</a> · <a href="{escape(pair["features"])}" download>RANSAC data (NPZ)</a></p></div>')
        rows.append(f'<tr><td>{title}</td><td>{pair["match_count"]}</td><td>{pair["inlier_count"]}</td><td>{pair["inlier_percent"]:.1f}%</td><td>{pair["inlier_rmse_px"]:.3f}</td></tr>')
    parameters = data['parameters']
    weakest = min(data['pairs'], key=lambda pair: pair['inlier_count'])
    template = Template((ROOT / 'templates' / 'b4.html').read_text(encoding='utf-8'))
    return template.substitute(mosaics='\n'.join(mosaics), pair_options='\n'.join(options),
                               pair_panels='\n'.join(panels), result_rows='\n'.join(rows),
                               threshold=f'{parameters["threshold_px"]:g}', seed=parameters['seed_base'],
                               min_hypotheses=parameters['min_valid_hypotheses'], max_trials=parameters['max_trials'],
                               weakest_pair=escape(weakest['title']), weakest_count=weakest['inlier_count'])


def render_page(sets, pairs):
    galleries = []
    for group in sets:
        galleries.append(f'<h3>{escape(group["scene"])}</h3><div class="photo-grid {"kitchen" if group["id"] == "2" else ""}">')
        for index, photo in enumerate(group["photos"], 1):
            source, alt = escape(photo["src"]), escape(photo["alt"])
            note = '<p class="photo-note">This is the link to the startup company. Welcome to visit!</p>' if group["id"] == "1" and index == 3 else ''
            galleries.append(f'<figure>{note}<a href="{source}" target="_blank" rel="noopener"><img src="{source}" alt="{alt}" width="{photo["size"][0]}" height="{photo["size"][1]}" loading="lazy"></a><figcaption>{group["scene"]} {index}</figcaption></figure>')
        galleries.append('</div>')
    selected = next((pair for pair in pairs if pair["id"] == "set_2_1_to_2"), pairs[0])
    options, rows = [], []
    for pair in pairs:
        label = f'{pair["scene"]} {pair["source_frame"]} → {pair["target_frame"]}'
        options.append(f'<option value="{pair["id"]}"{" selected" if pair is selected else ""}>{label}</option>')
        rows.append(f'<tr><td>{label}</td><td>{pair["num_pairs"]}</td><td>{pair["forward_rmse_px"]:.2f}</td><td>{pair["max_forward_error_px"]:.2f}</td></tr>')
    matrix = '\n'.join('[ ' + ' '.join(f'{value:14.6e}' for value in row) + ' ]' for row in selected['H'])
    system = '\n'.join('[ ' + ' '.join(f'{value:15.7g}' for value in [*row, selected['b'][i]]) + ' ]' for i, row in enumerate(selected['A']))
    template = Template((ROOT / 'templates' / 'index.html').read_text(encoding='utf-8'))
    html = template.substitute(photo_sets='\n'.join(galleries), pair_options='\n'.join(options),
                               matches_image=selected['matches_image'], matrix=matrix, system=system,
                               pair_caption=f'{selected["scene"]} {selected["source_frame"]} → {selected["target_frame"]} · {selected["num_pairs"]} point pairs. Click the image to enlarge.',
                               system_path=selected['system_path'], json_path=selected['json_path'], summary_rows='\n'.join(rows),
                               a3_section=render_a3_section(), a4_section=render_a4_section(),
                               b1_section=render_b1_section(), b2_section=render_b2_section(),
                               b3_section=render_b3_section(), b4_section=render_b4_section())
    (ROOT / 'index.html').write_text(html, encoding='utf-8')


def export_showcase_data():
    descriptions = {
        "1": ("Around the driveway", "Outdoor",
              "Four original views of a residential street, a parked truck, and a house. Building corners and vehicle details provide manual landmarks across the sequence.",
              ["Street and driveway with the edge of a parked truck", "Parked truck and trees beside the street", "Parked truck in front of a house", "House facade, garage, and driveway"]),
        "2": ("Across the kitchen", "Kitchen",
              "Three original views move across the kitchen scene. Cabinet corners, the range hood, and countertop details provide landmarks; nearby objects also make this a useful case for examining alignment limitations.",
              ["Kitchen view with refrigerator, cabinets, and stove", "Kitchen view with stove, cabinets, and island", "Kitchen view with countertop, cabinets, and doorway"]),
    }
    sets = []
    for folder in sorted((ROOT / "photos").iterdir()):
        if not folder.is_dir():
            continue
        images = sorted(path for path in folder.iterdir() if path.suffix.lower() in {".jpg", ".jpeg", ".png"})
        if not images:
            continue
        title, scene, description, alts = descriptions.get(folder.name, (f"Image set {folder.name}", f"Set {folder.name}", "Original source photographs.", []))
        photos = []
        for index, path in enumerate(images):
            with Image.open(path) as source:
                size = list(ImageOps.exif_transpose(source).size)
            photos.append({"src": path.relative_to(ROOT).as_posix(), "size": size,
                           "alt": alts[index] if index < len(alts) else f"{scene} photograph {index + 1}"})
        sets.append({"id": folder.name, "title": title, "scene": scene, "description": description,
                     "size": photos[0]["size"], "photos": photos})
    pairs = []
    for path in sorted((ROOT / "output" / "a2").glob("*/homography.json")):
        result = json.loads(path.read_text(encoding="utf-8"))
        result["id"] = path.parent.name
        source, target = Path(result["image1"]), Path(result["image2"])
        result["set_id"] = source.parent.name
        result["scene"] = next((item["scene"] for item in sets if item["id"] == result["set_id"]), f"Set {result['set_id']}")
        result["source_frame"], result["target_frame"] = source.stem, target.stem
        folder = path.parent.relative_to(ROOT).as_posix()
        for name, filename in (("matches_image", "correspondences.png"), ("reprojection_image", "reprojection.png"),
                               ("json_path", "homography.json"), ("errors_path", "point_errors.csv"), ("system_path", "linear_system.csv")):
            result[name] = f"{folder}/{filename}"
        A, b = build_linear_system(result["im1_pts"], result["im2_pts"])
        result["A"], result["b"] = A.tolist(), b.tolist()
        pairs.append(result)
    if not pairs:
        raise ValueError("No saved A.2 results. Run a2_recover_homographies.py first.")
    destination = ROOT / "assets" / "results.js"
    destination.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps({"sets": sets, "pairs": pairs}, ensure_ascii=False, indent=2, allow_nan=False).replace("</", "<\\/")
    destination.write_text("// Generated by build_showcase.py; rerun A.2 to refresh.\nwindow.SHOWCASE_DATA = " + serialized + ";\n", encoding="utf-8")
    render_page(sets, pairs)
    return destination


if __name__ == "__main__":
    print(export_showcase_data())
