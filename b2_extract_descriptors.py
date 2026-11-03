"""Extract B.2 descriptors from every saved B.1 ANMS selection."""

import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

from descriptors import extract_descriptors, grayscale


ROOT = Path(__file__).resolve().parent


def example_indices(points, scores, size, count=6):
    """Choose strong, spatially separated points for illustration only."""
    order = np.argsort(-scores, kind="stable")
    selected = []
    distance = min(size)*0.16
    for index in order:
        if not selected or np.linalg.norm(points[selected]-points[index], axis=1).min() >= distance:
            selected.append(int(index))
            if len(selected) == count:
                break
    for index in order:
        if len(selected) >= count:
            break
        if int(index) not in selected:
            selected.append(int(index))
    return selected


def make_examples(image, result, scores, directory):
    indices = example_indices(result['points'], scores, image.size)
    annotated = image.copy()
    draw = ImageDraw.Draw(annotated)
    try:
        font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 42)
    except OSError:
        font = ImageFont.load_default()
    gray = grayscale(np.asarray(image))
    examples = []
    relative = directory.relative_to(ROOT).as_posix()
    for number, index in enumerate(indices, 1):
        x, y = map(int, result['points'][index])
        draw.rectangle((x-20, y-20, x+19, y+19), outline="black", width=5)
        draw.rectangle((x-20, y-20, x+19, y+19), outline="#fbd54b", width=3)
        draw.text((x+24, y-25), str(number), font=font, fill="#fbd54b", stroke_width=2, stroke_fill="black")
        raw = np.clip(np.rint(gray[y-20:y+20, x-20:x+20]*255), 0, 255).astype(np.uint8)
        vector = result['descriptors'][index].reshape(8, 8)
        preview = np.clip(np.rint((vector+3)/6*255), 0, 255).astype(np.uint8)
        Image.fromarray(raw).save(directory / f"patch_{number}_40.png")
        Image.fromarray(preview).save(directory / f"patch_{number}_8.png")
        examples.append({"number": number, "descriptor_index": index, "point": [x, y],
                         "raw_patch": f"{relative}/patch_{number}_40.png",
                         "normalized_patch": f"{relative}/patch_{number}_8.png"})
    annotated.save(directory / "examples.jpg", quality=93)
    return examples


def process_image(record):
    with Image.open(ROOT / record['image']) as source:
        image = ImageOps.exif_transpose(source).convert('RGB')
    if list(image.size) != record['size']:
        raise ValueError(f"Image dimensions changed: {record['image']}")
    with np.load(ROOT / record['features']) as saved:
        b1_indices = saved['anms_indices']
        points = saved['points'][b1_indices]
        scores = saved['strengths'][b1_indices]
    data = extract_descriptors(np.asarray(image), points)
    output = ROOT / 'output' / 'b2' / record['id']
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output / 'descriptors.npz', **data,
                        b1_candidate_indices=b1_indices[data['source_indices']], image_size=np.array(image.size))
    examples = make_examples(image, data, scores[data['source_indices']], output)
    descriptors = data['descriptors']
    with (output / 'descriptors.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.writer(stream)
        writer.writerow(['descriptor_id', 'b1_candidate_id', 'x', 'y', 'sample_mean', 'sample_std']
                        + [f'd{i:02d}' for i in range(64)])
        for index, ((x, y), vector) in enumerate(zip(data['points'], descriptors)):
            writer.writerow([index, int(b1_indices[data['source_indices'][index]]), int(x), int(y),
                             data['means'][index], data['stds'][index]] + [f'{v:.9g}' for v in vector])
    relative = output.relative_to(ROOT).as_posix()
    result = {
        'id': record['id'], 'title': record['title'], 'image': record['image'], 'size': list(image.size),
        'b1_features': record['features'], 'input_count': len(points), 'descriptor_count': len(descriptors),
        'border_rejected': len(data['border_rejected']), 'low_contrast_rejected': len(data['contrast_rejected']),
        'descriptor_shape': list(descriptors.shape),
        'max_absolute_mean': float(np.abs(descriptors.mean(axis=1)).max()) if len(descriptors) else None,
        'max_std_error': float(np.abs(descriptors.std(axis=1)-1).max()) if len(descriptors) else None,
        'annotated': f'{relative}/examples.jpg', 'examples': examples,
        'features': f'{relative}/descriptors.npz', 'csv': f'{relative}/descriptors.csv',
        'record_path': f'{relative}/descriptors.json',
    }
    (output / 'descriptors.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    print(f'{record["title"]}: {len(points)} ANMS points -> {descriptors.shape} descriptors; '
          f'{result["border_rejected"]+result["low_contrast_rejected"]} skipped', flush=True)
    return result


def main():
    source = ROOT / 'output' / 'b1' / 'summary.json'
    if not source.exists():
        raise ValueError('Run b1_detect_corners.py before B.2.')
    b1 = json.loads(source.read_text(encoding='utf-8'))
    results = [process_image(record) for record in b1['examples']]
    summary = {
        'method': 'Axis-aligned blurred 40x40 window, 8x8 sampling, zero-mean/unit-standard-deviation normalization',
        'parameters': {'window_size': 40, 'descriptor_size': 8, 'gaussian_sigma': 2.0,
                       'sample_stride': 5, 'sample_offset_in_window': 2, 'min_std': 1e-6},
        'coordinates': 'EXIF-oriented original integer (x,y) points from B.1 ANMS; no rotation',
        'normalization': 'Population standard deviation (ddof=0); float32 output; incomplete or near-constant windows rejected',
        'preview': '40x40 unblurred grayscale crop and 8x8 normalized patch. Fixed display range [-3,3] is clipped for PNG only; stored descriptors are not clipped.',
        'example_selection': 'Up to six strong spatially separated valid ANMS points; illustration only, all valid points are extracted',
        'reference': 'https://www.microsoft.com/en-us/research/wp-content/uploads/2005/06/cvpr05.pdf#page=3',
        'examples': results,
    }
    output = ROOT / 'output' / 'b2'
    (output / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    with (output / 'summary.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.writer(stream)
        writer.writerow(['image', 'input_points', 'descriptors', 'dimensions', 'skipped', 'max_abs_mean', 'max_std_error'])
        for result in results:
            writer.writerow([result['title'], result['input_count'], result['descriptor_count'], 64,
                             result['border_rejected']+result['low_contrast_rejected'],
                             result['max_absolute_mean'], result['max_std_error']])
    from build_showcase import export_showcase_data
    export_showcase_data()
    print(f'B.2 results: {output}')


if __name__ == '__main__':
    main()
