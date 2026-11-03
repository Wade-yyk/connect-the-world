"""Match the B.2 descriptors for all adjacent photo pairs and draw matches."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

from feature_matching import match_descriptors


ROOT = Path(__file__).resolve().parent


def draw_matches(image1, image2, points1, points2, path, label1, label2):
    header, gap = 42, 32
    offset = image1.width+gap
    canvas = Image.new('RGB', (offset+image2.width, header+max(image1.height, image2.height)), '#171717')
    canvas.paste(image1, (0, header))
    canvas.paste(image2, (offset, header))
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype('C:/Windows/Fonts/arial.ttf', 25)
    except OSError:
        font = ImageFont.load_default()
    draw.text((12, 6), label1, fill='white', font=font)
    draw.text((offset+12, 6), label2, fill='white', font=font)
    palette = ('#fbd54b', '#00dcff', '#ff7b72', '#b5ef62', '#cda5ff', '#ffffff')
    for i, (p1, p2) in enumerate(zip(points1, points2)):
        color = palette[i % len(palette)]
        a, b = (int(p1[0]), int(p1[1])+header), (int(p2[0])+offset, int(p2[1])+header)
        draw.line((a, b), fill=color, width=2)
        for x, y in (a, b):
            draw.ellipse((x-6, y-6, x+6, y+6), fill='black')
            draw.ellipse((x-4, y-4, x+4, y+4), fill=color)
    canvas.save(path, quality=94)
    return list(canvas.size)


def load_features(record):
    with Image.open(ROOT / record['image']) as source:
        image = ImageOps.exif_transpose(source).convert('RGB')
    with np.load(ROOT / record['features']) as data:
        points, descriptors = data['points'].copy(), data['descriptors'].copy()
        if list(image.size) != data['image_size'].tolist() or len(points) != len(descriptors):
            raise ValueError(f"B.2 data do not match the source image: {record['image']}")
    if descriptors.shape[1:] != (64,) or points.shape != (len(descriptors), 2):
        raise ValueError('Expected 64-value B.2 descriptors and their Nx2 coordinates.')
    return image, points, descriptors


def process_pair(record1, record2, threshold, preview_limit):
    image1, points1, descriptors1 = load_features(record1)
    image2, points2, descriptors2 = load_features(record2)
    search = match_descriptors(descriptors1, descriptors2, ratio_threshold=threshold)
    matches = search['matches']
    indices1, indices2 = matches[:, 0], matches[:, 1]
    ratios = search['ratios'][indices1]
    accepted1, accepted2 = points1[indices1], points2[indices2]
    set_id = Path(record1['image']).parent.name
    frame1, frame2 = Path(record1['image']).stem, Path(record2['image']).stem
    identifier = f'set_{set_id}_{frame1}_to_{frame2}'
    title = f'{"Outdoor" if set_id == "1" else "Kitchen"} {frame1} → {frame2}'
    output = ROOT / 'output' / 'b3' / identifier
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output / 'matches.npz', matches=matches, im1_pts=accepted1,
                        im2_pts=accepted2, ratios=ratios,
                        distance1=search['distance1'][indices1], distance2=search['distance2'][indices1],
                        all_nearest_indices=search['nearest_indices'], all_second_indices=search['second_indices'],
                        all_distance1=search['distance1'], all_distance2=search['distance2'],
                        all_ratios=search['ratios'], accepted_mask=search['accepted'])
    with (output / 'matches.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.writer(stream)
        writer.writerow(['match_id', 'descriptor1', 'descriptor2', 'x1', 'y1', 'x2', 'y2', 'd1', 'd2', 'ratio'])
        for i, ((i1, i2), p1, p2, ratio) in enumerate(zip(matches, accepted1, accepted2, ratios)):
            writer.writerow([i, int(i1), int(i2), *p1.tolist(), *p2.tolist(),
                             search['distance1'][i1], search['distance2'][i1], ratio])
    with (output / 'queries.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.writer(stream)
        writer.writerow(['descriptor1', 'nearest_descriptor2', 'second_descriptor2', 'd1', 'd2', 'ratio', 'accepted'])
        for i in range(len(descriptors1)):
            writer.writerow([i, search['nearest_indices'][i], search['second_indices'][i],
                             search['distance1'][i], search['distance2'][i], search['ratios'][i], int(search['accepted'][i])])
    preview_indices = np.argsort(ratios, kind='stable')[:preview_limit]
    size = draw_matches(image1, image2, accepted1[preview_indices], accepted2[preview_indices],
                        output / 'matches_preview.jpg', record1['title'], record2['title'])
    draw_matches(image1, image2, accepted1, accepted2, output / 'matches_all.jpg', record1['title'], record2['title'])
    relative = output.relative_to(ROOT).as_posix()
    result = {
        'id': identifier, 'title': title, 'image1': record1['image'], 'image2': record2['image'],
        'features1': record1['features'], 'features2': record2['features'],
        'descriptor_counts': [len(descriptors1), len(descriptors2)],
        'ratio_threshold': threshold, 'squared_ratio_threshold': threshold**2,
        'match_count': len(matches), 'unique_target_count': len(np.unique(indices2)),
        'retained_percent': 100*len(matches)/len(descriptors1) if len(descriptors1) else 0,
        'median_ratio': float(np.median(ratios)) if len(ratios) else None,
        'preview_count': len(preview_indices), 'preview_match_indices': preview_indices.tolist(),
        'figure_size': size, 'matches_preview': f'{relative}/matches_preview.jpg',
        'matches_all': f'{relative}/matches_all.jpg', 'features': f'{relative}/matches.npz',
        'csv': f'{relative}/matches.csv', 'queries_csv': f'{relative}/queries.csv',
        'record_path': f'{relative}/matches.json',
        'matches': matches.tolist(), 'im1_pts': accepted1.tolist(), 'im2_pts': accepted2.tolist(),
        'ratios': ratios.tolist(),
    }
    (output / 'matches.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    console_title = title.replace('→', '->')
    print(f'{console_title}: {len(descriptors1)} queries -> {len(matches)} matches (d1/d2 < {threshold:g})', flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ratio', type=float, default=0.7)
    parser.add_argument('--preview-limit', type=int, default=40)
    args = parser.parse_args()
    if not 0 < args.ratio < 1 or args.preview_limit < 1:
        parser.error('Use 0 < --ratio < 1 and --preview-limit >= 1')
    path = ROOT / 'output' / 'b2' / 'summary.json'
    if not path.exists():
        raise ValueError('Run b2_extract_descriptors.py before B.3.')
    records = json.loads(path.read_text(encoding='utf-8'))['examples']
    groups = {}
    for record in records:
        groups.setdefault(Path(record['image']).parent.name, []).append(record)
    results = []
    for _, group in sorted(groups.items()):
        ordered = sorted(group, key=lambda r: int(Path(r['image']).stem))
        results.extend(process_pair(a, b, args.ratio, args.preview_limit) for a, b in zip(ordered, ordered[1:]))
    if not results:
        raise ValueError('At least two images from one scene are required.')
    summary = {
        'method': 'Exact left-to-right two-nearest-neighbor Euclidean matching with Lowe ratio test',
        'ratio_threshold': args.ratio, 'squared_ratio_threshold': args.ratio**2,
        'threshold_note': f'{args.ratio:g} in Euclidean distance is {args.ratio**2:g} in squared distance; Figure 6b uses squared errors.',
        'ambiguity': 'Reject a query if the second squared distance is <= 1e-12 (including two identical best descriptors).',
        'selection': 'One-way ratio test only; no mutual check, target deduplication, manual points, homography, or RANSAC.',
        'visualization': f'Preview shows up to {args.preview_limit} retained matches with smallest ratios; all retained matches are exported separately.',
        'reference': 'https://www.microsoft.com/en-us/research/wp-content/uploads/2005/06/cvpr05.pdf#page=4',
        'examples': results,
    }
    output = ROOT / 'output' / 'b3'
    (output / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    with (output / 'summary.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.writer(stream)
        writer.writerow(['pair', 'descriptors1', 'descriptors2', 'matches', 'retained_percent', 'median_ratio', 'ratio_threshold'])
        for result in results:
            writer.writerow([result['title'], *result['descriptor_counts'], result['match_count'],
                             result['retained_percent'], result['median_ratio'], args.ratio])
    from build_showcase import export_showcase_data
    export_showcase_data()
    print(f'B.3 results: {output}')


if __name__ == '__main__':
    main()
