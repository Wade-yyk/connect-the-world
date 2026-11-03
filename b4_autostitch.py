"""B.4: hand-written RANSAC and three manual/automatic mosaic comparisons."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from b3_match_features import draw_matches
from mosaicing import largest_valid_rectangle, stitch_mosaic
from ransac import ransac_homography


ROOT = Path(__file__).resolve().parent
PYRAMID_LEVELS = 6


def transforms_to_reference(images, reference, edges):
    """Compose adjacent source->target H matrices into a common reference."""
    reference_index = images.index(reference)
    transforms = []
    for index in range(len(images)):
        H = np.eye(3)
        if index < reference_index:
            for step in range(index, reference_index):
                H = edges[(images[step], images[step+1])] @ H
        else:
            for step in range(index-1, reference_index-1, -1):
                H = np.linalg.inv(edges[(images[step], images[step+1])]) @ H
        transforms.append(H/np.max(np.abs(H)))
    return transforms


def common_crop(manual_rgba, manual_origin, automatic, automatic_valid, automatic_origin):
    """Largest rectangle valid in both results, at identical reference pixels."""
    manual_origin, automatic_origin = np.array(manual_origin), np.array(automatic_origin)
    low = np.maximum(manual_origin, automatic_origin)
    high = np.minimum(manual_origin+np.array(manual_rgba.shape[1::-1]),
                      automatic_origin+np.array(automatic.shape[1::-1]))
    width, height = map(int, high-low)
    if min(width, height) <= 0:
        raise ValueError('Manual and automatic results have no shared canvas region.')
    mx, my = map(int, low-manual_origin)
    ax, ay = map(int, low-automatic_origin)
    valid = ((manual_rgba[my:my+height, mx:mx+width, 3] > 0)
             & automatic_valid[ay:ay+height, ax:ax+width])
    left, top, right, bottom = largest_valid_rectangle(valid)
    manual_box = [mx+left, my+top, mx+right, my+bottom]
    automatic_box = [ax+left, ay+top, ax+right, ay+bottom]
    reference_box = [int(low[0]+left), int(low[1]+top), int(low[0]+right), int(low[1]+bottom)]
    return (manual_rgba[my+top:my+bottom, mx+left:mx+right, :3],
            automatic[ay+top:ay+bottom, ax+left:ax+right],
            reference_box, manual_box, automatic_box)


def process_pair(pair, threshold, max_trials, seed):
    with np.load(ROOT / pair['features']) as data:
        source, target = data['im1_pts'].copy(), data['im2_pts'].copy()
    model = ransac_homography(source, target, threshold_px=threshold, max_iterations=max_trials,
                              min_iterations=min(200, max_trials), seed=seed)
    mask, errors, H = model['inliers'], model['errors'], model['H']
    output = ROOT / 'output' / 'b4' / 'pairs' / pair['id']
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output / 'ransac.npz', H=H, im1_pts=source, im2_pts=target,
                        inliers=mask, errors=errors, fit_mask=model['fit_mask'])
    np.savetxt(output / 'H.csv', H, delimiter=',', fmt='%.17g')
    with (output / 'point_errors.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.writer(stream)
        writer.writerow(['b3_match_id', 'x1', 'y1', 'x2', 'y2', 'forward_error_px', 'inlier', 'used_for_final_fit'])
        for i, (a, b, error) in enumerate(zip(source, target, errors)):
            writer.writerow([i, *a.tolist(), *b.tolist(), error, int(mask[i]), int(model['fit_mask'][i])])
    with Image.open(ROOT / pair['image1']) as image:
        image1 = ImageOps.exif_transpose(image).convert('RGB')
    with Image.open(ROOT / pair['image2']) as image:
        image2 = ImageOps.exif_transpose(image).convert('RGB')
    labels = [Path(pair[key]).stem for key in ('image1', 'image2')]
    size = draw_matches(image1, image2, source[mask], target[mask], output / 'inliers.jpg',
                        f'Image {labels[0]} - RANSAC inliers', f'Image {labels[1]} - {mask.sum()} / {len(mask)}')
    draw_matches(image1, image2, source[~mask], target[~mask], output / 'rejected.jpg',
                 f'Image {labels[0]} - rejected', f'Image {labels[1]} - {(~mask).sum()} / {len(mask)}')
    relative = output.relative_to(ROOT).as_posix()
    result = {
        'id': pair['id'], 'title': pair['title'], 'image1': pair['image1'], 'image2': pair['image2'],
        'b3_matches': pair['features'], 'b3_ratio_threshold': pair['ratio_threshold'],
        'direction': 'image1 -> image2', 'H': H.tolist(), 'match_count': len(source),
        'inlier_count': int(mask.sum()), 'inlier_percent': 100*float(mask.mean()),
        'inlier_rmse_px': float(np.sqrt(np.mean(errors[mask]**2))),
        'inlier_max_error_px': float(errors[mask].max()),
        'threshold_px': threshold, 'seed': seed, 'max_trials': max_trials,
        'trials': model['trials'], 'valid_hypotheses': model['valid_hypotheses'],
        'skipped_samples': model['skipped_samples'], 'refit_steps': model['refit_steps'],
        'consensus_stable': model['consensus_stable'],
        'inlier_match_indices': np.flatnonzero(mask).tolist(),
        'fit_match_indices': np.flatnonzero(model['fit_mask']).tolist(),
        'inliers_image': f'{relative}/inliers.jpg', 'rejected_image': f'{relative}/rejected.jpg',
        'figure_size': size, 'features': f'{relative}/ransac.npz',
        'errors_csv': f'{relative}/point_errors.csv', 'record_path': f'{relative}/ransac.json',
    }
    (output / 'ransac.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    print(f'{pair["id"]}: {mask.sum()}/{len(mask)} inliers, RMSE {result["inlier_rmse_px"]:.3f} px', flush=True)
    return result


def process_mosaic(manual, pairs):
    paths, reference = manual['sources'], manual['reference_image']
    edges = {(p['image1'], p['image2']): np.array(p['H']) for p in pairs}
    transforms = transforms_to_reference(paths, reference, edges)
    images = []
    for path in paths:
        with Image.open(ROOT / path) as source:
            images.append(np.array(ImageOps.exif_transpose(source).convert('RGB')))
    feather, overwrite, coverage, origin, laplacian = stitch_mosaic(images, transforms, pyramid_levels=PYRAMID_LEVELS)
    del overwrite
    output = ROOT / 'output' / 'b4' / 'mosaics' / manual['id']
    output.mkdir(parents=True, exist_ok=True)
    alpha = (coverage > 0).astype(np.uint8)*255
    for name, pixels in (('automatic_laplacian', laplacian), ('automatic_feather', feather)):
        Image.fromarray(np.dstack((pixels, alpha))).save(output / f'{name}.png')
    Image.fromarray(alpha).save(output / 'valid_mask.png')
    Image.fromarray(coverage).save(output / 'coverage.png')
    with Image.open(ROOT / manual['laplacian']) as image:
        manual_rgba = np.array(image.convert('RGBA'))
    manual_crop, automatic_crop, reference_box, manual_box, automatic_box = common_crop(
        manual_rgba, manual['reference_origin_xy'], laplacian, coverage > 0, origin)
    Image.fromarray(manual_crop).save(output / 'manual_comparison.png')
    Image.fromarray(automatic_crop).save(output / 'automatic_comparison.png')
    crop_height, crop_width = automatic_crop.shape[:2]
    # A fixed central/lower region in the shared crop; both details use the
    # exact same reference coordinates and contain no image resizing.
    left, top, right, bottom = int(crop_width*0.42), int(crop_height*0.45), int(crop_width*0.72), int(crop_height*0.85)
    for name, pixels in (('manual', manual_crop), ('automatic', automatic_crop)):
        Image.fromarray(pixels[top:bottom, left:right]).save(output / f'{name}_detail.png')
    relative = output.relative_to(ROOT).as_posix()
    shift = np.array([[1.,0.,-origin[0]], [0.,1.,-origin[1]], [0.,0.,1.]])
    result = {
        'id': manual['id'], 'title': manual['title'], 'sources': paths, 'reference_image': reference,
        'automatic_source_to_reference_H': [H.tolist() for H in transforms],
        'automatic_source_to_canvas_H': [(shift@H).tolist() for H in transforms],
        'automatic_origin_xy': list(origin), 'automatic_size': list(laplacian.shape[1::-1]),
        'manual_record': manual['record_path'], 'manual_full': manual['laplacian'],
        'automatic_full': f'{relative}/automatic_laplacian.png',
        'automatic_feather': f'{relative}/automatic_feather.png',
        'manual_comparison': f'{relative}/manual_comparison.png',
        'automatic_comparison': f'{relative}/automatic_comparison.png',
        'comparison_size': [crop_width,crop_height],
        'common_reference_box_ltrb': reference_box,
        'manual_crop_box_ltrb': manual_box, 'automatic_crop_box_ltrb': automatic_box,
        'detail_box_in_common_crop': [left,top,right,bottom],
        'manual_detail': f'{relative}/manual_detail.png', 'automatic_detail': f'{relative}/automatic_detail.png',
        'valid_mask': f'{relative}/valid_mask.png', 'coverage': f'{relative}/coverage.png',
        'record_path': f'{relative}/mosaic.json',
        'pair_ids': [p['id'] for p in pairs if p['image1'] in paths and p['image2'] in paths],
    }
    (output / 'mosaic.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    print(f'{manual["id"]}: automatic {result["automatic_size"]}, shared crop {result["comparison_size"]}', flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--threshold-px', type=float, default=3.0)
    parser.add_argument('--max-trials', type=int, default=5000)
    parser.add_argument('--seed', type=int, default=671)
    args = parser.parse_args()
    if not np.isfinite(args.threshold_px) or args.threshold_px <= 0 or args.max_trials < 1:
        parser.error('Use a positive --threshold-px and --max-trials >= 1')
    b3_path, a4_path = ROOT/'output/b3/summary.json', ROOT/'output/a4/summary.json'
    if not b3_path.exists() or not a4_path.exists():
        raise ValueError('Run B.3 and A.4 before B.4.')
    matches = json.loads(b3_path.read_text(encoding='utf-8'))
    manual = json.loads(a4_path.read_text(encoding='utf-8'))
    pairs = [process_pair(pair,args.threshold_px,args.max_trials,args.seed+i)
             for i,pair in enumerate(matches['examples'])]
    mosaics = [process_mosaic(example,pairs) for example in manual['examples']]
    if len(mosaics) < 3:
        raise ValueError('B.4 requires at least three automatic mosaics.')
    summary = {
        'method': 'Four-point RANSAC implemented from scratch; all-consensus least-squares refit using computeH',
        'parameters': {'threshold_px':args.threshold_px, 'max_trials':args.max_trials,
                       'min_valid_hypotheses':min(200,args.max_trials), 'confidence':0.999,
                       'seed_base':args.seed, 'pyramid_levels':PYRAMID_LEVELS},
        'inputs': 'Only B.3 automatic matches; B.3 ratio remains unchanged; manual correspondences are not used for automatic H.',
        'consensus': 'Original-resolution forward Euclidean reprojection error < threshold; degenerate four-point samples skipped; final mask recomputed after refit.',
        'comparison': 'Same original photos, reference view, manual bilinear warping and six-level Laplacian blending. Display crop is the largest rectangle valid in both results at identical reference coordinates.',
        'rmse_note': 'RMSE is calculated only on automatically selected inliers; it is not an independent alignment accuracy measurement.',
        'pairs':pairs, 'mosaics':mosaics,
    }
    output = ROOT/'output/b4'
    (output/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    with (output/'summary.csv').open('w',newline='',encoding='utf-8') as stream:
        writer=csv.writer(stream)
        writer.writerow(['pair','matches','inliers','inlier_percent','inlier_rmse_px','threshold_px','trials','seed'])
        for result in pairs:
            writer.writerow([result['title'],result['match_count'],result['inlier_count'],result['inlier_percent'],
                             result['inlier_rmse_px'],args.threshold_px,result['trials'],result['seed']])
    from build_showcase import export_showcase_data
    export_showcase_data()
    print(f'B.4 results: {output}')


if __name__ == '__main__':
    main()
