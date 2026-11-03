"""B.4: four-point RANSAC for a source-to-target homography, from scratch."""

from itertools import combinations

import numpy as np

from homography import computeH


def reprojection_errors(source, target, H):
    """Forward Euclidean errors in target pixels; points at infinity get inf."""
    H = np.asarray(H, dtype=float)
    H = H / np.max(np.abs(H))
    mapped = np.column_stack((source, np.ones(len(source)))) @ H.T
    scale = np.max(np.abs(mapped), axis=1)
    valid = (np.isfinite(mapped).all(axis=1)
             & (np.abs(mapped[:, 2]) > 64*np.finfo(float).eps*np.maximum(scale, np.finfo(float).tiny)))
    errors = np.full(len(source), np.inf)
    projected = mapped[valid, :2] / mapped[valid, 2, None]
    errors[valid] = np.linalg.norm(projected-target[valid], axis=1)
    return errors


def _degenerate(points):
    """Reject duplicate/near-duplicate points and any nearly collinear triple."""
    span_squared = float(np.sum(np.ptp(points, axis=0)**2))
    if span_squared == 0:
        return True
    for a, b in combinations(points, 2):
        if np.sum((a-b)**2) < 1e-8*span_squared:
            return True
    for a, b, c in combinations(points, 3):
        u, v = b-a, c-a
        area_twice = abs(u[0]*v[1]-u[1]*v[0])
        if area_twice < 1e-4*span_squared:
            return True
    return False


def ransac_homography(source, target, *, threshold_px=3.0, max_iterations=5000,
                      min_iterations=200, confidence=0.999, seed=671):
    """Sample four pairs, score consensus, and least-squares-refit its inliers.

    Inliers satisfy ||project(H*p)-q|| < threshold_px in original target pixels.
    The seed makes the sampling reproducible. An adaptive confidence estimate
    reduces the valid-hypothesis budget, subject to a minimum and a draw cap.
    """
    source, target = np.asarray(source, dtype=float), np.asarray(target, dtype=float)
    if (source.ndim != 2 or source.shape[1] != 2 or source.shape != target.shape
            or len(source) < 4 or not np.isfinite(source).all() or not np.isfinite(target).all()):
        raise ValueError('Supply at least four finite Nx2 point pairs.')
    for points in (source, target):
        if len(np.unique(points, axis=0)) < 4 or np.linalg.matrix_rank(points-points.mean(axis=0)) < 2:
            raise ValueError('The correspondences cannot determine a nondegenerate homography.')
    if (not np.isfinite(threshold_px) or threshold_px <= 0 or not 0 < confidence < 1
            or int(max_iterations) != max_iterations or max_iterations < 1
            or int(min_iterations) != min_iterations or not 1 <= min_iterations <= max_iterations):
        raise ValueError('Use a positive threshold, valid confidence, and integer iteration limits.')
    rng = np.random.default_rng(seed)
    best_score, best_mask = (-1, -np.inf), None
    required, trials, hypotheses, skipped = max_iterations, 0, 0, 0
    while trials < max_iterations and hypotheses < max(min_iterations, required):
        trials += 1
        sample = rng.choice(len(source), 4, replace=False)
        if _degenerate(source[sample]) or _degenerate(target[sample]):
            skipped += 1
            continue
        try:
            H = computeH(source[sample], target[sample])
        except (ValueError, np.linalg.LinAlgError):
            skipped += 1
            continue
        hypotheses += 1
        errors = reprojection_errors(source, target, H)
        inliers = errors < threshold_px
        count = int(inliers.sum())
        score = (count, -float(np.sum(errors[inliers]**2)))
        if count >= 4 and score > best_score:
            best_score, best_mask = score, inliers
            fraction = count/len(source)
            success = fraction**4
            required = 1 if success >= 1 else int(np.ceil(np.log1p(-confidence)/np.log1p(-success)))
            required = min(max_iterations, required)
    if best_mask is None:
        raise ValueError('RANSAC found no valid four-point model; check the feature matches.')

    # Refit and reclassify until the consensus stabilizes or repeats. Every
    # retained final candidate is fitted to all points in its recorded fit set.
    mask, seen, refined = best_mask, set(), []
    stable = False
    for _ in range(20):
        signature = mask.tobytes()
        if signature in seen or mask.sum() < 4:
            break
        seen.add(signature)
        try:
            H = computeH(source[mask], target[mask])
        except (ValueError, np.linalg.LinAlgError):
            break
        errors = reprojection_errors(source, target, H)
        classified = errors < threshold_px
        count = int(classified.sum())
        if count >= 4:
            refined.append(((count, -float(np.sum(errors[classified]**2))), H, classified, errors, mask.copy()))
        if np.array_equal(classified, mask):
            stable = True
            break
        mask = classified
    if not refined:
        raise ValueError('The consensus could not be refitted reliably.')
    chosen = refined[-1] if stable else max(refined, key=lambda item: item[0])
    _, H, inliers, errors, fit_mask = chosen
    return {'H': H, 'inliers': inliers, 'errors': errors, 'fit_mask': fit_mask,
            'trials': trials, 'valid_hypotheses': hypotheses, 'skipped_samples': skipped,
            'refit_steps': len(refined), 'consensus_stable': stable,
            'threshold_px': threshold_px, 'confidence': confidence, 'seed': seed}
