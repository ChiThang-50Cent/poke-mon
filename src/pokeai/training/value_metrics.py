"""Binary value calibration and train-only rating references, using NumPy only.

``turn_buckets`` are historical report names: their boundaries refer to the
zero-based decision index within each trajectory, NOT simulator turns.
UniversalState does not expose a simulator turn counter. Forced switches and
other decisions therefore advance this index just like ordinary decisions.
"""

import numpy as np


def _vector(values, name: str) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64)
    if result.ndim != 1 or not np.isfinite(result).all():
        raise ValueError(f"{name} must be a finite one-dimensional array")
    return result


def _inputs(probabilities, labels) -> tuple[np.ndarray, np.ndarray]:
    probabilities = _vector(probabilities, "probabilities")
    labels = _vector(labels, "labels")
    if probabilities.shape != labels.shape:
        raise ValueError("probabilities and labels must have the same shape")
    if ((probabilities < 0) | (probabilities > 1)).any():
        raise ValueError("probabilities must be in [0, 1]")
    if ((labels != 0) & (labels != 1)).any():
        raise ValueError("labels must be binary (0 or 1)")
    return probabilities, labels


def _roc_auc(probabilities: np.ndarray, labels: np.ndarray) -> float | None:
    positives = int(labels.sum())
    negatives = labels.size - positives
    if not positives or not negatives:
        return None
    order = np.argsort(probabilities, kind="stable")
    sorted_probabilities = probabilities[order]
    starts = np.r_[0, np.flatnonzero(np.diff(sorted_probabilities)) + 1]
    ends = np.r_[starts[1:], labels.size]
    # All tied observations receive the average of their one-based ranks.
    ranks = (starts + ends + 1) / 2
    positive_counts = np.add.reduceat(labels[order], starts)
    rank_sum = float(np.dot(ranks, positive_counts))
    return (rank_sum - positives * (positives + 1) / 2) / (positives * negatives)


def value_metrics(probabilities, labels) -> dict:
    """Return decision-weighted binary metrics and ten uniform reliability bins.

    Bins are [lower, upper), except the final bin includes probability 1.
    Accuracy uses P(win) >= 0.5. BCE clips probabilities to [1e-15, 1-1e-15]
    solely for logarithms. Empty inputs yield count=0 and undefined metrics
    (including empty-bin means) as None, making reports JSON serializable.
    ROC AUC uses average ranks for ties and is None for a one-class sample.
    """
    probabilities, labels = _inputs(probabilities, labels)
    count = probabilities.size
    bins = np.minimum((probabilities * 10).astype(np.int64), 9)
    counts = np.bincount(bins, minlength=10)
    sums = np.bincount(bins, weights=probabilities, minlength=10)
    observed = np.bincount(bins, weights=labels, minlength=10)
    reliability = [
        {
            "lower": index / 10,
            "upper": (index + 1) / 10,
            "count": int(counts[index]),
            "mean_probability": float(sums[index] / counts[index])
            if counts[index] else None,
            "observed_rate": float(observed[index] / counts[index])
            if counts[index] else None,
        }
        for index in range(10)
    ]
    result = dict.fromkeys(
        ("bce", "brier", "roc_auc", "accuracy", "mean_probability", "base_rate", "ece")
    )
    result.update(count=int(count), reliability=reliability)
    if not count:
        return result
    clipped = np.clip(probabilities, 1e-15, 1 - 1e-15)
    result.update(
        bce=float(-np.mean(labels * np.log(clipped)
                           + (1 - labels) * np.log1p(-clipped))),
        brier=float(np.mean((probabilities - labels) ** 2)),
        roc_auc=_roc_auc(probabilities, labels),
        accuracy=float(np.mean((probabilities >= 0.5) == labels)),
        mean_probability=float(probabilities.mean()),
        base_rate=float(labels.mean()),
        ece=float(np.abs(sums - observed).sum() / count),
    )
    return result


def trajectory_metrics(probabilities, labels, offsets) -> dict:
    """Add decision-index buckets and one first/final decision per trajectory.

    ``offsets`` is a nondecreasing integer array starting at zero and ending at
    the flat sample count. Empty trajectories are ignored for first/final
    metrics. ``turn_buckets`` uses zero-based decision indices 0–5, 6–15,
    16–30, and 31+; these are not simulator turn numbers.
    """
    probabilities, labels = _inputs(probabilities, labels)
    offsets = np.asarray(offsets)
    if (offsets.ndim != 1 or offsets.size == 0
            or not np.issubdtype(offsets.dtype, np.integer)
            or offsets[0] != 0 or offsets[-1] != probabilities.size
            or (np.diff(offsets) < 0).any()):
        raise ValueError("offsets must be nondecreasing integers spanning all samples")
    lengths = np.diff(offsets)
    indices = np.arange(probabilities.size) - np.repeat(offsets[:-1], lengths)
    result = value_metrics(probabilities, labels)
    result["turn_buckets"] = {}
    for name, lower, upper in (
        ("0-5", 0, 6), ("6-15", 6, 16), ("16-30", 16, 31),
        ("31+", 31, None),
    ):
        mask = indices >= lower
        if upper is not None:
            mask &= indices < upper
        result["turn_buckets"][name] = value_metrics(probabilities[mask], labels[mask])
    nonempty = lengths > 0
    for name, selected in (
        ("first_step", offsets[:-1][nonempty]),
        ("final_step", offsets[1:][nonempty] - 1),
    ):
        result[name] = value_metrics(probabilities[selected], labels[selected])
    return result


def _sigmoid(logits: np.ndarray) -> np.ndarray:
    exponential = np.exp(-np.abs(logits))
    return np.where(logits >= 0, 1 / (1 + exponential),
                    exponential / (1 + exponential))


def rating_reference(train_ratings, train_outcomes, val_ratings) -> tuple:
    """Fit a deterministic rating-only logistic reference on train trajectories.

    Inputs contain one scalar rating/outcome per trajectory; no validation
    outcomes are used. Ratings are standardized using training mean/std only.
    Newton IRLS minimizes summed logistic loss plus 0.5 * 1e-3 * ||beta||²,
    regularizing BOTH intercept and slope (finite even for one-class data).
    Returns (validation trajectory probabilities, JSON-serializable metadata).
    Metadata coefficients apply to standardized ratings, not raw rating units.
    """
    train_ratings = _vector(train_ratings, "train_ratings")
    train_outcomes = _vector(train_outcomes, "train_outcomes")
    val_ratings = _vector(val_ratings, "val_ratings")
    if not train_ratings.size or train_ratings.shape != train_outcomes.shape:
        raise ValueError("training ratings/outcomes must be nonempty and aligned")
    if ((train_outcomes != 0) & (train_outcomes != 1)).any():
        raise ValueError("train_outcomes must be binary (0 or 1)")
    mean = float(train_ratings.mean())
    scale = float(train_ratings.std()) or 1.0
    design = np.column_stack((np.ones(train_ratings.size),
                              (train_ratings - mean) / scale))
    coefficients = np.zeros(2)
    regularization = 1e-3
    max_iterations = 100
    tolerance = 1e-8

    def objective(beta):
        logits = design @ beta
        return float(np.sum(np.logaddexp(0, logits) - train_outcomes * logits)
                     + 0.5 * regularization * np.dot(beta, beta))

    converged = False
    for iteration in range(1, max_iterations + 1):
        fitted = _sigmoid(design @ coefficients)
        gradient = design.T @ (fitted - train_outcomes)
        gradient += regularization * coefficients
        if np.max(np.abs(gradient)) <= tolerance:
            converged = True
            break
        weights = fitted * (1 - fitted)
        hessian = design.T @ (weights[:, None] * design)
        hessian += regularization * np.eye(2)
        step = np.linalg.solve(hessian, gradient)
        # Backtracking keeps Newton stable on imbalanced or separable samples.
        fraction = 1.0
        current = objective(coefficients)
        decrease = float(np.dot(gradient, step))
        while fraction > 2 ** -30:
            candidate = coefficients - fraction * step
            if objective(candidate) <= current - 1e-4 * fraction * decrease:
                break
            fraction *= 0.5
        coefficients -= fraction * step
        if np.max(np.abs(fraction * step)) <= tolerance:
            converged = True
            break
    probabilities = _sigmoid(
        coefficients[0] + coefficients[1] * (val_ratings - mean) / scale
    )
    metadata = {
        "train_count": int(train_ratings.size),
        "train_base_rate": float(train_outcomes.mean()),
        "rating_mean": mean,
        "rating_scale": scale,
        "intercept": float(coefficients[0]),
        "slope": float(coefficients[1]),
        "coefficient_space": "standardized_rating",
        "regularization": regularization,
        "regularized_parameters": ["intercept", "slope"],
        "iterations": iteration,
        "converged": converged,
        "objective": objective(coefficients),
    }
    return probabilities, metadata
