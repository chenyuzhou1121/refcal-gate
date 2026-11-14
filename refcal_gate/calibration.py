"""Calibrate normal-reference patch distances and aggregate anomaly scores."""

from dataclasses import dataclass
import math
import numpy as np


SCORES = ("max_excess", "position_z", "local_covariance")


def discrete_quantile(values, quantile):
    """Return the value at one-based rank ceil(quantile * n), without interpolation."""
    if not 0 < quantile <= 1:
        raise ValueError("Quantile must be in (0, 1]")
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise ValueError("Quantile requires at least one value")
    return ordered[math.ceil(quantile * len(ordered)) - 1]


def patch_distances(query, references):
    """Return pooled and position-matched nearest-reference cosine distances.

    Unit-normalized query and reference arrays have shapes [P, D] and [R, P, D].
    Both returned arrays have shape [P]; pooled search spans every reference cell,
    while position-matched search compares only the same cell across images.
    """
    bank = references.reshape(-1, references.shape[-1])
    pooled = np.maximum(0.0, 1.0 - (query @ bank.T).max(axis=1))
    similarity = np.einsum("pd,rpd->rp", query, references)
    position = np.maximum(0.0, 1.0 - similarity.max(axis=0))
    return pooled, position


@dataclass
class Calibration:
    """Pooled offset and position/window statistics for a reference grid.

    position_mean and position_std have one value per cell. Each window lists
    four cell indices with a matching local mean vector and covariance matrix.
    """

    offset: float
    position_mean: np.ndarray
    position_std: np.ndarray
    windows: list[tuple[int, ...]]
    local_means: list[np.ndarray]
    local_covariances: list[np.ndarray]


def calibrate(references, grid_size):
    """Return Calibration for finite unit embeddings shaped [R, grid_size**2, D].

    grid_size is at least two and R is at least one.
    Multiple references use leave-one-image-out distances. A single reference
    uses other cells for pooled distances and adjacent cells for position stats.
    The offset is the empirical 0.95 quantile of pooled distances.
    """
    if references.ndim != 3 or references.shape[1] != grid_size * grid_size:
        raise ValueError("Reference embeddings must have shape [images, grid cells, features]")
    if len(references) < 1 or not np.isfinite(references).all():
        raise ValueError("Reference embeddings must be nonempty and finite")
    pooled, vectors = [], []
    if len(references) >= 2:
        for index, query in enumerate(references):
            remaining = np.delete(references, index, axis=0)
            global_distances, position = patch_distances(query, remaining)
            pooled.extend(global_distances.tolist())
            vectors.append(position)
        matrix = np.asarray(vectors, dtype=np.float32)
        position_mean = matrix.mean(axis=0)
        position_std = matrix.std(axis=0, ddof=0)
    else:
        # Fitting two references reaches this branch during leave-one-out scoring.
        # Exclude the cell itself and use Manhattan-adjacent cells for position stats.
        query = references[0]
        similarity = query @ query.T
        np.fill_diagonal(similarity, -1.0)
        pooled = np.maximum(0.0, 1.0 - similarity.max(axis=1)).tolist()
        position_mean, position_std, vector = [], [], []
        for index in range(len(query)):
            row, col = divmod(index, grid_size)
            neighbors = [other for other in range(len(query)) if other != index
                         and abs(other // grid_size - row) + abs(other % grid_size - col) <= 1]
            values = np.maximum(0.0, 1.0 - query[index] @ query[neighbors].T)
            position_mean.append(float(values.mean()))
            position_std.append(float(values.std(ddof=0)))
            vector.append(float(values.mean()))
        matrix = np.asarray([vector], dtype=np.float32)
        position_mean = np.asarray(position_mean, dtype=np.float32)
        position_std = np.asarray(position_std, dtype=np.float32)
    windows = [(row * grid_size + col, (row + 1) * grid_size + col,
                row * grid_size + col + 1, (row + 1) * grid_size + col + 1)
               for row in range(grid_size - 1) for col in range(grid_size - 1)]
    means, covariances = [], []
    for window in windows:
        local = matrix[:, window]
        mean = local.mean(axis=0)
        centered = local - mean
        covariance = centered.T @ centered / max(1, len(local) - 1)
        diagonal = np.diag(np.diag(covariance))
        # Shrink each 2x2-cell covariance toward its diagonal and add a ridge.
        shrunk = 0.75 * covariance + 0.25 * diagonal + 1e-4 * np.eye(len(window), dtype=np.float32)
        means.append(mean)
        covariances.append(shrunk)
    return Calibration(discrete_quantile(pooled, 0.95), position_mean, position_std,
                       windows, means, covariances)


def score_embedding(query, references, stats, score):
    """Return (nonnegative score, per-cell distances) for a calibrated query grid.

    max_excess subtracts the pooled offset from the largest pooled distance.
    position_z takes the largest positive standardized position distance.
    local_covariance takes the largest Mahalanobis distance across 2x2 windows.
    Inputs follow patch_distances shapes; stats must match the reference grid.
    """
    if score not in SCORES:
        raise ValueError(f"Unknown score: {score}")
    pooled, position = patch_distances(query, references)
    if score == "max_excess":
        value = max(0.0, float(pooled.max()) - stats.offset)
        distances = pooled
    elif score == "position_z":
        standardized = (position - stats.position_mean) / np.maximum(stats.position_std, 1e-6)
        value = max(0.0, float(standardized.max()))
        distances = position
    else:
        value = 0.0
        for window, mean, covariance in zip(stats.windows, stats.local_means, stats.local_covariances):
            delta = position[list(window)] - mean
            try:
                solved = np.linalg.solve(covariance, delta)
            except np.linalg.LinAlgError:
                # Retain a distance calculation if the covariance is singular.
                solved = np.linalg.pinv(covariance) @ delta
            value = max(value, math.sqrt(max(0.0, float(delta @ solved))))
        distances = position
    if not math.isfinite(value):
        raise ValueError("Nonfinite anomaly score")
    return value, [float(distance) for distance in distances]
