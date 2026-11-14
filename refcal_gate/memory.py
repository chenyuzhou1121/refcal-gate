"""Fit category reference memories and classify queries with empirical thresholds."""

import logging
import numpy as np

from .calibration import SCORES, calibrate, discrete_quantile, score_embedding


class ReferenceMemory:
    """Store grids and calibration for each category using a score and quantile.

    grid_size must be at least two and quantile must be in (0, 1]. Embeddings
    supplied to fit and predict are expected to be unit-normalized per cell.
    """

    def __init__(self, grid_size=3, score="max_excess", quantile=0.85):
        if score not in SCORES or grid_size < 2 or not 0 < quantile <= 1:
            raise ValueError("Invalid score, grid size or reference quantile")
        self.grid_size = grid_size
        self.score = score
        self.quantile = quantile
        self.references = {}
        self.statistics = {}
        self.normal_scores = {}
        self.thresholds = {}

    def fit(self, embeddings):
        """Fit from category -> [R, grid_size**2, D] arrays, R >= 2, and return self."""
        if not embeddings:
            raise ValueError("At least one category is required")
        self.references, self.statistics = {}, {}
        self.normal_scores, self.thresholds = {}, {}
        for category, values in embeddings.items():
            references = np.asarray(values, dtype=np.float32)
            if references.ndim != 3 or len(references) < 2:
                raise ValueError(f"At least two grid-encoded references are required: {category}")
            if references.shape[1] != self.grid_size ** 2 or not np.isfinite(references).all():
                raise ValueError(f"Invalid reference grid: {category}")
            scores = []
            # Each threshold sample is scored without its own reference image;
            # calibration is also recomputed from the remaining images.
            for index, query in enumerate(references):
                remaining = np.delete(references, index, axis=0)
                stats = calibrate(remaining, self.grid_size)
                value, _ = score_embedding(query, remaining, stats, self.score)
                scores.append(value)
            self.references[category] = references
            self.statistics[category] = calibrate(references, self.grid_size)
            self.normal_scores[category] = scores
            self.thresholds[category] = discrete_quantile(scores, self.quantile)
            logging.info("Calibrated category=%s references=%d threshold=%.6f",
                         category, len(references), self.thresholds[category])
        return self

    def predict(self, embedding, category):
        """Return a decision dict for a [grid_size**2, D] query in a fitted category.

        A score strictly above the category threshold is alarm; a tie is normal.
        margin is the absolute score-to-threshold distance used for routing.
        patch_distances is a row-major list of unaggregated cell distances.
        """
        if category not in self.references:
            raise ValueError(f"No normal reference memory for category: {category}")
        expected = self.references[category].shape[1:]
        if embedding.shape != expected or not np.isfinite(embedding).all():
            raise ValueError("Query embedding does not match the fitted reference memory")
        value, patches = score_embedding(embedding, self.references[category],
                                        self.statistics[category], self.score)
        threshold = self.thresholds[category]
        return {"category": category, "score": value, "threshold": threshold,
                "margin": abs(value - threshold),
                "predicted": "alarm" if value > threshold else "normal",
                "patch_distances": patches}
