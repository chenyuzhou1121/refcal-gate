"""Compute binary inspection metrics, AUROC, bootstrap intervals, and control summaries."""

from collections import Counter, defaultdict
import math
import random


LABELS = ("normal", "warning", "alarm")
SEVERITY = {label: index for index, label in enumerate(LABELS)}


def binary_metrics(truth, predicted):
    """Return metrics for nonempty paired normal/warning/alarm label sequences.

    warning and alarm are both positive for binary metrics. exact_accuracy
    compares all three labels. Undefined class rates are None; precision and F1
    use zero when their denominators vanish.
    """
    if not truth or len(truth) != len(predicted):
        raise ValueError("Metrics require nonempty paired labels")
    if any(label not in LABELS for label in list(truth) + list(predicted)):
        raise ValueError("Labels must be normal, warning or alarm")
    tp = tn = fp = fn = 0
    for actual, decision in zip(truth, predicted):
        positive, alarm = actual != "normal", decision != "normal"
        tp += int(positive and alarm)
        tn += int(not positive and not alarm)
        fp += int(not positive and alarm)
        fn += int(positive and not alarm)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else None
    return {"n": len(truth), "accuracy": (tp + tn) / len(truth),
            "exact_accuracy": sum(a == b for a, b in zip(truth, predicted)) / len(truth),
            "false_alarm_rate": fp / (fp + tn) if fp + tn else None,
            "miss_rate": fn / (tp + fn) if tp + fn else None,
            "precision": precision, "recall": recall,
            "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
            "tp": tp, "tn": tn, "fp": fp, "fn": fn}


def auroc(truth, scores):
    """Return binary AUROC with average ranks for ties, or None for one-class truth.

    Scores must be finite and paired with normal/warning/alarm labels. Larger
    scores indicate anomalies; warning and alarm both count as positive.
    """
    if not truth or len(truth) != len(scores):
        raise ValueError("AUROC requires nonempty paired labels and scores")
    if any(label not in LABELS for label in truth) or not all(math.isfinite(s) for s in scores):
        raise ValueError("Invalid AUROC labels or scores")
    ordered = sorted(zip(scores, [int(label != "normal") for label in truth]))
    positive = sum(label for _, label in ordered)
    negative = len(ordered) - positive
    if not positive or not negative:
        return None
    rank_sum, start = 0.0, 0
    while start < len(ordered):
        end = start + 1
        while end < len(ordered) and ordered[end][0] == ordered[start][0]:
            end += 1
        # All equal scores receive the mean of their one-based rank positions.
        rank_sum += (start + 1 + end) / 2 * sum(label for _, label in ordered[start:end])
        start = end
    return (rank_sum - positive * (positive + 1) / 2) / (positive * negative)


def bootstrap_accuracy(correct, repetitions=2000, seed=0):
    """Return a 95% percentile interval from seeded resampling of 0/1 correctness."""
    if not correct or repetitions < 1:
        raise ValueError("Bootstrap requires nonempty samples and positive repetitions")
    rng = random.Random(seed)
    values = sorted(sum(rng.choices(correct, k=len(correct))) / len(correct)
                    for _ in range(repetitions))
    return [values[max(0, math.ceil(0.025 * repetitions) - 1)],
            values[max(0, math.ceil(0.975 * repetitions) - 1)]]


def control_summary(rows):
    """Return metrics by control for paired query rows including query_only.

    Each control must cover the same unique sample IDs and ground-truth labels.
    severity_shift averages label-rank changes from each query-only prediction,
    using normal=0, warning=1, alarm=2. Latencies are averaged in milliseconds.
    """
    grouped = defaultdict(list)
    seen = set()
    for row in rows:
        key = row["control"], row["sample_id"]
        if key in seen:
            raise ValueError("Duplicate sample within a reference control")
        seen.add(key)
        grouped[row["control"]].append(row)
    query = {row["sample_id"]: row for row in grouped.get("query_only", [])}
    if not query:
        raise ValueError("Reference comparison requires query-only predictions")
    result = {}
    for control, items in grouped.items():
        if {row["sample_id"] for row in items} != set(query):
            raise ValueError("Reference controls must cover the same query samples")
        if any(row["ground_truth"] != query[row["sample_id"]]["ground_truth"] for row in items):
            raise ValueError("Reference controls disagree on ground-truth labels")
        metrics = binary_metrics([row["ground_truth"] for row in items],
                                 [row["predicted"] for row in items])
        metrics["prediction_counts"] = dict(Counter(row["predicted"] for row in items))
        metrics["severity_shift"] = sum(SEVERITY[row["predicted"]] -
                                        SEVERITY[query[row["sample_id"]]["predicted"]]
                                        for row in items) / len(items)
        metrics["mean_latency_ms"] = sum(row["latency_ms"] for row in items) / len(items)
        result[control] = metrics
    return result
