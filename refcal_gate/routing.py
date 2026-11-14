"""Select queries by edge margins and compute cascades from paired predictions."""

import math

from .metrics import binary_metrics, bootstrap_accuracy


def escalation_order(decisions):
    """Return indices sorted by abs(score - threshold), preserving ties in input order."""
    for decision in decisions:
        if not all(math.isfinite(decision[key]) for key in ("score", "threshold")):
            raise ValueError("Routing requires finite scores and reference thresholds")
    return sorted(range(len(decisions)),
                  key=lambda index: abs(decisions[index]["score"] - decisions[index]["threshold"]))


def select_cloud(decisions, cloud_fraction):
    """Return the lowest-margin index set of size floor(N * fraction), fraction in [0, 1]."""
    if not 0 <= cloud_fraction <= 1:
        raise ValueError("Cloud fraction must be in [0, 1]")
    count = math.floor(len(decisions) * cloud_fraction)
    return set(escalation_order(decisions)[:count])


def cascade_frontier(edge_rows, cloud_rows, bootstrap_repetitions=2000, seed=0):
    """Return metrics for every prefix of margin-ordered query-only escalation.

    Rows must have unique matching sample IDs and identical ground-truth labels.
    cloud_to_match is the first cloud fraction reaching cloud-only accuracy.
    The half-budget point uses floor(N / 2) calls and an optional bootstrap
    accuracy interval; zero repetitions omits the interval.
    """
    edge = {row["sample_id"]: row for row in edge_rows}
    cloud = {row["sample_id"]: row for row in cloud_rows}
    if not edge or len(edge) != len(edge_rows) or len(cloud) != len(cloud_rows):
        raise ValueError("Cascade inputs must be nonempty and have unique sample IDs")
    if set(edge) != set(cloud):
        raise ValueError("Edge and cloud files must cover exactly the same relative sample IDs")
    for key in edge:
        if edge[key]["ground_truth"] != cloud[key]["ground_truth"]:
            raise ValueError(f"Ground-truth mismatch for sample: {key}")
        if cloud[key].get("control") != "query_only":
            raise ValueError("Cascade cloud predictions must use query_only")
    keys = list(edge)
    decisions = [edge[key] for key in keys]
    truth = [edge[key]["ground_truth"] for key in keys]
    predicted = [edge[key]["predicted"] for key in keys]
    cloud_predicted = [cloud[key]["predicted"] for key in keys]
    edge_metrics = binary_metrics(truth, predicted)
    cloud_metrics = binary_metrics(truth, cloud_predicted)
    counts = {key: edge_metrics[key] for key in ("tp", "tn", "fp", "fn")}
    curve = []

    def point(calls):
        tp, tn, fp, fn = (counts[key] for key in ("tp", "tn", "fp", "fn"))
        return {"cloud_calls": calls, "cloud_rate": calls / len(keys),
                "accuracy": (tp + tn) / len(keys),
                "false_alarm_rate": fp / (fp + tn) if fp + tn else None,
                "miss_rate": fn / (tp + fn) if tp + fn else None, **counts}

    def bucket(actual, decision):
        if actual == "normal":
            return "tn" if decision == "normal" else "fp"
        return "fn" if decision == "normal" else "tp"

    order = escalation_order(decisions)
    curve.append(point(0))
    for calls, index in enumerate(order, 1):
        # Replace one edge decision at a time without recomputing all metrics.
        counts[bucket(truth[index], predicted[index])] -= 1
        counts[bucket(truth[index], cloud_predicted[index])] += 1
        curve.append(point(calls))
    match = next(row for row in curve if row["accuracy"] >= cloud_metrics["accuracy"] - 1e-9)
    half_count = math.floor(len(keys) * 0.5)
    half_indices = set(order[:half_count])
    correct = [int((cloud_predicted[index] != "normal" if index in half_indices
                    else predicted[index] != "normal") == (truth[index] != "normal"))
               for index in range(len(keys))]
    interval = bootstrap_accuracy(correct, bootstrap_repetitions, seed) if bootstrap_repetitions else None
    return {"n": len(keys), "edge_only": edge_metrics, "cloud_only": cloud_metrics,
            "cloud_to_match": match["cloud_rate"], "cloud_saving": 1 - match["cloud_rate"],
            "half_budget": {**curve[half_count], "accuracy_ci95": interval}, "curve": curve}
