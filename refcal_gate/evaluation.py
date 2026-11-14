"""Run dataset fitting, evaluation, cloud controls, and margin-based batch routing."""

from dataclasses import asdict
import json
import logging
from pathlib import Path
import time

from .dataset import read_dataset
from .gate import RefCalGate
from .metrics import auroc, binary_metrics, control_summary
from .routing import cascade_frontier, select_cloud


def write_json(path, value):
    """Write indented UTF-8 JSON, creating parents and rejecting nonfinite numbers."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def start_logging(directory):
    """Configure console and run.log handlers, replacing existing root handlers."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        handlers=[logging.FileHandler(directory / "run.log", encoding="utf-8"),
                                  logging.StreamHandler()], force=True)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)


def fit_gate(config, output):
    """Save an NPZ from configured references and return (fitted gate, dataset)."""
    dataset = read_dataset(config.data, include_queries=False)
    gate = RefCalGate(config.encoder, config.gate).fit(dataset.references, dataset.root)
    gate.save(output)
    logging.info("Saved gate categories=%d artifact=%s", len(dataset.references), output)
    return gate, dataset


def evaluate_gate(config, output, gate_path=None):
    """Write edge.json and return its metrics and per-query decision dictionary.

    An existing NPZ supplies settings and the reference IDs excluded from queries;
    otherwise fit and save gate.npz. The CLI creates the output directory.
    """
    output = Path(output)
    if gate_path:
        gate = RefCalGate.load(gate_path, config.encoder.device)
        dataset = read_dataset(config.data, reference_ids=gate.reference_ids)
    else:
        dataset = read_dataset(config.data)
        gate = RefCalGate(config.encoder, config.gate).fit(dataset.references, dataset.root)
        gate.save(output / "gate.npz")
    gate.encoder  # Load model weights before timing individual image predictions.
    rows = []
    for index, sample in enumerate(dataset.samples, 1):
        start = time.perf_counter()
        decision = gate.predict_path(sample.path, sample.category, dataset.root)
        elapsed = (time.perf_counter() - start) * 1000
        rows.append({"sample_id": sample.sample_id, "ground_truth": sample.label,
                     "latency_ms": elapsed, **decision})
        if index % 25 == 0 or index == len(dataset.samples):
            logging.info("Scored images=%d/%d", index, len(dataset.samples))
    truth = [row["ground_truth"] for row in rows]
    metrics = binary_metrics(truth, [row["predicted"] for row in rows])
    metrics["auroc"] = auroc(truth, [row["score"] for row in rows])
    per_category = {}
    for category in sorted(gate.memory.references):
        items = [row for row in rows if row["category"] == category]
        if items:
            actual = [row["ground_truth"] for row in items]
            per_category[category] = {**binary_metrics(actual, [row["predicted"] for row in items]),
                                      "auroc": auroc(actual, [row["score"] for row in items])}
    result = {"encoder": asdict(gate.encoder_config), "gate": asdict(gate.gate_config),
              "reference_ids": gate.reference_ids, "thresholds": gate.memory.thresholds,
              "reference_scores": gate.memory.normal_scores, "metrics": metrics,
              "per_category": per_category, "rows": rows}
    write_json(output / "edge.json", result)
    logging.info("Gate evaluation complete images=%d accuracy=%.6f auroc=%s",
                 len(rows), metrics["accuracy"], metrics["auroc"])
    return result


def run_cloud_controls(config, output, controls, blur_radius=12.0, gate_path=None):
    """Return cloud rows and write cloud.jsonl in an existing output directory.

    controls contains names from CONTROLS. Query-only rows also trigger a
    controls.json summary. Reference ranking uses whole-image CLIP embeddings.
    gate_path supplies the fitted reference IDs for reference selection and exclusion.
    """
    from PIL import Image
    from .cloud import CloudInspector
    from .controls import ReferenceControls
    from .encoders import FrozenGridEncoder

    gate = RefCalGate.load(gate_path, config.encoder.device) if gate_path else None
    dataset = read_dataset(config.data, reference_ids=gate.reference_ids if gate else None)
    inspector = CloudInspector(config.cloud)
    reference_controls = None
    if any(control != "query_only" for control in controls):
        retrieval_config = type(config.encoder)(model=config.encoder.retrieval_model,
                                               device=config.encoder.device,
                                               batch_size=config.encoder.batch_size)
        reference_controls = ReferenceControls(FrozenGridEncoder(retrieval_config),
                                               dataset.references, blur_radius)
    output = Path(output)
    rows = []
    with (output / "cloud.jsonl").open("w", encoding="utf-8") as stream:
        for index, sample in enumerate(dataset.samples, 1):
            with Image.open(sample.path) as source:
                query = source.convert("RGB")
            ranked = reference_controls.rank(query, sample.category) if reference_controls else []
            for control in controls:
                reference, reference_path = (reference_controls.select(sample, ranked, control)
                                             if reference_controls else (None, None))
                result = inspector.predict(query, reference)
                row = {"sample_id": sample.sample_id, "category": sample.category,
                       "ground_truth": sample.label, "control": control,
                       "reference_id": reference_path.relative_to(dataset.root).as_posix() if reference_path else None,
                       **result}
                stream.write(json.dumps(row, allow_nan=False) + "\n")
                stream.flush()
                rows.append(row)
                logging.info("Cloud images=%d/%d control=%s response_model=%s",
                             index, len(dataset.samples), control, result["response_model"])
    if "query_only" in controls:
        write_json(output / "controls.json", control_summary(rows))
    logging.info("Cloud evaluation complete rows=%d", len(rows))
    return rows


def read_cloud_rows(path):
    """Return JSON objects from nonblank lines of a cloud prediction file."""
    with Path(path).open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def build_frontier(edge_path, cloud_path, output, repetitions=2000, seed=0):
    """Join edge JSON and query-only cloud JSONL, write a frontier, and return it."""
    edge = json.loads(Path(edge_path).read_text(encoding="utf-8"))
    cloud = [row for row in read_cloud_rows(cloud_path) if row["control"] == "query_only"]
    result = cascade_frontier(edge["rows"], cloud, repetitions, seed)
    write_json(output, result)
    logging.info("Cascade frontier complete images=%d cloud_to_match=%.6f",
                 result["n"], result["cloud_to_match"])
    return result


def route_dataset(config, gate_path, output, cloud_fraction=None):
    """Write routed.json and return edge/cloud decisions for configured queries.

    Load the gate and exclude its saved reference IDs before sampling queries.
    Send floor(batch size * fraction) queries
    with smallest edge margins to query-only cloud inspection. A None fraction
    uses the routing configuration; final labels replace selected edge labels.
    """
    from PIL import Image

    gate = RefCalGate.load(gate_path, config.encoder.device)
    dataset = read_dataset(config.data, reference_ids=gate.reference_ids)
    fraction = config.routing.cloud_fraction if cloud_fraction is None else cloud_fraction
    rows = [{"sample_id": sample.sample_id, "ground_truth": sample.label,
             **gate.predict_path(sample.path, sample.category, dataset.root)} for sample in dataset.samples]
    selected = select_cloud(rows, fraction)
    inspector = None
    if selected:
        from .cloud import CloudInspector

        inspector = CloudInspector(config.cloud)
    for index, (sample, row) in enumerate(zip(dataset.samples, rows)):
        row["edge_predicted"] = row["predicted"]
        row["source"] = "cloud" if index in selected else "edge"
        if index in selected:
            with Image.open(sample.path) as image:
                result = inspector.predict(image.convert("RGB"))
            row["predicted"] = result["predicted"]
            row["cloud"] = result
        if (index + 1) % 25 == 0 or index + 1 == len(rows):
            logging.info("Routed images=%d/%d", index + 1, len(rows))
    result = {"cloud_calls": len(selected), "cloud_rate": len(selected) / len(rows),
              "metrics": binary_metrics([row["ground_truth"] for row in rows],
                                        [row["predicted"] for row in rows]), "rows": rows}
    write_json(Path(output) / "routed.json", result)
    logging.info("Batch routing complete images=%d cloud_calls=%d", len(rows), len(selected))
    return result
