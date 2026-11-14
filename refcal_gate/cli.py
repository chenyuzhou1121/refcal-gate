"""Define command-line arguments and dispatch data, gate, cloud, and API tasks."""

import argparse
import json
import logging
from pathlib import Path

from .config import load_config


def add_dataset_options(parser, include_queries=True):
    """Add dataset/device overrides, with query sampling options when requested."""
    parser.add_argument("--config", type=Path, required=True, help="TOML configuration file")
    parser.add_argument("--data-root", help="Image tree containing category/train/good and category/test/label")
    parser.add_argument("--categories", help="Comma-separated category directories")
    parser.add_argument("--references", type=int, help="Normal references per category")
    if include_queries:
        parser.add_argument("--per-category-limit", type=int, help="Maximum test images per category; zero selects all")
        parser.add_argument("--selection", help="JSON array of relative query IDs in evaluation order")
        parser.add_argument("--seed", type=int, help="Dataset sampling seed")
    parser.add_argument("--device", help="Encoder device, such as cpu, cuda:0 or auto")


def add_method_options(parser):
    """Add encoder-family, score, and reference-quantile overrides."""
    parser.add_argument("--backbone", choices=["clip", "dinov2"], help="Frozen image encoder family")
    parser.add_argument("--encoder-model", help="Public model identifier or local model directory")
    parser.add_argument("--score", choices=["max_excess", "position_z", "local_covariance"],
                        help="Reference-calibrated patch aggregation")
    parser.add_argument("--quantile", type=float, help="Normal-reference threshold quantile in (0, 1]")


def build_parser():
    """Return the argument parser for all supported subcommands."""
    parser = argparse.ArgumentParser(prog="refcal-gate",
                                     description="Reference-calibrated anomaly scoring and edge-cloud routing")
    commands = parser.add_subparsers(dest="command", required=True)
    fit = commands.add_parser("fit", help="Build reference memory and save a fitted gate")
    add_dataset_options(fit, include_queries=False)
    add_method_options(fit)
    fit.add_argument("--output", type=Path, required=True, help="Output fitted gate artifact")
    evaluate = commands.add_parser("evaluate", help="Score held-out images and report anomaly detection metrics")
    add_dataset_options(evaluate)
    add_method_options(evaluate)
    evaluate.add_argument("--gate", type=Path, help="Fitted gate supplying method settings and excluded reference IDs")
    evaluate.add_argument("--output", type=Path, required=True,
                          help="Directory for edge.json and run.log; also gate.npz when fitting a new gate")
    cloud = commands.add_parser("cloud", help="Run query-only or six-control VLM inspection")
    add_dataset_options(cloud)
    cloud.add_argument("--controls", default="query_only", help="Comma-separated controls or all")
    cloud.add_argument("--gate", type=Path, help="Use the gate's reference IDs for paired cloud evaluation")
    cloud.add_argument("--blur-radius", type=float, default=12.0, help="Gaussian blur radius for reference corruption")
    cloud.add_argument("--output", type=Path, required=True, help="Directory for cloud.jsonl and numeric summaries")
    frontier = commands.add_parser("frontier", help="Join edge and query-only predictions into a cascade frontier")
    frontier.add_argument("--edge", type=Path, required=True, help="edge.json from evaluate")
    frontier.add_argument("--cloud", type=Path, required=True, help="cloud.jsonl from cloud")
    frontier.add_argument("--output", type=Path, required=True, help="Output frontier JSON")
    frontier.add_argument("--bootstrap", type=int, default=2000, help="Accuracy bootstrap repetitions; zero disables")
    frontier.add_argument("--seed", type=int, default=0, help="Bootstrap random seed")
    summary = commands.add_parser("summarize-controls", help="Aggregate six reference controls into numeric metrics")
    summary.add_argument("--input", type=Path, required=True, help="cloud.jsonl containing paired controls")
    summary.add_argument("--output", type=Path, required=True, help="Output metrics JSON")
    route = commands.add_parser("route", help="Route a batch using only its edge margins and cloud budget")
    add_dataset_options(route)
    route.add_argument("--gate", type=Path, required=True, help="Fitted gate artifact")
    route.add_argument("--cloud-fraction", type=float, help="Fraction of batch sent to the query-only cloud VLM")
    route.add_argument("--output", type=Path, required=True, help="Directory for routed.json and run.log")
    predict = commands.add_parser("predict", help="Inspect one image with a fitted edge gate")
    predict.add_argument("--gate", type=Path, required=True, help="Fitted gate artifact")
    predict.add_argument("--image", type=Path, required=True, help="Query image")
    predict.add_argument("--data-root", type=Path, required=True, help="Image-tree root for reference sample ID exclusion")
    predict.add_argument("--category", required=True, help="Reference memory category")
    predict.add_argument("--device", default="auto", help="Encoder device")
    predict.add_argument("--output", type=Path, help="Optional output JSON; otherwise print to stdout")
    prepare = commands.add_parser("prepare", help="Convert dataset parquet shards into image trees")
    prepare.add_argument("--config", type=Path, required=True, help="TOML configuration file")
    prepare.add_argument("--categories", help="Comma-separated category directories")
    prepare.add_argument("--format", choices=["mvtec", "visa"], required=True, help="Parquet column schema")
    prepare.add_argument("--input-dir", type=Path, help="Local category.train.parquet and category.test.parquet shards")
    prepare.add_argument("--url-template", default="", help="HTTPS shard URL with {category} and {split} placeholders")
    prepare.add_argument("--output", type=Path, required=True, help="Output image-tree root")
    prepare.add_argument("--train-references", type=int, default=20, help="VisA train normals to extract per category")
    prepare.add_argument("--test-normal", type=int, default=10, help="VisA test normals to extract per category")
    prepare.add_argument("--test-anomaly", type=int, default=10, help="VisA test anomalies to extract per category")
    prepare.add_argument("--full-test", action="store_true", help="Extract every VisA test image")
    server = commands.add_parser("serve", help="Serve uploaded image batches through a minimal HTTP API")
    server.add_argument("--config", type=Path, required=True, help="TOML configuration for cloud calls and routing")
    server.add_argument("--gate", type=Path, required=True, help="Fitted gate artifact")
    server.add_argument("--host", default="127.0.0.1", help="HTTP bind address")
    server.add_argument("--port", type=int, default=8000, help="HTTP bind port")
    return parser


def configured(args):
    """Load Config and apply only the overrides supplied by a parsed command."""
    config = load_config(args.config)
    for option, attribute in (("data_root", "root"), ("references", "references_per_category"),
                              ("per_category_limit", "per_category_limit"), ("seed", "seed"),
                              ("selection", "selection")):
        value = getattr(args, option, None)
        if value is not None:
            setattr(config.data, attribute, value)
    if getattr(args, "categories", None) is not None:
        config.data.categories = [item.strip() for item in args.categories.split(",") if item.strip()]
    for option, attribute in (("backbone", "backbone"), ("encoder_model", "model"), ("device", "device")):
        value = getattr(args, option, None)
        if value is not None:
            setattr(config.encoder, attribute, value)
    if getattr(args, "backbone", None) and not getattr(args, "encoder_model", None):
        # A backbone override must use its own default rather than the TOML model.
        config.encoder.model = ""
    for option in ("score", "quantile"):
        value = getattr(args, option, None)
        if value is not None:
            setattr(config.gate, option, value)
    return config


def execute(args):
    """Execute a parsed command, writing artifacts to its requested output path."""
    from .evaluation import (build_frontier, evaluate_gate, fit_gate, read_cloud_rows,
                             route_dataset, run_cloud_controls, start_logging, write_json)

    if args.command == "predict":
        from .gate import RefCalGate

        result = RefCalGate.load(args.gate, args.device).predict_path(args.image, args.category, args.data_root)
        if args.output:
            write_json(args.output, result)
        else:
            print(json.dumps(result, allow_nan=False))
        return
    if args.command in {"frontier", "summarize-controls"}:
        start_logging(args.output.parent)
        if args.command == "frontier":
            if args.bootstrap < 0:
                raise ValueError("Bootstrap repetitions must be nonnegative")
            build_frontier(args.edge, args.cloud, args.output, args.bootstrap, args.seed)
        else:
            from .metrics import control_summary

            write_json(args.output, control_summary(read_cloud_rows(args.input)))
        return
    config = configured(args)
    if args.command == "serve":
        from .server import serve

        serve(args.gate, config, args.host, args.port)
        return
    directory = args.output.parent if args.command == "fit" else args.output
    start_logging(directory)
    logging.info("Starting command=%s", args.command)
    if args.command == "fit":
        fit_gate(config, args.output)
    elif args.command == "evaluate":
        evaluate_gate(config, args.output, args.gate)
    elif args.command == "cloud":
        from .controls import CONTROLS

        controls = list(CONTROLS) if args.controls == "all" else args.controls.split(",")
        if not controls or len(set(controls)) != len(controls) or set(controls) - set(CONTROLS):
            raise ValueError(f"Controls must be unique members of: {', '.join(CONTROLS)}")
        run_cloud_controls(config, args.output, controls, args.blur_radius, args.gate)
    elif args.command == "route":
        route_dataset(config, args.gate, args.output, args.cloud_fraction)
    elif args.command == "prepare":
        from .prepare import prepare_dataset

        result = prepare_dataset(config.data, args.format, args.output, args.input_dir, args.url_template,
                                 args.train_references, args.test_normal, args.test_anomaly, args.full_test)
        write_json(args.output / "counts.json", result)
        logging.info("Data preparation complete images=%d", sum(result.values()))


def main(argv=None):
    """Parse optional argv and return zero on success or one on execution failure."""
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        execute(args)
    except Exception:
        logging.exception("Command failed")
        return 1
    return 0
