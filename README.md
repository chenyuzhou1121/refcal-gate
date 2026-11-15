# RefCal-Gate

RefCal-Gate is an edge gate for industrial visual inspection that uses no VLM confidence.
It builds a per-category patch memory with frozen CLIP or DINOv2 grid embeddings and calibrates anomaly scores and thresholds from normal references alone.
Reference-Only Threshold Adaptation (ROTA) sets each category's threshold from leave-one-reference-out normal scores; the lowest-margin queries are sent to a query-only cloud VLM according to the batch budget.

Paper: RefCal-Gate: Confidence-Free Reference Calibration for Edge-Cloud Industrial VLM Inspection (ACCV 2026).

## Installation

Python 3.11 or newer is required. Core dependencies are NumPy, Pillow, PyTorch, and Transformers.

```sh
python3 -m pip install -e .
```

Optional extras provide the chat-completion client (`cloud`), parquet reader (`data`), and HTTP server (`server`).

```sh
python3 -m pip install -e '.[cloud,data,server]'
```

Model identifiers load pretrained weights through Transformers; local model directories are also accepted.

## Input images

Supply an image tree such as the following. Images and model weights are not bundled.

```text
data/images/part/
  train/good/          # Normal reference images
  test/good/query.png  # Normal query
  test/defect/         # Anomalous queries
```

Each label folder contains images directly. Supported suffixes are `.png`, `.jpg`, `.jpeg`, `.bmp`, `.tif`, and `.tiff`.
Normal folder names are `good`, `ok`, `normal`, `anomaly-free`, and `anomaly_free`, matched without case sensitivity; other test folders are labeled `alarm`.
When fitting, the first sorted normal training images form the reference set. If none exist, normal test images supply references and are excluded from queries.
At least two references are required per category; the example configuration uses five.

## Quick start

Run these commands from the repository root after supplying the image tree.

```sh
refcal-gate fit --config configs/example.toml --data-root data/images --output outputs/gate.npz
refcal-gate predict --gate outputs/gate.npz --data-root data/images --image data/images/part/test/good/query.png --category part
refcal-gate evaluate --config configs/example.toml --data-root data/images --gate outputs/gate.npz --output outputs/edge
```

`fit` writes NPZ format version 2 with reference embeddings, per-category reference IDs relative to the dataset root, and settings, plus a sibling `run.log`. Model weights remain separate.
`evaluate`, `route`, and `cloud --gate` use the saved reference IDs and exclude them before query sampling and selection.
`predict` identifies a held-out image using `--data-root` and excludes saved reference IDs. It prints JSON with `category`, `score`, `threshold`, `margin`, `predicted`, and row-major `patch_distances`; `--output` writes it to a file.
Scores strictly above the threshold produce `alarm`; ties produce `normal`. The margin is the absolute score-to-threshold distance.
`evaluate` writes `edge.json` with metrics and per-image rows, plus `run.log`. Omitting `--gate` fits a new gate and also writes `gate.npz`.
Binary metrics treat both `warning` and `alarm` as positive; `exact_accuracy` compares all three labels.

## Cloud inspection and routing

Set `REFCAL_API_BASE`, `REFCAL_CLOUD_MODEL`, and `REFCAL_API_KEY` for a vision-capable chat-completion API, or configure their variable names in TOML.
The endpoint and model in the example files are placeholders. Cloud calls send query images and, for reference controls, reference images to that endpoint.

```sh
refcal-gate cloud --config configs/example.toml --data-root data/images --gate outputs/gate.npz --output outputs/cloud
refcal-gate frontier --edge outputs/edge/edge.json --cloud outputs/cloud/cloud.jsonl --output outputs/frontier.json
refcal-gate summarize-controls --input outputs/cloud/cloud.jsonl --output outputs/controls.json
refcal-gate route --config configs/example.toml --data-root data/images --gate outputs/gate.npz --cloud-fraction 0.5 --output outputs/routed
refcal-gate serve --config configs/example.toml --gate outputs/gate.npz --host 127.0.0.1 --port 8000
```

`cloud` defaults to `query_only` and writes `cloud.jsonl`; including `query_only` also writes `controls.json`.
`--controls all` adds `correct_reference`, `same_category_distractor`, `blurred_reference`, `blank_reference`, and `wrong_category`.
These use the nearest reference, farthest same-category reference, blurred nearest reference, gray image, and a deterministic other-category reference. The last requires at least two categories.
`frontier` joins matching sample IDs and reports each margin-ordered cloud budget; `summarize-controls` aggregates paired control predictions.
`route` sends `floor(batch_size * cloud_fraction)` queries with the smallest margins to query-only cloud inspection and writes `routed.json`.
`serve` exposes `GET /categories` and multipart `POST /inspect`: repeat the `files` field, provide `categories` as a JSON array string in upload order, and optionally provide `cloud_fraction`.
The response contains `cloud_calls`, `count`, and ordered `predictions`. Install both `server` and `cloud` extras when serving with cloud routing.

## Parquet preparation

`prepare` accepts local `category.train.parquet` and `category.test.parquet` shards, or an HTTPS `--url-template` containing `{category}` and `{split}`.

```sh
refcal-gate prepare --config configs/mvtec.toml --categories bottle --format mvtec --input-dir data/shards --output data/prepared
```

The `mvtec` schema requires `object`, `split`, `defect`, and `image_path` with embedded `bytes` and an optional `path`.
The `visa` schema requires `label` (`0` normal, `1` anomaly) and `image` with embedded `bytes`.
VisA extraction defaults to 20 training normals, 10 test normals, and 10 test anomalies per category; `--train-references`, `--test-normal`, `--test-anomaly`, and `--full-test` control these counts.
VisA images are fitted within 1024x1024 and saved as JPEG; MVTec images are saved as PNG. Existing image files are not overwritten.
The output contains the image tree, `counts.json`, and `run.log`; downloaded shards are cached under `.parquet_cache/`.

## Configuration and layout

`configs/example.toml` documents every setting. Omitted settings use defaults from `refcal_gate/config.py`; unknown sections or keys are rejected.
Use `configs/visa.toml` or `configs/mvtec.toml` in place of `configs/example.toml` for the dataset settings; both specify decision-format JSON and the standard reference policy.
Relative paths resolve from the working directory. Dataset commands require `--config`; `--data-root` overrides `data.root` where offered.
`data.categories = []` discovers category directories; `prepare` requires explicit categories. References use sorted paths; `data.seed` controls query sampling and shuffling.
`data.per_category_limit` caps query sampling before `data.selection`, an optional path to a JSON file holding an array of relative query IDs such as `["part/test/good/query.png"]`; the array also sets query order.
`encoder` selects the backbone, model, device, grid, and batch size; `retrieval_model` selects the whole-image CLIP model for reference controls.
`gate.score` accepts `max_excess`, `position_z`, or `local_covariance`; `gate.quantile` selects an empirical leave-one-out normal-score threshold in `(0, 1]`.
`cloud` defaults to `prompt_mode = "json"` and `reference_policy = "standard"`. JSON requests `judgment`, `confidence`, `action`, `reasoning`, and `root_cause`; prediction uses `judgment`. `label` mode and the `sensitive` reference policy are also available.
`cloud.prompt_file` is a path to a JSON file holding an object of string overrides with keys `query_only`, `reference`, `query_only_user`, and `reference_user`.
`cloud.disable_thinking` sends `chat_template_kwargs.enable_thinking=false` when enabled. `routing.cloud_fraction` supplies the default batch cloud fraction in `[0, 1]`.
Loaded gate artifacts supply their own encoder and gate settings, with a device override available.

```text
refcal_gate/     CLI, configuration, data, encoding, calibration, gates, metrics,
                reference controls, cloud client, routing, evaluation, and server
configs/        Annotated example, MVTec, and VisA configurations
pyproject.toml  Package metadata, dependencies, extras, and CLI entry point
```

## License

MIT. See [LICENSE](LICENSE).
