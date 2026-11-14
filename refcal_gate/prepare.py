"""Convert MVTec-style or VisA-style parquet shards into category image trees."""

import io
import logging
from pathlib import Path
import re
import urllib.request


def safe_component(value):
    """Return an ASCII filename component or reject empty, traversal, and unsafe values."""
    value = str(value)
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", value) or value in {".", ".."}:
        raise ValueError("Dataset path components must contain only ASCII letters, digits, dots, dashes or underscores")
    return value


def fetch_shard(url_template, category, split, cache_dir):
    """Return a cached parquet path, downloading an HTTPS template URL if absent.

    The template substitutes category and split. Downloads use a partial file
    that is renamed only after the response is fully written.
    """
    category, split = safe_component(category), safe_component(split)
    url = url_template.format(category=category, split=split)
    if not url.startswith("https://"):
        raise ValueError("Dataset downloads require an HTTPS URL template")
    destination = Path(cache_dir) / f"{category}.{split}.parquet"
    if destination.exists():
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".parquet.part")
    logging.info("Downloading category=%s split=%s", category, split)
    with urllib.request.urlopen(url, timeout=120) as response, temporary.open("wb") as stream:
        while block := response.read(1024 * 1024):
            stream.write(block)
    temporary.replace(destination)
    return destination


def save_image(raw, destination, resize=False):
    """Save image bytes as RGB at a Path, refusing to overwrite existing files.

    resize=True fits the image within 1024x1024 without changing its aspect ratio.
    """
    from PIL import Image

    if destination.exists():
        raise ValueError(f"Refusing to overwrite an existing image: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(io.BytesIO(raw)) as source:
        image = source.convert("RGB")
        if resize:
            image.thumbnail((1024, 1024))
        image.save(destination, quality=95)


def prepare_dataset(config, dataset_format, output, input_dir, url_template,
                    train_references=20, test_normal=10, test_anomaly=10, full_test=False):
    """Extract category.train/test.parquet shards and return counts by image folder.

    config is DataConfig with explicit categories. dataset_format is mvtec or
    visa. A local input_dir takes precedence over an HTTPS url_template.
    MVTec rows need object, split, defect, and image_path (bytes and optional path).
    VisA rows need binary label (0=normal, 1=anomaly) and image (bytes).
    Only VisA uses the extraction limits, full_test, and image resizing.
    """
    import pyarrow.parquet as parquet

    if not config.categories:
        raise ValueError("Dataset preparation requires an explicit category list")
    if not input_dir and not url_template:
        raise ValueError("Provide a parquet input directory or an HTTPS URL template")
    if train_references < 2 or test_normal < 1 or test_anomaly < 1:
        raise ValueError("Use at least two train references and positive test sample counts")
    output = Path(output)
    counts = {}
    for category in config.categories:
        category = safe_component(category)
        for split in ("train", "test"):
            shard = (Path(input_dir) / f"{category}.{split}.parquet" if input_dir else
                     fetch_shard(url_template, category, split, output / ".parquet_cache"))
            rows = parquet.read_table(shard).to_pylist()
            if dataset_format == "mvtec":
                for index, row in enumerate(rows):
                    if row["object"] != category or row["split"] != split:
                        raise ValueError("MVTec shard category or split does not match its filename")
                    defect = safe_component(row["defect"] or "good")
                    payload = row["image_path"]
                    raw = payload.get("bytes")
                    if raw is None:
                        raise ValueError("MVTec parquet image bytes are missing")
                    source_name = str(payload.get("path") or "").replace("\\", "/")
                    stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", Path(source_name).stem)
                    stem = stem or f"{category}_{split}_{defect}_{index:05d}"
                    destination = output / category / split / defect / f"{safe_component(stem)}.png"
                    save_image(raw, destination)
                    key = f"{category}/{split}/{defect}"
                    counts[key] = counts.get(key, 0) + 1
            else:
                for kind, binary_label, limit in (("good", 0, train_references if split == "train" else test_normal),
                                                   ("anomaly", 1, test_anomaly)):
                    if split == "train" and binary_label:
                        continue
                    selected = [row for row in rows if int(row["label"]) == binary_label]
                    if not (split == "test" and full_test):
                        if len(selected) < limit:
                            raise ValueError(f"Too few {kind} samples in {category}.{split}")
                        selected = selected[:limit]
                    for index, row in enumerate(selected):
                        raw = row["image"].get("bytes")
                        if raw is None:
                            raise ValueError("VisA parquet image bytes are missing")
                        prefix = "ref" if split == "train" else "normal" if kind == "good" else "anomaly"
                        destination = output / category / split / kind / f"{prefix}_{index:03d}.jpg"
                        save_image(raw, destination, resize=True)
                    counts[f"{category}/{split}/{kind}"] = len(selected)
            logging.info("Prepared category=%s split=%s", category, split)
    return counts
