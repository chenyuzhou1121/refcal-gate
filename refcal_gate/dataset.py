"""Read category image trees into disjoint normal references and labeled queries."""

from dataclasses import dataclass
import json
from pathlib import Path
import random


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
NORMAL_LABELS = {"good", "ok", "normal", "anomaly-free", "anomaly_free"}


@dataclass(frozen=True)
class Sample:
    """One query with a root-relative POSIX ID and a normal or alarm label."""

    sample_id: str
    path: Path
    category: str
    label: str


@dataclass
class Dataset:
    """Resolved image root, category reference paths, and ordered query samples."""

    root: Path
    references: dict[str, list[Path]]
    samples: list[Sample]


def image_paths(directory):
    """Return sorted image files immediately inside a directory."""
    return sorted(path for path in directory.iterdir()
                  if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)


def read_dataset(config, include_queries=True, reference_ids=None):
    """Return a Dataset from DataConfig using train/test label directories.

    References are the first sorted normal training images, falling back to
    normal test images. Supplied reference_ids use the gate's fitted references
    instead of selecting a new set. References are excluded before query sampling
    and JSON selection; IDs are POSIX paths relative to the dataset root.
    With include_queries=False, only reference paths are returned.
    """
    root = Path(config.root).expanduser().resolve()
    if not root.is_dir():
        raise ValueError(f"Dataset root does not exist: {root}")
    if ((reference_ids is None and config.references_per_category < 2)
            or config.per_category_limit < 0):
        raise ValueError("Use at least two references and a nonnegative sample limit")
    categories = config.categories or sorted(path.name for path in root.iterdir()
                                            if path.is_dir() and ((path / "test").is_dir() or
                                                                  (path / "train").is_dir()))
    if len(set(categories)) != len(categories) or not categories:
        raise ValueError("Dataset categories must be nonempty and unique")
    rng = random.Random(config.seed)
    references, samples = {}, []
    excluded_ids = {value for ids in reference_ids.values() for value in ids} if reference_ids is not None else set()
    for category in categories:
        category_root = root / category
        if category_root.resolve().parent != root:
            raise ValueError("Category names must identify direct child directories")
        normal, anomaly, train = [], [], []
        test_root = category_root / "test"
        if not test_root.is_dir() and include_queries:
            raise ValueError(f"Missing test directory for category: {category}")
        for folder in sorted(test_root.iterdir()) if test_root.is_dir() else []:
            if folder.is_dir():
                target = normal if folder.name.lower() in NORMAL_LABELS else anomaly
                target.extend(image_paths(folder))
        if reference_ids is None:
            train_root = category_root / "train"
            if train_root.is_dir():
                for folder in sorted(train_root.iterdir()):
                    if folder.is_dir() and folder.name.lower() in NORMAL_LABELS:
                        train.extend(image_paths(folder))
            pool = sorted(train or normal)
            if len(pool) < config.references_per_category:
                raise ValueError(f"Too few normal references for category: {category}")
            references[category] = pool[:config.references_per_category]
            excluded_ids.update(path.relative_to(root).as_posix() for path in references[category])
        else:
            if category not in reference_ids:
                raise ValueError(f"No fitted references for category: {category}")
            references[category] = [root / value for value in reference_ids[category]]
        normal = [path for path in normal if path.relative_to(root).as_posix() not in excluded_ids]
        anomaly = [path for path in anomaly if path.relative_to(root).as_posix() not in excluded_ids]
        if not include_queries:
            continue
        limit = config.per_category_limit
        if limit and limit < len(normal) + len(anomaly):
            # Reserve roughly one third for normals, with at least five when
            # the limit is 15 or more; unused capacity is not backfilled.
            target = max(1, limit // 3)
            if limit >= 15:
                target = max(5, target)
            n_normal = min(len(normal), limit, target)
            n_anomaly = min(len(anomaly), limit - n_normal)
            normal = rng.sample(normal, n_normal) if len(normal) > n_normal else normal
            anomaly = rng.sample(anomaly, n_anomaly) if len(anomaly) > n_anomaly else anomaly
        for label, paths in (("normal", normal), ("alarm", anomaly)):
            samples.extend(Sample(path.relative_to(root).as_posix(), path, category, label)
                           for path in paths)
    rng.shuffle(samples)
    if config.selection and include_queries:
        selected = json.loads(Path(config.selection).read_text(encoding="utf-8"))
        if (not isinstance(selected, list)
                or any(not isinstance(item, str) for item in selected)
                or len(selected) != len(set(selected))):
            raise ValueError("Selection must be a JSON array of unique relative sample IDs")
        selected = [value for value in selected if value not in excluded_ids]
        by_id = {sample.sample_id: sample for sample in samples}
        missing = set(selected) - set(by_id)
        if missing:
            raise ValueError(f"Selection contains {len(missing)} unavailable samples")
        samples = [by_id[key] for key in selected]
    if not samples and include_queries:
        raise ValueError("No evaluation samples were selected")
    return Dataset(root, references, samples)
