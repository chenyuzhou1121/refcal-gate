"""Combine frozen image encoding, category memory, and portable gate artifacts."""

from dataclasses import asdict
import json
from pathlib import Path, PurePosixPath
import numpy as np

from .config import EncoderConfig, GateConfig
from .encoders import FrozenGridEncoder
from .memory import ReferenceMemory


class RefCalGate:
    """Image anomaly gate configured by EncoderConfig and GateConfig.

    Omitted configurations use defaults. The encoder is loaded on first use;
    fitting and prediction use a separate normal-reference memory per category.
    """

    def __init__(self, encoder_config=None, gate_config=None):
        self.encoder_config = encoder_config or EncoderConfig()
        self.gate_config = gate_config or GateConfig()
        self._encoder = None
        self.reference_ids = {}
        self.memory = ReferenceMemory(self.encoder_config.grid_size, self.gate_config.score,
                                      self.gate_config.quantile)

    @property
    def encoder(self):
        """Return the FrozenGridEncoder, loading its model on first access."""
        if self._encoder is None:
            self._encoder = FrozenGridEncoder(self.encoder_config)
        return self._encoder

    def fit(self, references, data_root):
        """Fit normal image paths and retain their IDs relative to data_root."""
        root = Path(data_root).expanduser().resolve()
        reference_ids = {category: [Path(path).expanduser().resolve().relative_to(root).as_posix()
                                    for path in paths] for category, paths in references.items()}
        embeddings = {category: np.stack([self.encoder.encode_path(path) for path in paths])
                      for category, paths in references.items()}
        self.memory.fit(embeddings)
        self._set_reference_ids(reference_ids)
        self.encoder_config.model = self.encoder.model_name
        return self

    def _set_reference_ids(self, reference_ids):
        """Validate and store one root-relative sample identity per reference grid."""
        if not isinstance(reference_ids, dict) or set(reference_ids) != set(self.memory.references):
            raise ValueError("Reference IDs must cover every fitted category")
        for category, ids in reference_ids.items():
            if (not isinstance(ids, list) or len(ids) != len(self.memory.references[category])
                    or any(not isinstance(value, str) for value in ids)):
                raise ValueError("Reference IDs must match the fitted reference arrays")
            if len(set(ids)) != len(ids):
                raise ValueError("Reference sample IDs must be unique")
            for value in ids:
                path = PurePosixPath(value)
                if (path.is_absolute() or len(path.parts) < 2 or path.parts[0] != category
                        or ".." in path.parts or "\\" in value or path.as_posix() != value):
                    raise ValueError("Reference IDs must be canonical root-relative POSIX paths")
        self.reference_ids = {category: list(ids) for category, ids in reference_ids.items()}

    def predict_image(self, image, category):
        """Return a score, threshold, margin, label, and cell distances for a PIL image."""
        return self.memory.predict(self.encoder.encode_grid(image.convert("RGB")), category)

    def predict_path(self, path, category, data_root):
        """Predict a held-out image path, identifying references relative to data_root."""
        path = Path(path).expanduser().resolve()
        sample_id = path.relative_to(Path(data_root).expanduser().resolve()).as_posix()
        if any(sample_id in ids for ids in self.reference_ids.values()):
            raise ValueError("Reference samples are excluded from prediction queries")
        return self.memory.predict(self.encoder.encode_path(path), category)

    def save(self, path):
        """Write NPZ format 2 with reference grids, root-relative IDs, and settings."""
        if not self.memory.references:
            raise ValueError("Fit the gate before saving")
        categories = sorted(self.memory.references)
        self._set_reference_ids(self.reference_ids)
        metadata = {"format_version": 2, "encoder": asdict(self.encoder_config),
                    "gate": asdict(self.gate_config), "categories": categories,
                    "reference_ids": self.reference_ids}
        arrays = {f"references_{index}": self.memory.references[category]
                  for index, category in enumerate(categories)}
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as stream:
            np.savez_compressed(stream, metadata=np.array(json.dumps(metadata)), **arrays)

    @classmethod
    def load(cls, path, device=None):
        """Load NPZ format 2, restore reference IDs, and rebuild calibration."""
        with np.load(path, allow_pickle=False) as archive:
            metadata = json.loads(str(archive["metadata"].item()))
            if metadata.get("format_version") != 2:
                raise ValueError("Unsupported gate artifact format")
            categories = metadata["categories"]
            if not categories or len(set(categories)) != len(categories):
                raise ValueError("Gate categories must be nonempty and unique")
            encoder = EncoderConfig(**metadata["encoder"])
            if device:
                encoder.device = device
            instance = cls(encoder, GateConfig(**metadata["gate"]))
            embeddings = {category: archive[f"references_{index}"].copy()
                          for index, category in enumerate(categories)}
        instance.memory.fit(embeddings)
        instance._set_reference_ids(metadata.get("reference_ids"))
        return instance
