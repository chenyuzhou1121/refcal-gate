"""Load typed settings from TOML and resolve cloud environment overrides."""

from dataclasses import dataclass, field, fields
import os
from pathlib import Path
import tomllib


@dataclass
class DataConfig:
    """Image-tree location, category selection, and query sampling settings."""

    root: str = "/path/to/data"
    categories: list[str] = field(default_factory=list)
    references_per_category: int = 5
    per_category_limit: int = 0
    seed: int = 42
    selection: str = ""


@dataclass
class EncoderConfig:
    """Frozen encoder family, model identifiers, device, and crop batching."""

    backbone: str = "clip"
    model: str = ""
    retrieval_model: str = "openai/clip-vit-base-patch32"
    device: str = "auto"
    grid_size: int = 3
    batch_size: int = 32

    def model_name(self):
        """Return the explicit model identifier or the backbone's default."""
        defaults = {"clip": "openai/clip-vit-base-patch32", "dinov2": "facebook/dinov2-base"}
        return self.model or defaults[self.backbone]


@dataclass
class GateConfig:
    """Score variant and empirical normal-reference threshold quantile."""

    score: str = "max_excess"
    quantile: float = 0.85


@dataclass
class CloudConfig:
    """Chat-completion endpoint, credential variable name, and prompt settings."""

    base_url: str = "https://api.example.com/v1"
    base_url_env: str = "REFCAL_API_BASE"
    model: str = "vision-model"
    model_env: str = "REFCAL_CLOUD_MODEL"
    api_key_env: str = "REFCAL_API_KEY"
    max_tokens: int = 2048
    timeout: float = 120.0
    temperature: float = 0.0
    prompt_mode: str = "json"
    reference_policy: str = "standard"
    disable_thinking: bool = False
    prompt_file: str = ""

    def endpoint(self):
        """Return the endpoint, with its named environment variable taking priority."""
        return os.environ.get(self.base_url_env, self.base_url)

    def model_name(self):
        """Return the model, with its named environment variable taking priority."""
        return os.environ.get(self.model_env, self.model)


@dataclass
class RoutingConfig:
    """Fraction of each batch to send to the cloud, rounded down to a count."""

    cloud_fraction: float = 0.5


@dataclass
class Config:
    """Group data, encoder, gate, cloud, and routing settings."""

    data: DataConfig = field(default_factory=DataConfig)
    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    gate: GateConfig = field(default_factory=GateConfig)
    cloud: CloudConfig = field(default_factory=CloudConfig)
    routing: RoutingConfig = field(default_factory=RoutingConfig)


def load_config(path):
    """Read a TOML path into Config, filling defaults and rejecting unknown keys."""
    with Path(path).open("rb") as stream:
        raw = tomllib.load(stream)
    schema = {
        "data": DataConfig, "encoder": EncoderConfig, "gate": GateConfig,
        "cloud": CloudConfig, "routing": RoutingConfig,
    }
    unknown = set(raw) - set(schema)
    if unknown:
        raise ValueError(f"Unknown configuration sections: {sorted(unknown)}")
    sections = {}
    for name, cls in schema.items():
        values = raw.get(name, {})
        unknown = set(values) - {item.name for item in fields(cls)}
        if unknown:
            raise ValueError(f"Unknown {name} settings: {sorted(unknown)}")
        sections[name] = cls(**values)
    return Config(**sections)
