"""Encode whole images or regular image crops with frozen CLIP or DINOv2 models."""

import logging


class FrozenGridEncoder:
    """Load an EncoderConfig model and return unit-normalized image features.

    The auto device selects CUDA when available, otherwise CPU. Grid features
    encode each crop as an image using the model's own image processor.
    """

    def __init__(self, config):
        import torch
        from transformers import AutoImageProcessor, AutoModel, CLIPModel, CLIPProcessor

        if config.backbone not in {"clip", "dinov2"}:
            raise ValueError("Backbone must be clip or dinov2")
        if config.grid_size < 2 or config.batch_size < 1:
            raise ValueError("Grid size must be at least two and batch size must be positive")
        self.config = config
        self.torch = torch
        self.device = config.device
        if self.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_name = config.model_name()
        logging.info("Loading encoder backbone=%s model=%s device=%s",
                     config.backbone, self.model_name, self.device)
        if config.backbone == "clip":
            self.model = CLIPModel.from_pretrained(self.model_name).to(self.device).eval()
            self.processor = CLIPProcessor.from_pretrained(self.model_name)
        else:
            self.model = AutoModel.from_pretrained(self.model_name).to(self.device).eval()
            self.processor = AutoImageProcessor.from_pretrained(self.model_name)
        self.model.requires_grad_(False)

    def encode_images(self, images):
        """Return a finite float32 [N, D] NumPy array for a nonempty PIL image list."""
        import numpy as np

        if not images:
            raise ValueError("At least one image is required")
        features = []
        torch = self.torch
        with torch.inference_mode():
            for start in range(0, len(images), self.config.batch_size):
                inputs = self.processor(images=images[start:start + self.config.batch_size],
                                        return_tensors="pt").to(self.device)
                if self.config.backbone == "clip":
                    feature = self.model.get_image_features(**inputs)
                    if not isinstance(feature, torch.Tensor):
                        # Accept model outputs that wrap the projected features.
                        output = feature
                        feature = getattr(output, "image_embeds", None)
                        if feature is None:
                            feature = getattr(output, "pooler_output", None)
                else:
                    output = self.model(**inputs)
                    feature = getattr(output, "pooler_output", None)
                    if feature is None:
                        feature = output.last_hidden_state[:, 0]
                if feature is None:
                    raise TypeError("Encoder did not return image embeddings")
                feature = feature / feature.norm(dim=-1, keepdim=True).clamp_min(1e-12)
                features.append(feature.float().cpu().numpy())
        result = np.concatenate(features)
        if not np.isfinite(result).all():
            raise ValueError("Encoder returned nonfinite embeddings")
        return result

    def encode_grid(self, image):
        """Return [grid_size**2, D] features for row-major, nonoverlapping PIL crops."""
        grid = self.config.grid_size
        width, height = image.size
        if width < grid or height < grid:
            raise ValueError("Image dimensions must be at least the grid size")
        crops = [image.crop((round(col * width / grid), round(row * height / grid),
                             round((col + 1) * width / grid), round((row + 1) * height / grid)))
                 for row in range(grid) for col in range(grid)]
        return self.encode_images(crops)

    def encode_path(self, path):
        """Open an image path as RGB and return its row-major grid features."""
        from PIL import Image

        with Image.open(path) as image:
            return self.encode_grid(image.convert("RGB"))
