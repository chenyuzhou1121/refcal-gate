"""Send query/reference images to a chat-completion API and parse inspection labels."""

import base64
import io
import json
import os
from pathlib import Path
import re
import time

from .metrics import LABELS


QUERY_SYSTEM = (
    "You are an advanced industrial visual quality inspection model for complex analysis.\n"
    "Decide whether the product in the IMAGE is normal, warning, or alarm, and perform\n"
    "root-cause analysis. The judgment is image-first: base it on what you actually see. "
    "Output JSON only.\n\n"
    "Judgment (image-first):\n"
    "- If you can see a visible product anomaly, damage, abnormal residue, color\n"
    "  shift, geometry change, or structural irregularity in the image, output\n"
    '  "warning" (subtle/ambiguous) or "alarm" (obvious/severe) by severity.\n'
    '- If the image looks clean and intact, output "normal".'
)
REFERENCE_INTRO = (
    "You are an advanced industrial visual quality inspection model for complex cloud analysis.\n"
    "You receive TWO images: image 1 is the QUERY sample to judge; image 2 is a\n"
    "KNOWN-GOOD REFERENCE of the same product category. Decide whether the QUERY\n"
    "is normal, warning, or alarm by COMPARING it against the reference, and perform\n"
    "root-cause analysis. Output JSON only.\n\n"
)
REFERENCE_SYSTEM = REFERENCE_INTRO + (
    "Rules (image-first, comparative):\n"
    "- If the query looks substantially like the reference (same surface pattern,\n"
    '  same color, same structure, no extra marks) -> "normal".\n'
    "- If the query shows a visible anomaly that is NOT present in the reference,\n"
    '  output "warning" (subtle) or "alarm" (obvious).\n'
    "- Natural variation that also appears in the reference (printed speckles,\n"
    "  texture, repeated internal structure) is NORMAL."
)
SENSITIVE_SYSTEM = REFERENCE_INTRO + (
    "Rules (image-first, comparative, HIGH SENSITIVITY - an undetected real anomaly is\n"
    "worse than a false alarm):\n"
    '- Output "normal" ONLY if the query is essentially identical to the reference\n'
    "  with NO extra marks, spots, color shift, or geometric change.\n"
    '- Any visible deviation the reference does not show should output "warning"\n'
    '  (subtle) or "alarm" (obvious).\n'
    "- When uncertain whether a mark is a real anomaly or natural variation, err\n"
    '  toward "warning". Do NOT explain away a suspicious mark as acceptable\n'
    "  variation unless the reference clearly shows the same thing."
)
DECISION_INSTRUCTION = (
    "Output strict JSON (pick one value, no alternatives):\n"
    '{"judgment":"normal/warning/alarm","confidence":0-1,'
    '"action":"maintain/adjust_flow/reduce_input/increase_cooling/emergency_stop",'
    '"reasoning":"analysis reason","root_cause":"root cause (when anomalous)"}'
)


def image_url(image):
    """Return an API image content block containing a PIL image as an RGB PNG data URL."""
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encoded}"}}


def parse_label(content):
    """Parse a label or judgment JSON after removing think blocks and code fences.

    Return normal, warning, or alarm; reject responses without one valid label.
    """
    cleaned = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", cleaned, flags=re.IGNORECASE).strip()
    try:
        payload = json.loads(cleaned)
    except json.JSONDecodeError:
        payload = None
    label = payload.get("judgment") if isinstance(payload, dict) else cleaned.lower().rstrip(". ")
    if label not in LABELS:
        raise ValueError("Cloud response must contain exactly one inspection label")
    return label


class CloudInspector:
    """Create a synchronous client from CloudConfig and optional prompt overrides.

    The API key is read from the configured environment variable. A prompt JSON
    may override query_only, reference, query_only_user, and reference_user text.
    """

    def __init__(self, config):
        from openai import OpenAI

        key = os.environ.get(config.api_key_env)
        if not key:
            raise ValueError(f"Set the API credential environment variable: {config.api_key_env}")
        if config.prompt_mode not in {"label", "json"}:
            raise ValueError("Cloud prompt mode must be label or json")
        if config.reference_policy not in {"standard", "sensitive"}:
            raise ValueError("Reference policy must be standard or sensitive")
        if config.max_tokens < 1 or config.timeout <= 0:
            raise ValueError("Cloud token budget and timeout must be positive")
        self.config = config
        self.model = config.model_name()
        self.client = OpenAI(base_url=config.endpoint(), api_key=key, timeout=config.timeout)
        self.prompts = {}
        if config.prompt_file:
            self.prompts = json.loads(Path(config.prompt_file).read_text(encoding="utf-8"))
            if not isinstance(self.prompts, dict) or not all(isinstance(value, str) for value in self.prompts.values()):
                raise ValueError("Prompt file must be a JSON object of string values")

    def predict(self, query, reference=None):
        """Inspect a PIL query and optional normal reference, returning a result dict.

        JSON mode requests judgment, confidence, action, reasoning, and root_cause;
        only judgment supplies the prediction. Query is image 1, reference image 2.
        Results include the label,
        latency in milliseconds, model names, finish reason, and token count.
        Empty or length-truncated responses raise ValueError.
        """
        has_reference = reference is not None
        system = REFERENCE_SYSTEM if has_reference else QUERY_SYSTEM
        if has_reference and self.config.reference_policy == "sensitive":
            system = SENSITIVE_SYSTEM
        default_user = (
            "The first image is the QUERY to judge; the second image is a "
            "known-good REFERENCE of the same category. Judge the QUERY relative to the REFERENCE."
            if has_reference else "An image has been provided for visual inspection."
        )
        name = "reference" if has_reference else "query_only"
        system = self.prompts.get(name, system)
        user = self.prompts.get(f"{name}_user", default_user)
        if self.config.prompt_mode == "label":
            system = system.replace(" Output JSON only.", "")
            instruction = "Output exactly one lowercase label and nothing else: normal, warning, or alarm."
            user_instruction = "Output exactly one label: normal, warning, or alarm. /no_think"
        else:
            instruction = DECISION_INSTRUCTION
            user_instruction = "Perform deep analysis and output JSON. /no_think"
        content = [image_url(query)]
        if has_reference:
            content.append(image_url(reference))
        content.append({"type": "text", "text": user + "\n\n" + user_instruction})
        extra = (
            {"extra_body": {"chat_template_kwargs": {"enable_thinking": False}}}
            if self.config.disable_thinking else {}
        )
        start = time.perf_counter()
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "system", "content": system + "\n\n" + instruction},
                      {"role": "user", "content": content}],
            temperature=self.config.temperature, max_tokens=self.config.max_tokens, **extra)
        latency_ms = (time.perf_counter() - start) * 1000
        choice = response.choices[0]
        usage = response.usage
        if choice.finish_reason == "length" or not (choice.message.content or "").strip():
            tokens = usage.completion_tokens if usage else None
            raise ValueError(
                f"Incomplete cloud response: finish_reason={choice.finish_reason}, "
                f"completion_tokens={tokens}"
            )
        return {"predicted": parse_label(choice.message.content), "latency_ms": latency_ms,
                "requested_model": self.model, "response_model": response.model,
                "finish_reason": choice.finish_reason,
                "completion_tokens": usage.completion_tokens if usage else None}
