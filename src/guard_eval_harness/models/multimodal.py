"""Shared helpers for multimodal guard adapters."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
import re
from typing import Any, Iterable, Literal, Mapping, Sequence

from guard_eval_harness.models.templates import score_from_text
from guard_eval_harness.schemas import (
    MediaPart,
    Message,
    NormalizedSample,
    TextPart,
)


_REMOTE_URI_PREFIXES = ("http://", "https://", "data:")
_LLAMA_GUARD_CATEGORY_PATTERN = re.compile(
    r"\bS\d{1,2}\b",
    flags=re.IGNORECASE,
)
_MARKDOWN_JSON_PATTERN = re.compile(
    r"^```(?:json)?\s*(?P<body>.*?)\s*```$",
    flags=re.DOTALL | re.IGNORECASE,
)


@dataclass(slots=True)
class ParsedGuardOutput:
    """Normalized parsed output from a multimodal guard."""

    unsafe_score: float
    predicted_categories: tuple[str, ...] = ()
    category_scores: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


def export_message_for_hf(
    message: Message,
    *,
    image_mode: Literal["auto", "path", "url", "placeholder"] = "auto",
    ensure_text_block_for_images: bool = False,
) -> dict[str, Any]:
    """Convert one normalized message into HF processor chat content."""
    if isinstance(message.content, str):
        return {
            "role": message.role,
            "content": [{"type": "text", "text": message.content}],
        }

    exported_parts: list[dict[str, Any]] = []
    saw_text = False
    saw_image = False
    for part in message.content:
        if isinstance(part, TextPart):
            exported_parts.append({"type": "text", "text": part.text})
            saw_text = True
            continue

        if not isinstance(part, MediaPart):
            continue
        if part.media.modality != "image":
            raise ValueError(
                f"unsupported media modality: {part.media.modality}"
            )
        exported_parts.append(
            _export_image_part(
                part.media.uri,
                image_mode=image_mode,
            )
        )
        saw_image = True

    if ensure_text_block_for_images and saw_image and not saw_text:
        exported_parts.append({"type": "text", "text": ""})

    return {
        "role": message.role,
        "content": exported_parts,
    }


def sample_to_hf_messages(
    sample: NormalizedSample,
    *,
    image_mode: Literal["auto", "path", "url", "placeholder"] = "auto",
    ensure_text_block_for_images: bool = False,
) -> list[dict[str, Any]]:
    """Convert one normalized sample into HF processor chat messages."""
    return [
        export_message_for_hf(
            message,
            image_mode=image_mode,
            ensure_text_block_for_images=ensure_text_block_for_images,
        )
        for message in sample.messages
    ]


def load_sample_images(sample: NormalizedSample) -> list[Any]:
    """Load all sample images as detached RGB PIL images."""
    try:
        from PIL import Image  # type: ignore[import-untyped]
    except ImportError as exc:
        raise ImportError(
            "Pillow is required for multimodal Hugging Face adapters"
        ) from exc

    images: list[Any] = []
    for message in sample.messages:
        for image_ref in message.image_refs:
            uri = image_ref.uri
            if uri.startswith(_REMOTE_URI_PREFIXES):
                raise ValueError(
                    "remote image URIs are not supported by this adapter "
                    f"without prior dataset resolution: {uri}"
                )
            path = Path(uri)
            if not path.exists():
                raise FileNotFoundError(f"image not found: {path}")
            with Image.open(path) as image:
                images.append(image.convert("RGB"))
    return images


def model_device(model: Any) -> Any | None:
    """Return the best available device for one loaded model."""
    device = getattr(model, "device", None)
    if device is not None and str(device) != "meta":
        return device

    parameters = getattr(model, "parameters", None)
    if callable(parameters):
        try:
            first_parameter = next(parameters())
        except StopIteration:
            return None
        return getattr(first_parameter, "device", None)
    return None


def move_batch_to_device(inputs: Any, device: Any | None) -> Any:
    """Move one processor batch to a concrete device when possible."""
    if device is None:
        return inputs
    if hasattr(inputs, "to"):
        return inputs.to(device)
    if isinstance(inputs, Mapping):
        return {
            key: value.to(device) if hasattr(value, "to") else value
            for key, value in inputs.items()
        }
    return inputs


def decode_generated_texts(
    *,
    processor: Any,
    outputs: Any,
    input_ids: Any,
) -> list[str]:
    """Decode only generated continuation tokens for each batch row."""
    generated_tokens = outputs[:, input_ids.shape[-1] :]
    batch_decode = getattr(processor, "batch_decode", None)
    if callable(batch_decode):
        return [
            str(text)
            for text in batch_decode(
                generated_tokens,
                skip_special_tokens=True,
            )
        ]

    decode = getattr(processor, "decode", None)
    if not callable(decode):
        raise ValueError("processor does not provide decode helpers")
    return [
        str(decode(token_row, skip_special_tokens=True))
        for token_row in generated_tokens
    ]


def parse_llama_guard_output(text: str) -> ParsedGuardOutput:
    """Parse `safe` / `unsafe\\nS{N}` style verdicts."""
    raw_output = text.strip()
    if not raw_output:
        raise ValueError("empty Llama Guard output")

    lines = [line.strip() for line in raw_output.splitlines() if line.strip()]
    lowered_lines = [line.lower() for line in lines]
    categories = _dedupe_preserving_order(
        match.group(0).upper()
        for match in _LLAMA_GUARD_CATEGORY_PATTERN.finditer(raw_output)
    )

    unsafe_score: float | None = None
    if "unsafe" in lowered_lines:
        unsafe_score = 1.0
    elif "safe" in lowered_lines:
        unsafe_score = 0.0
    elif categories:
        unsafe_score = 1.0
    else:
        inferred = score_from_text(raw_output)
        if inferred is None:
            raise ValueError(
                f"could not parse Llama Guard output: {raw_output!r}"
            )
        unsafe_score = 1.0 if inferred >= 0.5 else 0.0

    if unsafe_score < 0.5:
        categories = ()

    return ParsedGuardOutput(
        unsafe_score=unsafe_score,
        predicted_categories=categories,
        category_scores={category: 1.0 for category in categories},
        metadata={"raw_output": raw_output},
    )


_GUARDREASONER_RESULT_PATTERN = re.compile(
    r"<result>\s*(?P<body>.*?)\s*</result>",
    flags=re.DOTALL | re.IGNORECASE,
)
_GUARDREASONER_REQUEST_PATTERN = re.compile(
    r"Request:\s*(harmful|unharmful|none|harmless)",
    flags=re.IGNORECASE,
)
_GUARDREASONER_RESPONSE_PATTERN = re.compile(
    r"Response:\s*(harmful|unharmful|none|harmless)",
    flags=re.IGNORECASE,
)


def parse_guardreasoner_vl_output(
    text: str,
) -> ParsedGuardOutput:
    """Parse GuardReasoner-VL ``<result>`` block output."""
    raw_output = text.strip()
    if not raw_output:
        raise ValueError("empty GuardReasoner-VL output")

    result_match = _GUARDREASONER_RESULT_PATTERN.search(
        raw_output
    )
    search_text = (
        result_match.group("body")
        if result_match
        else raw_output
    )

    request_match = _GUARDREASONER_REQUEST_PATTERN.search(
        search_text
    )
    response_match = _GUARDREASONER_RESPONSE_PATTERN.search(
        search_text
    )

    request_harmful = False
    response_harmful = False
    if request_match:
        verdict = request_match.group(1).lower()
        request_harmful = verdict == "harmful"
    if response_match:
        verdict = response_match.group(1).lower()
        response_harmful = verdict == "harmful"

    if not request_match and not response_match:
        lowered = search_text.lower()
        has_harmful = bool(
            re.search(r"\bharmful\b", lowered)
        )
        has_unharmful = (
            "unharmful" in lowered or "harmless" in lowered
        )
        if has_harmful and not has_unharmful:
            request_harmful = True
        elif has_harmful and has_unharmful:
            # Mixed signals — default to unsafe to avoid
            # false negatives on malformed output.
            request_harmful = True
        elif has_unharmful:
            request_harmful = False
        else:
            inferred = score_from_text(raw_output)
            if inferred is None:
                raise ValueError(
                    "could not parse GuardReasoner-VL "
                    f"output: {raw_output!r}"
                )
            unsafe_score = (
                1.0 if inferred >= 0.5 else 0.0
            )
            return ParsedGuardOutput(
                unsafe_score=unsafe_score,
                metadata={"raw_output": raw_output},
            )

    unsafe_score = (
        1.0
        if (request_harmful or response_harmful)
        else 0.0
    )
    return ParsedGuardOutput(
        unsafe_score=unsafe_score,
        metadata={
            "raw_output": raw_output,
            "request_harmful": request_harmful,
            "response_harmful": response_harmful,
        },
    )


def parse_llavaguard_output(text: str) -> ParsedGuardOutput:
    """Parse LlavaGuard JSON output with rating/category/rationale."""
    raw_output = text.strip()
    if not raw_output:
        raise ValueError("empty LlavaGuard output")

    payload = _json_mapping_from_text(raw_output)
    rating = _normalized_text(
        payload.get("rating")
        or payload.get("label")
        or payload.get("safety_rating")
    )
    if rating is None:
        inferred = score_from_text(raw_output)
        if inferred is None:
            raise ValueError(
                f"could not parse LlavaGuard rating from {raw_output!r}"
            )
        unsafe_score = 1.0 if inferred >= 0.5 else 0.0
    else:
        lowered_rating = rating.lower()
        if lowered_rating.startswith("unsafe"):
            unsafe_score = 1.0
        elif lowered_rating.startswith("safe"):
            unsafe_score = 0.0
        else:
            inferred = score_from_text(rating)
            if inferred is None:
                raise ValueError(f"unsupported LlavaGuard rating: {rating!r}")
            unsafe_score = 1.0 if inferred >= 0.5 else 0.0

    categories = ()
    category_value = payload.get("category") or payload.get("categories")
    if unsafe_score >= 0.5:
        categories = _normalized_categories(category_value)

    metadata = {
        "raw_output": raw_output,
        "parsed_output": payload,
    }
    rationale = _normalized_text(payload.get("rationale"))
    if rationale is not None:
        metadata["rationale"] = rationale

    return ParsedGuardOutput(
        unsafe_score=unsafe_score,
        predicted_categories=categories,
        category_scores={category: 1.0 for category in categories},
        metadata=metadata,
    )


def _export_image_part(
    uri: str,
    *,
    image_mode: Literal["auto", "path", "url", "placeholder"],
) -> dict[str, Any]:
    """Export one image content block for HF chat templates."""
    if image_mode == "placeholder":
        return {"type": "image"}
    if image_mode == "path":
        return {"type": "image", "path": uri}
    if image_mode == "url":
        return {"type": "image", "url": uri}
    if uri.startswith(_REMOTE_URI_PREFIXES):
        return {"type": "image", "url": uri}
    return {"type": "image", "path": uri}


def _dedupe_preserving_order(values: Iterable[str]) -> tuple[str, ...]:
    """Return a stable tuple with duplicates removed in order."""
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        normalized = value.strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        ordered.append(normalized)
    return tuple(ordered)


def _json_mapping_from_text(text: str) -> dict[str, Any]:
    """Extract and parse one JSON object from raw generated text."""
    stripped = text.strip()
    markdown_match = _MARKDOWN_JSON_PATTERN.match(stripped)
    if markdown_match is not None:
        stripped = markdown_match.group("body").strip()

    candidates: list[str] = [stripped]
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidates.append(stripped[start : end + 1])

    for candidate in candidates:
        try:
            payload = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, Mapping):
            return dict(payload)
    raise ValueError(f"could not parse JSON object from {text!r}")


def _normalized_text(value: Any) -> str | None:
    """Normalize one optional text field."""
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    return text


def _normalized_categories(value: Any) -> tuple[str, ...]:
    """Normalize one category or category list into a stable tuple."""
    if value is None:
        return ()
    if isinstance(value, str):
        return _dedupe_preserving_order((value,))
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return _dedupe_preserving_order(str(item) for item in value)
    return _dedupe_preserving_order((str(value),))
