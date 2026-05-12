"""Typed contracts shared across the harness."""

from __future__ import annotations

from typing import Annotated, Any, Literal, Union

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationInfo,
    field_validator,
    model_validator,
)

from guard_eval_harness.config.models import (
    ResolvedExecutionConfig,
    ResolvedModelConfig,
    ResolvedOutputConfig,
)


Role = Literal["system", "user", "assistant", "tool"]


class HarnessModel(BaseModel):
    """Common pydantic base class for shared contracts."""

    model_config = ConfigDict(extra="forbid")


# --- Multimodal content types ---


class MediaRef(HarnessModel):
    """Reference to a resolved media artifact."""

    modality: Literal["image", "audio"]
    uri: str = Field(min_length=1)
    sha256: str | None = None
    mime_type: str | None = None
    width: int | None = None
    height: int | None = None
    duration_seconds: float | None = Field(default=None, ge=0.0)
    sample_rate_hz: int | None = Field(default=None, ge=1)
    channels: int | None = Field(default=None, ge=1)


class TextPart(HarnessModel):
    """Text content part within a multimodal message."""

    type: Literal["text"] = "text"
    text: str


class MediaPart(HarnessModel):
    """Media content part within a multimodal message."""

    type: Literal["media"] = "media"
    media: MediaRef


ContentPart = Annotated[
    Union[TextPart, MediaPart], Field(discriminator="type")
]


# --- Messages ---


def _coerce_openai_content_part(part: Any) -> Any:
    """Normalize an OpenAI-style content dict to typed ContentPart."""
    if not isinstance(part, dict):
        return part
    ptype = part.get("type")
    if ptype == "text":
        return part
    if ptype == "image_url":
        image_url_value = part.get("image_url", {})
        if isinstance(image_url_value, str):
            url = image_url_value
        elif isinstance(image_url_value, dict):
            url = image_url_value.get("url", "")
        else:
            url = ""
        return {
            "type": "media",
            "media": {"modality": "image", "uri": url},
        }
    if ptype == "image":
        url = part.get("url", "")
        return {
            "type": "media",
            "media": {"modality": "image", "uri": url},
        }
    if ptype == "audio":
        audio_value = (
            part.get("audio_url")
            if part.get("audio_url") is not None
            else part.get("audio")
        )
        if isinstance(audio_value, str):
            url = audio_value
        elif isinstance(audio_value, dict):
            url = (
                audio_value.get("url")
                or audio_value.get("audio_url")
                or audio_value.get("audio")
                or ""
            )
        else:
            url = part.get("url", "")
        return {
            "type": "media",
            "media": {"modality": "audio", "uri": url},
        }
    return part


class Message(HarnessModel):
    """Normalized message contract used across all datasets.

    ``content`` accepts either a plain string (backward-compatible
    text-only path) or a list of typed content parts for multimodal
    messages.
    """

    role: Role
    content: str | list[ContentPart]
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def coerce_openai_content(cls, values: Any) -> Any:
        """Normalize OpenAI-style content arrays at the boundary."""
        if isinstance(values, dict):
            raw = values.get("content")
            if isinstance(raw, list) and raw and isinstance(
                raw[0], dict
            ):
                values = dict(values)
                values["content"] = [
                    _coerce_openai_content_part(p) for p in raw
                ]
        return values

    @model_validator(mode="after")
    def validate_content(self) -> "Message":
        """Reject empty content except for empty assistant turns."""
        if isinstance(self.content, str):
            if self.content == "" and self.role == "assistant":
                return self
            if not self.content.strip():
                raise ValueError(
                    "message content must not be empty"
                )
        else:
            if self.role == "assistant":
                return self
            has_text = any(
                isinstance(p, TextPart) and p.text.strip()
                for p in self.content
            )
            has_media = any(
                isinstance(p, MediaPart) for p in self.content
            )
            if not has_text and not has_media:
                raise ValueError(
                    "message content must contain at least one "
                    "non-empty text part or media part"
                )
        return self

    @property
    def text_content(self) -> str:
        """Extract concatenated text from content."""
        if isinstance(self.content, str):
            return self.content
        return " ".join(
            p.text
            for p in self.content
            if isinstance(p, TextPart)
        )

    @property
    def image_refs(self) -> list[MediaRef]:
        """Extract image MediaRefs from content."""
        if isinstance(self.content, str):
            return []
        return [
            p.media
            for p in self.content
            if isinstance(p, MediaPart)
            and p.media.modality == "image"
        ]

    @property
    def media_refs(self) -> list[MediaRef]:
        """Extract all media refs from content."""
        if isinstance(self.content, str):
            return []
        return [
            p.media
            for p in self.content
            if isinstance(p, MediaPart)
        ]

    @property
    def audio_refs(self) -> list[MediaRef]:
        """Extract audio MediaRefs from content."""
        if isinstance(self.content, str):
            return []
        return [
            p.media
            for p in self.content
            if isinstance(p, MediaPart)
            and p.media.modality == "audio"
        ]


class UnsafeLabel(HarnessModel):
    """Binary label contract for dataset normalization."""

    unsafe: bool


PREDICT_METADATA_BLOCKLIST: frozenset[str] = frozenset(
    {
        # Generic local-file and HF label columns.
        "raw_label",
        "label",
        "labels",
        "label_name",
        "category_labels",
        "binary_label",
        "majority_label",
        "safety_label",
        "image_safety_label",
        "image_safe",
        "safe",
        "unsafe",
        "is_safe",
        "is_unsafe",
        "toxicity",
    }
)

PREDICT_METADATA_BLOCKLIST_BY_DATASET: dict[str, frozenset[str]] = {
    "agent_harm": frozenset({"config"}),
    "aegis_ai_content_safety_dataset_2": frozenset(
        {"violated_categories"}
    ),
    "beaver_tails_330k": frozenset({"active_categories"}),
    "circleguardbench_public": frozenset(
        {"prompt_verdict", "default_answer_verdict"}
    ),
    "civil_comments": frozenset(
        {
            "toxicity",
            "severe_toxicity",
            "obscene",
            "threat",
            "insult",
            "identity_attack",
            "sexual_explicit",
        }
    ),
    "dices_990": frozenset({"degree_of_harm"}),
    "hatexplain": frozenset(
        {"hate_speech_count", "offensive_language_count", "neither_count"}
    ),
    "implicit_hate": frozenset({"class"}),
    "jigsaw_toxicity": frozenset(
        {
            "toxic",
            "severe_toxic",
            "obscene",
            "threat",
            "insult",
            "identity_hate",
        }
    ),
    "local_image_dir": frozenset({"label_directory"}),
    "measuring_hate_speech": frozenset({"hate_speech_score"}),
    "openai_moderation_eval": frozenset(
        {"S", "H", "V", "HR", "SH", "S3", "H2", "V2"}
    ),
    "olid": frozenset({"subtask_b", "subtask_c"}),
    "or_bench": frozenset({"subset"}),
    "pku_safe_rlhf": frozenset({"active_categories"}),
    "real_toxicity_prompts": frozenset(
        {"prompt_toxicity", "continuation_toxicity"}
    ),
    "social_bias_frames": frozenset({"offensiveYN"}),
    "toxic_chat": frozenset({"human_annotation", "jailbreaking"}),
    "wildguardmix": frozenset(
        {
            "prompt_harm_label",
            "response_harm_label",
            "response_refusal_label",
        }
    ),
    "wildjailbreak": frozenset({"data_type"}),
    "xstest": frozenset({"type"}),
}

PREDICT_METADATA_AUDIT_BLOCKLIST: frozenset[str] = (
    PREDICT_METADATA_BLOCKLIST | frozenset({"label_directory"})
)


def sanitize_predict_metadata(
    metadata: dict[str, Any] | None,
    dataset: str | None = None,
) -> dict[str, Any]:
    """Drop ground-truth-shaped metadata from predict-path samples."""
    if not metadata:
        return {}
    blocklist = set(PREDICT_METADATA_BLOCKLIST)
    if dataset is not None:
        blocklist.update(
            PREDICT_METADATA_BLOCKLIST_BY_DATASET.get(dataset, ())
        )
    return {
        key: value
        for key, value in metadata.items()
        if key not in blocklist
    }


class PredictSample(HarnessModel):
    """Predict-path view of a sample. No ground truth — what the model sees."""

    id: str
    dataset: str
    split: str
    messages: list[Message] = Field(min_length=1)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("id", "dataset", "split")
    @classmethod
    def validate_identifier(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("value must not be empty")
        return cleaned

    @field_validator("metadata")
    @classmethod
    def validate_metadata(
        cls,
        value: dict[str, Any],
        info: ValidationInfo,
    ) -> dict[str, Any]:
        dataset = info.data.get("dataset") if info.data else None
        return sanitize_predict_metadata(value, dataset=dataset)


class SampleGroundTruth(HarnessModel):
    """Score-path view of a sample. Joined to predictions by sample_id."""

    sample_id: str
    label: UnsafeLabel
    category_labels: tuple[str, ...] = ()

    @field_validator("sample_id")
    @classmethod
    def validate_sample_id(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("sample_id must not be empty")
        return cleaned


class NormalizedSample(HarnessModel):
    """Canonical normalized dataset row."""

    id: str
    dataset: str
    split: str
    messages: list[Message] = Field(min_length=1)
    label: UnsafeLabel
    category_labels: tuple[str, ...] = ()
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("id", "dataset", "split")
    @classmethod
    def validate_identifier(cls, value: str) -> str:
        """Reject empty identifiers."""
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("value must not be empty")
        return cleaned

    def to_predict_sample(self) -> "PredictSample":
        return PredictSample(
            id=self.id,
            dataset=self.dataset,
            split=self.split,
            messages=self.messages,
            metadata=sanitize_predict_metadata(
                self.metadata,
                dataset=self.dataset,
            ),
        )

    def to_ground_truth(self) -> "SampleGroundTruth":
        return SampleGroundTruth(
            sample_id=self.id,
            label=self.label,
            category_labels=self.category_labels,
        )


class NormalizedPrediction(HarnessModel):
    """Canonical adapter output contract."""

    sample_id: str
    unsafe_score: float = Field(ge=0.0, le=1.0)
    unsafe_label: bool
    threshold: float = Field(ge=0.0, le=1.0, default=0.5)
    latency_ms: float = Field(ge=0.0, default=0.0)
    predicted_categories: tuple[str, ...] = ()
    category_scores: dict[str, float] = Field(
        default_factory=dict
    )
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("sample_id")
    @classmethod
    def validate_sample_id(cls, value: str) -> str:
        """Reject empty sample identifiers."""
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("sample_id must not be empty")
        return cleaned

    @model_validator(mode="after")
    def validate_threshold_alignment(self) -> "NormalizedPrediction":
        """Keep `unsafe_label` aligned with `unsafe_score` semantics."""
        expected = self.unsafe_score >= self.threshold
        if self.unsafe_label != expected:
            raise ValueError(
                "unsafe_label must match unsafe_score >= threshold"
            )
        return self


class DatasetMetadata(HarnessModel):
    """Dataset-level metadata written into manifests and reports."""

    name: str
    display_name: str
    version: str | None = None
    source_uri: str | None = None
    license: str | None = None
    splits: tuple[str, ...] = ("test",)
    sample_count: int | None = Field(default=None, ge=0)
    unsafe_count: int | None = Field(default=None, ge=0)
    languages: tuple[str, ...] = ()
    categories: tuple[str, ...] = ()
    input_modalities: tuple[str, ...] = ("text",)
    metric_eligibility: dict[str, bool] = Field(
        default_factory=lambda: {"binary_classification": True}
    )
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_counts(self) -> "DatasetMetadata":
        """Ensure unsafe counts never exceed total samples."""
        if (
            self.sample_count is not None
            and self.unsafe_count is not None
            and self.unsafe_count > self.sample_count
        ):
            raise ValueError(
                "unsafe_count cannot exceed sample_count"
            )
        return self


class AdapterCapabilities(HarnessModel):
    """Capability flags declared by each adapter."""

    adapter_name: str
    probability_scores: bool
    batching: bool
    concurrency: bool
    cost_estimation: bool
    token_accounting: bool
    supported_input_modalities: tuple[str, ...] = ("text",)
    supports_category_outputs: bool = False
    requires_ground_truth: bool = False
    notes: tuple[str, ...] = ()


class RunEnvironment(HarnessModel):
    """Execution environment details stored in the run manifest."""

    python_version: str
    platform: str
    hostname: str


class RunManifest(HarnessModel):
    """Top-level run manifest contract."""

    tool_version: str
    run_name: str
    run_dir: str
    status: Literal["completed", "failed", "partial"]
    started_at: str
    finished_at: str
    resolved_config_sha256: str
    model: ResolvedModelConfig
    execution: ResolvedExecutionConfig
    output: ResolvedOutputConfig
    threshold: float = Field(ge=0.0, le=1.0)
    datasets: list[DatasetMetadata] = Field(default_factory=list)
    adapter_capabilities: AdapterCapabilities
    environment: RunEnvironment
    warnings: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
