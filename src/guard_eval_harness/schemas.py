"""Typed contracts shared across the harness."""

from typing import Annotated, Any, Literal, Sequence, Union

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

from guard_eval_harness.config import (
    PREDICT_METADATA_FIELD_DENYLIST,
)


Role = Literal["system", "user", "assistant", "tool"]


class HarnessModel(BaseModel):
    """Common pydantic base class for shared contracts."""

    model_config = ConfigDict(extra="forbid")


# --- Multimodal content types ---


class MediaRef(HarnessModel):
    """Reference to a resolved media artifact (image)."""

    modality: Literal["image"]
    uri: str = Field(min_length=1)
    sha256: str | None = None
    mime_type: str | None = None
    width: int | None = None
    height: int | None = None


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
    return part


class Message(HarnessModel):
    """Normalized message contract used across all datasets."""

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
                raise ValueError("message content must not be empty")
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
            p.text for p in self.content if isinstance(p, TextPart)
        )

    @property
    def image_refs(self) -> list[MediaRef]:
        """Extract image MediaRefs from content."""
        if isinstance(self.content, str):
            return []
        return [
            p.media
            for p in self.content
            if isinstance(p, MediaPart) and p.media.modality == "image"
        ]

    @property
    def media_refs(self) -> list[MediaRef]:
        """Extract all media refs from content."""
        if isinstance(self.content, str):
            return []
        return [
            p.media for p in self.content if isinstance(p, MediaPart)
        ]


class UnsafeLabel(HarnessModel):
    """Binary label contract for dataset normalization."""

    unsafe: bool


class PredictSample(HarnessModel):
    """Model-facing sample view. Ground truth is structurally absent."""

    id: str
    dataset: str
    split: str
    messages: list[Message] = Field(min_length=1)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("id", "dataset", "split")
    @classmethod
    def validate_identifier(cls, value: str) -> str:
        """Reject empty identifiers."""
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("value must not be empty")
        return cleaned

    @field_validator("metadata")
    @classmethod
    def validate_metadata(
        cls,
        value: dict[str, Any],
    ) -> dict[str, Any]:
        """Reject generic label-shaped metadata at the model boundary."""
        invalid = sorted(
            key
            for key in value
            if key in PREDICT_METADATA_FIELD_DENYLIST
        )
        if invalid:
            names = ", ".join(invalid)
            raise ValueError(
                "PredictSample metadata cannot include label-like fields: "
                f"{names}"
            )
        return value


class SampleGroundTruth(HarnessModel):
    """Score-side view of one sample, joined to predictions by sample id."""

    sample_id: str
    label: UnsafeLabel
    category_labels: tuple[str, ...] = ()

    @field_validator("sample_id")
    @classmethod
    def validate_sample_id(cls, value: str) -> str:
        """Reject empty sample identifiers."""
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

    def to_predict_sample(
        self,
        *,
        predict_metadata_fields: Sequence[str] = (),
    ) -> PredictSample:
        """Return the model-facing sample view with allowlisted metadata."""
        metadata = {
            key: self.metadata[key]
            for key in predict_metadata_fields
            if key in self.metadata
        }
        return PredictSample(
            id=self.id,
            dataset=self.dataset,
            split=self.split,
            messages=self.messages,
            metadata=metadata,
        )

    def to_ground_truth(self) -> SampleGroundTruth:
        """Return the score-side ground truth for metrics."""
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
    category_scores: dict[str, float] = Field(default_factory=dict)
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
