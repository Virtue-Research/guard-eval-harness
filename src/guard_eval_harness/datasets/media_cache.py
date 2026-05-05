"""Content-addressed media caching utilities."""

from __future__ import annotations

import hashlib
import mimetypes
import shutil
import uuid
import wave
from pathlib import Path
from typing import Any

_DEFAULT_CACHE_DIR = Path.home() / ".cache" / "guard-eval-harness" / "media"


def default_cache_dir() -> Path:
    """Return the default content-addressed media cache directory."""
    return _DEFAULT_CACHE_DIR


def compute_sha256(path: Path) -> str:
    """Compute the SHA-256 hex digest for a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def image_dimensions(path: Path) -> tuple[int, int]:
    """Return ``(width, height)`` for an image file.

    Requires Pillow at runtime.  The import is deferred so that
    text-only usage never needs PIL installed.
    """
    try:
        from PIL import Image  # type: ignore[import-untyped]
    except ImportError as exc:
        raise ImportError(
            "Pillow is required for image support: "
            "pip install Pillow"
        ) from exc
    with Image.open(path) as img:
        return img.size  # (width, height)


def audio_metadata(
    path: Path,
) -> tuple[float | None, int | None, int | None, str | None]:
    """Return ``(duration_seconds, sample_rate_hz, channels, mime_type)``."""
    mime_type, _encoding = mimetypes.guess_type(path.name)

    try:
        import soundfile as sf  # type: ignore[import-untyped]
    except ImportError:
        sf = None

    if sf is not None:
        info = sf.info(path.as_posix())
        duration_seconds: float | None = None
        if info.frames and info.samplerate:
            duration_seconds = float(info.frames) / float(info.samplerate)
        return (
            duration_seconds,
            int(info.samplerate) if info.samplerate else None,
            int(info.channels) if info.channels else None,
            mime_type,
        )

    if path.suffix.lower() == ".wav":
        with wave.open(path.as_posix(), "rb") as handle:
            frame_count = handle.getnframes()
            sample_rate_hz = handle.getframerate()
            channels = handle.getnchannels()
        duration_seconds = None
        if frame_count and sample_rate_hz:
            duration_seconds = float(frame_count) / float(sample_rate_hz)
        return (
            duration_seconds,
            int(sample_rate_hz) if sample_rate_hz else None,
            int(channels) if channels else None,
            mime_type or "audio/wav",
        )

    raise ImportError(
        "Audio support requires soundfile for non-WAV formats: "
        "pip install soundfile"
    )


def load_audio_waveform(
    path: Path,
    *,
    target_sample_rate: int | None = None,
) -> tuple[Any, int]:
    """Load one audio file as a mono float waveform plus sample rate."""
    try:
        import numpy as np
        import soundfile as sf  # type: ignore[import-untyped]
    except ImportError as exc:
        raise ImportError(
            "Audio support requires numpy and soundfile: "
            "pip install numpy soundfile"
        ) from exc

    waveform, sample_rate_hz = sf.read(
        path.as_posix(),
        always_2d=False,
        dtype="float32",
    )
    if getattr(waveform, "ndim", 1) > 1:
        waveform = np.mean(waveform, axis=1)

    resolved_sample_rate = int(sample_rate_hz)
    if (
        target_sample_rate is not None
        and resolved_sample_rate != int(target_sample_rate)
    ):
        try:
            import librosa  # type: ignore[import-untyped]
        except ImportError as exc:
            raise ImportError(
                "Resampling audio requires librosa: pip install librosa"
            ) from exc
        waveform = librosa.resample(
            waveform,
            orig_sr=resolved_sample_rate,
            target_sr=int(target_sample_rate),
        )
        resolved_sample_rate = int(target_sample_rate)

    return waveform, resolved_sample_rate


def resolve_pil_image(
    image: Any,
    cache_dir: Path | None = None,
) -> tuple[Path, str]:
    """Persist a PIL Image to the content-addressed cache.

    Returns ``(local_path, sha256)``.
    """
    try:
        from PIL import Image  # type: ignore[import-untyped]
    except ImportError as exc:
        raise ImportError(
            "Pillow is required for image support: "
            "pip install Pillow"
        ) from exc

    if not isinstance(image, Image.Image):
        raise TypeError(
            f"Expected PIL.Image.Image, got {type(image).__name__}"
        )

    cache = cache_dir or default_cache_dir()
    cache.mkdir(parents=True, exist_ok=True)

    # Save to a temporary file first, compute hash, then rename.
    tmp_path = cache / f"_tmp_{uuid.uuid4().hex}.png"
    try:
        image.save(tmp_path, format="PNG")
    except OSError as exc:
        if "cannot write mode" not in str(exc):
            raise
        tmp_path.unlink(missing_ok=True)
        # Fall back for modes that Pillow cannot encode as PNG directly.
        target_mode = "RGBA" if "A" in image.getbands() else "RGB"
        image.convert(target_mode).save(tmp_path, format="PNG")
    digest = compute_sha256(tmp_path)
    final_path = cache / f"{digest}.png"
    if not final_path.exists():
        shutil.move(str(tmp_path), str(final_path))
    else:
        tmp_path.unlink(missing_ok=True)
    return final_path, digest


def resolve_image_bytes(
    payload: bytes | bytearray,
    cache_dir: Path | None = None,
) -> tuple[Path, str]:
    """Persist encoded image bytes to the content-addressed cache."""
    if not isinstance(payload, (bytes, bytearray)):
        raise TypeError(
            f"Expected image bytes, got {type(payload).__name__}"
        )

    cache = cache_dir or default_cache_dir()
    cache.mkdir(parents=True, exist_ok=True)

    suffix = _image_suffix(bytes(payload))
    tmp_path = cache / f"_tmp_{uuid.uuid4().hex}{suffix}"
    try:
        tmp_path.write_bytes(bytes(payload))
        digest = compute_sha256(tmp_path)
        final_path = cache / f"{digest}{suffix}"
        if not final_path.exists():
            shutil.move(str(tmp_path), str(final_path))
        else:
            tmp_path.unlink(missing_ok=True)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise
    return final_path, digest


def _image_suffix(payload: bytes) -> str:
    if payload.startswith(b"\x89PNG\r\n\x1a\n"):
        return ".png"
    if payload.startswith(b"\xff\xd8\xff"):
        return ".jpg"
    if payload.startswith((b"GIF87a", b"GIF89a")):
        return ".gif"
    if payload.startswith(b"RIFF") and payload[8:12] == b"WEBP":
        return ".webp"
    if payload.startswith(b"BM"):
        return ".bmp"
    if payload.startswith((b"II*\x00", b"MM\x00*")):
        return ".tiff"
    return ".img"


def resolve_local_image(
    path: Path,
    cache_dir: Path | None = None,
) -> tuple[Path, str]:
    """Compute the SHA-256 for a local image file.

    If the file is outside the cache directory it is **not** copied.
    Returns ``(path, sha256)``.
    """
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")
    digest = compute_sha256(path)
    return path, digest


def resolve_local_audio(
    path: Path,
    cache_dir: Path | None = None,
) -> tuple[Path, str]:
    """Compute the SHA-256 for a local audio file."""
    if not path.exists():
        raise FileNotFoundError(f"Audio not found: {path}")
    digest = compute_sha256(path)
    return path, digest
