"""Content-addressed media caching utilities (image-only)."""

import hashlib
import shutil
import uuid
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


