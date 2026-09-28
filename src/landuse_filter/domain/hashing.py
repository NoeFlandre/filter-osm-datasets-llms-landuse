"""Content addressing shared by every table the pipeline writes."""

import hashlib


def sha256_text(text: str) -> str:
    """Hex sha256 of the exact UTF-8 bytes of ``text`` (no normalisation)."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_parts(*parts: str) -> str:
    """Hex sha256 of ``parts`` joined by a separator no part can contain."""
    return sha256_text("\x1f".join(parts))
