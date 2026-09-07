"""Minimal, strict JWS (compact serialization) verification for EdDSA.

Why JWS rather than signing canonicalised JSON: the signing input is the exact
ASCII of ``header.payload``, so Next.js and Python never have to agree on how to
serialise numbers, Unicode escaping or key order. We verify bytes we were given,
not bytes we re-derived.

Why hand-rolled rather than PyJWT: it avoids a dependency, and the entire attack
surface of a JWT library -- algorithm confusion, ``alg: none``, unhandled
``crit`` -- is closed here by a hard allowlist of exactly one algorithm. The
signature check itself uses ``cryptography``.

Rules enforced, in order:
  1. three segments, base64url without padding
  2. header ``alg`` is exactly "EdDSA"; anything else is refused outright
  3. no unrecognised ``crit`` header
  4. **signature verified before the payload is parsed as claims**
  5. claims validated by the caller
"""

from __future__ import annotations

import base64
import json
from typing import Any

ALGORITHM = "EdDSA"


class JwsError(Exception):
    """Malformed or unverifiable token. Safe to show to the caller."""


def b64url_decode(segment: str) -> bytes:
    padding = "=" * (-len(segment) % 4)
    try:
        return base64.urlsafe_b64decode(segment + padding)
    except Exception as exc:
        raise JwsError("Invalid base64url segment.") from exc


def peek_header(token: str) -> dict[str, Any]:
    """Read the header WITHOUT verifying anything.

    Only legitimate use: selecting the key by ``kid``. Never trust the result
    for an access decision.
    """
    parts = token.split(".")
    if len(parts) != 3:
        raise JwsError("Malformed token: expected three segments.")
    try:
        header = json.loads(b64url_decode(parts[0]))
    except ValueError as exc:
        raise JwsError("Header is not valid JSON.") from exc
    if not isinstance(header, dict):
        raise JwsError("Header is not an object.")
    return header


def verify(token: str, public_key: bytes) -> dict[str, Any]:
    """Verify an EdDSA JWS and return its claims."""
    try:
        from cryptography.exceptions import InvalidSignature
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
    except ImportError:
        raise JwsError(
            "The 'cryptography' package is required. Install it into ComfyUI's "
            "Python environment: pip install cryptography"
        ) from None

    parts = token.split(".")
    if len(parts) != 3:
        raise JwsError("Malformed token: expected three segments.")
    header_b64, payload_b64, signature_b64 = parts

    header = peek_header(token)
    if header.get("alg") != ALGORITHM:
        # Closes algorithm confusion and 'alg: none' in one line.
        raise JwsError(f"Unsupported algorithm: {header.get('alg')!r}.")
    if "crit" in header:
        raise JwsError("Unsupported 'crit' header.")

    signing_input = f"{header_b64}.{payload_b64}".encode("ascii")
    signature = b64url_decode(signature_b64)

    try:
        Ed25519PublicKey.from_public_bytes(public_key).verify(signature, signing_input)
    except InvalidSignature:
        raise JwsError("Signature verification failed.") from None
    except ValueError as exc:
        raise JwsError(f"Invalid public key: {exc}") from exc

    # Only now is it safe to look at the payload.
    try:
        claims = json.loads(b64url_decode(payload_b64))
    except ValueError as exc:
        raise JwsError("Payload is not valid JSON.") from exc
    if not isinstance(claims, dict):
        raise JwsError("Payload is not an object.")
    return claims
