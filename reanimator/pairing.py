"""Pairing: bridge-issued nonces, signed assertion verification, and tokens.

Threat model (docs/plan-local-bridge.md §4, §5): assume the cloud is hostile and
assume other local processes are hostile. Neither a valid assertion alone nor a
valid token alone is enough to start using someone's GPU -- a human has to click
Allow inside ComfyUI, on the machine itself.
"""

from __future__ import annotations

import secrets
import threading
import time
from dataclasses import dataclass, asdict
from typing import Any, Literal

from . import config, keys

NONCE_TTL_SECONDS = 120
APPROVAL_TTL_SECONDS = 180
TOKEN_TTL_SECONDS = 30 * 24 * 3600
RENEW_WINDOW_SECONDS = 7 * 24 * 3600     # renew silently within a week of expiry

ApprovalState = Literal["pending", "approved", "rejected", "expired"]


class PairingError(Exception):
    """Rejected pairing attempt. The message is safe to return to the caller."""

    def __init__(self, message: str, code: str = "pairing_failed") -> None:
        super().__init__(message)
        self.code = code


def _now() -> int:
    return int(time.time())


# --------------------------------------------------------------------------
# Nonces
# --------------------------------------------------------------------------

class NonceStore:
    """Single-use, short-lived nonces.

    The bridge issues these -- not the backend. A backend-issued nonce would let
    a captured assertion be replayed against the bridge later; one we issued and
    then consume cannot be replayed at all.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._issued: dict[str, int] = {}

    def issue(self) -> tuple[str, int]:
        nonce = secrets.token_urlsafe(24)
        expires = _now() + NONCE_TTL_SECONDS
        with self._lock:
            self._prune_locked()
            self._issued[nonce] = expires
        return nonce, expires

    def consume(self, nonce: str) -> None:
        with self._lock:
            self._prune_locked()
            expires = self._issued.pop(nonce, None)
        if expires is None:
            raise PairingError("Unknown or already-used nonce.", "bad_nonce")
        if expires < _now():
            raise PairingError("Nonce expired.", "bad_nonce")

    def _prune_locked(self) -> None:
        now = _now()
        for key in [k for k, v in self._issued.items() if v < now]:
            del self._issued[key]


# --------------------------------------------------------------------------
# Assertion verification
# --------------------------------------------------------------------------

ISSUER = "https://reanimator.app"
REQUIRED_CLAIMS = ("iss", "aud", "sub", "email", "origin", "nonce", "iat", "exp", "jti")
CLOCK_SKEW_SECONDS = 60
MAX_ASSERTION_LIFETIME = 300


class _JtiStore:
    """Assertion ids already seen. Belt and braces alongside the nonce."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._seen: dict[str, int] = {}

    def claim(self, jti: str, expires_at: int) -> None:
        with self._lock:
            now = _now()
            for key in [k for k, v in self._seen.items() if v < now]:
                del self._seen[key]
            if jti in self._seen:
                raise PairingError("Assertion already used.", "replayed")
            self._seen[jti] = expires_at


_jtis = _JtiStore()


def verify_assertion(token: str, nonce_store: NonceStore) -> dict[str, Any]:
    """Verify an EdDSA JWS pairing assertion and return its claims.

    Order matters. The signature is checked before any claim is trusted, and the
    nonce is consumed *last* so a malformed, misdirected or badly-signed attempt
    cannot burn the nonce the legitimate browser is still holding.
    """
    from . import jws

    if not isinstance(token, str) or not token:
        raise PairingError("Missing assertion.")

    try:
        header = jws.peek_header(token)      # untrusted: used only to pick the key
        key = keys.find(str(header.get("kid", "")))
        if key is None:
            raise PairingError(
                f"Unknown signing key {header.get('kid')!r}. Update Reanimator Bridge.",
                "unknown_key",
            )
        if keys.is_placeholder(key):
            raise PairingError(
                "This build carries a placeholder signing key and cannot pair. "
                "Install a released build of Reanimator Bridge.",
                "placeholder_key",
            )
        claims = jws.verify(token, key.raw)
    except jws.JwsError as exc:
        raise PairingError(str(exc), "bad_signature") from None

    missing = [c for c in REQUIRED_CLAIMS if c not in claims]
    if missing:
        raise PairingError(f"Assertion is missing: {', '.join(missing)}.")

    if claims["iss"] != ISSUER:
        raise PairingError(f"Unexpected issuer: {claims['iss']}.", "bad_issuer")

    # Binds the assertion to THIS bridge install, not merely to this nonce.
    if claims["aud"] != config.audience():
        raise PairingError("Assertion was not issued for this device.", "bad_audience")

    if claims["origin"] not in config.allowed_origins():
        raise PairingError(f"Origin not allowed: {claims['origin']}.", "bad_origin")

    now = _now()
    try:
        exp = int(claims["exp"])
        iat = int(claims["iat"])
    except (TypeError, ValueError):
        raise PairingError("Assertion has invalid timestamps.") from None
    if exp < now - CLOCK_SKEW_SECONDS:
        raise PairingError("Assertion expired.", "expired")
    if iat > now + CLOCK_SKEW_SECONDS:
        raise PairingError("Assertion is not yet valid.", "expired")
    if exp - iat > MAX_ASSERTION_LIFETIME:
        raise PairingError("Assertion lifetime is too long.", "expired")

    _jtis.claim(str(claims["jti"]), exp + CLOCK_SKEW_SECONDS)

    # Everything else has passed; only now is the nonce spent.
    nonce_store.consume(str(claims["nonce"]))
    return claims


# --------------------------------------------------------------------------
# Approvals awaiting a human click inside ComfyUI
# --------------------------------------------------------------------------

@dataclass
class Approval:
    request_id: str
    account_id: str
    email: str
    origin: str
    device_label: str
    created_at: int
    expires_at: int
    state: ApprovalState = "pending"
    token: str | None = None

    def public(self) -> dict[str, Any]:
        """What the ComfyUI panel shows in the Allow/Reject dialog."""
        return {
            "requestId": self.request_id,
            "email": self.email,
            "origin": self.origin,
            "device": self.device_label,
            "createdAt": self.created_at,
            "expiresAt": self.expires_at,
            "state": self.state,
        }


class ApprovalQueue:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._items: dict[str, Approval] = {}

    def create(self, claims: dict[str, Any], device_label: str) -> Approval:
        now = _now()
        approval = Approval(
            request_id=secrets.token_urlsafe(16),
            account_id=str(claims["sub"]),
            email=str(claims["email"]),
            origin=str(claims["origin"]),
            device_label=device_label,
            created_at=now,
            expires_at=now + APPROVAL_TTL_SECONDS,
        )
        with self._lock:
            self._expire_locked()
            self._items[approval.request_id] = approval
        return approval

    def get(self, request_id: str) -> Approval | None:
        with self._lock:
            self._expire_locked()
            return self._items.get(request_id)

    def pending(self) -> list[dict[str, Any]]:
        with self._lock:
            self._expire_locked()
            return [a.public() for a in self._items.values() if a.state == "pending"]

    def resolve(self, request_id: str, approved: bool) -> Approval:
        with self._lock:
            self._expire_locked()
            approval = self._items.get(request_id)
            if approval is None:
                raise PairingError("No such pairing request.", "unknown_request")
            if approval.state != "pending":
                raise PairingError(
                    f"Request already {approval.state}.", "already_resolved"
                )
            approval.state = "approved" if approved else "rejected"
            return approval

    def _expire_locked(self) -> None:
        now = _now()
        for key, item in list(self._items.items()):
            if item.state == "pending" and item.expires_at < now:
                item.state = "expired"
            # Keep resolved entries briefly so the browser can read the outcome.
            if item.expires_at + 300 < now:
                del self._items[key]


# --------------------------------------------------------------------------
# Tokens
# --------------------------------------------------------------------------

@dataclass
class PairedBrowser:
    token: str
    account_id: str
    email: str
    origin: str
    label: str
    created_at: int
    expires_at: int
    last_seen: int = 0

    def public(self) -> dict[str, Any]:
        """Never includes the token itself."""
        data = asdict(self)
        data.pop("token")
        data["tokenPrefix"] = self.token[:8]
        return data


class TokenStore:
    """Bearer tokens, persisted in the bridge config so pairing survives restart."""

    def __init__(self) -> None:
        self._lock = threading.Lock()

    def _read(self) -> list[dict[str, Any]]:
        return list(config.load().get("paired", []))

    def _write(self, rows: list[dict[str, Any]]) -> None:
        config.update(paired=rows)

    def issue(self, approval: Approval, label: str) -> PairedBrowser:
        now = _now()
        entry = PairedBrowser(
            token=secrets.token_urlsafe(32),
            account_id=approval.account_id,
            email=approval.email,
            origin=approval.origin,
            label=label,
            created_at=now,
            expires_at=now + TOKEN_TTL_SECONDS,
            last_seen=now,
        )
        with self._lock:
            rows = self._read()
            rows.append(asdict(entry))
            self._write(rows)
        return entry

    def validate(self, token: str | None, origin: str | None) -> PairedBrowser:
        if not token:
            raise PairingError("Missing bearer token.", "unauthenticated")
        with self._lock:
            rows = self._read()
            for row in rows:
                if not secrets.compare_digest(row["token"], token):
                    continue
                entry = PairedBrowser(**row)
                if entry.expires_at < _now():
                    raise PairingError("Token expired.", "token_expired")
                if origin is not None and entry.origin != origin:
                    # Distinct from "bad_origin": the bridge accepts this origin
                    # fine, the token just belongs to a different one. Telling
                    # the user the bridge is refusing them would be a lie, and
                    # the fix is to pair again, not to change any setting.
                    raise PairingError(
                        f"This token was issued for {entry.origin}, "
                        f"not {origin}. Pair again.",
                        "token_origin_mismatch",
                    )
                entry.last_seen = _now()
                row["last_seen"] = entry.last_seen
                self._write(rows)
                return entry
        raise PairingError("Unknown token.", "unauthenticated")

    def renew(self, token: str) -> PairedBrowser:
        with self._lock:
            rows = self._read()
            for row in rows:
                if not secrets.compare_digest(row["token"], token):
                    continue
                entry = PairedBrowser(**row)
                if entry.expires_at + RENEW_WINDOW_SECONDS < _now():
                    raise PairingError(
                        "Token too old to renew. Pair again.", "token_expired"
                    )
                entry.expires_at = _now() + TOKEN_TTL_SECONDS
                row["expires_at"] = entry.expires_at
                self._write(rows)
                return entry
        raise PairingError("Unknown token.", "unauthenticated")

    def revoke(self, *, token: str | None = None, prefix: str | None = None) -> bool:
        """Revoke by full token (from the browser) or by prefix (from the panel)."""
        with self._lock:
            rows = self._read()
            kept = [
                r
                for r in rows
                if not (
                    (token is not None and secrets.compare_digest(r["token"], token))
                    or (prefix is not None and r["token"].startswith(prefix))
                )
            ]
            if len(kept) == len(rows):
                return False
            self._write(kept)
            return True

    def list(self) -> list[dict[str, Any]]:
        return [PairedBrowser(**r).public() for r in self._read()]


nonces = NonceStore()
approvals = ApprovalQueue()
tokens = TokenStore()
