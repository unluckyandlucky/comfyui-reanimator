"""Reanimator's Ed25519 signing keys, embedded in the bridge release.

The bridge trusts a *list* of keys, not one, so a key can be rotated without
turning every existing install into a forced upgrade: publish a release that
carries both the new and the previous key, wait for adoption, then drop the old
one. See docs/plan-local-bridge.md §11.

Keys are public. Nothing secret belongs in this file.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass


@dataclass(frozen=True)
class PublicKey:
    key_id: str
    raw: bytes          # 32-byte Ed25519 public key
    retired: bool = False


def _b64(value: str) -> bytes:
    return base64.b64decode(value)


# The public half of the key Reanimator signs pairing assertions with. The
# private half lives in the Worker's secret store (BRIDGE_SIGNING_KEY) and never
# here; generate a new pair with tools/generate_keypair.py.
#
# BRIDGE_SIGNING_KID on the server has to name a key_id from this list, and the
# server's default is "dev-2026-07" -- a deployment that forgets the variable
# signs with an id no released bridge trusts, and every pairing fails with an
# unknown key while the server logs nothing wrong.
TRUSTED_KEYS: tuple[PublicKey, ...] = (
    PublicKey(
        key_id="prod-2026-07",
        raw=_b64("pkXhvqyg21y6Svn6B7QLSXiVs1GR7snlQSV67f6gfJI="),
    ),
)


def find(key_id: str) -> PublicKey | None:
    for key in TRUSTED_KEYS:
        if key.key_id == key_id and not key.retired:
            return key
    return None


def is_placeholder(key: PublicKey) -> bool:
    """True while the repo still carries the all-zero development key."""
    return key.raw == b"\x00" * 32
