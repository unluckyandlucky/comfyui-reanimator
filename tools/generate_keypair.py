"""Generate the Ed25519 keypair that signs pairing assertions.

    python comfyui-reanimator/tools/generate_keypair.py --kid prod-2026-07

Prints two values in the two formats they are actually consumed in:

  * the PRIVATE key as base64 PKCS8, for `crypto.subtle.importKey("pkcs8", ...)`
    in the Worker;
  * the PUBLIC key as base64 raw 32 bytes, for `Ed25519PublicKey.from_public_bytes`
    in the bridge.

The private key is printed once and never written to disk. Put it straight into
the Worker secret store:

    npx wrangler secret put BRIDGE_SIGNING_KEY

It must never be committed. The public half is not secret and belongs in
reanimator/keys.py, shipped inside every bridge release.

Rotation: add the new key to TRUSTED_KEYS alongside the old one, release, wait
for adoption, switch BRIDGE_SIGNING_KID, then mark the old key retired=True in a
later release. Skipping the overlap turns rotation into a forced upgrade for
every user (docs/plan-local-bridge.md §11).
"""

from __future__ import annotations

import argparse
import base64
import sys

try:
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
except ImportError:
    sys.exit("This tool needs 'cryptography': pip install cryptography")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--kid",
        default="prod-2026-07",
        help="Key identifier, carried in the JWS header (default: %(default)s)",
    )
    args = parser.parse_args()

    private = Ed25519PrivateKey.generate()

    pkcs8 = private.private_bytes(
        encoding=serialization.Encoding.DER,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    )
    raw_public = private.public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )

    print(f"\n  Key id: {args.kid}\n")
    print("=" * 72)
    print("PRIVATE  ->  Worker secret. Shown once, never stored.")
    print("=" * 72)
    print("\n  npx wrangler secret put BRIDGE_SIGNING_KEY")
    print("  npx wrangler secret put BRIDGE_SIGNING_KID")
    print(f"\nBRIDGE_SIGNING_KEY={base64.b64encode(pkcs8).decode()}")
    print(f"BRIDGE_SIGNING_KID={args.kid}\n")

    print("=" * 72)
    print("PUBLIC  ->  reanimator/keys.py, shipped with the bridge.")
    print("=" * 72)
    print(
        f"""
    PublicKey(
        key_id="{args.kid}",
        raw=_b64("{base64.b64encode(raw_public).decode()}"),
    ),
"""
    )

    # Prove the pair round-trips before anyone wires it into two codebases.
    signature = private.sign(b"reanimator-selftest")
    private.public_key().verify(signature, b"reanimator-selftest")
    print("Self-test: sign/verify OK.\n")


if __name__ == "__main__":
    main()
