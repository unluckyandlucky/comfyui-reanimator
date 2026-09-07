"""Reanimator Bridge — let reanimator.app drive this machine's ComfyUI.

Nothing in this package uploads user media anywhere. See
docs/plan-local-bridge.md for the architecture and threat model.
"""

from .capabilities import BRIDGE_VERSION, PROTOCOL_VERSION

__all__ = ["BRIDGE_VERSION", "PROTOCOL_VERSION"]
