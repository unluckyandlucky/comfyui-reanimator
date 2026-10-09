"""ComfyUI entry point for Reanimator Bridge.

ComfyUI imports this at startup. We register the panel API on its server and
start our own loopback server for the browser-facing API.

It is a service first. The one node it exports, ReanimatorH3Sequencer, exists
because the minimax-h3-keyframes template needs a variable number of keys and
a template cannot chain a variable number of nodes (see reanimator/nodes.py).
"""

from __future__ import annotations

import asyncio
import logging

from .reanimator.nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

log = logging.getLogger("reanimator.bridge")
WEB_DIRECTORY = "./web"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]

_runner = None
_port: int | None = None


def _current_port() -> int | None:
    return _port


async def _boot() -> None:
    global _runner, _port
    from .reanimator import server
    from .reanimator.workflow import geometry

    # Declared in pyproject, but what matters is whether it imports in the
    # interpreter ComfyUI is actually running. Without it the bridge cannot pad
    # or crop frames, and every generation would come back the wrong size --
    # better to say so once, at startup, than once per generation.
    if not geometry.pillow_available():
        log.error(
            "Reanimator Bridge: Pillow is missing from this Python environment. "
            "Frames cannot be padded or cropped, so results will come back at "
            "the model's resolution instead of the frame's. Install it with: "
            "python -m pip install Pillow"
        )

    try:
        _runner, _port = await server.start()
    except Exception:
        # A failed bridge must never prevent ComfyUI from starting.
        log.exception("Reanimator Bridge failed to start")


def _setup() -> None:
    try:
        from server import PromptServer  # ComfyUI
    except ImportError:
        log.warning("Reanimator Bridge: not running inside ComfyUI, skipping setup")
        return

    from .reanimator import panel_routes

    instance = PromptServer.instance
    panel_routes.register(instance, _current_port)

    loop = getattr(instance, "loop", None) or asyncio.get_event_loop()
    loop.create_task(_boot())


_setup()
