"""Routes for the approval UI *inside* ComfyUI.

These DO belong on PromptServer: the ComfyUI frontend is same-origin with it, so
there is no CORS involved and no configuration for the user. The separation is
deliberate --

    reanimator.app  ->  loopback server (server.py)   cross-origin, token-gated
    ComfyUI's UI    ->  PromptServer  (this module)   same-origin, local only

The Allow/Reject decision must happen here, on the machine, which is what makes
a stolen assertion useless on its own.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from aiohttp import web

from . import capabilities, config, media
from .pairing import PairingError, approvals, tokens

log = logging.getLogger("reanimator.bridge")


def _json(payload: dict[str, Any], status: int = 200) -> web.Response:
    return web.json_response(payload, status=status)


def register(prompt_server: Any, port_getter) -> None:
    """Attach the panel API to ComfyUI's own aiohttp app."""
    routes = prompt_server.routes

    @routes.get("/reanimator/panel/status")
    async def status(_: web.Request) -> web.Response:
        cfg = config.load()
        return _json(
            {
                "bridgeVersion": capabilities.BRIDGE_VERSION,
                "port": port_getter(),
                "deviceLabel": cfg.get("device_label"),
                "devOrigins": bool(cfg.get("dev_origins")),
                "allowedOrigins": list(config.allowed_origins()),
                "pending": approvals.pending(),
                "paired": tokens.list(),
            }
        )

    @routes.post("/reanimator/panel/resolve")
    async def resolve(request: web.Request) -> web.Response:
        """Allow or Reject a pending pairing request."""
        body = await request.json()
        request_id = body.get("requestId")
        approve = bool(body.get("approve"))
        if not isinstance(request_id, str):
            return _json({"error": "Missing requestId."}, status=400)
        try:
            approval = approvals.resolve(request_id, approve)
        except PairingError as exc:
            return _json({"error": str(exc), "code": exc.code}, status=400)
        log.info(
            "Pairing %s for %s", "approved" if approve else "rejected", approval.email
        )
        return _json({"state": approval.state})

    @routes.post("/reanimator/panel/revoke")
    async def revoke(request: web.Request) -> web.Response:
        """Revoke a paired browser from the ComfyUI side."""
        body = await request.json()
        prefix = body.get("tokenPrefix")
        if not isinstance(prefix, str) or len(prefix) < 6:
            return _json({"error": "Missing tokenPrefix."}, status=400)
        return _json({"revoked": tokens.revoke(prefix=prefix)})

    @routes.post("/reanimator/panel/dev-origins")
    async def set_dev_origins(request: web.Request) -> web.Response:
        """Accept localhost origins as well as reanimator.app.

        Persisted, unlike the REANIMATOR_BRIDGE_DEV environment variable, which
        ComfyUI Desktop drops depending on how the instance was launched.
        Pairing still needs approval here, so this widens who may *ask*, never
        who may connect unattended.
        """
        body = await request.json()
        enabled = bool(body.get("enabled"))
        config.update(dev_origins=enabled)
        log.info("Reanimator dev origins %s", "enabled" if enabled else "disabled")
        return _json({"devOrigins": enabled, "origins": list(config.allowed_origins())})

    @routes.get("/reanimator/panel/project-root")
    async def get_project_root(_: web.Request) -> web.Response:
        root = media.project_root()
        return _json(
            {
                "projectRoot": str(root) if root else None,
                "mediaCount": len(media.list_media(root)) if root else 0,
            }
        )

    @routes.post("/reanimator/panel/project-root")
    async def set_project_root(request: web.Request) -> web.Response:
        """Authorize a folder for local media.

        The path is typed here, in ComfyUI, and never travels from the browser --
        the browser cannot know a real path, and a native file dialog is not an
        option on a headless install.
        """
        body = await request.json()
        raw = str(body.get("path", "")).strip()
        if not raw:
            config.update(project_root=None)
            media.capabilities.revoke_all()
            return _json({"projectRoot": None})

        path = Path(raw).expanduser()
        try:
            resolved = path.resolve(strict=True)
        except (OSError, RuntimeError):
            return _json({"error": f"No such folder: {raw}"}, status=400)
        if not resolved.is_dir():
            return _json({"error": "Not a folder."}, status=400)

        config.update(project_root=str(resolved))
        # Old capabilities pointed into the previous root.
        media.capabilities.revoke_all()
        log.info("Reanimator project root set to %s", resolved)
        return _json(
            {"projectRoot": str(resolved), "mediaCount": len(media.list_media(resolved))}
        )

    @routes.post("/reanimator/panel/device-label")
    async def device_label(request: web.Request) -> web.Response:
        body = await request.json()
        label = str(body.get("label", "")).strip()
        if not 1 <= len(label) <= 64:
            return _json({"error": "Label must be 1-64 characters."}, status=400)
        config.update(device_label=label)
        return _json({"deviceLabel": label})
