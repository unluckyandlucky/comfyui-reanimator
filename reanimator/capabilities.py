"""What this machine can do, reported to the editor after pairing.

Everything here is describing hardware and software versions. No user content.
"""

from __future__ import annotations

import platform
import sys
from typing import Any

from . import config, projects

BRIDGE_VERSION = "0.1.4"
PROTOCOL_VERSION = 1

# Commands this build understands. The editor reads this and degrades instead of
# erroring when it is talking to an older bridge.
SUPPORTED_COMMANDS: tuple[str, ...] = (
    "get_capabilities",
    "list_templates",
    "validate",
    "put_input",
    "run",
    "interrupt",
    "read_output",
    "project_storage",
)


def _comfyui_version() -> str | None:
    try:
        from comfy.cli_args import args  # noqa: F401  (import proves we are inside ComfyUI)
    except Exception:
        return None
    for module_name, attr in (
        ("comfyui_version", "__version__"),
        ("comfy.model_management", "__version__"),
    ):
        try:
            module = __import__(module_name, fromlist=[attr])
            value = getattr(module, attr, None)
            if value:
                return str(value)
        except Exception:
            continue
    return "unknown"


def _devices() -> list[dict[str, Any]]:
    try:
        import torch
    except ImportError:
        return []

    devices: list[dict[str, Any]] = []
    try:
        if torch.cuda.is_available():
            for index in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(index)
                free, total = torch.cuda.mem_get_info(index)
                devices.append(
                    {
                        "index": index,
                        "name": props.name,
                        "backend": "cuda",
                        "vramTotal": int(total),
                        "vramFree": int(free),
                    }
                )
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            devices.append({"index": 0, "name": "Apple Silicon", "backend": "mps"})
    except Exception:
        # Never let hardware probing break the endpoint.
        pass
    return devices


def _torch_version() -> str | None:
    try:
        import torch

        return str(torch.__version__)
    except ImportError:
        return None


def hello() -> dict[str, Any]:
    """Unauthenticated. Discovery only -- must not leak anything identifying.

    ``bridgeInstanceId`` is included because the browser needs it to ask the
    backend for an assertion whose audience matches this install. It is a random
    per-install value, not derived from hardware or user, so it identifies the
    bridge to its own pairing flow and nothing else.
    """
    return {
        "service": "reanimator-bridge",
        "bridgeVersion": BRIDGE_VERSION,
        "protocol": PROTOCOL_VERSION,
        "bridgeInstanceId": config.bridge_instance_id(),
        "paired": bool(config.load().get("paired")),
    }


def _templates() -> list[dict[str, Any]]:
    from .workflow import templates

    try:
        return templates.summaries()
    except Exception:
        # A broken template must not make the device look disconnected.
        return []


def _local_project_storage() -> dict[str, Any]:
    """Never let a storage problem look like a disconnected device.

    A raised exception here would take the whole capabilities response down,
    and the editor would report BRIDGE OFFLINE for a machine whose only problem
    is a full disk. The block itself says what is wrong.
    """
    try:
        return projects.capability()
    except Exception as exc:  # pragma: no cover - defensive
        return {
            "contract": projects.CONTRACT,
            "available": False,
            "root": None,
            "writable": False,
            "freeBytes": None,
            "reason": f"Local project storage could not be checked: {exc}",
        }


def capabilities() -> dict[str, Any]:
    """Authenticated. The full picture, shown in the device selector."""
    cfg = config.load()
    return {
        "bridgeVersion": BRIDGE_VERSION,
        "protocol": PROTOCOL_VERSION,
        "commands": list(SUPPORTED_COMMANDS),
        "deviceLabel": cfg.get("device_label"),
        "os": f"{platform.system()} {platform.release()}",
        "python": sys.version.split()[0],
        "torch": _torch_version(),
        "comfyuiVersion": _comfyui_version(),
        "devices": _devices(),
        "templates": _templates(),
        # Declared, and only ever from a real test write. An editor that finds
        # this key missing must treat the machine as having no local storage --
        # see docs/contract-local-project-storage.md §1.
        "localProjectStorage": _local_project_storage(),
    }
