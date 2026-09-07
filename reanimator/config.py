"""Persistent bridge configuration, stored inside ComfyUI's user directory.

Everything here is local to the machine and never leaves it.
"""

from __future__ import annotations

import json
import logging
import os
import threading
from pathlib import Path
from typing import Any

# The browser-facing loopback server. Deliberately NOT 8765 -- legacy serve.py
# uses that port and the two must be able to run side by side.
DEFAULT_PORT = 8771
FALLBACK_PORTS = (8772, 8773)

# The only web origin allowed to talk to the bridge.
ALLOWED_ORIGIN = "https://reanimator.app"

# Extra origins accepted only when REANIMATOR_BRIDGE_DEV=1, for local web development.
DEV_ORIGINS = ("http://localhost:3000", "http://127.0.0.1:3000")

CONFIG_FILENAME = "bridge.json"

_lock = threading.RLock()


def dev_mode() -> bool:
    """True when localhost origins are accepted as well as reanimator.app.

    Two ways in. The environment variable is fine for a terminal launch, but
    ComfyUI Desktop starts the process from its own hub, where the variable is
    easily lost — so the panel toggle, which persists in the config, is the one
    that actually survives day to day.
    """
    if os.environ.get("REANIMATOR_BRIDGE_DEV") == "1":
        return True
    return bool(load().get("dev_origins"))


def allowed_origins() -> tuple[str, ...]:
    return (ALLOWED_ORIGIN, *DEV_ORIGINS) if dev_mode() else (ALLOWED_ORIGIN,)


def _comfy_user_dir() -> Path:
    """ComfyUI's user directory, falling back to a sibling of this package."""
    try:
        import folder_paths  # provided by ComfyUI at runtime

        base = Path(folder_paths.get_user_directory())
    except Exception:
        base = Path(__file__).resolve().parents[2] / "user"
    return base


def state_dir() -> Path:
    d = _comfy_user_dir() / "reanimator"
    d.mkdir(parents=True, exist_ok=True)
    return d


def config_path() -> Path:
    return state_dir() / CONFIG_FILENAME


_DEFAULTS: dict[str, Any] = {
    "port": DEFAULT_PORT,
    "device_label": None,       # defaults to the hostname on first read
    "bridge_instance_id": None,  # generated once, identifies THIS install
    "project_root": None,        # user-authorized folder for local projects
    "paired": [],                # see pairing.TokenStore
    "external_media": [],        # files registered from outside project_root
    "dev_origins": False,        # also accept localhost:3000 (development)
}


_cache: dict[str, Any] | None = None


def load() -> dict[str, Any]:
    """Cached in memory: allowed_origins() runs on every request, including each
    Range request while a video streams, and a file read per chunk is silly.
    save() is the only writer and invalidates the cache itself."""
    global _cache
    with _lock:
        if _cache is not None:
            return dict(_cache)

        path = config_path()
        data = dict(_DEFAULTS)
        if path.exists():
            try:
                # utf-8-sig, not utf-8: PowerShell's Out-File and Notepad both
                # write a BOM, and plain utf-8 would choke on it. Silently
                # falling back to defaults here means losing the user's pairings
                # and their device id -- and the symptom appears somewhere else
                # entirely, hours later.
                data.update(json.loads(path.read_text(encoding="utf-8-sig")))
            except (OSError, ValueError) as exc:
                # Never stop ComfyUI from starting, but never lose the file
                # quietly either: keep a copy and say so, loudly.
                logging.getLogger("reanimator.bridge").error(
                    "Could not read %s (%s). Falling back to defaults; the "
                    "previous file is kept as %s.bad",
                    path, exc, path,
                )
                try:
                    path.replace(path.with_suffix(".json.bad"))
                except OSError:
                    pass

        dirty = False
        if not data.get("device_label"):
            import socket

            data["device_label"] = socket.gethostname()
            dirty = True
        if not data.get("bridge_instance_id"):
            # Binds a pairing assertion to this specific install: an assertion
            # minted for another machine's bridge will fail the audience check.
            import secrets

            data["bridge_instance_id"] = secrets.token_urlsafe(18)
            dirty = True
        if dirty:
            save(data)          # save() refreshes the cache itself
        else:
            _cache = dict(data)
        return dict(data)


def bridge_instance_id() -> str:
    return str(load()["bridge_instance_id"])


def audience() -> str:
    return f"reanimator-bridge:{bridge_instance_id()}"


def reset_cache() -> None:
    """Drop the in-memory copy. Needed by tests, which repoint the state dir."""
    global _cache
    with _lock:
        _cache = None


def save(data: dict[str, Any]) -> None:
    """Atomic write, so an interrupted save cannot corrupt the config."""
    global _cache
    with _lock:
        _cache = dict(data)
        path = config_path()
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
        os.replace(tmp, path)
        try:
            os.chmod(path, 0o600)
        except OSError:
            pass  # Windows and some filesystems do not support this


def update(**fields: Any) -> dict[str, Any]:
    with _lock:
        data = load()
        data.update(fields)
        save(data)
        return data
