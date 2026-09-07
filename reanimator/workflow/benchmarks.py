"""What this machine has actually measured. Local state, never distributed.

A benchmark is a fact about one computer. Shipping it inside a template's
manifest makes it look like a property of the template, and the next person to
install that template on different hardware inherits a number nobody measured
there. So the manifest declares what is *stable* -- which weight formats exist,
what they weigh, where to get them -- and everything empirical lives here, in
``<comfy user dir>/reanimator/benchmarks.json``.

Recorded for diagnosis only. It deliberately does NOT choose the checkpoint:
ranking variants by these rows once demoted the one that had just proved
faster, because a cold run's total seconds got compared against a warm run's
seconds-per-step. Selection is static (templates.select_checkpoint) until
there is data from more than one machine.

The key is the full combination, because every part of it has been observed to
move the numbers: the GPU model and its compute capability decide which ops run
natively, the ComfyUI and torch builds decide how those ops are dispatched
(0.28.3 turned into 0.29.2 by itself mid-session), and quality, resolution and
picture count decide how much work there is to do.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from typing import Any, Mapping

from .. import config

log = logging.getLogger("reanimator.bridge")

FILENAME = "benchmarks.json"
MAX_ROWS = 500

# Everything that has to match for a stored measurement to describe this run.
KEY_FIELDS = (
    "templateId", "checkpointVariant", "device", "computeCapability",
    "comfyui", "torch", "quality", "modelResolution", "pictures",
)

_lock = threading.RLock()
_cache: list[dict[str, Any]] | None = None


def path():
    return config.state_dir() / FILENAME


def _load() -> list[dict[str, Any]]:
    global _cache
    with _lock:
        if _cache is not None:
            return _cache
        rows: list[dict[str, Any]] = []
        p = path()
        if p.exists():
            try:
                # utf-8-sig for the same reason config.py uses it.
                data = json.loads(p.read_text(encoding="utf-8-sig"))
                if isinstance(data, dict):
                    data = data.get("runs") or []
                rows = [r for r in data if isinstance(r, dict)]
            except (OSError, ValueError) as exc:
                log.error("Could not read %s (%s); starting a fresh benchmark log", p, exc)
        _cache = rows
        return rows


def reset_cache() -> None:
    global _cache
    with _lock:
        _cache = None


def key(**fields: Any) -> dict[str, Any]:
    return {f: fields.get(f) for f in KEY_FIELDS}


def _same(row: Mapping[str, Any], wanted: Mapping[str, Any]) -> bool:
    """A field the caller did not specify is a wildcard; a field it did
    specify must match exactly. Nothing is inferred from 'close enough'."""
    return all(
        value is None or str(row.get(field)) == str(value)
        for field, value in wanted.items()
    )


def record(entry: Mapping[str, Any]) -> None:
    """Append one measurement. Never blocks a run: a benchmark that cannot be
    written is a lost data point, not a failed generation."""
    row = dict(entry)
    row.setdefault("timestamp", int(time.time()))
    try:
        with _lock:
            rows = _load()
            rows.append(row)
            del rows[:-MAX_ROWS]
            tmp = path().with_suffix(".tmp")
            tmp.write_text(json.dumps({"runs": rows}, indent=2), encoding="utf-8")
            tmp.replace(path())
    except Exception:
        log.warning("Could not store benchmark", exc_info=True)


def find(**wanted: Any) -> list[dict[str, Any]]:
    """Successful measurements matching every field given, newest first."""
    rows = [r for r in _load() if r.get("success") and _same(r, wanted)]
    return sorted(rows, key=lambda r: r.get("timestamp") or 0, reverse=True)
