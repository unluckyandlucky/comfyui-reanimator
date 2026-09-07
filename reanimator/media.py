"""Local media: path confinement, opaque capability URLs, and Range streaming.

Two rules govern this module.

**A filesystem path never appears in a URL.** ``/media?path=C:\\Users\\...`` would
be a traversal hole and an accidental disk browser. Instead the bridge grants an
opaque, unguessable, read-only, revocable capability id bound to exactly one
already-validated file.

**The capability id is the credential for media reads.** ``<video src>`` cannot
carry an ``Authorization`` header, so putting the general bearer token in the URL
would leak it into history, logs and referrers. A capability is scoped to a
single file, so leaking one costs one file, not the account.
"""

from __future__ import annotations

import mimetypes
import re
import secrets
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from aiohttp import web

from . import config

CHUNK_SIZE = 256 * 1024
CAPABILITY_TTL_SECONDS = 12 * 3600

VIDEO_SUFFIXES = frozenset({".mp4", ".mov", ".webm", ".mkv", ".m4v", ".avi"})
IMAGE_SUFFIXES = frozenset({".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"})
AUDIO_SUFFIXES = frozenset({".wav", ".mp3", ".m4a", ".aac", ".flac", ".ogg"})
ALLOWED_SUFFIXES = VIDEO_SUFFIXES | IMAGE_SUFFIXES | AUDIO_SUFFIXES

_RANGE_RE = re.compile(r"^bytes=(\d*)-(\d*)$")


class MediaError(Exception):
    def __init__(self, message: str, status: int = 400) -> None:
        super().__init__(message)
        self.status = status


# --------------------------------------------------------------------------
# Path confinement
# --------------------------------------------------------------------------

def project_root() -> Path | None:
    raw = config.load().get("project_root")
    if not raw:
        return None
    try:
        return Path(raw).resolve(strict=True)
    except (OSError, RuntimeError):
        return None


def resolve_within(root: Path, relative: str) -> Path:
    """Resolve ``relative`` under ``root``, refusing anything that escapes.

    ``Path.resolve()`` follows symlinks, so comparing the *resolved* candidate
    against the *resolved* root closes symlink escape as well as ``..``.
    """
    if not relative or relative.startswith(("/", "\\")) or ":" in relative:
        raise MediaError("Invalid media reference.", 400)

    candidate = (root / relative).resolve()
    root_resolved = root.resolve()
    if candidate != root_resolved and root_resolved not in candidate.parents:
        raise MediaError("Media reference escapes the project root.", 403)
    if not candidate.is_file():
        raise MediaError("No such file.", 404)
    if candidate.suffix.lower() not in ALLOWED_SUFFIXES:
        raise MediaError(f"File type not allowed: {candidate.suffix}", 415)
    return candidate


def list_media(root: Path, limit: int = 2000) -> list[dict[str, Any]]:
    """Media files under the authorized root. Relative names only, no paths."""
    entries: list[dict[str, Any]] = []
    root_resolved = root.resolve()
    for path in sorted(root.rglob("*")):
        if len(entries) >= limit:
            break
        try:
            if not path.is_file() or path.suffix.lower() not in ALLOWED_SUFFIXES:
                continue
            resolved = path.resolve()
            if root_resolved not in resolved.parents:
                continue  # a symlink pointing outside the root
            stat = resolved.stat()
        except OSError:
            continue
        entries.append(
            {
                "name": path.name,
                "ref": path.relative_to(root).as_posix(),
                "bytes": stat.st_size,
                "modified": int(stat.st_mtime),
                "kind": _kind(path.suffix.lower()),
            }
        )
    return entries


def _kind(suffix: str) -> str:
    if suffix in VIDEO_SUFFIXES:
        return "video"
    if suffix in IMAGE_SUFFIXES:
        return "image"
    return "audio"


# --------------------------------------------------------------------------
# Capabilities
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Capability:
    id: str
    path: Path
    mime: str
    size: int
    expires_at: int


class CapabilityStore:
    """In-memory only: capabilities die with the process, which is correct.

    They are per-session read handles, not persistent grants.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._items: dict[str, Capability] = {}

    def grant(self, path: Path) -> Capability:
        stat = path.stat()
        mime = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        cap = Capability(
            id=secrets.token_urlsafe(32),
            path=path,
            mime=mime,
            size=stat.st_size,
            expires_at=int(time.time()) + CAPABILITY_TTL_SECONDS,
        )
        with self._lock:
            self._prune_locked()
            self._items[cap.id] = cap
        return cap

    def resolve(self, capability_id: str) -> Capability:
        with self._lock:
            self._prune_locked()
            cap = self._items.get(capability_id)
        if cap is None:
            # Deliberately indistinguishable from "expired": do not confirm that
            # an id ever existed.
            raise MediaError("Unknown or expired media reference.", 404)
        return cap

    def revoke(self, capability_id: str) -> bool:
        with self._lock:
            return self._items.pop(capability_id, None) is not None

    def revoke_all(self) -> int:
        with self._lock:
            count = len(self._items)
            self._items.clear()
            return count

    def _prune_locked(self) -> None:
        now = int(time.time())
        for key in [k for k, v in self._items.items() if v.expires_at < now]:
            del self._items[key]


capabilities = CapabilityStore()


# --------------------------------------------------------------------------
# Range-aware streaming
# --------------------------------------------------------------------------

def parse_range(header: str, size: int) -> tuple[int, int] | None:
    """Parse a single byte range. Returns (start, end) inclusive, or None.

    Raises MediaError(416) when the range is syntactically valid but
    unsatisfiable, which is what ``<video>`` needs to recover cleanly.
    """
    match = _RANGE_RE.match(header.strip())
    if not match:
        return None  # multi-range or malformed: serve the whole thing instead

    first, last = match.group(1), match.group(2)
    if not first and not last:
        return None

    if not first:                      # suffix range: last N bytes
        length = int(last)
        if length == 0:
            raise MediaError("Unsatisfiable range.", 416)
        start = max(0, size - length)
        end = size - 1
    else:
        start = int(first)
        end = int(last) if last else size - 1

    if start >= size or start > end:
        raise MediaError("Unsatisfiable range.", 416)
    return start, min(end, size - 1)


async def serve(request: web.Request, cap: Capability) -> web.StreamResponse:
    """Stream a capability, honouring Range with 206 / Content-Range / 416."""
    try:
        size = cap.path.stat().st_size
    except OSError:
        raise MediaError("File is no longer available.", 410) from None

    range_header = request.headers.get("Range")
    span: tuple[int, int] | None = None
    if range_header:
        try:
            span = parse_range(range_header, size)
        except MediaError as exc:
            if exc.status == 416:
                return web.Response(
                    status=416,
                    headers={
                        "Content-Range": f"bytes */{size}",
                        "Accept-Ranges": "bytes",
                    },
                )
            raise

    headers = {
        "Accept-Ranges": "bytes",
        "Content-Type": cap.mime,
        "Cache-Control": "no-store",
        "X-Content-Type-Options": "nosniff",
    }

    if span is None:
        start, end, status = 0, size - 1, 200
    else:
        start, end = span
        status = 206
        headers["Content-Range"] = f"bytes {start}-{end}/{size}"

    length = end - start + 1
    headers["Content-Length"] = str(length)

    # CORS goes on BEFORE prepare(), not by the middleware.
    #
    # prepare() writes the status line and headers to the socket. Anything the
    # middleware sets afterwards is mutating an object whose headers have
    # already gone out, so it changes nothing on the wire. A Python client never
    # notices; a browser refuses the response and reports "Failed to fetch",
    # which is how a generation that had already finished on the GPU looked like
    # the bridge being unreachable.
    #
    # The same omission would taint the canvas for <video crossorigin>, so every
    # media read needs this, not just result downloads.
    origin = request.headers.get("Origin")
    if origin and origin in config.allowed_origins():
        headers["Access-Control-Allow-Origin"] = origin
        headers["Access-Control-Expose-Headers"] = (
            "Content-Range, Accept-Ranges, Content-Length"
        )
        headers["Vary"] = "Origin"

    response = web.StreamResponse(status=status, headers=headers)
    await response.prepare(request)

    if request.method == "HEAD":
        return response

    remaining = length
    with cap.path.open("rb") as handle:
        handle.seek(start)
        while remaining > 0:
            chunk = handle.read(min(CHUNK_SIZE, remaining))
            if not chunk:
                break
            await response.write(chunk)
            remaining -= len(chunk)
    await response.write_eof()
    return response
