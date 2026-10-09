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

import re
from dataclasses import dataclass
from pathlib import Path

from aiohttp import web

from . import config

CHUNK_SIZE = 256 * 1024


_RANGE_RE = re.compile(r"^bytes=(\d*)-(\d*)$")


class MediaError(Exception):
    def __init__(self, message: str, status: int = 400) -> None:
        super().__init__(message)
        self.status = status


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
