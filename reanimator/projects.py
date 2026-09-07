"""Local project storage: the bytes of a project that never leave this machine.

Implements docs/contract-local-project-storage.md v1. The editor already speaks
this contract; until now nothing answered on this side, which is why Local GPU
could generate but could not keep anything it generated.

Two rules govern the module, and they are the same two the contract opens with.

**Absence of capability means "not ready", never "assume yes."** Every reason a
write could fail -- no folder, no permission, no disk -- is reported by
:func:`capability` *before* the user draws anything, and it is reported from an
actual write, not from inspecting permission bits.

**Saved means persisted and verified.** The sha256 is recomputed here from the
body that arrived, and the file is renamed into place only after it matches.
A truncated upload can never become a file the editor later calls saved.
"""

from __future__ import annotations

import hashlib
import os
import re
import secrets
import shutil
from pathlib import Path
from typing import Any, Iterable

from . import config

# Version of the contract document this implementation speaks. The editor
# refuses to write when its number and this one disagree, which is the correct
# reaction: better stopped and explained than writing on assumptions that do
# not match.
CONTRACT = 1

# Both ids come from the browser, so both are validated before anything touches
# the disk. UUIDs from Supabase carry hyphens, which this allows; what it does
# not allow is a dot, a slash or a drive letter, so no id can ever name a path.
ID_RE = re.compile(r"^[A-Za-z0-9_-]{8,64}$")

# Closed table. The extension is derived from the content type the bridge
# accepts and NEVER from anything the client names -- a filename from the
# browser is exactly how a .png upload becomes a .py on disk.
EXT_BY_MIME: dict[str, str] = {
    "image/png": ".png",
    "image/jpeg": ".jpg",
    "image/webp": ".webp",
    "video/mp4": ".mp4",
    "video/webm": ".webm",
}
KNOWN_EXTS = tuple(dict.fromkeys(EXT_BY_MIME.values()))
# Reading is the same closed table backwards, not mimetypes.guess_type(): the
# registry differs per machine and per Windows install, and a served type that
# disagrees with the one that was written is how an <img> silently stops
# decoding on one computer only.
MIME_BY_EXT: dict[str, str] = {}
for _mime, _ext in EXT_BY_MIME.items():
    MIME_BY_EXT.setdefault(_ext, _mime)

# Kept at or below the app-wide client_max_size in server.build_app(), which is
# runner.MAX_INPUT_BYTES + 1 MB. Above that aiohttp 413s the request before any
# handler runs, and the caller gets a limit nobody wrote down instead of the
# message below.
MAX_ASSET_BYTES = 64 * 1024 * 1024
MAX_VERIFY_REFS = 500


class StorageError(Exception):
    def __init__(self, message: str, status: int, code: str) -> None:
        super().__init__(message)
        self.status = status
        self.code = code


# --------------------------------------------------------------------------
# Where it lives
# --------------------------------------------------------------------------

def root() -> Path:
    """The folder that holds every locally stored project.

    Inside the bridge's own state directory by default, not the user's media
    folder: this is data the bridge owns and writes, whereas ``project_root``
    is a folder the user authorized us to *read*. Mixing them would make a
    generated result appear in the source listing as if the user had put it
    there.

    ``local_projects_root`` in bridge.json moves it, for an install whose user
    directory sits on a small system drive.
    """
    configured = config.load().get("local_projects_root")
    if configured:
        return Path(str(configured)).expanduser()
    return config.state_dir() / "projects"


def _ensure_root() -> Path:
    path = root()
    path.mkdir(parents=True, exist_ok=True)
    return path


def capability() -> dict[str, Any]:
    """What ``/rb/v1/capabilities`` reports about local project storage.

    ``writable`` comes from writing and deleting a real file, every time this
    is asked, not from a permission check done once at startup. A disk that
    filled up an hour ago has to show up here, in the panel the user reads
    before they trust the mode -- not in the first save that loses their work.
    """
    block: dict[str, Any] = {
        "contract": CONTRACT,
        "available": False,
        "root": None,
        "writable": False,
        "freeBytes": None,
        "reason": None,
    }
    try:
        path = _ensure_root()
    except OSError as exc:
        block["reason"] = f"The local project folder could not be created: {exc}"
        return block

    block["available"] = True
    block["root"] = str(path)
    try:
        block["freeBytes"] = int(shutil.disk_usage(path).free)
    except OSError:
        pass

    probe = path / f".writetest-{secrets.token_hex(6)}"
    try:
        probe.write_bytes(b"reanimator")
        probe.unlink()
    except OSError as exc:
        block["reason"] = f"The local project folder is not writable: {exc}"
        return block

    block["writable"] = True
    return block


# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------

def _checked(project_id: str, asset_id: str) -> tuple[str, str]:
    if not ID_RE.match(project_id or ""):
        raise StorageError("Invalid project id.", 422, "bad_project_id")
    if not ID_RE.match(asset_id or ""):
        raise StorageError("Invalid asset id.", 422, "bad_asset_id")
    return project_id, asset_id


def _within(base: Path, candidate: Path) -> Path:
    """Confinement, same rule as media.resolve_within.

    The id regexes already make traversal impossible, so this is not what stops
    ``..``; what it stops is a *symlinked* project folder quietly redirecting
    writes somewhere else on the disk.
    """
    base_resolved = base.resolve()
    resolved = candidate.resolve()
    if resolved != base_resolved and base_resolved not in resolved.parents:
        raise StorageError("Asset path escapes the project root.", 403, "outside_root")
    return resolved


def project_dir(project_id: str, create: bool = False) -> Path:
    base = _ensure_root() if create else root()
    directory = base / project_id
    if create:
        directory.mkdir(parents=True, exist_ok=True)
    return _within(base, directory) if directory.exists() else directory


def mime_of(path: Path) -> str:
    return MIME_BY_EXT.get(path.suffix.lower(), "application/octet-stream")


def find_asset(project_id: str, asset_id: str) -> Path | None:
    """The stored file for an id, whatever type it was written as.

    The editor addresses an asset by id alone -- it never learns, and never
    needs to learn, which extension the bridge chose from the content type.
    """
    _checked(project_id, asset_id)
    directory = root() / project_id
    for ext in KNOWN_EXTS:
        candidate = directory / f"{asset_id}{ext}"
        if candidate.is_file():
            return _within(root(), candidate)
    return None


# --------------------------------------------------------------------------
# Write, read, verify, delete
# --------------------------------------------------------------------------

def write(
    project_id: str, asset_id: str, content_type: str | None, data: bytes,
    declared_sha256: str | None,
) -> dict[str, Any]:
    """Persist one asset. Atomically, and only if the bytes hash as promised."""
    _checked(project_id, asset_id)

    mime = (content_type or "").split(";")[0].strip().lower()
    ext = EXT_BY_MIME.get(mime)
    if ext is None:
        raise StorageError(
            f"Unsupported content type: {mime or 'unknown'}", 415, "bad_media_type"
        )
    if len(data) > MAX_ASSET_BYTES:
        raise StorageError(
            f"Asset is larger than {MAX_ASSET_BYTES // (1024 * 1024)} MB.",
            413, "too_large",
        )

    digest = hashlib.sha256(data).hexdigest()
    # Recomputed, never trusted. Without this "saved" would mean "sent", and a
    # write cut off halfway would pass for a good file until the day someone
    # opened it.
    if declared_sha256 and declared_sha256.strip().lower() != digest:
        raise StorageError(
            "The bytes received do not match the sha256 that was declared.",
            409, "sha256_mismatch",
        )

    directory = project_dir(project_id, create=True)
    final = _within(root(), directory / f"{asset_id}{ext}")

    if final.is_file():
        try:
            if hashlib.sha256(final.read_bytes()).hexdigest() == digest:
                # Re-sending identical bytes is a no-op, not an error: the
                # editor retries, and a retry that fails would be worse than
                # the problem it is recovering from.
                return _stored(project_id, asset_id, final, digest, mime)
        except OSError:
            pass  # unreadable: just overwrite it below

    # Temp file plus rename. A bridge that dies mid-write cannot leave half a
    # PNG behind under a name the editor will later read as complete.
    tmp = directory / f".{asset_id}.{secrets.token_hex(4)}.part"
    try:
        tmp.write_bytes(data)
        os.replace(tmp, final)
    except OSError as exc:
        try:
            tmp.unlink()
        except OSError:
            pass
        if getattr(exc, "errno", None) == 28:      # ENOSPC
            free = None
            try:
                free = int(shutil.disk_usage(directory).free)
            except OSError:
                pass
            raise StorageError(
                "There is not enough space on this disk to save that file"
                + (f" ({free} bytes free)." if free is not None else "."),
                507, "no_space",
            ) from exc
        raise StorageError(f"Could not write the asset: {exc}", 500, "write_failed") from exc

    return _stored(project_id, asset_id, final, digest, mime)


def _stored(
    project_id: str, asset_id: str, path: Path, digest: str, mime: str
) -> dict[str, Any]:
    # `ref` is relative and informative. The editor addresses assets by id, so
    # it never sends nor receives an absolute path -- the same project opened
    # on another machine would find a Windows path meaningless anyway.
    return {
        "ref": f"{project_id}/{path.name}",
        "assetId": asset_id,
        "sha256": digest,
        "bytes": path.stat().st_size,
        "mime": mime,
    }


def verify(project_id: str, refs: Iterable[Any]) -> list[dict[str, Any]]:
    """Which of these assets are still on this disk, and still the right bytes.

    One call for the whole project: forty GETs just to find out what survived
    is a request storm and a loading bar nobody needs.
    """
    items = list(refs or [])
    if len(items) > MAX_VERIFY_REFS:
        raise StorageError(
            f"Too many refs in one call (limit {MAX_VERIFY_REFS}).", 422, "too_many_refs"
        )

    results: list[dict[str, Any]] = []
    for item in items:
        asset_id = str((item or {}).get("assetId") or "")
        wanted = str((item or {}).get("sha256") or "").strip().lower()
        if not ID_RE.match(asset_id):
            raise StorageError("Invalid asset id.", 422, "bad_asset_id")
        path = find_asset(project_id, asset_id)
        if path is None:
            results.append({"assetId": asset_id, "present": False,
                            "sha256Matches": False, "bytes": 0})
            continue
        try:
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            size = path.stat().st_size
        except OSError:
            results.append({"assetId": asset_id, "present": False,
                            "sha256Matches": False, "bytes": 0})
            continue
        results.append({
            "assetId": asset_id,
            "present": True,
            # No declared hash means the caller only asked whether it is there.
            "sha256Matches": (digest == wanted) if wanted else True,
            "sha256": digest,
            "bytes": size,
        })
    return results


# --------------------------------------------------------------------------
# Inventory and per-file deletion  (the orphan collector needs both)
#
# WHY THIS IS ABOUT FILES AND NOT ABOUT ASSET IDS. `abc123.png` and
# `abc123.webp` can both exist -- write() picks the extension from the content
# type, and find_asset() then returns whichever comes first in KNOWN_EXTS. A
# collector that reasoned in asset ids would see one live asset where there are
# two files, keep both, and if it ever deleted "the asset" it would take
# whichever file the extension order happened to favour. So the inventory is
# physical: one entry per file on disk, and deletion names the file it means.
# --------------------------------------------------------------------------

def list_files(project_id: str) -> list[dict[str, Any]]:
    """Every file this project owns, one entry per file on disk.

    Only files whose name is a known asset id plus a known extension are
    listed. Anything else in the folder -- a leftover `.part`, something the
    user dropped there, a file from a future version of this bridge -- is not
    ours to describe and therefore not ours to offer for deletion.
    """
    if not ID_RE.match(project_id or ""):
        raise StorageError("Invalid project id.", 422, "bad_project_id")
    directory = root() / project_id
    if not directory.is_dir():
        return []
    _within(root(), directory)

    items: list[dict[str, Any]] = []
    for path in sorted(directory.iterdir()):
        if not path.is_file():
            continue
        ext = path.suffix.lower()
        if ext not in MIME_BY_EXT or not ID_RE.match(path.stem):
            continue
        try:
            size = path.stat().st_size
        except OSError:
            continue
        items.append({
            "assetId": path.stem,
            "filename": path.name,
            "mime": MIME_BY_EXT[ext],
            "bytes": size,
        })
    return items


def delete_file(project_id: str, filename: str) -> bool:
    """Delete exactly this file. Not "whatever is stored for this id".

    The caller has an inventory and knows which file it means; resolving the
    name again here -- by id, by extension order -- would reintroduce the
    ambiguity the inventory exists to remove.
    """
    if not ID_RE.match(project_id or ""):
        raise StorageError("Invalid project id.", 422, "bad_project_id")
    name = filename or ""
    stem, _, ext = name.rpartition(".")
    ext = f".{ext.lower()}"
    # Structural, not defensive: a name that passes this cannot contain a
    # separator, a `..`, a drive letter or a NUL, so there is no path to
    # traverse. `_within` below still guards a symlinked project folder.
    if ext not in MIME_BY_EXT or not ID_RE.match(stem):
        raise StorageError("Not a project file name.", 422, "bad_file_name")

    directory = root() / project_id
    if not directory.is_dir():
        return False
    path = _within(root(), directory / name)
    if not path.is_file():
        # Already gone is the outcome the caller wanted, not an error.
        return False
    try:
        path.unlink()
    except OSError as exc:
        raise StorageError(f"Could not delete the file: {exc}", 500, "delete_failed") from exc
    return True


def delete(project_id: str, asset_id: str) -> bool:
    """Explicit cleanup only. Prefer delete_file(): this one resolves by id.

    Kept because the v1 contract declares it, but it inherits find_asset()'s
    extension order, so it is not what the collector uses.

    Never cascades from a cloud project deletion: an accidental click in a
    browser must not be able to take files off the user's disk.
    """
    path = find_asset(project_id, asset_id)
    if path is None:
        return False
    try:
        path.unlink()
    except OSError as exc:
        raise StorageError(f"Could not delete the asset: {exc}", 500, "delete_failed") from exc
    return True
