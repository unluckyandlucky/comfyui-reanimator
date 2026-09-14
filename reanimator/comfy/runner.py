"""Run a bound workflow on the local ComfyUI, in process.

**Never over HTTP to a guessed port.** ComfyUI's port is assigned per instance by
the Desktop hub; it was 8000 on the machine this was built on, not the 8188
everybody assumes (docs/local-bridge-handoff.md §1, §3). ``serve.py`` gets away
with ``COMFYUI_URL`` because it is a separate process the user configures. A
custom node is already *inside* ComfyUI, so it can reach the queue directly and
be right on every install.

Ported from ``serve.py``: queueing (``comfyui_queue``, line 211), reading history
(``comfyui_history``, line 226) and output parsing (lines 658-703). The logic is
the same; the transport is not.

Every ComfyUI internal touched here is reached through ``_prompt_server()`` /
``_folder_paths()``, which raise :class:`ComfyUnavailable` outside ComfyUI. That
keeps the module importable -- and testable -- on a machine with no ComfyUI at
all.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
import mimetypes
import re
import secrets
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping

log = logging.getLogger("reanimator.bridge")

INPUT_PREFIX = "rb_"
INPUT_SUFFIXES = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
                  ".webp": "image/webp"}
# What write_input names a file, and so the only thing read_input and
# delete_input will touch. Kept identical to validate.INPUT_FILENAME_RE.
INPUT_NAME_RE = re.compile(r"^rb_[a-f0-9]{8,64}\.(png|jpg|jpeg|webp)$")
MAX_INPUT_BYTES = 64 * 1024 * 1024
MAX_INPUTS_PER_RUN = 16

POLL_INTERVAL_SECONDS = 0.25
# 15 minutes was catastrophically wrong. On a card with no room to spare, a
# four-step Qwen 2511 edit took 15:22 and 15:02 -- and the timeout does not just
# stop waiting, it calls interrupt(), so it destroyed two finished generations
# 22 and 2 seconds before they landed. A timeout exists to stop an infinite
# wait, not to impose a performance budget: too long merely annoys, too short
# throws away work the user already paid for in GPU time.
DEFAULT_TIMEOUT_SECONDS = 60 * 60
# How long a prompt may be neither in the queue nor in the history before we
# call it lost. Without this a job someone cleared by hand waits forever.
LOST_GRACE_SECONDS = 10.0

OBJECT_INFO_TTL_SECONDS = 15
MAX_RUNS_KEPT = 40
RUN_TTL_SECONDS = 3600
MAX_ACTIVE_RUNS = 4

# Keys ComfyUI uses in a node's output block. serve.py only looked at 'video'
# and 'images'; the rest cost nothing and stop a template silently producing
# "no output" on a node that saves a webp or an audio track.
OUTPUT_KEYS = ("images", "gifs", "video", "videos", "audio", "files")
# Same list serve.py checked at line 684: a SaveImage node can perfectly well
# have written an mp4, and reporting that as an image sends the editor looking
# for a still that is not there.
VIDEO_SUFFIXES = (".mp4", ".webm", ".mov", ".mkv", ".m4v")


class ComfyUnavailable(RuntimeError):
    """Not running inside ComfyUI, or ComfyUI has not finished starting."""


class RunError(Exception):
    def __init__(self, message: str, code: str = "run_failed") -> None:
        super().__init__(message)
        self.code = code


# --------------------------------------------------------------------------
# ComfyUI handles
# --------------------------------------------------------------------------

def _prompt_server() -> Any:
    try:
        from server import PromptServer          # ComfyUI, at runtime
    except ImportError as exc:                   # pragma: no cover - outside ComfyUI
        raise ComfyUnavailable(
            "The Reanimator Bridge is not running inside ComfyUI."
        ) from exc
    instance = getattr(PromptServer, "instance", None)
    if instance is None or getattr(instance, "prompt_queue", None) is None:
        raise ComfyUnavailable("ComfyUI is still starting up. Try again in a moment.")
    return instance


def _folder_paths() -> Any:
    try:
        import folder_paths                      # ComfyUI, at runtime
    except ImportError as exc:                   # pragma: no cover - outside ComfyUI
        raise ComfyUnavailable(
            "The Reanimator Bridge is not running inside ComfyUI."
        ) from exc
    return folder_paths


# --------------------------------------------------------------------------
# object_info
# --------------------------------------------------------------------------

_object_info_lock = threading.Lock()
_object_info_cache: tuple[float, dict[str, Any]] | None = None


def _build_object_info() -> dict[str, Any]:
    """What ComfyUI's ``GET /object_info`` returns, obtained without HTTP.

    ``node_info`` is the exact function that route uses, so preferring it keeps
    the validator reading the same shape the rest of the world sees. The
    fallback builds the only part the validator needs -- ``input`` -- straight
    from ``INPUT_TYPES()``, which is also where the model filename lists come
    from, so a missing ``.safetensors`` is still caught.
    """
    try:
        import nodes                              # ComfyUI, at runtime
    except ImportError as exc:                    # pragma: no cover - outside ComfyUI
        # Named, not an ImportError traceback: the editor shows this sentence.
        raise ComfyUnavailable(
            "The Reanimator Bridge is not running inside ComfyUI."
        ) from exc

    node_info = None
    try:
        from server import node_info as _node_info
        node_info = _node_info
    except Exception:
        pass

    info: dict[str, Any] = {}
    for name, node_class in dict(nodes.NODE_CLASS_MAPPINGS).items():
        try:
            if node_info is not None:
                info[name] = node_info(name)
            else:
                info[name] = {
                    "input": node_class.INPUT_TYPES(),
                    "output": list(getattr(node_class, "RETURN_TYPES", ()) or ()),
                    "name": name,
                }
        except Exception:
            # One broken custom node must not blind the validator to the other
            # five hundred. Registering the name alone still lets the
            # "is this class installed?" check pass, which is the truth.
            info.setdefault(name, {"input": {}, "name": name})
    return info


def object_info(refresh: bool = False) -> dict[str, Any]:
    """Cached briefly: a validate immediately followed by a run would otherwise
    re-scan every models folder twice, and the user can add a file between runs
    but not between those two calls."""
    global _object_info_cache
    with _object_info_lock:
        now = time.monotonic()
        if not refresh and _object_info_cache and now - _object_info_cache[0] < OBJECT_INFO_TTL_SECONDS:
            return _object_info_cache[1]
        info = _build_object_info()
        _object_info_cache = (now, info)
        return info


# --------------------------------------------------------------------------
# Inputs
# --------------------------------------------------------------------------

def quantization_formats() -> set[str] | None:
    """Which quantized weight formats THIS build of ComfyUI can read at all.

    ``comfy.quant_ops.QUANT_ALGOS`` is the registry ComfyUI itself indexes when
    it loads a quantized checkpoint. A format that is not a key there is not
    "slow" -- it is a ``KeyError`` deep inside UNETLoader, which reaches the
    user as ``UNETLoader: 'int8_tensorwise'`` a minute into a run that was
    never going to work.

    **Membership, not nativeness.** ComfyUI keeps a separate ``_disabled`` set
    for formats it knows but cannot run natively on this GPU: those are
    emulated -- slower, and they DO produce an image. That is exactly the
    fp8mixed-on-Ampere case this project already measured. Screening those out
    would refuse the one checkpoint that works.

    Returning ``None`` means "cannot tell": an older ComfyUI without the
    module, or the bridge imported outside ComfyUI entirely. The caller then
    filters nothing, because a guess in either direction is worse -- refusing
    everything would break installs that work today, and no bytes are at risk
    the way they are with storage.
    """
    try:
        from comfy.quant_ops import QUANT_ALGOS
    except Exception:
        return None
    try:
        return {str(name) for name in QUANT_ALGOS}
    except Exception:
        return None


_profile_cache: dict[str, Any] | None = None


def device_profile() -> dict[str, Any]:
    """What a stored benchmark has to match to be believed here.

    Compute capability rather than "it is Ampere": within one architecture the
    available native ops still differ, and the whole reason INT8 beat FP8 on the
    development machine was which ops ComfyUI could run natively. The ComfyUI
    and torch versions are in the key because both have changed under us mid-
    session -- 0.28.3 became 0.29.2 on its own -- and either can move these
    numbers.
    """
    global _profile_cache
    if _profile_cache is not None:
        return _profile_cache

    profile: dict[str, Any] = {}
    try:
        import torch
        profile["torch"] = str(torch.__version__)
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            profile["device"] = props.name
            profile["computeCapability"] = f"{props.major}.{props.minor}"
            profile["vramTotalMb"] = props.total_memory // 1048576
            profile["backend"] = "cuda"
    except Exception:
        pass
    try:
        import comfyui_version
        profile["comfyui"] = str(comfyui_version.__version__)
    except Exception:
        pass

    _profile_cache = profile
    return profile


def model_exists(folder: str, filename: str) -> bool:
    """Is this model file actually on disk?

    ``folder_paths.get_full_path`` ends in ``os.path.isfile``, so it answers the
    real question. ``/object_info``'s choice lists do not: a name can appear
    there with no file behind it, and then the pre-flight blesses a run that
    ComfyUI aborts a minute later while loading.
    """
    try:
        return _folder_paths().get_full_path(folder, filename) is not None
    except Exception:
        return False


def input_dir() -> Path:
    return Path(_folder_paths().get_input_directory())


def write_input(data: bytes, suffix: str) -> str:
    """Write bytes into ComfyUI's ``input/`` under a name the bridge invented.

    The caller never chooses the filename. That single rule removes traversal,
    absolute paths, overwriting someone's existing input, and the whole class of
    "the extension says .png but it is a .py" problems in one go -- the name is
    generated, the extension comes from a fixed allowlist, and the size is
    capped before anything touches the disk.
    """
    suffix = suffix.lower()
    if suffix not in INPUT_SUFFIXES:
        raise RunError(f"Unsupported image type: {suffix}", "bad_input_type")
    if not data:
        raise RunError("Empty upload.", "bad_input")
    if len(data) > MAX_INPUT_BYTES:
        raise RunError(
            f"Image is larger than {MAX_INPUT_BYTES // (1024 * 1024)} MB.",
            "input_too_large",
        )

    directory = input_dir()
    directory.mkdir(parents=True, exist_ok=True)
    name = f"{INPUT_PREFIX}{secrets.token_hex(12)}{suffix}"
    target = directory / name
    tmp = target.with_suffix(target.suffix + ".part")
    tmp.write_bytes(data)
    tmp.replace(target)
    return name


def _input_path(name: Any) -> Path | None:
    """The file a bridge input name refers to, or None if it is not one.

    A prefix test was the whole check here, and ``rb_x/../../photo.png`` passes
    a prefix test: the Comfy-Org registry review found a paired client could
    read a file outside input/ that way, and have the cleanup delete it. The
    name must be exactly the shape write_input invents -- no separator, no dots
    but the suffix -- and must still resolve inside the folder.
    """
    if not isinstance(name, str) or not INPUT_NAME_RE.match(name):
        return None
    directory = input_dir().resolve()
    path = (directory / name).resolve()
    if path.parent != directory:
        return None
    return path


def read_input(name: str) -> bytes:
    """Read back a file this bridge wrote into ComfyUI's input folder."""
    path = _input_path(name)
    if path is None:
        raise RunError("Not a bridge input.", "bad_input")
    return path.read_bytes()


def delete_input(name: str) -> None:
    """Best effort. A leftover input costs disk; a crash here costs the run."""
    try:
        path = _input_path(name)
        if path is not None:
            path.unlink(missing_ok=True)
    except (OSError, ComfyUnavailable):
        pass


# --------------------------------------------------------------------------
# Queueing
# --------------------------------------------------------------------------

async def _validate_with_comfy(prompt_id: str, graph: Mapping[str, Any]) -> list[str] | None:
    """Let ComfyUI check the prompt too, and tell us which nodes are outputs.

    ``execution.validate_prompt`` has changed signature more than once -- it took
    ``(prompt)``, then ``(prompt_id, prompt)``, then a third argument, and it
    became a coroutine along the way. Reading the signature is ugly but the
    alternative is a bridge that breaks on a ComfyUI update with a TypeError
    nobody can interpret.
    """
    try:
        import execution
    except ImportError:                          # pragma: no cover - outside ComfyUI
        return None
    validate = getattr(execution, "validate_prompt", None)
    if validate is None:
        return None

    try:
        parameters = [
            p for p in inspect.signature(validate).parameters.values()
            if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
        ]
    except (TypeError, ValueError):               # pragma: no cover
        return None
    required = [p for p in parameters if p.default is inspect.Parameter.empty]

    if len(required) >= 3:
        result = validate(prompt_id, graph, None)
    elif len(required) == 2:
        result = validate(prompt_id, graph)
    else:
        result = validate(graph)
    if inspect.isawaitable(result):
        result = await result

    valid = bool(result[0])
    error = result[1] if len(result) > 1 else None
    outputs = list(result[2]) if len(result) > 2 and result[2] else []
    node_errors = result[3] if len(result) > 3 else None
    if not valid:
        raise RunError(_rejection_message(error, node_errors), "rejected_by_comfyui")
    return [str(node_id) for node_id in outputs]


def _rejection_message(error: Any, node_errors: Any) -> str:
    """Turn validate_prompt's refusal into something a person can act on.

    The top-level ``error`` is almost always the generic "Prompt outputs failed
    validation". Everything that identifies the actual problem -- which node,
    which input, what was wrong with it -- is in ``node_errors``, so dropping it
    leaves the user with a sentence that rules nothing out. That defeats the
    entire reason for validating before queueing.
    """
    top = "ComfyUI rejected this workflow."
    if isinstance(error, Mapping):
        top = str(error.get("message") or top)
        details = error.get("details")
        if details:
            top = f"{top} ({details})"
    elif error:
        top = str(error)

    parts: list[str] = []
    if isinstance(node_errors, Mapping):
        for node_id, info in node_errors.items():
            if not isinstance(info, Mapping):
                continue
            class_type = info.get("class_type") or "?"
            for item in info.get("errors") or []:
                if not isinstance(item, Mapping):
                    continue
                text = item.get("message") or item.get("type") or "invalid"
                extra = item.get("details")
                parts.append(
                    f"node {node_id} ({class_type}): {text}" + (f" — {extra}" if extra else "")
                )
    if parts:
        return f"{top} " + "; ".join(parts[:6])
    return top


def _queue_wants_sensitive() -> bool:
    """Whether this ComfyUI expects the 6th ``sensitive`` element.

    ``SENSITIVE_EXTRA_DATA_KEYS`` arrived in ``execution`` together with that
    element, so its presence is a reliable proxy for the queue tuple's arity --
    far better than comparing version strings, which fork and get patched.
    """
    try:
        import execution
    except ImportError:                          # pragma: no cover - outside ComfyUI
        return False
    return hasattr(execution, "SENSITIVE_EXTRA_DATA_KEYS")


def queue_item(
    number: int, prompt_id: str, graph: Mapping[str, Any], outputs: list[str]
) -> tuple:
    """Build the tuple ComfyUI's prompt_worker unpacks.

    GET THIS WRONG AND THE WORKER THREAD DIES. It reads ``item[5]`` directly
    (``main.py``), so a short tuple raises IndexError inside the worker, which
    is not our thread and not our try/except: the traceback lands in ComfyUI's
    log, the thread exits, and from then on the instance accepts prompts and
    executes none of them -- ours *and* the ones the user queues from the
    ComfyUI window. The bridge sees the item sitting in the queue and waits out
    its full timeout, so the editor just says "Generating…" forever.

    ``sensitive`` is empty because the bridge sets no extra_data at all; the
    element still has to be there.
    """
    item = (number, prompt_id, dict(graph), {}, list(outputs))
    return item + ({},) if _queue_wants_sensitive() else item


def _worker_is_alive() -> bool:
    """Best effort: is ComfyUI's prompt_worker thread still running?

    Pure diagnosis. A dead worker is indistinguishable from a busy queue to
    everything else in this module, so without it the failure above costs a
    fifteen-minute timeout and produces the least informative message possible.
    Python 3.10+ puts the target's name into an auto-generated thread name, so
    matching on it is cheap; when nothing matches we assume alive, because a
    false "ComfyUI is broken" would be worse than no check at all.
    """
    try:
        for thread in threading.enumerate():
            target = getattr(thread, "_target", None)
            if "prompt_worker" in thread.name or getattr(target, "__name__", "") == "prompt_worker":
                return thread.is_alive()
    except Exception:                             # pragma: no cover - defensive
        pass
    return True


async def queue(graph: Mapping[str, Any], output_node_ids: Iterable[str]) -> str:
    """Put a prompt on ComfyUI's own queue and return its prompt id."""
    server = _prompt_server()
    prompt_id = str(uuid.uuid4())

    outputs = await _validate_with_comfy(prompt_id, graph)
    if not outputs:
        outputs = [str(node_id) for node_id in output_node_ids]
    if not outputs:
        raise RunError("This workflow has no output node.", "no_output")

    if not _worker_is_alive():
        raise RunError(
            "ComfyUI's queue worker is not running, so nothing will execute. "
            "Restart ComfyUI.",
            "worker_dead",
        )

    number = getattr(server, "number", 0)
    try:
        server.number = number + 1
    except Exception:                             # pragma: no cover
        pass
    # No client_id: nothing of ours is listening on ComfyUI's websocket, and
    # naming a client that does not exist only makes ComfyUI queue up progress
    # messages for a socket that will never read them.
    server.prompt_queue.put(queue_item(number, prompt_id, graph, outputs))
    return prompt_id


def history(prompt_id: str) -> dict[str, Any] | None:
    entries = _prompt_server().prompt_queue.get_history(prompt_id=prompt_id)
    if not entries:
        return None
    entry = entries.get(prompt_id)
    return entry if isinstance(entry, Mapping) else None


def _in_queue(prompt_id: str) -> bool:
    try:
        running, pending = _prompt_server().prompt_queue.get_current_queue()
    except Exception:                             # pragma: no cover
        return True                               # assume the best, keep waiting
    for item in list(running) + list(pending):
        if len(item) > 1 and str(item[1]) == prompt_id:
            return True
    return False


def entry_error(entry: Mapping[str, Any]) -> str | None:
    """The failure message ComfyUI recorded, or None if the run succeeded.

    Ported from serve.py 697-702, which only inspected ``status_str``. The
    per-node messages are where the useful text actually lives ("value not in
    list", "out of memory"), so they are pulled out too.
    """
    status = entry.get("status")
    if not isinstance(status, Mapping):
        return None
    if status.get("status_str") != "error" and status.get("completed", True):
        return None

    for message in status.get("messages") or []:
        if not (isinstance(message, (list, tuple)) and len(message) >= 2):
            continue
        kind, payload = message[0], message[1]
        if kind != "execution_error" or not isinstance(payload, Mapping):
            continue
        text = payload.get("exception_message") or payload.get("exception_type")
        node = payload.get("node_type") or payload.get("node_id")
        if text:
            return f"{node}: {text}" if node else str(text)
    if status.get("status_str") == "error":
        return "ComfyUI reported an error but gave no message."
    return None


async def interrupt(prompt_id: str | None = None) -> None:
    server = _prompt_server()
    if prompt_id:
        try:
            server.prompt_queue.delete_queue_item(
                lambda item: len(item) > 1 and str(item[1]) == prompt_id
            )
        except Exception:                         # pragma: no cover
            pass
    try:
        import nodes
        nodes.interrupt_processing()
    except Exception:                             # pragma: no cover
        try:
            import comfy.model_management as mm
            mm.interrupt_current_processing()
        except Exception:
            pass


# --------------------------------------------------------------------------
# Outputs
# --------------------------------------------------------------------------

def parse_outputs(entry: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Flatten a history entry's outputs. Port of serve.py 670-692.

    Each item keeps ComfyUI's own ``filename`` / ``subfolder`` / ``type``, which
    are resolved later by :func:`resolve_output`. They are never handed to the
    browser: the run layer swaps them for an opaque ref, the same way media.py
    refuses to put a path in a URL.
    """
    items: list[dict[str, Any]] = []
    for node_id, node_out in (entry.get("outputs") or {}).items():
        if not isinstance(node_out, Mapping):
            continue
        for key in OUTPUT_KEYS:
            for item in node_out.get(key) or []:
                if not isinstance(item, Mapping):
                    continue
                filename = str(item.get("filename") or "")
                if not filename:
                    continue
                kind = (
                    "video"
                    if filename.lower().endswith(VIDEO_SUFFIXES)
                    else {"images": "image", "gifs": "video", "videos": "video",
                          "files": "file"}.get(key, key)
                )
                items.append(
                    {
                        "node": str(node_id),
                        "filename": filename,
                        "subfolder": str(item.get("subfolder") or ""),
                        "type": str(item.get("type") or "output"),
                        "kind": kind,
                        "mime": mimetypes.guess_type(filename)[0]
                        or "application/octet-stream",
                    }
                )
    return items


def corrected_output(run: "Run", ref: str, path: Path) -> Path | None:
    """The result with the padding removed, at exactly the source resolution.

    Written into the bridge's own state directory rather than over ComfyUI's
    output: that file belongs to ComfyUI and to the user's history, and the
    bridge has no business rewriting it. Cached per ref so refetching costs
    nothing.
    """
    from ..workflow import geometry
    from .. import config

    cached = run.corrected.get(ref)
    if cached and cached.is_file():
        return cached
    if run.transform is None:
        return None
    try:
        data = geometry.unpad(path.read_bytes(), run.transform)
        folder = config.state_dir() / "outputs"
        folder.mkdir(parents=True, exist_ok=True)
        target = folder / f"{run.id}_{ref}.png"
        tmp = target.with_suffix(".part")
        tmp.write_bytes(data)
        tmp.replace(target)
        run.corrected[ref] = target
        return target
    except Exception:
        # A failed crop must not swallow the generation: serve the raw result
        # and let the caller see the wrong size rather than nothing at all.
        log.warning("Could not restore the source resolution for %s", run.id, exc_info=True)
        return None


def resolve_output(item: Mapping[str, Any]) -> Path:
    """Turn an output record into a real path, confined to ComfyUI's own folders.

    ``filename`` and ``subfolder`` come from ComfyUI, not from the browser, but
    they still pass through a node's code -- a custom node is free to report any
    string it likes. Confining here means a hostile or buggy node cannot turn a
    result download into an arbitrary file read.
    """
    folder_paths = _folder_paths()
    kind = str(item.get("type") or "output")
    try:
        base = Path(folder_paths.get_directory_by_type(kind))
    except Exception:
        base = None
    if base is None:
        raise RunError(f"Unknown output location: {kind}", "bad_output")

    base = base.resolve()
    candidate = (base / str(item.get("subfolder") or "") / str(item["filename"])).resolve()
    if candidate != base and base not in candidate.parents:
        raise RunError("Output escapes ComfyUI's output folder.", "bad_output")
    if not candidate.is_file():
        raise RunError("The result file is no longer there.", "output_missing")
    return candidate


# --------------------------------------------------------------------------
# Runs
# --------------------------------------------------------------------------

@dataclass
class Run:
    id: str
    template_id: str
    state: str = "queued"                 # queued | running | succeeded | failed | canceled
    prompt_id: str | None = None
    error: str | None = None
    error_code: str | None = None
    outputs: list[dict[str, Any]] = field(default_factory=list)
    files: dict[str, dict[str, Any]] = field(default_factory=dict)   # ref -> output record
    inputs: list[str] = field(default_factory=list)
    corrected: dict[str, Path] = field(default_factory=dict)
    # Links the manifest authorised unplugging for this run, and which ones
    # actually went. Reported and logged: a silently pruned workflow is a
    # workflow nobody can debug afterwards.
    pruned: list[dict[str, Any]] = field(default_factory=list)
    quality: str | None = None
    intent: str | None = None
    checkpoint: dict[str, Any] | None = None
    model_resolution: str | None = None
    geometry: dict[str, Any] | None = None
    transform: Any = None
    pictures: int | None = None
    # Set when the requested quality could not run and a different one was
    # substituted. Reported, never silent: the user asked for a 4-step preview
    # and is about to wait for a 40-step render.
    fallback: dict[str, Any] | None = None
    telemetry: dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    finished_at: float | None = None
    task: Any = None

    def as_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "runId": self.id,
            "templateId": self.template_id,
            "state": self.state,
            "createdAt": int(self.created_at),
        }
        if self.pruned:
            payload["pruned"] = self.pruned
        if self.quality:
            payload["quality"] = self.quality
        if self.intent:
            payload["intent"] = self.intent
        if self.checkpoint:
            payload["checkpoint"] = self.checkpoint
        if self.fallback:
            payload["fallback"] = self.fallback
        if self.telemetry:
            payload["telemetry"] = self.telemetry
        if self.geometry:
            payload["resolution"] = self.geometry
        if self.error:
            payload["error"] = self.error
            payload["errorCode"] = self.error_code
        if self.state == "succeeded":
            payload["outputs"] = self.outputs
        return payload


class RunStore:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._runs: dict[str, Run] = {}

    def create(self, template_id: str, inputs: list[str]) -> Run:
        with self._lock:
            active = sum(1 for r in self._runs.values() if r.state in ("queued", "running"))
            if active >= MAX_ACTIVE_RUNS:
                raise RunError(
                    "Too many generations are already running on this device.",
                    "too_many_runs",
                )
            self._prune_locked()
            run = Run(id=secrets.token_urlsafe(12), template_id=template_id, inputs=list(inputs))
            self._runs[run.id] = run
            return run

    def get(self, run_id: str) -> Run:
        with self._lock:
            run = self._runs.get(run_id)
        if run is None:
            raise RunError("No such run.", "unknown_run")
        return run

    def _prune_locked(self) -> None:
        now = time.time()
        stale = [
            run_id for run_id, run in self._runs.items()
            if run.state in ("succeeded", "failed", "canceled")
            and now - (run.finished_at or run.created_at) > RUN_TTL_SECONDS
        ]
        if len(self._runs) - len(stale) > MAX_RUNS_KEPT:
            finished = sorted(
                (r for r in self._runs.values() if r.state not in ("queued", "running")),
                key=lambda r: r.finished_at or r.created_at,
            )
            for run in finished[: len(self._runs) - MAX_RUNS_KEPT]:
                stale.append(run.id)
        for run_id in set(stale):
            run = self._runs.pop(run_id, None)
            if run:
                for name in run.inputs:
                    delete_input(name)
                for path in run.corrected.values():
                    try:
                        path.unlink(missing_ok=True)
                    except OSError:
                        pass


runs = RunStore()


def _record_benchmark(run: Run) -> None:
    """Store what this run cost, keyed to this machine.

    Only successful runs: a failure says nothing about how fast a variant is,
    and letting one poison the selection would make the bridge avoid a perfectly
    good checkpoint forever because of one out-of-memory afternoon.
    """
    try:
        from ..workflow import benchmarks
        checkpoint = run.checkpoint or {}
        profile = device_profile()
        tel = run.telemetry or {}
        benchmarks.record({
            "templateId": run.template_id,
            "checkpointVariant": checkpoint.get("variant"),
            "device": profile.get("device"),
            "computeCapability": profile.get("computeCapability"),
            "comfyui": profile.get("comfyui"),
            "torch": profile.get("torch"),
            "quality": run.quality,
            "modelResolution": run.model_resolution,
            "pictures": run.pictures,
            "totalSeconds": tel.get("seconds"),
            "peakVramMb": tel.get("devicePeakMb"),
            "peakRamMb": tel.get("processRamPeakMb"),
            "success": True,
        })
    except Exception:                             # pragma: no cover - never block a run
        log.debug("Could not record benchmark", exc_info=True)


async def _drive(run: Run, graph: Mapping[str, Any], output_node_ids: list[str], timeout: float) -> None:
    try:
        begin_telemetry(run)
        run.prompt_id = await queue(graph, output_node_ids)
        run.state = "running"
        entry = await _wait(run, timeout)
        error = entry_error(entry)
        if error:
            run.state, run.error, run.error_code = "failed", error, "comfyui_error"
            return
        parsed = parse_outputs(entry)
        if not parsed:
            run.state = "failed"
            run.error = "The workflow finished without producing an output."
            run.error_code = "no_output"
            return
        for item in parsed:
            ref = secrets.token_urlsafe(9)
            run.files[ref] = item
            run.outputs.append(
                {"ref": ref, "kind": item["kind"], "mime": item["mime"], "node": item["node"]}
            )
        run.state = "succeeded"
    except RunError as exc:
        run.state, run.error, run.error_code = "failed", str(exc), exc.code
    except ComfyUnavailable as exc:
        run.state, run.error, run.error_code = "failed", str(exc), "comfy_unavailable"
    except asyncio.CancelledError:
        run.state, run.error, run.error_code = "canceled", "Canceled.", "canceled"
        raise
    except Exception as exc:                      # pragma: no cover - defensive
        log.exception("Local run %s failed", run.id)
        run.state, run.error, run.error_code = "failed", str(exc), "internal"
    finally:
        run.finished_at = time.time()
        end_telemetry(run)
        # AFTER end_telemetry, not before: recording first stored a row with
        # every timing null, which then could not be compared with anything.
        if run.state == "succeeded":
            _record_benchmark(run)
        # The frames are on disk in ComfyUI's input/ only for the length of the
        # run. Leaving them would quietly build a copy of the user's footage in
        # a folder they never chose.
        for name in run.inputs:
            delete_input(name)
        run.inputs = []


def _torch():
    try:
        import torch
        return torch if torch.cuda.is_available() else None
    except Exception:
        return None


def begin_telemetry(run: Run) -> None:
    """Bridge and ComfyUI share one process (one PID owns both ports), so
    torch's own counters measure the generation directly -- no NVML needed for
    the allocator figures. The device total still comes from mem_get_info,
    because torch cannot see what other processes hold."""
    torch = _torch()
    run.telemetry = {"startedAt": time.time()}
    if torch is None:
        return
    try:
        torch.cuda.reset_peak_memory_stats()
        free, total = torch.cuda.mem_get_info()
        run.telemetry["deviceTotalMb"] = total // 1048576
        run.telemetry["deviceUsedBeforeMb"] = (total - free) // 1048576
    except Exception:
        pass


def sample_telemetry(run: Run) -> None:
    torch = _torch()
    if torch is None:
        return
    try:
        free, total = torch.cuda.mem_get_info()
        used = (total - free) // 1048576
        run.telemetry["devicePeakMb"] = max(run.telemetry.get("devicePeakMb", 0), used)
    except Exception:
        pass
    try:
        import psutil
        rss = psutil.Process().memory_info().rss // 1048576
        run.telemetry["processRamPeakMb"] = max(run.telemetry.get("processRamPeakMb", 0), rss)
    except Exception:
        pass


def end_telemetry(run: Run) -> None:
    torch = _torch()
    run.telemetry["seconds"] = round(time.time() - run.telemetry.get("startedAt", time.time()), 1)
    if torch is None:
        return
    try:
        run.telemetry["peakAllocatedMb"] = torch.cuda.max_memory_allocated() // 1048576
        run.telemetry["peakReservedMb"] = torch.cuda.max_memory_reserved() // 1048576
    except Exception:
        pass


async def _wait(run: Run, timeout: float) -> Mapping[str, Any]:
    deadline = time.monotonic() + timeout
    lost_since: float | None = None
    while True:
        sample_telemetry(run)
        entry = history(run.prompt_id or "")
        if entry is not None:
            return entry

        if _in_queue(run.prompt_id or ""):
            lost_since = None
        else:
            now = time.monotonic()
            lost_since = lost_since if lost_since is not None else now
            if now - lost_since > LOST_GRACE_SECONDS:
                raise RunError(
                    "The job disappeared from ComfyUI's queue. It may have been "
                    "cleared from the ComfyUI window.",
                    "run_lost",
                )

        if time.monotonic() > deadline:
            await interrupt(run.prompt_id)
            raise RunError("Timed out waiting for ComfyUI.", "timeout")
        await asyncio.sleep(POLL_INTERVAL_SECONDS)


def start(
    template_id: str,
    graph: Mapping[str, Any],
    output_node_ids: list[str],
    inputs: list[str],
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
) -> Run:
    """Queue a bound graph and return immediately.

    The browser polls ``GET /rb/v1/run/:id``. Holding one HTTP request open for
    the minutes an image edit takes would work on loopback and fail the moment
    anything -- a proxy, a sleeping laptop, a browser tab throttle -- decides it
    has waited long enough, and there would be no way to recover the result.
    """
    if len(inputs) > MAX_INPUTS_PER_RUN:
        raise RunError("Too many input images.", "too_many_inputs")
    run = runs.create(template_id, inputs)
    run.task = asyncio.ensure_future(_drive(run, graph, output_node_ids, timeout))
    return run


async def cancel(run: Run) -> None:
    if run.state in ("succeeded", "failed", "canceled"):
        return
    await interrupt(run.prompt_id)
    if run.task is not None:
        run.task.cancel()
    run.state = "canceled"
    run.error = "Canceled."
    run.error_code = "canceled"
    run.finished_at = time.time()
