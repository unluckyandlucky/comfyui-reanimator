"""The browser-facing loopback server.

This is a SEPARATE aiohttp application, not routes on ComfyUI's PromptServer.
See docs/plan-local-bridge.md §3: registering on PromptServer would (a) hit
ComfyUI's origin-only middleware, reintroducing the --enable-cors-header burden
this design exists to remove, and (b) be exposed on 0.0.0.0 whenever the user
runs ComfyUI with --listen.

Four independent barriers guard every authenticated route:
  bind 127.0.0.1 only  ·  exact Origin  ·  bearer token  ·  command allowlist
"""

from __future__ import annotations

import logging
import re
import secrets
from collections import Counter
from typing import Any, Awaitable, Callable, Mapping

from aiohttp import web

from . import capabilities, config, media, projects
from .comfy import runner
from .comfy.runner import ComfyUnavailable, RunError
from .media import MediaError
from .pairing import PairingError, approvals, nonces, tokens
from .projects import StorageError
from .workflow import binder, geometry, templates, validate
from .workflow.binder import BindError
from .workflow.templates import TemplateError

log = logging.getLogger("reanimator.bridge")

Handler = Callable[[web.Request], Awaitable[web.StreamResponse]]

# Routes reachable without a bearer token. Everything else requires one.
PUBLIC_ROUTES = frozenset(
    {
        "/rb/v1/hello",
        "/rb/v1/pair/nonce",
        "/rb/v1/pair",
    }
)


def _json(payload: dict[str, Any], status: int = 200) -> web.Response:
    return web.json_response(payload, status=status)


def _error(code: str, message: str, status: int) -> web.Response:
    return _json({"error": {"code": code, "message": message}}, status=status)


def _rejected_origin(origin: str) -> web.Response:
    """403 that the browser will actually hand to JavaScript.

    Without Access-Control-Allow-Origin here, a rejected origin reaches the page
    as a bare "Failed to fetch" -- indistinguishable from "no bridge running",
    "browser blocked local access" and "an extension ate it". The caller then
    shows a message that sends the user off reinstalling for nothing.

    Echoing the origin back on the *rejection* leaks nothing: the body says only
    that the origin is not allowed, and every real endpoint stays blocked. It
    buys an error message the user can act on, which is worth far more than
    hiding the bridge's existence from a site that could time it out anyway.
    """
    response = _json(
        {
            "error": {
                "code": "bad_origin",
                "message": f"Origin not allowed: {origin}",
                "allowed": list(config.allowed_origins()),
            }
        },
        status=403,
    )
    response.headers["Access-Control-Allow-Origin"] = origin
    response.headers["Access-Control-Allow-Headers"] = (
        "Content-Type, Authorization, Range, X-Sha256"
    )
    response.headers["Vary"] = "Origin"
    return response


# --------------------------------------------------------------------------
# Middleware
# --------------------------------------------------------------------------

@web.middleware
async def cors_middleware(request: web.Request, handler: Handler) -> web.StreamResponse:
    """Exact-origin CORS. Never a wildcard -- see D2/D8."""
    origin = request.headers.get("Origin")
    allowed = config.allowed_origins()

    if request.method == "OPTIONS":
        if origin not in allowed:
            return _rejected_origin(origin)
        return web.Response(
            status=204,
            headers={
                "Access-Control-Allow-Origin": origin,
                # DELETE has to be listed or the browser refuses the preflight
                # and never sends it -- which is how discarding an unused input
                # silently stopped happening and left the user's frames sitting
                # in ComfyUI's input folder. X-Sha256 likewise: without it the
                # verified write in docs/contract-local-project-storage.md
                # cannot leave the page at all.
                "Access-Control-Allow-Methods": "GET, HEAD, POST, PUT, DELETE, OPTIONS",
                "Access-Control-Allow-Headers": (
                    "Content-Type, Authorization, Range, X-Sha256"
                ),
                "Access-Control-Expose-Headers": (
                    "Content-Range, Accept-Ranges, Content-Length"
                ),
                "Access-Control-Max-Age": "600",
                "Vary": "Origin",
            },
        )

    # Defense in depth ONLY -- never authentication. A local process can forge
    # any Origin it likes. What actually protects the user is the combination of
    # binding 127.0.0.1, the bearer token, human-approved pairing, and the
    # command/template allowlist. This check just stops a stray browser tab.
    if origin is None:
        return _error("bad_origin", "Origin not allowed.", 403)
    if origin not in allowed:
        return _rejected_origin(origin)

    try:
        response = await handler(request)
    except web.HTTPException as exc:
        # aiohttp raises these from *outside* our handlers -- the 413 when a PUT
        # exceeds client_max_size is the one that bites. Letting it propagate
        # sends a response with no CORS headers, and the browser then hides the
        # status and the body from JavaScript: the user gets "Failed to fetch"
        # for a problem the server described perfectly well. HTTPException is a
        # Response, so it can simply be decorated and returned.
        response = exc
    response.headers["Access-Control-Allow-Origin"] = origin
    # Without this, <video crossorigin="anonymous"> taints the canvas and every
    # toDataURL() in the keyframe pipeline throws a SecurityError.
    response.headers["Access-Control-Expose-Headers"] = (
        "Content-Range, Accept-Ranges, Content-Length"
    )
    response.headers["Vary"] = "Origin"
    response.headers.setdefault("Cache-Control", "no-store")
    response.headers.setdefault("X-Content-Type-Options", "nosniff")
    return response


@web.middleware
async def auth_middleware(request: web.Request, handler: Handler) -> web.StreamResponse:
    if request.method == "OPTIONS" or request.path in PUBLIC_ROUTES:
        return await handler(request)
    # /rb/v1/pair/<id> is public: the browser polls it before it has a token.
    if request.path.startswith("/rb/v1/pair/"):
        return await handler(request)
    # /rb/v1/media/<capability> carries its own credential in the path, because
    # <video src> cannot send an Authorization header. See media.py.
    if request.path.startswith("/rb/v1/media/"):
        return await handler(request)

    header = request.headers.get("Authorization", "")
    token = header[7:].strip() if header.lower().startswith("bearer ") else None
    try:
        request["paired"] = tokens.validate(token, request.headers.get("Origin"))
    except PairingError as exc:
        return _error(exc.code, str(exc), 401)
    return await handler(request)


@web.middleware
async def error_middleware(request: web.Request, handler: Handler) -> web.StreamResponse:
    try:
        return await handler(request)
    except PairingError as exc:
        return _error(exc.code, str(exc), 400)
    except MediaError as exc:
        return _error("media_error", str(exc), exc.status)
    except StorageError as exc:
        # Its own code, not a blanket one: a 409 sha mismatch is retried by the
        # editor, a 422 is a bug in the editor, and a 507 is a message the user
        # can act on. One code for all three would make them indistinguishable.
        return _error(exc.code, str(exc), exc.status)
    except TemplateError as exc:
        return _error(exc.code, str(exc), 404)
    except BindError as exc:
        return _error(exc.code, str(exc), 400)
    except ComfyUnavailable as exc:
        return _error("comfy_unavailable", str(exc), 503)
    except RunError as exc:
        return _error(exc.code, str(exc), 404 if exc.code == "unknown_run" else 400)
    except geometry.GeometryError as exc:
        # 422, and the message says what to do about it. A canvas the user typed
        # is a user-facing problem; letting it fall through to a generic 500
        # hides a sentence they could have acted on.
        return _error(exc.code, str(exc), 422)
    except web.HTTPException:
        raise
    except Exception:
        log.exception("Unhandled error in %s %s", request.method, request.path)
        return _error("internal", "Internal bridge error.", 500)


# --------------------------------------------------------------------------
# Routes
# --------------------------------------------------------------------------

routes = web.RouteTableDef()


@routes.get("/rb/v1/hello")
async def hello(_: web.Request) -> web.Response:
    return _json(capabilities.hello())


@routes.get("/rb/v1/pair/nonce")
async def pair_nonce(_: web.Request) -> web.Response:
    nonce, expires_at = nonces.issue()
    return _json({"nonce": nonce, "expiresAt": expires_at})


@routes.post("/rb/v1/pair")
async def pair(request: web.Request) -> web.Response:
    """Verify the backend-signed JWS assertion, then wait for a human in ComfyUI."""
    body = await request.json()
    assertion = body.get("assertion")
    if not isinstance(assertion, str):
        return _error("bad_request", "Missing 'assertion' (compact JWS string).", 400)

    from .pairing import verify_assertion

    try:
        claims = verify_assertion(assertion, nonces)
    except PairingError as exc:
        # Logged here too: the browser may not surface it, and this is the one
        # place with enough context to diagnose a misconfigured signing key.
        log.warning("Pairing rejected (%s): %s", exc.code, exc)
        raise
    approval = approvals.create(claims, config.load()["device_label"])
    log.info("Pairing requested by %s (%s)", claims["email"], claims["origin"])
    return _json(
        {
            "requestId": approval.request_id,
            "expiresAt": approval.expires_at,
            "device": approval.device_label,
        }
    )


@routes.get("/rb/v1/pair/{request_id}")
async def pair_status(request: web.Request) -> web.Response:
    """Polled by the browser while the dialog is open in ComfyUI."""
    approval = approvals.get(request.match_info["request_id"])
    if approval is None:
        return _error("unknown_request", "No such pairing request.", 404)

    payload: dict[str, Any] = {"state": approval.state, "device": approval.device_label}
    if approval.state == "approved":
        if approval.token is None:
            label = request.query.get("label") or approval.origin
            approval.token = tokens.issue(approval, label).token
        payload["token"] = approval.token
    return _json(payload)


@routes.post("/rb/v1/token/renew")
async def token_renew(request: web.Request) -> web.Response:
    header = request.headers.get("Authorization", "")
    token = header[7:].strip()
    entry = tokens.renew(token)
    return _json({"expiresAt": entry.expires_at})


@routes.post("/rb/v1/token/revoke")
async def token_revoke(request: web.Request) -> web.Response:
    """'Forget this browser', invoked from the editor."""
    header = request.headers.get("Authorization", "")
    token = header[7:].strip()
    return _json({"revoked": tokens.revoke(token=token)})


@routes.get("/rb/v1/capabilities")
async def get_capabilities(_: web.Request) -> web.Response:
    return _json(capabilities.capabilities())


# --------------------------------------------------------------------------
# Generation (Phase 2)
#
# The cloud sends a templateId and parameters. It never sends a graph -- see
# docs/plan-local-bridge.md §5. A ComfyUI custom node is arbitrary Python, so a
# bridge that runs a pushed workflow is an RCE target for whoever compromises
# reanimator.app, and "we would notice" is not a security control.
# --------------------------------------------------------------------------

# Client-chosen input ids. They are dictionary keys and nothing else: the file
# on disk is named by the bridge (runner.write_input), so this only has to be
# short and unambiguous, never safe as a path.
INPUT_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
MAX_INPUTS_HELD = 64

# input id -> filename inside ComfyUI's input/. Process-lifetime only, exactly
# like media capabilities: these are handles for one editing session.
_inputs: dict[str, str] = {}


def _drop_input(input_id: str) -> None:
    entry = _inputs.pop(input_id, None)
    if entry:
        runner.delete_input(entry["file"])


@routes.get("/rb/v1/templates")
async def list_templates(_: web.Request) -> web.Response:
    return _json({"templates": templates.summaries()})


@routes.put("/rb/v1/input/{input_id}")
async def put_input(request: web.Request) -> web.Response:
    """Raw image bytes, straight into ComfyUI's input/ folder."""
    input_id = request.match_info["input_id"]
    if not INPUT_ID_RE.match(input_id):
        return _error("bad_request", "Invalid input id.", 400)

    content_type = (request.headers.get("Content-Type") or "").split(";")[0].strip().lower()
    suffix = {
        "image/png": ".png",
        "image/jpeg": ".jpg",
        "image/webp": ".webp",
    }.get(content_type)
    if suffix is None:
        return _error(
            "bad_input_type",
            f"Unsupported image type: {content_type or 'unknown'}",
            415,
        )

    data = await request.read()
    _drop_input(input_id)                     # replacing an id must not leak the old file
    if len(_inputs) >= MAX_INPUTS_HELD:
        for stale in list(_inputs)[: len(_inputs) - MAX_INPUTS_HELD + 1]:
            _drop_input(stale)

    filename = runner.write_input(data, suffix)
    # Measured from the decoded image, never taken from the request: a declared
    # resolution that disagrees with the file would put every later crop off by
    # exactly the amount of the lie.
    try:
        width, height = geometry.measure(data)
    except geometry.GeometryError as exc:
        runner.delete_input(filename)
        return _error(exc.code, str(exc), 400)
    _inputs[input_id] = {"file": filename, "width": width, "height": height}
    return _json({"inputId": input_id, "bytes": len(data),
                  "resolution": [width, height]})


@routes.delete("/rb/v1/input/{input_id}")
async def delete_input(request: web.Request) -> web.Response:
    """Discard an input the editor uploaded but never used.

    Safe next to the PUT above: aiohttp auto-registers HEAD alongside GET, and
    only alongside GET, so a second method on the same path is not the duplicate
    that silently stops the whole bridge from starting.
    """
    input_id = request.match_info["input_id"]
    if not INPUT_ID_RE.match(input_id):
        return _error("bad_request", "Invalid input id.", 400)
    had = input_id in _inputs
    _drop_input(input_id)
    return _json({"deleted": had})


def _frames_note(
    template: templates.Template, body: Mapping[str, Any], values: Mapping[str, Any]
) -> dict[str, Any] | None:
    """What was asked for and what will actually be rendered.

    Reported whether or not it changed. "It came back longer than my timeline"
    is only a mystery while the number that did the changing is invisible; the
    editor can put "62 frames -> 65" on screen and the surprise is gone.
    """
    slot = str(template.frames.get("slot") or "")
    if not slot or slot not in values:
        return None
    requested = (body.get("parameters") or {}).get(slot)
    rendered = values[slot]
    if not isinstance(requested, int) or isinstance(requested, bool):
        return {"rendered": rendered}
    return {"requested": requested, "rendered": rendered, "padded": rendered - requested}


def _picture_count(values: Mapping[str, Any]) -> int:
    """Numbered image slots only.

    A prefix test counted $IMAGE_PATHS as one picture, and that slot is the
    whole keyframe list in a single widget -- so a fifty-key sequence reported
    the same number as a one-image edit, to a checkpoint selector that uses it
    to decide how much VRAM the run needs.
    """
    return sum(1 for name in values if binder.IMAGE_SLOT_RE.match(name))


def _input(input_id: Any) -> dict[str, Any]:
    entry = _inputs.get(str(input_id))
    if entry is None:
        raise BindError(f"Unknown input '{input_id}'. Upload it again.", "unknown_input")
    return entry


def _filename(input_id: Any) -> str:
    return str(_input(input_id)["file"])


def _size_of(filename: str) -> tuple[int, int]:
    for record in _inputs.values():
        if record["file"] == filename:
            return int(record["width"]), int(record["height"])
    # Not in the registry: measure the file rather than guess. A wrong size
    # here would resize a keyframe that did not need it.
    return geometry.measure(runner.read_input(filename))


def _target_size(sizes: list[tuple[int, int]]) -> tuple[int, int]:
    """The size the shot is actually in: the one most of the keys agree on.

    First in the caller's order breaks a tie. Taking the largest instead would
    let one key exported at the wrong resolution drag the whole render up to
    it, which costs VRAM and minutes for a shot nobody asked to enlarge.
    """
    counts = Counter(sizes)
    most = max(counts.values())
    return next(size for size in sizes if counts[size] == most)


def _match_collected_sizes(
    template: templates.Template, values: dict[str, Any], mutate: bool,
) -> tuple[list[str], list[dict[str, Any]]]:
    """Make every picture in a collected role the same size.

    The loader that takes the whole list batches it into one tensor, and mixed
    shapes cannot be batched. WhatDreamsCost's MultiImageLoader does not fail
    on that: multi_image_loader.py:171 prints a console warning and substitutes
    torch.zeros((1, 64, 64, 3)). That single 64x64 black frame is what reaches
    the sequencers AND the node the output resolution is derived from, so the
    run does not break -- it finishes nine minutes later as a black clip with
    none of the keyframes in it, reported as a success.

    Fitting is geometry.fit_into: scale to fit, centre, black bars, no crop and
    no stretch. When the odd key shares the others' aspect ratio -- the ordinary
    case, 640x360 against 848x480 -- the bars are zero pixels wide and this is a
    plain LANCZOS resample. A key of a genuinely different shape does get bars,
    and that is REPORTED rather than swallowed: bars on one key and not on the
    rest are visible in the finished clip, so the caller has to see it coming.
    """
    replaced: list[str] = []
    notes: list[dict[str, Any]] = []

    for role in template.roles:
        if not template.role_collects(str(role)):
            continue
        slot = (template.role_slots(str(role)) or [None])[0]
        names = values.get(slot) if slot else None
        if not isinstance(names, list) or len(names) < 2:
            continue

        sizes = [_size_of(str(name)) for name in names]
        target = _target_size(sizes)
        if all(size == target for size in sizes):
            continue

        for index, (name, size) in enumerate(zip(list(names), sizes)):
            if size == target:
                continue
            # The bars fit_into will actually leave, in pixels. Comparing the
            # aspect ratios exactly instead called 640x360 against 848x480
            # letterboxed -- true, by three pixels, which is not what the word
            # means to anyone reading it. Real frame sizes are rarely exact
            # sixteen-ninths, so an exact test flags nearly every resize and
            # the flag stops carrying information.
            scale = min(target[0] / size[0], target[1] / size[1])
            bars = (target[0] - max(1, round(size[0] * scale)),
                    target[1] - max(1, round(size[1] * scale)))
            notes.append({
                "role": str(role), "keyframe": index + 1,
                "from": f"{size[0]}x{size[1]}", "to": f"{target[0]}x{target[1]}",
                "bars": list(bars),
                "letterboxed": bars[0] > target[0] / 100 or bars[1] > target[1] / 100,
            })
            if not mutate:
                # /validate is a dry run. Rewriting the uploads here consumed
                # them, and the /run that followed could not find the files it
                # had just been told were fine.
                continue
            old = str(name)
            fitted = geometry.fit_into(runner.read_input(old), target)
            new = runner.write_input(fitted, ".png")
            names[index] = new
            replaced.append(new)
            runner.delete_input(old)

    return replaced, notes


def _values_for(
    template: templates.Template, body: dict[str, Any]
) -> tuple[dict[str, Any], set[str]]:
    """Turn a request body into slot values, plus which roles were supplied.

    Input ids become ``$IMAGE_n`` here rather than in the caller, so a filename
    inside ComfyUI's input folder is never something the browser can name -- it
    can only refer to an id it registered.

    Two shapes are accepted. ``inputs`` names the roles explicitly and is what
    the editor sends; ``inputIds`` is the positional list, kept working so an
    older editor does not break mid-upgrade. Positional is exactly the implicit
    contract worth getting rid of, so it maps through the manifest's role order
    rather than being interpreted a second, independent way.
    """
    parameters = body.get("parameters")
    if parameters is not None and not isinstance(parameters, dict):
        raise BindError("'parameters' must be an object.", "bad_request")

    values: dict[str, Any] = {}
    for name, value in (parameters or {}).items():
        if isinstance(name, str) and name.startswith("$"):
            values[name] = value
        elif name != "instruction":
            raise BindError(f"Not a slot name: {name!r}", "unknown_slot")

    supplied: set[str] = set()
    inputs = body.get("inputs")
    if isinstance(inputs, dict):
        for role, given in inputs.items():
            slots = template.role_slots(str(role))
            if not slots:
                raise BindError(f"This template has no '{role}' input.", "unknown_role")
            given_list = given if isinstance(given, list) else [given]
            given_list = [g for g in given_list if g]
            if not given_list:
                continue
            # zip() stops at the shorter side, so handing a role more images
            # than it has slots dropped the extras without a word -- and the
            # caller was told 202. The editor already refuses to truncate for
            # exactly this reason; doing it here silently just moved the same
            # failure one layer down, where nobody can see it. A template with
            # three image slots cannot take four images: say so.
            allowed = template.role_max(str(role))
            if len(given_list) > allowed:
                raise BindError(
                    f"This template takes {allowed} "
                    f"image{'' if allowed == 1 else 's'} for '{role}' and "
                    f"{len(given_list)} were given. The extra "
                    f"one{'' if len(given_list) - allowed == 1 else 's'} would not "
                    f"reach the model.",
                    "too_many_inputs",
                )
            if template.role_collects(str(role)):
                # One slot, the whole list. The order is the caller's, and it
                # is the order the timings are numbered against -- image 3 and
                # keyframe 3 are the same key, so nothing here may reorder.
                values[slots[0]] = [_filename(i) for i in given_list]
            else:
                for slot, input_id in zip(slots, given_list):
                    values[slot] = _filename(input_id)
            supplied.add(str(role))
    else:
        input_ids = body.get("inputIds") or []
        if not isinstance(input_ids, list):
            raise BindError("'inputIds' must be a list.", "bad_request")
        # Walk the manifest's roles in declaration order and fill them.
        order = [s for role in template.roles for s in template.role_slots(role)]
        order = order or [f"$IMAGE_{i}" for i in range(1, len(input_ids) + 1)]
        for slot, input_id in zip(order, input_ids):
            values[slot] = _filename(input_id)
        for role in template.roles:
            if any(s in values for s in template.role_slots(role)):
                supplied.add(role)

    # The length, in the arithmetic this preset can actually render. The editor
    # sends the span it means; rounding it there would bake one model's rule
    # into a contract meant to outlive that model.
    frame_slot = str(template.frames.get("slot") or "")
    if frame_slot and frame_slot in values:
        wanted = values[frame_slot]
        if isinstance(wanted, bool) or not isinstance(wanted, int) or wanted < 1:
            raise BindError(
                f"{frame_slot} must be a whole number of frames.", "bad_value"
            )
        values[frame_slot] = template.legal_frame_count(wanted)

    # One timing, every stage. A multi-stage video workflow samples the same
    # shot two or three times, each with its own sequencer, and all of them
    # must be told the same keyframes -- a stage left on the template's shipped
    # timings would re-time the shot halfway through the render and the output
    # would be a video nobody asked for. The editor sends the timing ONCE; that
    # a preset happens to need three copies is the bridge's business.
    if "$SEQUENCER" in values:
        for name in template.slots:
            if binder.SEQUENCER_SLOT_RE.match(name) and name not in values:
                values[name] = values["$SEQUENCER"]

    if "$SEED" not in values and (template.slots.get("$SEED") or {}).get("randomizePerRun"):
        # A fixed seed would make every retry return the identical image, and
        # "generate again" is the main way a user works around a bad result.
        values["$SEED"] = secrets.randbelow(2**53)
    return values, supplied


def _load_template(body: dict[str, Any]) -> tuple[templates.Template, dict[str, Any] | None]:
    """The preset for this request.

    An explicit templateId still works -- it is how a test or a power user pins
    one. Without it the bridge resolves intent + hardware + installed models
    itself, because that is knowledge only the bridge has.
    """
    if not body.get("templateId"):
        template, fallback = templates.resolve_for_intent(
            body.get("intent"), runner.device_profile(), runner.model_exists
        )
        if template is None:
            raise TemplateError(
                _no_preset_message(fallback),
                (fallback or {}).get("code") or "no_runnable_preset",
            )
        return template, fallback

    template = templates.get(str(body.get("templateId") or ""))
    wanted_hash = body.get("templateHash")
    if wanted_hash and str(wanted_hash) != template.hash:
        # The cloud pinned a version this machine does not have installed.
        raise TemplateError(
            "This template has been updated on this device. Reload the editor.",
            "template_mismatch",
        )
    return template, None


def _no_preset_message(fallback: dict[str, Any] | None) -> str:
    """Say which file or which limit, not just "no preset"."""
    blocked = (fallback or {}).get("blocked") or []
    first = blocked[0] if blocked else {}
    if first.get("code") == "missing_model":
        size = f" ({first['gb']} GB)" if first.get("gb") else ""
        return (f"Nothing can run this locally yet: '{first.get('file')}' is not "
                f"installed. Put it in models/{first.get('folder')}/{size}.")
    if first.get("code") == "insufficient_vram":
        return (f"This action needs about {first.get('requiredVramGb')} GB of VRAM "
                f"and this GPU has {first.get('availableVramGb')} GB. Use "
                f"Reanimator Cloud for now.")
    if (fallback or {}).get("code") == "intent_not_supported":
        return f"No installed preset serves '{(fallback or {}).get('intent')}'."
    return "No local preset can run this action."


def _resolved(template, report, intent, quality_fallback, preset_fallback) -> dict[str, Any]:
    """What the bridge decided, and on what grounds.

    One block, always the same shape, so the editor never has to reconstruct the
    decision from the pieces -- reconstructing it is exactly how a client ends
    up with its own second selector.
    """
    checkpoint = _checkpoint_summary(getattr(report, "checkpoint", None))
    return {
        "intent": intent,
        "preset": template.id,
        "quality": getattr(report, "quality", None),
        "checkpoint": checkpoint,
        # Two things can be substituted: the preset, when the first choice
        # cannot run here, and the quality, when its models are missing.
        "fallback": preset_fallback or quality_fallback,
    }


def _incompatible_message(template, checkpoint: dict[str, Any]) -> str:
    """Say which format, and whether anything else could have run.

    "Incompatible checkpoint" on its own sends the user looking at their GPU.
    The actual problem is the ComfyUI build: it does not know how to read this
    file, and updating it -- or installing the alternative the manifest already
    names -- is what fixes it.
    """
    ops = ", ".join(checkpoint.get("unsupportedOps") or []) or "an unknown format"
    other = [
        s for s in (checkpoint.get("skipped") or [])
        if s.get("code") == "missing_model"
    ]
    tail = ""
    if other:
        names = ", ".join(str(s.get("file")) for s in other)
        tail = (f" This preset can also run on {names}, which is not installed "
                f"on this machine.")
    return (
        f"'{checkpoint.get('file')}' is stored as {ops}, which this ComfyUI "
        f"cannot read. Nothing was queued. Update ComfyUI, or use Reanimator "
        f"Cloud for this one.{tail}"
    )


def _checkpoint_summary(checkpoint: dict[str, Any] | None) -> dict[str, Any] | None:
    """What was selected and on what grounds. No absolute path, ever.

    referenceMeasurement is passed through under that name deliberately: it
    describes the machine the number was taken on, not the one about to run,
    and a field called "measurement" invited exactly that confusion.
    """
    if not checkpoint:
        return None
    out: dict[str, Any] = {
        "variant": checkpoint.get("variant"),
        "file": checkpoint.get("file"),
        "selectedBy": checkpoint.get("selectedBy"),
    }
    # What was passed over, and why. A silent automatic fallback is its own
    # bug: the user is running something other than the manifest's first
    # choice, and only this says so.
    for key in ("compatible", "unsupportedOps", "fallbackFrom", "skipped"):
        if checkpoint.get(key) is not None:
            out[key] = checkpoint[key]
    if checkpoint.get("referenceMeasurement"):
        out["referenceMeasurement"] = checkpoint["referenceMeasurement"]
    return out


def _consumed_input_ids(body: dict[str, Any]) -> list[str]:
    """Every input id this request used, whichever shape it arrived in."""
    inputs = body.get("inputs")
    if isinstance(inputs, dict):
        out: list[str] = []
        for given in inputs.values():
            for item in (given if isinstance(given, list) else [given]):
                if item:
                    out.append(str(item))
        return out
    return [str(i) for i in (body.get("inputIds") or [])]


def _apply_geometry(
    template: templates.Template, body: dict[str, Any], values: dict[str, Any],
    mutate: bool,
) -> tuple[geometry.Transform | None, list[str]]:
    """Pad the frames so the scaler stops cropping them, in place.

    The clean frame and the annotated frame get the *same call* -- not the same
    algorithm, the same Transform object. They are the same picture twice, and
    padding them independently is how the strokes drift off what they point at.

    The continuity board deliberately does NOT get that transform. It is a
    separate reference that may be any shape, so it is letterboxed into the
    canvas on its own terms. Forcing the frame's padding onto it would be
    meaningless arithmetic applied to an unrelated image.
    """
    roles = body.get("inputs") if isinstance(body.get("inputs"), dict) else None
    clean_slot = (template.role_slots("cleanFrame") or [None])[0]
    if not clean_slot or clean_slot not in values:
        return None, []

    def entry_for(slot: str) -> dict[str, Any] | None:
        if roles:
            for role, given in roles.items():
                slots = template.role_slots(str(role))
                given_list = given if isinstance(given, list) else [given]
                for s_name, input_id in zip(slots, [g for g in given_list if g]):
                    if s_name == slot:
                        return _input(input_id)
        for input_id, rec in _inputs.items():
            if rec["file"] == values.get(slot):
                return rec
        return None

    clean = entry_for(clean_slot)
    if not clean:
        return None, []

    source = (int(clean["width"]), int(clean["height"]))
    annotated_slot = (template.role_slots("annotatedFrame") or [None])[0]
    annotated = entry_for(annotated_slot) if annotated_slot and annotated_slot in values else None
    if annotated and (int(annotated["width"]), int(annotated["height"])) != source:
        raise BindError(
            f"The clean frame is {source[0]}x{source[1]} and the annotated frame is "
            f"{annotated['width']}x{annotated['height']}. They are the same picture "
            f"twice and must match exactly, or the strokes will not land where "
            f"they were drawn.",
            "frame_size_mismatch",
        )

    transform = geometry.plan(source, pad_mode=str(body.get("padMode") or "black"))
    replaced: list[str] = []
    if not mutate:
        # /validate is a dry run. Rewriting the uploads here consumed them, and
        # the /run that followed could not find the files it had just been told
        # were fine.
        return transform, replaced
    if transform.has_padding:
        for slot in [clean_slot] + ([annotated_slot] if annotated else []):
            old = str(values[slot])
            padded = geometry.pad(runner.read_input(old), transform)
            new = runner.write_input(padded, ".png")
            values[slot] = new
            replaced.append(new)
            runner.delete_input(old)

    # The board goes into the same canvas by its own fit, so the model sees one
    # consistent frame size without anything being stretched or cut.
    for slot in template.role_slots("contextFrames"):
        if slot in values:
            old = str(values[slot])
            fitted = geometry.fit_into(runner.read_input(old), transform.padded)
            new = runner.write_input(fitted, ".png")
            values[slot] = new
            replaced.append(new)
            runner.delete_input(old)

    return transform, replaced


def _prepare(body: dict[str, Any], refresh: bool):
    """Everything /validate and /run must agree on, in one place.

    Two checks kept deliberately separate (see validate.py):
      1. **the whole template**, every node including the disconnected ones,
         against the manifest's class allowlist;
      2. **the graph that will actually execute**, after pruning.

    refresh=True on the run path: the user may well have just installed the
    model the last attempt complained about, and a cached "no you haven't"
    sends them back to a problem they already fixed.
    """
    template, preset_fallback = _load_template(body)
    info = runner.object_info(refresh=refresh)
    intent = body.get("intent")

    security = validate.check_template_security(template, info)
    if not security.ok:
        security.preset_fallback = preset_fallback
        return template, security, {}, None, intent

    fit = validate.check_device_fit(template, runner.device_profile())
    if not fit.ok:
        fit.preset_fallback = preset_fallback
        return template, fit, {}, None, intent

    values, supplied = _values_for(template, body)

    intent_report = validate.check_intent(template, intent, supplied)
    if not intent_report.ok:
        intent_report.preset_fallback = preset_fallback
        return template, intent_report, values, None, intent

    # Quality decides which branch runs, so it decides which files are needed.
    quality, fallback = template.resolve_quality(
        str(body.get("quality") or "final"), runner.model_exists
    )
    blocked = fallback and (
        body.get("allowQualityFallback") is False or fallback.get("to") is None
    )
    if blocked:
        report = validate.Report()
        report.error(
            "missing_dependency",
            f"'{fallback['from']}' cannot run: {fallback['reason']}.",
            detail={"fallback": fallback},
        )
        report.preset_fallback = preset_fallback
        return template, report, values, fallback, intent

    quality_slot = template.quality_slot
    if quality_slot and quality_slot not in values:
        value = template.quality_value(quality)
        if value is not None:
            values[quality_slot] = value

    # Which weight format to load. Chosen from the manifest against this
    # machine's measured profile -- never by the caller, who has no business
    # naming a file in the models folder.
    checkpoint = template.select_checkpoint(
        runner.device_profile(), runner.model_exists,
        override=body.get("checkpointVariant"),
        quality=quality,
        pictures=_picture_count(values),
        supported_ops=runner.quantization_formats(),
    )
    # Refuse here, before a single node is queued. ComfyUI would discover this
    # while loading the weights and report it as a KeyError from UNETLoader --
    # a minute of the user's time to say something the manifest already knew.
    if checkpoint and checkpoint.get("compatible") is False:
        report = validate.Report()
        report.error(
            "incompatible_checkpoint",
            _incompatible_message(template, checkpoint),
            detail={"checkpoint": _checkpoint_summary(checkpoint)},
        )
        report.checkpoint = checkpoint
        report.preset_fallback = preset_fallback
        return template, report, values, fallback, intent

    slot = (template.checkpoints.get("slot") or "$CHECKPOINT") if checkpoint else None
    if checkpoint and slot:
        values[str(slot)] = str(checkpoint["file"])

    # The prompt describes the images, so it is built from the same set of
    # supplied roles that decides which inputs stay plugged in. One decision,
    # not two that can drift apart and leave the text describing a Picture 3
    # that no longer exists.
    transform, padded_files = _apply_geometry(template, body, values, mutate=refresh)
    # A collected role is one loader taking the whole list, so its pictures
    # have to agree on a size before anything else looks at them.
    matched, resized = _match_collected_sizes(template, values, mutate=refresh)
    padded_files += matched

    prompt_slot = (template.manifest.get("prompt") or {}).get("slot")
    instruction = (body.get("parameters") or {}).get("instruction")
    if prompt_slot and instruction is not None:
        text = validate.compose_prompt(template, str(instruction), supplied, intent)
        if transform and transform.has_padding:
            # Without saying so, the model paints scene into the bars. The
            # editor learned this the hard way on the cloud path.
            text = (
                "The black border is temporary padding outside the original "
                "frame. Do not extend the scene into it and do not treat it as "
                "part of the image.\n\n" + text
            )
        values[str(prompt_slot)] = text

    report = validate.validate(
        template, values, info, runner.model_exists, selected_checkpoint=checkpoint
    )
    report.quality = quality
    report.checkpoint = checkpoint
    report.transform = transform
    report.padded_files = padded_files
    report.resized = resized
    report.preset_fallback = preset_fallback
    return template, report, values, fallback, intent


@routes.post("/rb/v1/validate")
async def validate_run(request: web.Request) -> web.Response:
    body = await request.json()
    template, report, values, fallback, intent = _prepare(body, refresh=False)
    payload: dict[str, Any] = {
        "templateId": template.id, "intent": intent,
        "resolved": _resolved(template, report, intent, fallback, getattr(report, "preset_fallback", None)),
        **report.as_dict(),
    }
    if fallback:
        payload["fallback"] = fallback
    if getattr(report, "checkpoint", None):
        payload["checkpoint"] = _checkpoint_summary(report.checkpoint)
    frames = _frames_note(template, body, values)
    if frames:
        payload["frames"] = frames
    if getattr(report, "resized", None):
        payload["resized"] = report.resized
    prompt_slot = (template.manifest.get("prompt") or {}).get("slot")
    if prompt_slot and values.get(str(prompt_slot)) is not None:
        payload["composedPrompt"] = values[str(prompt_slot)]
    return _json(payload)


@routes.post("/rb/v1/run")
async def start_run(request: web.Request) -> web.Response:
    body = await request.json()
    template, report, values, fallback, intent = _prepare(body, refresh=True)
    quality = getattr(report, "quality", None)
    if not report.ok:
        first = report.errors[0]
        payload: dict[str, Any] = {
            # The specific code, not a blanket one: "this action needs an
            # annotated frame" and "this template uses a node you have not
            # installed" call for completely different reactions in the editor.
            "error": {"code": first.code, "message": first.message},
            "templateId": template.id,
            "intent": intent,
            "resolved": _resolved(template, report, intent, fallback, getattr(report, "preset_fallback", None)),
            **report.as_dict(),
        }
        if fallback:
            payload["fallback"] = fallback
        if getattr(report, "resized", None):
            payload["resized"] = report.resized
        return _json(payload, status=422)
    if report.pruned:
        log.info("Pruned %s: %s", template.id, report.pruned)

    consumed = _consumed_input_ids(body)
    # The padded copies are what ended up in the graph, so they are what has to
    # be deleted afterwards. Listing the originals would leave the real files
    # behind and try to remove names that no longer exist.
    used = [_inputs[i]["file"] for i in consumed if i in _inputs]
    used += [f for f in getattr(report, "padded_files", []) if f not in used]
    run = runner.start(
        template.id,
        report.graph,
        binder.output_node_ids(report.slots),
        inputs=used,
    )
    run.pruned = report.pruned
    run.transform = getattr(report, "transform", None)
    run.geometry = run.transform.as_dict() if run.transform else None
    run.model_resolution = (
        "x".join(map(str, run.transform.model)) if run.transform else None
    )
    run.quality = quality
    run.fallback = fallback
    run.intent = intent
    run.checkpoint = _checkpoint_summary(getattr(report, 'checkpoint', None))
    run.pictures = _picture_count(values)
    # The files now belong to the run, which deletes them when it finishes.
    for input_id in consumed:
        _inputs.pop(input_id, None)

    payload = run.as_dict()
    payload["resolved"] = _resolved(template, report, intent, fallback, getattr(report, "preset_fallback", None))
    frames = _frames_note(template, body, values)
    if frames:
        payload["frames"] = frames
    if getattr(report, "resized", None):
        payload["resized"] = report.resized
    if report.warnings:
        payload["warnings"] = [w.as_dict() for w in report.warnings]
    return _json(payload, status=202)


@routes.get("/rb/v1/run/{run_id}")
async def run_status(request: web.Request) -> web.Response:
    return _json(runner.runs.get(request.match_info["run_id"]).as_dict())


@routes.post("/rb/v1/run/{run_id}/interrupt")
async def run_interrupt(request: web.Request) -> web.Response:
    run = runner.runs.get(request.match_info["run_id"])
    await runner.cancel(run)
    return _json(run.as_dict())


@routes.get("/rb/v1/output/{run_id}/{ref}")
async def run_output(request: web.Request) -> web.StreamResponse:
    """Result bytes.

    Reuses media.serve, so a generated video gets Range, 206 and 416 for free
    and <video> can seek it without downloading the whole file. The ref is
    opaque and per-run -- no filename, no subfolder, no path anywhere in the URL.
    """
    run = runner.runs.get(request.match_info["run_id"])
    ref = request.match_info["ref"]
    item = run.files.get(ref)
    if item is None:
        raise MediaError("Unknown or expired result reference.", 404)

    path = runner.resolve_output(item)
    # Undo the padding and restore the source resolution exactly. Done here so
    # ComfyUI's own output file is left untouched, and cached so a second fetch
    # does not resample the image again.
    if run.transform is not None and item.get("kind") == "image":
        corrected = runner.corrected_output(run, ref, path)
        if corrected is not None:
            path = corrected
    cap = media.Capability(
        id=request.match_info["ref"],
        path=path,
        mime=item["mime"],
        size=path.stat().st_size,
        expires_at=0,
    )
    return await media.serve(request, cap)


# --------------------------------------------------------------------------
# Local media
# --------------------------------------------------------------------------

@routes.get("/rb/v1/project/media")
async def project_media(_: web.Request) -> web.Response:
    """List media under the user-authorized project root.

    The browser cannot know a real path -- it only ever sees `ref`, a name
    relative to the root, which is meaningless outside this bridge.
    """
    root = media.project_root()
    if root is None:
        return _error(
            "no_project_root",
            "No project folder chosen yet. Pick one in the Reanimator panel "
            "inside ComfyUI.",
            409,
        )
    return _json({"root": root.name, "items": media.list_media(root)})


@routes.post("/rb/v1/project/media/grant")
async def grant_media(request: web.Request) -> web.Response:
    """Exchange a `ref` for an opaque, single-file, read-only capability URL."""
    root = media.project_root()
    if root is None:
        return _error("no_project_root", "No project folder chosen yet.", 409)

    body = await request.json()
    ref = body.get("ref")
    if not isinstance(ref, str):
        return _error("bad_request", "Missing 'ref'.", 400)

    path = media.resolve_within(root, ref)
    cap = media.capabilities.grant(path)
    return _json(
        {
            "capabilityId": cap.id,
            "url": f"/rb/v1/media/{cap.id}",
            "mime": cap.mime,
            "bytes": cap.size,
            "expiresAt": cap.expires_at,
        }
    )


@routes.post("/rb/v1/project/media/revoke")
async def revoke_media(request: web.Request) -> web.Response:
    body = await request.json()
    capability_id = body.get("capabilityId")
    if capability_id is None:
        return _json({"revoked": media.capabilities.revoke_all()})
    return _json({"revoked": media.capabilities.revoke(str(capability_id))})


@routes.get("/rb/v1/media/{capability_id}")
async def read_media(request: web.Request) -> web.StreamResponse:
    """Range-aware read. The capability id in the path IS the credential.

    No @routes.head here: aiohttp registers HEAD alongside every GET, and adding
    it explicitly raises "method HEAD is already registered" at startup.
    media.serve() checks request.method, so HEAD is handled either way.
    """
    cap = media.capabilities.resolve(request.match_info["capability_id"])
    return await media.serve(request, cap)


# --------------------------------------------------------------------------
# Local project storage  (docs/contract-local-project-storage.md)
#
# NOT the same thing as /rb/v1/project/media above, despite the one letter
# between them. That one LISTS a folder the user authorized us to read; this
# one WRITES the project's own bytes into a folder the bridge owns. Local GPU
# promised the user's material never leaves the machine, and this is the half
# that gives it somewhere to land.
#
# Every route here is token-gated like the rest of /rb/v1: /rb/v1/projects/ is
# not in PUBLIC_ROUTES and matches neither of auth_middleware's prefix
# exemptions (/rb/v1/pair/ and /rb/v1/media/).
# --------------------------------------------------------------------------

@routes.post("/rb/v1/projects/{project_id}/assets/verify")
async def verify_project_assets(request: web.Request) -> web.Response:
    """What survived on this disk, for the whole project in one call.

    Registered BEFORE the {asset_id} routes so the literal wins outright, and
    on a word no valid asset id could collide with anyway: projects.ID_RE
    demands at least eight characters and "verify" is six.
    """
    body = await request.json()
    refs = body.get("refs")
    if not isinstance(refs, list):
        return _error("bad_request", "'refs' must be a list.", 400)
    results = projects.verify(request.match_info["project_id"], refs)
    return _json({"results": results})


@routes.get("/rb/v1/projects/{project_id}/files")
async def list_project_files(request: web.Request) -> web.Response:
    """The physical inventory: one entry per file, not per asset id.

    What the orphan collector compares against. It has to be files, because
    one asset id can have two of them (see projects.list_files).
    """
    return _json({"items": projects.list_files(request.match_info["project_id"])})


@routes.delete("/rb/v1/projects/{project_id}/files/{filename}")
async def delete_project_file(request: web.Request) -> web.Response:
    """Delete exactly the named file."""
    projects.delete_file(
        request.match_info["project_id"], request.match_info["filename"]
    )
    return web.Response(status=204)


@routes.put("/rb/v1/projects/{project_id}/assets/{asset_id}")
async def put_project_asset(request: web.Request) -> web.Response:
    """Raw bytes in, verified reference out."""
    data = await request.read()
    stored = projects.write(
        request.match_info["project_id"],
        request.match_info["asset_id"],
        request.headers.get("Content-Type"),
        data,
        request.headers.get("X-Sha256"),
    )
    return _json(stored)


@routes.get("/rb/v1/projects/{project_id}/assets/{asset_id}")
async def get_project_asset(request: web.Request) -> web.StreamResponse:
    """Reuses media.serve, so Range, 206 and 416 come for free."""
    path = projects.find_asset(
        request.match_info["project_id"], request.match_info["asset_id"]
    )
    if path is None:
        raise MediaError("No such asset in this project.", 404)
    cap = media.Capability(
        id=request.match_info["asset_id"],
        path=path,
        mime=projects.mime_of(path),
        size=path.stat().st_size,
        expires_at=0,
    )
    return await media.serve(request, cap)


@routes.delete("/rb/v1/projects/{project_id}/assets/{asset_id}")
async def delete_project_asset(request: web.Request) -> web.Response:
    projects.delete(
        request.match_info["project_id"], request.match_info["asset_id"]
    )
    # 204 whether or not it was there: deleting something already gone is the
    # outcome the caller wanted, not an error worth a retry loop.
    return web.Response(status=204)


# --------------------------------------------------------------------------
# Lifecycle
# --------------------------------------------------------------------------

def build_app() -> web.Application:
    # ORDER MATTERS. aiohttp applies middlewares in reverse, so the first entry
    # is the OUTERMOST. cors_middleware has to be outermost or every error
    # response produced by error_middleware escapes without CORS headers, and
    # the browser then hides the status and body from JavaScript -- turning a
    # perfectly clear 400 into an opaque "Failed to fetch".
    app = web.Application(
        middlewares=[cors_middleware, error_middleware, auth_middleware],
        # aiohttp's default is 1 MB, which a single 1024x1024 keyframe can
        # exceed. The real cap is runner.MAX_INPUT_BYTES, enforced in
        # write_input with a message the editor can show; this only has to be
        # loose enough that aiohttp does not 413 the request out from under it.
        client_max_size=runner.MAX_INPUT_BYTES + 1024 * 1024,
    )
    app.add_routes(routes)
    return app


async def start(port: int | None = None) -> tuple[web.AppRunner, int]:
    """Bind 127.0.0.1 only, trying the configured port then the fallbacks."""
    cfg = config.load()
    candidates = [port or cfg.get("port", config.DEFAULT_PORT), *config.FALLBACK_PORTS]

    runner = web.AppRunner(build_app(), access_log=None)
    await runner.setup()

    last_error: OSError | None = None
    for candidate in candidates:
        try:
            site = web.TCPSite(runner, host="127.0.0.1", port=candidate)
            await site.start()
        except OSError as exc:
            last_error = exc
            continue
        if candidate != cfg.get("port"):
            config.update(port=candidate)
        log.info("Reanimator Bridge listening on http://127.0.0.1:%d", candidate)
        return runner, candidate

    await runner.cleanup()
    raise RuntimeError(
        f"Could not bind any of {candidates} on 127.0.0.1: {last_error}"
    )
