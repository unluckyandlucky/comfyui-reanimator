"""Installed templates: the only workflows this bridge will ever run.

A template is the user's own working ComfyUI workflow, exported in API format,
with a handful of nodes retitled (see docs/local-bridge-handoff.md §5). Nothing
here downloads or installs anything: templates ship inside the bridge release,
so their trust comes from the package itself.

Lookup is by id through a dict built by scanning the templates directory. An id
never becomes a path component, so a crafted `templateId` cannot reach outside
this folder -- an unknown id is simply a miss.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

log = logging.getLogger("reanimator.bridge")

TEMPLATES_DIR = Path(__file__).resolve().parents[2] / "templates"

# Ids are echoed back to the editor and used as dict keys. Keeping them to a
# strict slug means they can never be mistaken for a path, even by future code.
ID_RE = re.compile(r"^[a-z0-9][a-z0-9-]{1,63}$")


class TemplateError(Exception):
    def __init__(self, message: str, code: str = "template_error") -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class Template:
    id: str
    manifest: dict[str, Any]
    graph: dict[str, Any]
    hash: str

    @property
    def name(self) -> str:
        return str(self.manifest.get("name") or self.id)

    @property
    def kind(self) -> str:
        return str(self.manifest.get("kind") or "image")

    @property
    def images(self) -> int:
        """How many input images the workflow has slots for."""
        return int(self.manifest.get("images") or 0)

    @property
    def slots(self) -> dict[str, dict[str, Any]]:
        return dict(self.manifest.get("slots") or {})

    @property
    def requires(self) -> dict[str, Any]:
        return dict(self.manifest.get("requires") or {})

    @property
    def roles(self) -> dict[str, Any]:
        return dict(self.manifest.get("roles") or {})

    @property
    def intents(self) -> dict[str, dict[str, Any]]:
        """Which actions this template serves, and what each one demands.

        Accepts the short form ``["edit_pose"]`` -- meaning "serves this intent,
        with the roles' own required flags" -- as well as the full form, where an
        intent tightens them: edit_pose needs a drawing, edit_keyframe does not.
        """
        raw = self.manifest.get("intents")
        if isinstance(raw, dict):
            return {k: dict(v) if isinstance(v, dict) else {} for k, v in raw.items()}
        if isinstance(raw, list):
            return {str(name): {} for name in raw}
        return {}

    def required_roles(self, intent: str | None) -> list[str]:
        spec = self.intents.get(intent or "", {}) if intent else {}
        if spec.get("requires"):
            return [str(r) for r in spec["requires"]]
        return [
            name for name, role in self.roles.items()
            if isinstance(role, dict) and role.get("required")
        ]

    def role_slots(self, name: str) -> list[str]:
        role = self.roles.get(name)
        if not isinstance(role, dict):
            return []
        return [str(s) for s in (role.get("slots") or ([role["slot"]] if role.get("slot") else []))]

    def role_max(self, name: str) -> int:
        """How many images this role can actually take.

        Slots are fixed per role: $IMAGE_2 belongs to annotatedFrame and is
        detached when it is absent, never handed to somebody else. So the limit
        is NOT "images minus the ones already used" -- inferring it that way
        promises a second reference that has nowhere to bind, and the model
        never sees it.

        A manifest may also declare `max` explicitly. Whichever is smaller wins:
        the declaration cannot conjure a slot that does not exist, and a slot
        cannot be used past a limit the template states.

        A **collecting** role breaks the one-image-per-slot arithmetic on
        purpose: its single slot is a loader that takes a whole list, so the
        limit is what the manifest declares and nothing else. Counting slots
        there would cap a fifty-keyframe sequencer at one image.
        """
        role = self.roles.get(name)
        slots = len(self.role_slots(name))
        if not isinstance(role, dict):
            return slots
        declared = role.get("max")
        if isinstance(declared, bool) or not isinstance(declared, int) or declared < 0:
            declared = None
        if role.get("collect"):
            return declared if declared is not None else 0
        if declared is None:
            return slots
        return min(declared, slots) if slots else 0

    def role_collects(self, name: str) -> bool:
        """True when this role's one slot takes the whole list of images."""
        role = self.roles.get(name)
        return bool(isinstance(role, dict) and role.get("collect"))

    @property
    def frames(self) -> dict[str, Any]:
        """How long a run may be, in this model's own arithmetic.

        LTX wants 8n+1 frames; the next local video model will want something
        else. Keeping the rule here is what lets the editor send the span it
        actually means and stay ignorant of which model is installed.
        """
        block = self.manifest.get("frames")
        return dict(block) if isinstance(block, dict) else {}

    def legal_frame_count(self, wanted: int) -> int:
        """Round UP to the nearest length this preset can render.

        Up, never down: rounding down would drop the tail of the shot -- the
        last keyframe could fall outside the video and the sequencer would
        refuse it, or worse, land on the final frame and be seen for an instant.
        """
        rule = self.frames
        if not rule:
            return int(wanted)
        minimum = int(rule.get("min") or 1)
        step = int(rule.get("step") or 1)
        offset = int(rule.get("offset") or 0)
        count = max(minimum, int(wanted))
        # DOWN, and capped, when the frames come from an existing clip: a
        # re-shot video cannot be longer than the video it re-shoots, and the
        # frames past its end would be generated from nothing.
        down = rule.get("round") == "down"
        maximum = rule.get("max")
        if isinstance(maximum, int) and not isinstance(maximum, bool):
            count = min(count, maximum)
        if step > 1:
            remainder = (count - offset) % step
            if remainder:
                count += -remainder if down else step - remainder
        return max(count, minimum)

    @property
    def quality(self) -> dict[str, Any]:
        """The quality LEVELS only.

        The manifest block also carries "slot", which names the widget the level
        is written into. Treating that as a level makes "slot" a candidate
        quality and the fallback picks it -- silently, because it is a perfectly
        good string.
        """
        block = self.manifest.get("quality") or {}
        return {k: v for k, v in block.items() if isinstance(v, dict)}

    @property
    def quality_slot(self) -> str | None:
        block = self.manifest.get("quality") or {}
        slot = block.get("slot")
        return str(slot) if isinstance(slot, str) else None

    def models_for_quality(self, quality: str) -> list[dict[str, Any]]:
        """Conditional models this quality setting will actually load."""
        return [
            entry
            for entry in self.requires.get("conditionalModels") or []
            if isinstance(entry, dict) and quality in (entry.get("requiredFor") or [])
        ]

    def resolve_quality(
        self, wanted: str, file_exists: Any
    ) -> tuple[str, dict[str, Any] | None]:
        """Pick the quality that can actually run, and say so if it changed.

        Preview means "take the Lightning branch". If that file is not
        installed, queueing anyway produces a run we already know will fail --
        so this steps down to Final instead and reports the substitution.
        A warning followed by a doomed execution is not an acceptable answer.
        """
        wanted = wanted if wanted in self.quality else "final"
        if file_exists is None:
            return wanted, None

        missing = [
            entry for entry in self.models_for_quality(wanted)
            if not file_exists(str(entry.get("folder")), str(entry.get("file")))
        ]
        if not missing:
            return wanted, None

        alternative = next(
            (name for name in self.quality if name != wanted and not [
                e for e in self.models_for_quality(name)
                if not file_exists(str(e.get("folder")), str(e.get("file")))
            ]),
            None,
        )
        reason = f"{missing[0].get('file')} is not installed"
        if alternative is None:
            return wanted, {"from": wanted, "to": None, "reason": reason}
        return alternative, {"from": wanted, "to": alternative, "reason": reason}

    @property
    def tier(self) -> str:
        """Which hardware bracket this preset targets. Two, deliberately:
        'full' (24 GB+) and, when it exists, 'lite' (12-16 GB)."""
        return str(self.manifest.get("tier") or "full")

    @property
    def minimum_vram_gb(self) -> float | None:
        value = self.manifest.get("minimumVramGb")
        return float(value) if isinstance(value, (int, float)) else None

    @property
    def checkpoints(self) -> dict[str, Any]:
        return dict(self.requires.get("checkpoints") or {})

    def select_checkpoint(
        self,
        profile: Mapping[str, Any] | None = None,
        file_exists: Any = None,
        override: str | None = None,
        quality: str | None = None,
        model_resolution: str | None = None,
        pictures: int | None = None,
        supported_ops: set[str] | None = None,
    ) -> dict[str, Any] | None:
        """Pick the weight format to load. Static, by design.

        The manifest names a ``preferred`` variant and an ordered list of
        ``alternatives``; the first one that is **installed and loadable here**
        wins. An earlier version of this ranked variants by locally recorded
        benchmarks and it was a bad trade: one cold run wrote a total-seconds
        row, the comparison put it against a per-step row, and the selector
        demoted the checkpoint it had just measured as faster. Telemetry is
        still recorded -- for diagnosis, not for control. Deciding by
        measurement needs data from more than one machine before it is worth
        the failure modes.

        "Installed" was never enough on its own. A variant declares the
        quantization formats it needs in ``requiresNativeOps``; a ComfyUI that
        does not know one of them does not run it slowly, it dies loading it.
        ``supported_ops`` is what this build actually understands
        (runner.quantization_formats). ``None`` means nobody could tell, and
        then nothing is screened out -- the same behaviour as before this
        check existed.
        """
        block = self.checkpoints
        variants = block.get("variants") or {}
        if not variants:
            return None

        def installed(entry: Mapping[str, Any]) -> bool:
            if file_exists is None:
                return True
            return bool(file_exists(str(entry.get("folder")), str(entry.get("file"))))

        def missing_ops(entry: Mapping[str, Any]) -> list[str]:
            if supported_ops is None:
                return []
            return [
                str(op) for op in (entry.get("requiresNativeOps") or [])
                if str(op) not in supported_ops
            ]

        def result(name: str, selected_by: str, **extra: Any) -> dict[str, Any]:
            entry = dict(variants[name])
            absent = missing_ops(entry)
            return {
                "variant": name, **entry, "selectedBy": selected_by,
                "installed": installed(entry),
                # Whether this machine can load it at all, which is a different
                # question from whether the file is on disk.
                "compatible": not absent,
                **({"unsupportedOps": absent} if absent else {}),
                # Kept in the response because it is useful context, and named
                # so nobody mistakes it for a measurement of the machine that is
                # about to run: it belongs to the reference machine.
                "referenceMeasurement": entry.get("referenceMeasurement"),
                **extra,
            }

        # An explicit variant is the user saying "I know". It is still reported
        # as incompatible when it is, so the run refuses instead of crashing.
        if override and override in variants:
            return result(override, "user-override")

        preferred = str(block.get("preferred") or next(iter(variants)))
        order = [preferred, *[str(n) for n in block.get("alternatives") or []]]
        order += [n for n in variants if n not in order]

        # Why each earlier candidate was passed over. This is the whole value of
        # falling back automatically: silently loading a different checkpoint
        # than the manifest prefers, and never saying so, is its own bug.
        skipped: list[dict[str, Any]] = []
        for name in order:
            if name not in variants:
                continue
            entry = variants[name]
            absent = missing_ops(entry)
            if absent:
                skipped.append({"variant": name, "code": "unsupported_quantization",
                                "missingOps": absent})
                continue
            if not installed(entry):
                skipped.append({"variant": name, "code": "missing_model",
                                "file": entry.get("file")})
                continue
            extra: dict[str, Any] = {}
            if skipped:
                extra["fallbackFrom"] = skipped[0]
                extra["skipped"] = skipped
            return result(
                name, "preferred" if name == preferred else "alternative", **extra
            )

        # Nothing usable. Return the preferred one anyway -- with the reasons
        # attached -- so the caller can name the file, its download URL, or the
        # formats this build cannot read, instead of a bare "no checkpoint".
        return result(preferred, "preferred", skipped=skipped)

    def quality_value(self, quality: str) -> Any:
        block = self.quality.get(quality)
        return block.get("value") if isinstance(block, dict) else None

    def detach_plan(self, supplied: set[str]) -> list[dict[str, str]]:
        """Links to unplug because the slots that feed them got no value.

        Driven entirely by the installed manifest. The request can say "there is
        no context frame this time"; it can never name a wire to cut.
        """
        removals: list[dict[str, str]] = []
        for role in self.roles.values():
            if not isinstance(role, dict):
                continue
            slots = role.get("slots") or ([role["slot"]] if role.get("slot") else [])
            role_supplied = any(slot in supplied for slot in slots)
            for removal in role.get("detachWhenAbsent") or []:
                if not (isinstance(removal, dict) and removal.get("slot") and removal.get("input")):
                    continue
                # By default a removal fires when the whole role is empty. With
                # `whenSlotAbsent` it fires when that ONE slot got nothing: a
                # first/last-frame preset has one role with two slots, and one
                # image must unplug the last-frame input while the first stays.
                only = removal.get("whenSlotAbsent")
                if only is not None and str(only) not in slots:
                    continue  # names a slot of another role: a manifest typo, not a wire
                absent = (str(only) not in supplied) if only is not None else not role_supplied
                if absent:
                    removals.append(
                        {"slot": str(removal["slot"]), "input": str(removal["input"])}
                    )
        return removals

    def summary(self) -> dict[str, Any]:
        """What the editor is told. No paths, no graph."""
        return {
            "id": self.id,
            "name": self.name,
            # The MODEL this preset implements ("ltx", "minimax-h3-flf"), which is
            # what the editor shows. Two presets can serve the same intent with
            # different models, so choosing by intent alone could run MiniMax
            # under a menu that says LTX.
            **({"model": self.manifest["model"]} if self.manifest.get("model") else {}),
            "kind": self.kind,
            # The rate this preset renders at, when it has one. The editor
            # cannot check a project against it otherwise, and an unchecked
            # mismatch is the worst kind: the keys land on the frames they were
            # asked for, so the shot comes back correct and playing at the
            # wrong speed, with nothing on screen connecting the two.
            **({"fps": self.manifest["fps"]} if self.manifest.get("fps") else {}),
            "version": self.manifest.get("version", 1),
            "description": self.manifest.get("description"),
            "images": self.images,
            "hash": self.hash,
            "slots": sorted(self.slots),
            # The editor picks a template from this list, so what it needs to
            # choose by has to be in it. Without them the only distinguishing
            # field was the id, and selecting on a literal id is how the client
            # ended up preferring the older template forever.
            "intents": sorted(self.intents),
            "roles": sorted(self.roles),
            # How many images each role takes, stated rather than left to be
            # inferred. The editor was working it out as images-minus-the-ones-
            # in-use, which reads like arithmetic and is wrong: slots belong to
            # roles, so a free $IMAGE_2 is not a spare reference slot. Saying it
            # here is the only way the editor can offer exactly what will
            # actually reach the model.
            "roleMax": {name: self.role_max(name) for name in sorted(self.roles)},
            "priority": int(self.manifest.get("priority") or 0),
            "tier": self.tier,
            # So a menu can leave out what this card cannot run: the cloud
            # variants ship in the same folder and asked for 48 GB on a 3090.
            "minimumVramGb": self.minimum_vram_gb,
        }


_lock = threading.RLock()
_cache: dict[str, Template] | None = None
_cache_stamp: tuple | None = None


def _stamp(base: Path) -> tuple:
    """Fingerprint of the templates folder: names and modification times.

    Without this, dropping a template into the folder does nothing until ComfyUI
    is restarted -- and restarting ComfyUI to pick up a JSON file is the kind of
    step that gets left out of the instructions and then wastes somebody's
    afternoon.
    """
    try:
        return tuple(
            sorted((p.name, p.stat().st_mtime_ns, p.stat().st_size)
                   for p in base.glob("*.json"))
        )
    except OSError:
        return ()


def _hash(manifest_bytes: bytes, graph_bytes: bytes) -> str:
    """Covers the manifest as well as the graph.

    Hashing only the graph would let a manifest edit -- which is what declares
    the required models and the slot contract -- pass unnoticed by a client that
    pinned a hash.
    """
    digest = hashlib.sha256()
    digest.update(manifest_bytes)
    digest.update(b"\x00")
    digest.update(graph_bytes)
    return f"sha256:{digest.hexdigest()}"


MODEL_SUFFIXES = (
    ".safetensors", ".ckpt", ".pt", ".pth", ".bin", ".gguf", ".sft", ".onnx",
)


def native_model_paths(graph: dict[str, Any], sep: str = os.sep) -> dict[str, Any]:
    """Model filenames in subfolders, written with this platform's separator.

    ComfyUI lists a model in a subfolder as ``ltx2\\file`` on Windows and
    ``ltx2/file`` on Linux, and compares the widget value to that list as a
    plain string. Templates are exported from whichever machine built them, so
    without this a graph made on Windows reports its LoRA as missing on the
    Linux cloud worker, with the file sitting right there. Edits in place.
    """
    for node in graph.values():
        inputs = node.get("inputs") if isinstance(node, dict) else None
        if not isinstance(inputs, dict):
            continue
        for widget, value in inputs.items():
            if (
                isinstance(value, str)
                and value.lower().endswith(MODEL_SUFFIXES)
                and ("/" in value or "\\" in value)
            ):
                inputs[widget] = value.replace("\\", "/").replace("/", sep)
    return graph


def _load_one(manifest_path: Path) -> Template | None:
    graph_path = manifest_path.with_name(
        manifest_path.name[: -len(".manifest.json")] + ".json"
    )
    if not graph_path.is_file():
        log.warning("Template %s has no graph next to it; skipped", manifest_path.name)
        return None

    try:
        # utf-8-sig for the same reason config.py uses it: an editor that adds a
        # BOM would otherwise make a perfectly good template unreadable.
        manifest_bytes = manifest_path.read_bytes()
        graph_bytes = graph_path.read_bytes()
        manifest = json.loads(manifest_bytes.decode("utf-8-sig"))
        graph = json.loads(graph_bytes.decode("utf-8-sig"))
    except (OSError, ValueError) as exc:
        log.error("Template %s is unreadable (%s); skipped", manifest_path.name, exc)
        return None

    template_id = str(manifest.get("id") or "")
    if not ID_RE.match(template_id):
        log.error("Template %s has an invalid id %r; skipped", manifest_path.name, template_id)
        return None
    if not isinstance(graph, dict) or not graph:
        log.error("Template %s has an empty graph; skipped", template_id)
        return None

    return Template(
        id=template_id,
        manifest=manifest,
        graph=native_model_paths(graph),
        hash=_hash(manifest_bytes, graph_bytes),
    )


def load_all(directory: Path | None = None) -> dict[str, Template]:
    """Every installed template, keyed by id.

    Cached, but the cache re-checks the folder's modification times, so a
    template added or edited on disk is picked up on the next call.
    """
    global _cache, _cache_stamp
    with _lock:
        base = directory or TEMPLATES_DIR
        stamp = _stamp(base)
        if directory is None and _cache is not None and stamp == _cache_stamp:
            return dict(_cache)

        found: dict[str, Template] = {}
        if base.is_dir():
            for manifest_path in sorted(base.glob("*.manifest.json")):
                template = _load_one(manifest_path)
                if template is not None:
                    found[template.id] = template

        if directory is None:
            _cache = dict(found)
            _cache_stamp = stamp
        return found


def reset() -> None:
    global _cache, _cache_stamp
    with _lock:
        _cache = None
        _cache_stamp = None


def get(template_id: str) -> Template:
    if not isinstance(template_id, str) or not ID_RE.match(template_id):
        raise TemplateError("Unknown template.", "unknown_template")
    template = load_all().get(template_id)
    if template is None:
        raise TemplateError(f"Template not installed: {template_id}", "unknown_template")
    return template


def summaries() -> list[dict[str, Any]]:
    return [t.summary() for t in sorted(load_all().values(), key=lambda t: t.id)]


def blocking_reason(
    template: Template, profile: Mapping[str, Any] | None, file_exists: Any
) -> dict[str, Any] | None:
    """Why this preset cannot run here, or None if it can."""
    minimum = template.minimum_vram_gb
    have_mb = (profile or {}).get("vramTotalMb")
    if minimum and isinstance(have_mb, (int, float)) and have_mb / 1024 + 0.5 < minimum:
        return {
            "code": "insufficient_vram", "preset": template.id, "tier": template.tier,
            "requiredVramGb": minimum, "availableVramGb": round(have_mb / 1024, 1),
        }
    if file_exists is not None:
        for entry in template.requires.get("models") or []:
            if isinstance(entry, dict) and not file_exists(
                str(entry.get("folder")), str(entry.get("file"))
            ):
                return {"code": "missing_model", "preset": template.id,
                        "file": entry.get("file"), "folder": entry.get("folder"),
                        "gb": entry.get("gb"), "url": entry.get("url")}
        checkpoint = template.select_checkpoint(profile, file_exists)
        if checkpoint and not checkpoint.get("installed"):
            return {"code": "missing_model", "preset": template.id,
                    "file": checkpoint.get("file"), "folder": checkpoint.get("folder"),
                    "gb": checkpoint.get("gb"), "url": checkpoint.get("url")}
    return None


def resolve_for_intent(
    intent: str | None,
    profile: Mapping[str, Any] | None = None,
    file_exists: Any = None,
) -> tuple[Template | None, dict[str, Any] | None]:
    """Pick the preset for an action. This decision belongs to the bridge.

    The browser knows what the user is doing; only the bridge knows what is
    installed, what the GPU is, and which manifest declares what. A client-side
    selector is a second, independent implementation of this that drifts the
    moment a preset is added -- which it did: local-gpu.js preferred a template
    by literal id and quietly kept choosing the older one, reporting the
    mismatch as a malformed request rather than a bad choice.

    Returns the template and, when the obvious choice was unusable, a fallback
    record saying which one and why.
    """
    # Only templates that DECLARE the intent. Treating "declares nothing" as
    # "serves everything" made the pre-roles template a candidate for
    # edit_pose, and it has no cleanFrame to bind -- the request would be
    # accepted and then fail describing itself as malformed.
    everything = load_all().values()
    candidates = [t for t in everything if intent and intent in t.intents]
    if not intent:
        candidates = [t for t in everything if t.intents]
    if not candidates:
        return None, {"code": "intent_not_supported", "intent": intent,
                      "available": sorted({i for t in load_all().values() for i in t.intents})}

    candidates.sort(key=lambda t: (-int(t.manifest.get("priority") or 0), t.id))
    blocked: list[dict[str, Any]] = []
    for index, template in enumerate(candidates):
        reason = blocking_reason(template, profile, file_exists)
        if reason is None:
            fallback = None
            if index > 0:
                fallback = {"from": candidates[0].id, "to": template.id,
                            "reason": blocked[0]}
            return template, fallback
        blocked.append(reason)

    # Nothing runnable: hand back the first choice's reason, because that is the
    # one the user would have to fix to get the preset they actually want.
    return None, {"code": "no_runnable_preset", "intent": intent,
                  "blocked": blocked[:3], "recommend": "cloud"}
