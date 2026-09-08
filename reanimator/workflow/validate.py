"""Pre-flight validation against ComfyUI's own ``/object_info``.

Everything here happens **before the prompt is queued**, and that is the whole
point. ComfyUI will of course fail on a missing ``.safetensors`` too -- after the
user has waited for a model to load, and with an error buried in a console they
may not be looking at. The bridge can say *"you are missing this file"* in one
second instead.

The other half is the security half (docs/plan-local-bridge.md §5). The cloud
supplies parameters, so parameters are treated as hostile:

* a value may only reach a slot the template author retitled;
* an image slot takes a bridge-generated filename and nothing else -- no paths,
  no traversal, no absolute paths, no drive letters;
* numbers are checked against the widget's declared range;
* a choice widget's value must be one of the choices ComfyUI reports.

``object_info`` is passed in rather than fetched, because the runner is the only
module that knows how to reach ComfyUI, and because a test needs to hand this a
machine that does not exist.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Mapping

from . import binder
from .templates import Template

# Files ComfyUI loads from a models folder. Used only to word the error: a
# missing checkpoint is "install this model", a bad sampler name is "invalid
# choice", and telling the user the wrong one wastes their afternoon.
MODEL_SUFFIXES = (
    ".safetensors", ".ckpt", ".pt", ".pth", ".bin", ".gguf", ".sft", ".onnx",
)

# The bridge generates every input filename itself (see comfy/runner.py). This
# is what a legitimate one looks like; anything else is refused outright.
INPUT_FILENAME_RE = re.compile(r"^rb_[a-f0-9]{8,64}\.(png|jpg|jpeg|webp)$")

MAX_PROMPT_LENGTH = 20000


@dataclass(frozen=True)
class Issue:
    code: str
    message: str
    node: str | None = None
    slot: str | None = None
    detail: dict[str, Any] | None = None

    def as_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"code": self.code, "message": self.message}
        if self.node:
            out["node"] = self.node
        if self.slot:
            out["slot"] = self.slot
        if self.detail:
            out.update(self.detail)
        return out


@dataclass
class Report:
    errors: list[Issue] = field(default_factory=list)
    warnings: list[Issue] = field(default_factory=list)
    slots: dict[str, binder.Slot] = field(default_factory=dict)
    # The graph exactly as it would be queued: bound, and pruned of the optional
    # inputs nobody supplied. Validating anything else means validating a
    # workflow that is not the one about to run.
    graph: dict[str, Any] = field(default_factory=dict)
    pruned: list[dict[str, Any]] = field(default_factory=list)
    quality: str | None = None
    checkpoint: dict[str, Any] | None = None
    preset_fallback: dict[str, Any] | None = None
    transform: Any = None
    padded_files: list[str] = field(default_factory=list)
    # Keyframes the bridge had to bring to a common size, so the caller can
    # see it happened instead of wondering why a key came back letterboxed.
    resized: list[dict[str, Any]] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors

    def error(self, code: str, message: str, **kwargs: Any) -> None:
        self.errors.append(Issue(code, message, **kwargs))

    def warn(self, code: str, message: str, **kwargs: Any) -> None:
        self.warnings.append(Issue(code, message, **kwargs))

    def as_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "errors": [i.as_dict() for i in self.errors],
            "warnings": [i.as_dict() for i in self.warnings],
            "slots": {name: slot.as_dict() for name, slot in sorted(self.slots.items())},
            # Reported, never silent: a workflow that was quietly pruned is a
            # workflow nobody can debug afterwards.
            "pruned": self.pruned,
            "quality": self.quality,
        }


# --------------------------------------------------------------------------
# object_info shape
# --------------------------------------------------------------------------
#
# ComfyUI has emitted two shapes for an input spec over the years:
#
#   ["INT", {"min": 0, "max": 4294967295}]        a typed widget
#   [["euler", "heun"], {"default": "euler"}]     a choice, options inline
#   {"type": "COMBO", "options": [...]}           the newer dict form
#
# Reading only one of them means the validator silently stops validating on the
# other -- worse than not having it, because the report still says "ok".

def _spec_parts(spec: Any) -> tuple[str | None, list[Any] | None, dict[str, Any]]:
    """-> (type name, choices, metadata)."""
    if isinstance(spec, Mapping):
        options = spec.get("options")
        if isinstance(options, list):
            return "COMBO", list(options), dict(spec)
        return (str(spec.get("type")) if spec.get("type") else None), None, dict(spec)

    if isinstance(spec, (list, tuple)) and spec:
        head = spec[0]
        meta = spec[1] if len(spec) > 1 and isinstance(spec[1], Mapping) else {}
        if isinstance(head, (list, tuple)):
            return "COMBO", list(head), dict(meta)
        return str(head), None, dict(meta)

    return None, None, {}


def _class_inputs(object_info: Mapping[str, Any], class_type: str) -> dict[str, Any]:
    node = object_info.get(class_type)
    if not isinstance(node, Mapping):
        return {}
    inputs = node.get("input")
    if not isinstance(inputs, Mapping):
        return {}
    merged: dict[str, Any] = {}
    for section in ("required", "optional"):
        block = inputs.get(section)
        if isinstance(block, Mapping):
            merged.update(block)
    return merged


def _looks_like_model_file(value: Any) -> bool:
    return isinstance(value, str) and value.lower().endswith(MODEL_SUFFIXES)


def _optional_models(template: Template) -> set[str]:
    entries = template.requires.get("optionalModels") or []
    return {
        str(entry.get("file"))
        for entry in entries
        if isinstance(entry, Mapping) and entry.get("file")
    }


def _looks_like_path(value: str) -> bool:
    return (
        "/" in value
        or "\\" in value
        or ".." in value
        or bool(re.match(r"^[A-Za-z]:", value))
        or value.startswith("~")
        or "\x00" in value
    )


# --------------------------------------------------------------------------
# The checks
# --------------------------------------------------------------------------

def _check_declared_models(
    template: Template, file_exists: Any, report: Report,
    selected_checkpoint: dict[str, Any] | None = None,
) -> None:
    """Check the manifest's model list against the DISK, not against ComfyUI.

    ComfyUI's choice lists are not proof that a file is there. A name can sit in
    ``/object_info`` with nothing behind it -- observed on the development
    machine with ``qwen_image_edit_2511_bf16.safetensors``, listed by
    ``UNETLoader`` and present in no models folder at all. Trusting the list
    means the pre-flight says "ready", the user waits, and ComfyUI then fails
    loading a file: exactly the experience validating early exists to prevent.

    ``file_exists(folder, filename) -> bool`` is injected because only the
    runner knows how to ask ComfyUI where its model folders are. When it is not
    supplied -- in tests, or outside ComfyUI -- this check is skipped rather
    than guessed at.
    """
    if file_exists is None:
        return

    optional = _optional_models(template)
    conditional = {
        str(entry.get("file")): entry
        for entry in template.requires.get("conditionalModels") or []
        if isinstance(entry, Mapping) and entry.get("file")
    }

    entries = list(template.requires.get("models") or [])
    entries += list(conditional.values())
    # Only the variant that will actually load. Reporting the others missing
    # would tell the user to download 20 GB they do not need.
    if selected_checkpoint:
        entries.append(selected_checkpoint)

    for entry in entries:
        if not isinstance(entry, Mapping):
            continue
        folder, filename = entry.get("folder"), entry.get("file")
        if not folder or not filename:
            continue
        try:
            if file_exists(str(folder), str(filename)):
                continue
        except Exception:                     # pragma: no cover - never block on this
            continue

        size = entry.get("gb")
        where = f"models/{folder}/"
        detail = {"file": str(filename), "folder": str(folder), "url": entry.get("url")}
        # A file only some quality settings load is not a reason to refuse the
        # ones that do not; the run layer decides whether it is needed today.
        conditional_entry = conditional.get(str(filename))
        issue = report.warn if (conditional_entry or str(filename) in optional) else report.error
        suffix = ""
        if conditional_entry and conditional_entry.get("requiredFor"):
            suffix = f" It is only needed for: {', '.join(conditional_entry['requiredFor'])}."
        issue(
            "missing_model",
            f"'{filename}' is not installed. Put it in {where}"
            + (f" ({size} GB)." if size else ".")
            + suffix,
            detail=detail,
        )


def _check_node_classes(
    template: Template, object_info: Mapping[str, Any], report: Report
) -> None:
    seen: set[str] = set()
    for node_id, node in template.graph.items():
        if not isinstance(node, Mapping):
            continue
        class_type = str(node.get("class_type") or "")
        if not class_type or class_type in seen:
            continue
        seen.add(class_type)
        if class_type not in object_info:
            report.error(
                "missing_node",
                f"This workflow needs the node '{class_type}', which is not "
                f"installed in ComfyUI.",
                node=str(node_id),
                detail={"classType": class_type},
            )

    for class_type in template.requires.get("nodes") or []:
        if class_type not in object_info and class_type not in seen:
            report.error(
                "missing_node",
                f"This template requires the node '{class_type}', which is not "
                f"installed in ComfyUI.",
                detail={"classType": class_type},
            )


def _check_slots(template: Template, report: Report) -> dict[str, binder.Slot]:
    declared = template.slots
    try:
        slots = binder.find_slots(template.graph, declared.keys())
    except binder.BindError as exc:
        report.error(exc.code, str(exc))
        return {}

    for name, spec in declared.items():
        slot = slots.get(name)
        if slot is None:
            if spec.get("required"):
                report.error(
                    "missing_slot",
                    f"This workflow has no {name} node. Retitle the node that "
                    f"should receive it.",
                    slot=name,
                )
            continue
        # Two families have no single widget on purpose: $OUTPUT is read
        # rather than written, and a sequencer is timed through a whole
        # family of numbered widgets that _write_sequencer names itself.
        # Only $OUTPUT was exempt here, so every keyframe video preset was
        # refused before it could queue.
        if slot.family not in ("output", "sequencer") and slot.widget is None:
            report.error(
                "unknown_widget",
                f"{name} is on node {slot.node_id} ({slot.class_type}), but that "
                f"node has no editable field to receive the value.",
                node=slot.node_id,
                slot=name,
            )
    return slots


def _check_widget_names(
    template: Template,
    slots: Mapping[str, binder.Slot],
    object_info: Mapping[str, Any],
    report: Report,
) -> None:
    for name, slot in slots.items():
        if slot.widget is None:
            continue
        spec = _class_inputs(object_info, slot.class_type)
        if not spec:
            continue        # the class is missing; already reported
        if slot.widget not in spec:
            report.error(
                "unknown_widget",
                f"{name} writes to '{slot.widget}' on {slot.class_type}, but this "
                f"ComfyUI build has no such field. The node was probably updated.",
                node=slot.node_id,
                slot=name,
                detail={"widget": slot.widget, "known": sorted(spec)[:24]},
            )


def _check_widget_values(
    template: Template,
    graph: Mapping[str, Any],
    object_info: Mapping[str, Any],
    report: Report,
    skip: set[tuple[str, str]] | None = None,
    live_nodes: list[str] | None = None,
) -> None:
    """Every literal value in the graph, against what ComfyUI says it accepts.

    This is where a missing model file is actually caught: ComfyUI builds a
    loader's choice list by scanning the models folders, so a file that is not
    installed simply is not in the list.

    ``skip`` holds the image slots. ``LoadImage``'s choices are the contents of
    ComfyUI's ``input/`` folder, captured when object_info was built -- which
    can easily predate the frame the bridge wrote three milliseconds ago. The
    filename came from the bridge and was matched against INPUT_FILENAME_RE in
    :func:`_check_parameters`, so checking it against a stale directory listing
    would only ever produce a false "invalid choice".
    """
    optional = _optional_models(template)
    skip = skip or set()

    # Only nodes that will actually execute. ComfyUI walks back from the
    # outputs, so a node nothing reaches is never run and its widget values are
    # irrelevant -- including the placeholder filename on a LoadImage that was
    # just unplugged, which would otherwise be reported as an invalid choice and
    # block a workflow that is perfectly fine.
    live = binder.reachable(graph, live_nodes or [])

    for node_id, node in binder.ordered_nodes(graph):
        if live and node_id not in live:
            continue
        class_type = str(node.get("class_type") or "")
        spec_map = _class_inputs(object_info, class_type)
        if not spec_map:
            continue
        for widget, value in node.get("inputs", {}).items():
            if binder.is_link(value) or (str(node_id), widget) in skip:
                continue
            spec = spec_map.get(widget)
            if spec is None:
                report.warn(
                    "unknown_widget",
                    f"Node {node_id} ({class_type}) sets '{widget}', which this "
                    f"ComfyUI build does not know about.",
                    node=str(node_id),
                )
                continue
            _check_one_value(
                node_id, class_type, widget, value, spec, optional, report
            )


def _check_one_value(
    node_id: str,
    class_type: str,
    widget: str,
    value: Any,
    spec: Any,
    optional_models: set[str],
    report: Report,
) -> None:
    type_name, choices, meta = _spec_parts(spec)

    if choices is not None:
        if value in choices:
            return
        if _looks_like_model_file(value):
            # Optional models are declared exactly so this is a warning: the
            # qwen template ships with its Lightning LoRA switched off, and
            # refusing to run because an unused file is absent would block a
            # workflow that works perfectly.
            issue = report.warn if str(value) in optional_models else report.error
            issue(
                "missing_model",
                f"The model file '{value}' is not installed "
                f"({class_type}.{widget}).",
                node=str(node_id),
                detail={"file": str(value), "widget": widget},
            )
        else:
            report.error(
                "invalid_choice",
                f"Node {node_id} ({class_type}) sets {widget}='{value}', which "
                f"this ComfyUI build does not offer.",
                node=str(node_id),
                detail={"widget": widget, "choices": [str(c) for c in choices[:24]]},
            )
        return

    if type_name in ("INT", "FLOAT") and isinstance(value, (int, float)) and not isinstance(value, bool):
        low, high = meta.get("min"), meta.get("max")
        if isinstance(low, (int, float)) and value < low:
            report.error(
                "out_of_range",
                f"Node {node_id} ({class_type}) sets {widget}={value}, below the "
                f"minimum of {low}.",
                node=str(node_id),
                detail={"widget": widget, "min": low, "max": high},
            )
        elif isinstance(high, (int, float)) and value > high:
            report.error(
                "out_of_range",
                f"Node {node_id} ({class_type}) sets {widget}={value}, above the "
                f"maximum of {high}.",
                node=str(node_id),
                detail={"widget": widget, "min": low, "max": high},
            )


def _check_parameters(
    template: Template,
    values: Mapping[str, Any],
    slots: Mapping[str, binder.Slot],
    report: Report,
) -> None:
    declared = template.slots
    for name, value in values.items():
        if name not in declared:
            # Not merely unknown: the cloud is asking to write somewhere the
            # template author never exposed.
            report.error(
                "unknown_slot",
                f"This template does not accept {name}.",
                slot=name,
            )
            continue
        if name not in slots:
            continue                     # already reported as missing_slot
        family = binder.family_of(name)

        if family == "image":
            if not isinstance(value, str) or not INPUT_FILENAME_RE.match(value):
                report.error(
                    "bad_input_reference",
                    f"{name} must be an input registered with this bridge.",
                    slot=name,
                )
            continue

        if family == "text":
            if not isinstance(value, str):
                report.error("bad_value", f"{name} must be text.", slot=name)
            elif len(value) > MAX_PROMPT_LENGTH:
                report.error(
                    "bad_value",
                    f"{name} is longer than {MAX_PROMPT_LENGTH} characters.",
                    slot=name,
                )
            continue

        if family == "seed":
            if isinstance(value, bool) or not isinstance(value, int):
                report.error("bad_value", f"{name} must be a whole number.", slot=name)
            continue

        # The two list-shaped slots. Reported here as well as refused in the
        # binder because /validate is a dry run the editor asks BEFORE spending
        # the user's time: "keyframe 3 has no frame number" belongs on screen
        # next to the timeline, not in an exception thrown a minute later.
        if family == "imagelist":
            items = value if isinstance(value, (list, tuple)) else [value]
            if not items:
                report.error("bad_value", f"{name} was given no images.", slot=name)
            for item in items:
                if not isinstance(item, str) or not INPUT_FILENAME_RE.match(item):
                    report.error(
                        "bad_input_reference",
                        f"{name} must list inputs registered with this bridge.",
                        slot=name,
                    )
                    break
            continue

        if family == "sequencer":
            if not isinstance(value, (list, tuple)) or not value:
                report.error(
                    "bad_value",
                    f"{name} must be a non-empty list of keyframe timings.",
                    slot=name,
                )
                continue
            frames: list[int] = []
            for index, item in enumerate(value, start=1):
                frame = item.get("frame") if isinstance(item, Mapping) else None
                if isinstance(frame, bool) or not isinstance(frame, int) or frame < 0:
                    report.error(
                        "bad_value",
                        f"Keyframe {index} has no frame number.",
                        slot=name,
                    )
                    frames = []
                    break
                frames.append(frame)
            # Two keys on the same frame is not a timing an animator can mean,
            # and the sequencer would quietly let the later one win.
            if len(set(frames)) != len(frames):
                report.error(
                    "bad_value",
                    f"{name} puts two keyframes on the same frame.",
                    slot=name,
                )
            continue

        if isinstance(value, str) and _looks_like_path(value):
            report.error(
                "bad_value",
                f"{name} looks like a file path, which is never accepted.",
                slot=name,
            )
        elif not isinstance(value, binder.SCALAR_TYPES):
            report.error("bad_value", f"{name} must be a single value.", slot=name)


# --------------------------------------------------------------------------

def check_template_security(
    template: Template, object_info: Mapping[str, Any]
) -> Report:
    """Layer 1: the whole template, including nodes nothing reaches.

    Deliberately separate from the executable-graph check, and deliberately run
    over *every* node. A class_type that is not on the manifest's allowlist is
    refused even if it is disconnected -- otherwise a tampered template could
    carry a node that today is unreachable and tomorrow is one pruning rule away
    from executing. ComfyUI custom nodes are arbitrary Python, so the allowlist
    is the thing standing between a modified template file and code execution.
    """
    report = Report()
    allowed = set(template.requires.get("nodes") or [])

    for node_id, node in binder.ordered_nodes(template.graph):
        class_type = str(node.get("class_type") or "")
        if not class_type:
            report.error("bad_node", f"Node {node_id} declares no class_type.", node=node_id)
            continue
        if allowed and class_type not in allowed:
            report.error(
                "node_not_allowed",
                f"Node {node_id} uses '{class_type}', which this template does "
                f"not declare. Refusing to run it.",
                node=node_id,
                detail={"classType": class_type},
            )
        elif class_type not in object_info:
            report.error(
                "missing_node",
                f"This workflow needs the node '{class_type}', which is not "
                f"installed in ComfyUI.",
                node=node_id,
                detail={"classType": class_type},
            )
    return report


def check_device_fit(template: Template, profile: Mapping[str, Any] | None) -> Report:
    """Does this machine clear the preset's hardware bar?

    Two tiers, on purpose: 'full' at 24 GB and, once it exists, 'lite' at 12-16.
    Nothing adaptive -- a static bracket the user can read off a spec sheet
    beats a rule derived from one benchmark on one computer.
    """
    report = Report()
    minimum = template.minimum_vram_gb
    if not minimum or not profile:
        return report
    have_mb = profile.get("vramTotalMb")
    if not isinstance(have_mb, (int, float)):
        return report
    have_gb = have_mb / 1024
    if have_gb + 0.5 < minimum:        # 24576 MB reads as 24.0, not 23.9
        report.error(
            "insufficient_vram",
            f"This preset needs about {minimum:g} GB of VRAM and this GPU has "
            f"{have_gb:.1f} GB. There is no lighter preset installed yet, so "
            f"use Reanimator Cloud for now.",
            detail={"tier": template.tier, "requiredVramGb": minimum,
                    "availableVramGb": round(have_gb, 1), "recommend": "cloud"},
        )
    return report


def check_intent(
    template: Template, intent: str | None, supplied_roles: set[str]
) -> Report:
    """Which roles this action cannot run without.

    edit_pose is an instruction expressed as a drawing, so a request with no
    annotated frame is not a cheap version of it -- it is a different request,
    and running it would silently return an unedited frame.
    """
    report = Report()
    if intent and template.intents and intent not in template.intents:
        report.error(
            "intent_not_supported",
            f"This template does not serve '{intent}'.",
            detail={"intent": intent, "supported": sorted(template.intents)},
        )
        return report

    for role in template.required_roles(intent):
        if role not in supplied_roles:
            report.error(
                "missing_role",
                f"'{intent or 'this action'}' needs {role}.",
                detail={"role": role, "intent": intent},
            )
    return report


def _when_holds(condition: Any, roles: set[str], intent: str | None) -> bool:
    """Is one `when` clause true for this request?

    A clause is a role name, ``intent:<name>``, or ``always``; ``!`` negates it,
    and a list means every clause must hold. That is the whole grammar, and it
    is deliberately small -- but it has to reach this far, because two of the
    things the preamble must say depend on what is ABSENT:

    * which Picture number the context frame is. With no annotated frame the
      context is Picture 2, not Picture 3, and a preamble that points at a
      Picture nobody connected sends the model looking for an image that is
      not there.
    * whether Picture 1 is a frame to preserve or a drawing to realise. Saying
      "preserve" over somebody's pencil lines is an instruction the model obeys
      perfectly, by handing the drawing back nearly untouched.
    """
    if isinstance(condition, (list, tuple)):
        return all(_when_holds(c, roles, intent) for c in condition)
    if condition in (None, "always"):
        return True
    text = str(condition)
    if text.startswith("!"):
        return not _when_holds(text[1:], roles, intent)
    if text.startswith("intent:"):
        return intent == text[len("intent:"):]
    return text in roles


def compose_prompt(
    template: Template,
    instruction: str,
    supplied_roles: set[str],
    intent: str | None = None,
) -> str:
    """Build the prompt from the manifest, describing only the images present.

    TextEncodeQwenImageEditPlus prepends 'Picture 1:', 'Picture 2:'… itself, one
    per connected image. A preamble that describes a Picture 3 nobody connected
    points the model at nothing, and the model will invent something to satisfy
    it -- so the lines and the wires have to come from the same decision.
    """
    spec = template.manifest.get("prompt") or {}
    lines: list[str] = []
    for line in spec.get("lines") or []:
        if not isinstance(line, dict):
            continue
        if _when_holds(line.get("when"), supplied_roles, intent):
            lines.append(str(line.get("text") or ""))
    preamble = " ".join(l for l in lines if l)
    instruction = (instruction or "").strip()
    if preamble and instruction:
        return f"{preamble}\n\n{instruction}"
    return instruction or preamble


def check_after_pruning(
    graph: Mapping[str, Any],
    output_nodes: list[str],
    object_info: Mapping[str, Any],
) -> Report:
    """Re-check a graph that :func:`binder.detach` has unplugged inputs from.

    Two things can go wrong that ordinary validation would not notice, because
    ordinary validation assumes the template is intact:

    * the **output stops being reachable** -- unplug the wrong wire and the run
      completes having executed almost nothing;
    * a node that still runs **loses an input ComfyUI requires**, which ComfyUI
      would reject at queue time with a message pointing at a node the user
      never edited.
    """
    report = Report()

    if not output_nodes:
        report.error("no_output", "Pruning left the workflow with no output node.")
        return report

    live = binder.reachable(graph, output_nodes)
    for node_id in output_nodes:
        if node_id not in graph:
            report.error(
                "output_unreachable",
                "Pruning removed the node the result is read from.",
                node=node_id,
            )
            return report

    for node_id in sorted(live):
        node = graph.get(node_id)
        if not isinstance(node, Mapping):
            continue
        class_type = str(node.get("class_type") or "")
        spec = object_info.get(class_type)
        if not isinstance(spec, Mapping):
            continue
        required = spec.get("input")
        required = required.get("required") if isinstance(required, Mapping) else None
        if not isinstance(required, Mapping):
            continue
        present = node.get("inputs") or {}
        for name in required:
            if name not in present:
                report.error(
                    "missing_input",
                    f"After removing the unused inputs, node {node_id} "
                    f"({class_type}) has no '{name}'.",
                    node=node_id,
                    detail={"input": name},
                )

        for name, value in present.items():
            if binder.is_link(value) and str(value[0]) not in graph:
                report.error(
                    "dangling_link",
                    f"Node {node_id} ({class_type}) still points at node "
                    f"{value[0]}, which is not in the workflow.",
                    node=node_id,
                )
    return report


def validate(
    template: Template,
    values: Mapping[str, Any],
    object_info: Mapping[str, Any],
    file_exists: Any = None,
    prune: list[dict[str, str]] | None = None,
    selected_checkpoint: dict[str, Any] | None = None,
) -> Report:
    """Full pre-flight. Never raises for a bad template or bad values -- the
    caller wants the whole list of problems, not the first one."""
    report = Report()

    _check_declared_models(template, file_exists, report, selected_checkpoint)
    _check_node_classes(template, object_info, report)
    slots = _check_slots(template, report)
    report.slots = dict(slots)
    _check_widget_names(template, slots, object_info, report)
    _check_parameters(template, values, slots, report)

    # Bind, then prune, THEN check values -- in that order, so what gets
    # inspected is byte-for-byte what would be queued. Checking before pruning
    # reports the placeholder filename on a LoadImage that is about to be
    # unplugged, and refuses a workflow that would have run perfectly.
    graph: dict[str, Any] = dict(template.graph)
    outputs = binder.output_node_ids(slots)
    if report.ok:
        try:
            graph = binder.fill(template.graph, values, slots)
            plan = prune if prune is not None else template.detach_plan(set(values))
            if plan:
                graph, report.pruned = binder.detach(graph, plan, slots)
        except binder.BindError as exc:
            report.error(exc.code, str(exc))
            graph = dict(template.graph)

    report.graph = graph

    image_slots = {
        (slot.node_id, slot.widget)
        for name, slot in slots.items()
        if slot.family == "image" and slot.widget and name in values
    }
    _check_widget_values(template, graph, object_info, report, image_slots, outputs)

    if report.pruned and report.ok:
        after = check_after_pruning(graph, outputs, object_info)
        report.errors.extend(after.errors)
        report.warnings.extend(after.warnings)

    return report
