"""Put values into a template's slots.

**Binding is by ``_meta.title``, never by node id.** ComfyUI re-numbers nodes on
every export, and subgraph ids like ``11:8:411`` change shape entirely when the
subgraph is edited. Titles are typed by a human and survive.

Retitling five nodes is the whole contract with the template author:

    $IMAGE_1   the frame to edit          LoadImage
    $PROMPT    the instruction            any node with a text widget
    $NEGATIVE  the negative instruction
    $SEED      vary the result            any node with a seed widget
    $OUTPUT    where the result is read   SaveImage / SaveVideo

A title may name the widget explicitly with a colon -- ``$SEED:seed`` -- which is
what a node with several plausible widgets needs.

When a title is missing, a **duck-typing fallback** finds the node by shape
instead: a ``LoadImage`` is an image input, a ``SaveImage`` is an output, a node
carrying ``insert_frame_1`` is a keyframe sequencer, whatever feeds a sampler's
``positive`` input is the prompt. The fallback exists so a user's own workflow
mostly works before they have retitled anything; it is a convenience, never a
security boundary. What keeps this safe is that only *values* are ever written,
never structure -- see :func:`bind`.
"""

from __future__ import annotations

import copy
import re
from dataclasses import dataclass
from typing import Any, Iterable, Mapping

SLOT_RE = re.compile(r"^\$([A-Z][A-Z0-9_]*?)(?:_(\d+))?(?::([A-Za-z0-9_]+))?$")
IMAGE_SLOT_RE = re.compile(r"^\$IMAGE_(\d+)$")
# One widget holding many filenames, one per line. A basename and nothing else:
# the loader that reads this widget tries the string as an absolute path FIRST
# and only then falls back to ComfyUI's input folder, so a value with a
# separator in it would read any file on the disk.
SAFE_INPUT_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
# Widgets a $SEQUENCER value may write, and nothing else. Per-image timing on
# these nodes is a family of numbered widgets, not one value, so it cannot
# travel as a scalar -- but every individual write below still is one.
# A multi-stage video workflow has one sequencer PER STAGE, all of them timing
# the same keyframes. They are numbered rather than sharing a title because a
# duplicate title is refused -- deliberately, so a slot always names one node.
SEQUENCER_SLOT_RE = re.compile(r"^\$SEQUENCER(?:_\d+)?$")
SEQUENCER_TIMING_RE = re.compile(r"^(?:insert_frame|insert_second|strength)_(\d+)$")

# Widget names to try, in order, for each family of slot. First one that exists
# on the node and is not already wired to another node wins.
TEXT_WIDGETS = ("prompt", "text", "string", "value")
IMAGE_WIDGETS = ("image", "image_path", "filename")
IMAGELIST_WIDGETS = ("image_paths", "images", "paths")
SEED_WIDGETS = ("seed", "noise_seed", "rand_seed")

# Nodes that terminate a graph. PreviewImage last: it writes to temp/ and is a
# poor result source, so it is only used when nothing better exists.
OUTPUT_CLASSES = (
    "SaveImage",
    "SaveVideo",
    "SaveAnimatedWEBP",
    "SaveAnimatedPNG",
    "SaveWEBM",
    "VHS_VideoCombine",
    "SaveAudio",
    "PreviewImage",
)
IMAGE_INPUT_CLASSES = ("LoadImage", "LoadImageOutput", "LoadImageMask")

SCALAR_TYPES = (str, int, float, bool)
MAX_TEXT_LENGTH = 20000


class BindError(Exception):
    def __init__(self, message: str, code: str = "bind_error") -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class Slot:
    name: str            # "$IMAGE_1", "$PROMPT", ...
    node_id: str
    class_type: str
    widget: str | None   # None for $OUTPUT: nothing is written into it
    family: str          # image | text | seed | output | sequencer | value
    matched_by: str      # "title" | "duck"

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "node": self.node_id,
            "classType": self.class_type,
            "widget": self.widget,
            "family": self.family,
            "matchedBy": self.matched_by,
        }


# --------------------------------------------------------------------------
# Graph helpers
# --------------------------------------------------------------------------

def is_link(value: Any) -> bool:
    """True for a wire to another node's output: ``["102:38", 0]``."""
    return (
        isinstance(value, list)
        and len(value) == 2
        and isinstance(value[0], (str, int))
        and isinstance(value[1], int)
        and not isinstance(value[1], bool)
    )


def title_of(node: Mapping[str, Any]) -> str:
    meta = node.get("_meta")
    if isinstance(meta, Mapping):
        return str(meta.get("title") or "")
    return ""


def _inputs(node: Mapping[str, Any]) -> Mapping[str, Any]:
    values = node.get("inputs")
    return values if isinstance(values, Mapping) else {}


def _widgets(node: Mapping[str, Any]) -> dict[str, Any]:
    """Inputs holding a literal value, i.e. everything that is not a wire."""
    return {k: v for k, v in _inputs(node).items() if not is_link(v)}


def ordered_nodes(graph: Mapping[str, Any]) -> list[tuple[str, Mapping[str, Any]]]:
    """Stable iteration order regardless of JSON key ordering.

    Node ids are strings like "78" and "102:38", so a plain sort would put "102"
    before "78". Sorting on the numeric segments keeps "$IMAGE_2 is the second
    LoadImage" meaning what a human would expect when reading the workflow.
    """
    def key(item: tuple[str, Any]) -> tuple[Any, ...]:
        parts = str(item[0]).split(":")
        return tuple((0, int(p)) if p.isdigit() else (1, p) for p in parts)

    return sorted(
        ((str(k), v) for k, v in graph.items() if isinstance(v, Mapping)), key=key
    )


def parse_slot_title(title: str) -> tuple[str, str | None] | None:
    """``"$SEED:seed"`` -> ``("$SEED", "seed")``. Returns None for a plain title."""
    if not title.startswith("$"):
        return None
    match = SLOT_RE.match(title.strip())
    if not match:
        return None
    base, index, widget = match.groups()
    name = f"${base}_{index}" if index else f"${base}"
    return name, widget


def parse_slot_titles(title: str) -> list[tuple[str, str | None]]:
    """Every slot a title declares, in order.

    One node, one title -- but a node can carry two widgets worth exposing, and
    a KSampler carrying both the seed and the denoise is the ordinary case.
    Space-separated slots let the template SAY so:

        "$SEED:seed $DENOISE:denoise"

    The alternative was to title the node for one and let duck-typing find the
    other. That works today and would keep working right up until the template
    grew a second sampler, at which point the untitled one binds to whichever
    comes first -- silently, and to a widget that still accepts the value.
    """
    out: list[tuple[str, str | None]] = []
    for piece in title.split():
        parsed = parse_slot_title(piece)
        if parsed is not None:
            out.append(parsed)
    return out


def family_of(slot_name: str) -> str:
    if IMAGE_SLOT_RE.match(slot_name):
        return "image"
    if slot_name in ("$PROMPT", "$NEGATIVE"):
        return "text"
    if slot_name == "$SEED":
        return "seed"
    if slot_name == "$OUTPUT":
        return "output"
    if SEQUENCER_SLOT_RE.match(slot_name):
        return "sequencer"
    if slot_name in ("$IMAGE_PATHS", "$IMAGES"):
        return "imagelist"
    return "value"


def _pick_widget(node: Mapping[str, Any], family: str) -> str | None:
    candidates = {
        "image": IMAGE_WIDGETS,
        "imagelist": IMAGELIST_WIDGETS,
        "text": TEXT_WIDGETS,
        "seed": SEED_WIDGETS,
    }.get(family)
    widgets = _widgets(node)
    if candidates:
        for name in candidates:
            if name in widgets:
                return name
    if family in ("output", "sequencer"):
        return None
    # A generic $NAME slot on a primitive: bind its only literal input.
    if len(widgets) == 1:
        return next(iter(widgets))
    return None


# --------------------------------------------------------------------------
# Duck-typed fallbacks
# --------------------------------------------------------------------------

def _sampler_nodes(nodes: list[tuple[str, Mapping[str, Any]]]) -> list[tuple[str, Mapping[str, Any]]]:
    return [
        (node_id, node)
        for node_id, node in nodes
        if is_link(_inputs(node).get("positive")) and is_link(_inputs(node).get("negative"))
    ]


def _follow(nodes_by_id: Mapping[str, Mapping[str, Any]], link: Any) -> str | None:
    if not is_link(link):
        return None
    node_id = str(link[0])
    return node_id if node_id in nodes_by_id else None


def _duck(
    slot_name: str,
    family: str,
    nodes: list[tuple[str, Mapping[str, Any]]],
    taken: set[str],
) -> tuple[str, Mapping[str, Any]] | None:
    nodes_by_id = {node_id: node for node_id, node in nodes}

    if family == "output":
        for wanted in OUTPUT_CLASSES:
            for node_id, node in nodes:
                if node.get("class_type") == wanted and node_id not in taken:
                    return node_id, node
        return None

    if family == "image":
        index = int(IMAGE_SLOT_RE.match(slot_name).group(1))  # type: ignore[union-attr]
        loaders = [
            (node_id, node)
            for node_id, node in nodes
            if node.get("class_type") in IMAGE_INPUT_CLASSES
        ]
        return loaders[index - 1] if 0 < index <= len(loaders) else None

    if family == "sequencer":
        for node_id, node in nodes:
            # `taken` matters here: a multi-stage workflow has one sequencer per
            # stage, and without this every $SEQUENCER_n resolved to the first
            # of them -- stages two and three would then run the template's
            # shipped timings instead of the user's.
            if node_id not in taken and "insert_frame_1" in _inputs(node):
                return node_id, node
        return None

    if family == "imagelist":
        for node_id, node in nodes:
            if node_id in taken:
                continue
            if any(name in _widgets(node) for name in IMAGELIST_WIDGETS):
                return node_id, node
        return None

    if family == "text":
        wire = "positive" if slot_name == "$PROMPT" else "negative"
        for _, sampler in _sampler_nodes(nodes):
            target = _follow(nodes_by_id, _inputs(sampler).get(wire))
            if target and target not in taken:
                node = nodes_by_id[target]
                if _pick_widget(node, "text"):
                    return target, node
        return None

    if family == "seed":
        for node_id, sampler in _sampler_nodes(nodes):   # the sampler's own seed first
            if _pick_widget(sampler, "seed"):
                return node_id, sampler
        for node_id, node in nodes:
            if node_id not in taken and _pick_widget(node, "seed"):
                return node_id, node
        return None

    return None


# --------------------------------------------------------------------------
# Slot discovery
# --------------------------------------------------------------------------

def find_slots(
    graph: Mapping[str, Any], wanted: Iterable[str] | None = None
) -> dict[str, Slot]:
    """Locate every slot in ``graph``.

    Titles win. ``wanted`` (typically the manifest's slot names) additionally
    enables the duck-typed fallback for the slots that no title claimed.
    """
    nodes = ordered_nodes(graph)
    slots: dict[str, Slot] = {}
    taken: set[str] = set()

    for node_id, node in nodes:
        for parsed in parse_slot_titles(title_of(node)):
            name, explicit_widget = parsed
            if name in slots:
                raise BindError(
                    f"Template declares {name} twice (nodes {slots[name].node_id} "
                    f"and {node_id}).",
                    "duplicate_slot",
                )
            family = family_of(name)
            widget = explicit_widget or _pick_widget(node, family)
            if explicit_widget and explicit_widget not in _inputs(node):
                raise BindError(
                    f"{name} names widget '{explicit_widget}', which node {node_id} "
                    f"({node.get('class_type')}) does not have.",
                    "unknown_widget",
                )
            slots[name] = Slot(
                name=name,
                node_id=node_id,
                class_type=str(node.get("class_type") or ""),
                widget=widget,
                family=family,
                matched_by="title",
            )
            taken.add(node_id)

    for name in wanted or ():
        if name in slots:
            continue
        found = _duck(name, family_of(name), nodes, taken)
        if found is None:
            continue
        node_id, node = found
        family = family_of(name)
        slots[name] = Slot(
            name=name,
            node_id=node_id,
            class_type=str(node.get("class_type") or ""),
            widget=_pick_widget(node, family),
            family=family,
            matched_by="duck",
        )
        taken.add(node_id)

    return slots


def output_node_ids(slots: Mapping[str, Slot]) -> list[str]:
    slot = slots.get("$OUTPUT")
    return [slot.node_id] if slot else []


def reachable(graph: Mapping[str, Any], roots: Iterable[str]) -> set[str]:
    """Nodes that actually contribute to ``roots``.

    ComfyUI walks backwards from the output nodes, so anything not in this set
    is never executed. That is what makes :func:`detach` safe without deleting
    nodes: unplug a LoadImage and it stops being reachable, which stops it being
    run, without touching the node table at all.
    """
    seen: set[str] = set()
    stack = [str(r) for r in roots]
    while stack:
        node_id = stack.pop()
        if node_id in seen or node_id not in graph:
            continue
        seen.add(node_id)
        node = graph[node_id]
        if not isinstance(node, Mapping):
            continue
        for value in _inputs(node).values():
            if is_link(value):
                stack.append(str(value[0]))
    return seen


def detach(
    graph: Mapping[str, Any],
    removals: Iterable[Mapping[str, str]],
    slots: Mapping[str, Slot],
) -> tuple[dict[str, Any], list[dict[str, str]]]:
    """Unplug the inputs listed in ``removals``. Returns the new graph and a log.

    This is the only place the bridge changes a workflow's *shape* rather than
    its values, so it is deliberately the narrowest possible operation:

    * it only ever **deletes** an entry from a node's ``inputs``;
    * it never adds a node, removes a node, or touches a ``class_type``;
    * the removals come from the installed manifest, never from the request --
      the cloud can say "there is no context frame", never "unplug this wire".

    Orphaned nodes are left in place on purpose. ComfyUI executes backwards from
    the outputs, so an unplugged LoadImage simply never runs, and leaving it
    there keeps the node table byte-identical to the template the user installed.
    """
    pruned = copy.deepcopy(dict(graph))
    log: list[dict[str, str]] = []

    for removal in removals:
        slot_name = str(removal.get("slot") or "")
        input_name = str(removal.get("input") or "")
        slot = slots.get(slot_name)
        if slot is None or not input_name:
            continue
        node = pruned.get(slot.node_id)
        if not isinstance(node, dict):
            continue
        inputs = node.get("inputs")
        if not isinstance(inputs, dict) or input_name not in inputs:
            continue
        was = inputs.pop(input_name)
        log.append(
            {
                "slot": slot_name,
                "node": slot.node_id,
                "input": input_name,
                "wasLinkedTo": str(was[0]) if is_link(was) else None,
            }
        )

    # Belt and braces: prove we only unplugged. If this ever fires, something
    # edited the graph that had no business doing so.
    if set(pruned) != set(graph):
        raise BindError("Pruning changed the node set.", "prune_unsafe")
    for node_id, node in pruned.items():
        if node.get("class_type") != graph[node_id].get("class_type"):
            raise BindError("Pruning changed a node's type.", "prune_unsafe")

    return pruned, log


# --------------------------------------------------------------------------
# Binding
# --------------------------------------------------------------------------

def _node_for(bound: dict[str, Any], slot: Slot, name: str) -> dict[str, Any]:
    node = bound.get(slot.node_id)
    if not isinstance(node, dict):
        raise BindError(f"{name} points at a node that is gone.", "unknown_slot")
    return node


def _set_widget(node: dict[str, Any], slot_name: str, widget: str, value: Any) -> None:
    """One scalar into one existing, unwired widget. The only way anything is written."""
    inputs = node.setdefault("inputs", {})
    if widget not in inputs:
        raise BindError(
            f"{slot_name} names widget '{widget}', which this node does not have.",
            "unknown_widget",
        )
    if is_link(inputs[widget]):
        raise BindError(
            f"{slot_name} writes '{widget}', which is driven by another node in "
            f"this workflow and cannot be set directly.",
            "slot_is_wired",
        )
    inputs[widget] = value


def _write_image_list(bound: dict[str, Any], slot: Slot, name: str, value: Any) -> None:
    """Many input filenames into one multiline widget.

    Every line is checked against SAFE_INPUT_NAME_RE, and that check is the
    point of this function rather than a nicety. The loaders that read a widget
    like this resolve the string as an absolute path first and fall back to the
    input folder second, so one ``..`` or one drive letter turns a keyframe list
    into an arbitrary file read. Names reaching here come from the bridge's own
    upload registry, so the check should never fire -- which is exactly why it
    has to be here rather than only at the caller.
    """
    if slot.widget is None:
        raise BindError(f"{name} is a read-only slot.", "readonly_slot")
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, (list, tuple)):
        raise BindError(f"{name} takes a list of input names.", "bad_value")
    names: list[str] = []
    for item in value:
        if not isinstance(item, str) or not SAFE_INPUT_NAME_RE.match(item):
            raise BindError(
                f"{name} takes plain input filenames; {item!r} is not one.",
                "bad_value",
            )
        names.append(item)
    if not names:
        raise BindError(f"{name} was given no images.", "bad_value")
    _set_widget(_node_for(bound, slot, name), name, slot.widget, "\n".join(names))


def _write_sequencer(bound: dict[str, Any], slot: Slot, name: str, value: Any) -> None:
    """Per-image timing into the numbered widgets of a keyframe sequencer.

    The shape is the generic one the editor speaks -- ``[{"frame": 0,
    "strength": 1.0}, ...]`` -- and turning it into ``insert_frame_1`` and
    friends is this module's job, not the caller's. That is what keeps the
    editor from having to know which node this template uses.

    ``insert_mode`` is forced to "frames" and never taken from the request. An
    animator's timing is in frames; letting a value pick "seconds" would make
    the same list of numbers mean something completely different, silently.
    """
    node = _node_for(bound, slot, name)
    if not isinstance(value, (list, tuple)) or not value:
        raise BindError(f"{name} takes a non-empty list of keyframe timings.", "bad_value")
    if len(value) > 50:
        raise BindError(f"{name} takes at most 50 keyframes.", "bad_value")

    # Capacity is read from EVERY input, not just the unwired ones. Asking
    # _widgets() here made a wired insert_frame_2 look like a sequencer with
    # room for one key, and the caller was told "too many keyframes" about a
    # template that holds fifty. Whether a widget can be written is
    # _set_widget's question, and it answers it with slot_is_wired.
    present = _inputs(node)
    timings: list[tuple[int, float]] = []
    for index, item in enumerate(value, start=1):
        if not isinstance(item, Mapping):
            raise BindError(f"{name}[{index}] must be an object.", "bad_value")
        frame = item.get("frame")
        if isinstance(frame, bool) or not isinstance(frame, int) or frame < 0:
            raise BindError(
                f"{name}[{index}] needs a whole frame number of 0 or more.", "bad_value"
            )
        strength = item.get("strength", 1.0)
        if isinstance(strength, bool) or not isinstance(strength, (int, float)):
            raise BindError(f"{name}[{index}] has a strength that is not a number.", "bad_value")
        if not 0.0 <= float(strength) <= 1.0:
            raise BindError(f"{name}[{index}] needs a strength between 0 and 1.", "bad_value")
        if f"insert_frame_{index}" not in present:
            raise BindError(
                f"This template's sequencer holds {index - 1} keyframes and "
                f"{len(value)} were given.",
                "too_many_keyframes",
            )
        timings.append((int(frame), float(strength)))

    if "num_images" in present:
        _set_widget(node, name, "num_images", len(timings))
    if "insert_mode" in present:
        _set_widget(node, name, "insert_mode", "frames")
    for index, (frame, strength) in enumerate(timings, start=1):
        _set_widget(node, name, f"insert_frame_{index}", frame)
        if f"strength_{index}" in present:
            _set_widget(node, name, f"strength_{index}", strength)


def bind(
    graph: Mapping[str, Any],
    values: Mapping[str, Any],
    slots: Mapping[str, Slot] | None = None,
) -> dict[str, Any]:
    """Return a copy of ``graph`` with ``values`` written into their slots.

    Three rules make this safe to drive from a parameter dict that came off the
    network, and they are the reason the cloud never needs to send a graph:

    * only **scalars** are ever written, so a value cannot become a wire, a node
      or a nested structure;
    * an input that is currently a **link is never overwritten**, so a value
      cannot detach a node and re-route the graph;
    * an **unknown slot name is an error**, so a value cannot reach a node the
      template author did not deliberately expose.

    The input graph is never mutated: templates are loaded once and cached, and
    a run that edited them in place would leak into the next one.
    """
    resolved = dict(slots) if slots is not None else find_slots(graph, values.keys())
    bound = copy.deepcopy(dict(graph))

    for name, value in values.items():
        slot = resolved.get(name)
        if slot is None:
            raise BindError(f"This template has no {name} slot.", "unknown_slot")

        # Two slots take a value that is a list, because what they describe IS
        # a list: the keyframe images, and the timing of each one. They do not
        # loosen the rules above -- every write they perform is still a single
        # scalar into a widget named by this module, never by the caller.
        if slot.family == "imagelist":
            _write_image_list(bound, slot, name, value)
            continue
        if slot.family == "sequencer":
            _write_sequencer(bound, slot, name, value)
            continue

        if slot.widget is None:
            raise BindError(f"{name} is a read-only slot.", "readonly_slot")
        if not isinstance(value, SCALAR_TYPES):
            raise BindError(
                f"{name} takes a single text, number or boolean value.", "bad_value"
            )
        if isinstance(value, str) and len(value) > MAX_TEXT_LENGTH:
            raise BindError(f"{name} is too long.", "bad_value")

        node = bound.get(slot.node_id)
        if not isinstance(node, dict):
            raise BindError(f"{name} points at a node that is gone.", "unknown_slot")
        inputs = node.setdefault("inputs", {})
        if slot.widget not in inputs:
            raise BindError(
                f"{name} names widget '{slot.widget}', which node {slot.node_id} "
                f"does not have.",
                "unknown_widget",
            )
        if is_link(inputs[slot.widget]):
            # Writing here would silently disconnect whatever feeds it, and the
            # workflow would run to completion producing something else.
            raise BindError(
                f"{name} is driven by another node in this workflow and cannot "
                f"be set directly.",
                "slot_is_wired",
            )
        inputs[slot.widget] = value

    return bound
