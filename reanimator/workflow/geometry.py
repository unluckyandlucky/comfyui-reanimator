"""Resolution: pad in, crop back out, and never lose a pixel of the frame.

``FluxKontextImageScale`` resizes to the nearest of a fixed list of resolutions
with ``crop="center"`` -- it *crops* the frame to the bucket's aspect ratio and
then stretches what is left. For an 848x478 frame that is 10 rows off the top
and 10 off the bottom: 4.2% of the picture, gone before generation starts, and
the result comes back at 1392x752 which does not fit the frame it came from.

Worse, the clean frame and the annotated frame are cropped independently. They
match today only because they happen to be the same size; the moment anything
else enters with a different shape, the strokes stop sitting over what they
point at -- and that reads as "the model ignored my drawing", not as a
geometry bug.

So the bridge pads to the bucket's aspect ratio first. Bars instead of a crop:
nothing is lost, the scaler becomes a plain resize, and the padding is removed
again afterwards with the exact integers used on the way in.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# comfy_extras/nodes_flux.py PREFERRED_KONTEXT_RESOLUTIONS, mirrored so the
# bridge can predict what the node will do. If ComfyUI changes this list the
# prediction goes stale, which is why plan() also verifies the outcome instead
# of trusting the arithmetic -- see comfy_would_crop().
PREFERRED_KONTEXT_RESOLUTIONS: tuple[tuple[int, int], ...] = (
    (672, 1568), (688, 1504), (720, 1456), (752, 1392), (800, 1328),
    (832, 1248), (880, 1184), (944, 1104), (1024, 1024), (1104, 944),
    (1184, 880), (1248, 832), (1328, 800), (1392, 752), (1456, 720),
    (1504, 688), (1568, 672),
)

PAD_MODES = ("black", "edge", "reflect")

# Until the editor gained blank projects, the frame size always came from a file
# -- a video or a loaded image -- so it was implicitly sane. Now the user types
# it into a form, and these are the shapes that arrive: odd numbers, squares,
# nothing like a multiple of 8, and whatever a stray keypress produces. Every
# limit below exists because the alternative is a malformed tensor deep inside
# ComfyUI, which surfaces as a stack trace about dimensions the user never saw.
MIN_SOURCE_EDGE = 16
MAX_SOURCE_EDGE = 8192
MAX_SOURCE_PIXELS = 40_000_000

# How much of the padded canvas the picture must occupy. A 4000x10 strip pads to
# 4000x1714 -- 99.4% bars -- and the model then spends its whole capacity
# rendering black while the frame is four pixels tall. It does not fail; it
# produces confident rubbish, which is worse.
MIN_COVERAGE = 0.5


def supported_aspect_range() -> tuple[float, float]:
    ratios = [w / h for w, h in PREFERRED_KONTEXT_RESOLUTIONS]
    return min(ratios) * MIN_COVERAGE, max(ratios) / MIN_COVERAGE


class GeometryError(Exception):
    def __init__(self, message: str, code: str = "geometry_error") -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class Transform:
    """Every number used, not just the aspect ratio.

    The inverse is computed from these integers rather than re-derived from
    ratios, so the way back cannot drift from the way in.
    """

    source: tuple[int, int]
    padded: tuple[int, int]
    model: tuple[int, int]
    padding: dict[str, int] = field(default_factory=dict)
    mode: str = "fit_pad"
    pad_mode: str = "black"

    @property
    def has_padding(self) -> bool:
        return any(v > 0 for v in self.padding.values())

    def as_dict(self) -> dict[str, Any]:
        return {
            "source": list(self.source),
            "padded": list(self.padded),
            "model": list(self.model),
            "output": list(self.source),      # the contract: output IS source
            "padding": dict(self.padding),
            "mode": self.mode,
            "padMode": self.pad_mode,
        }


def preferred_bucket(width: int, height: int) -> tuple[int, int]:
    """The resolution FluxKontextImageScale would pick. Same tie-break as the
    node: min() over (distance, w, h), so equal distances resolve identically."""
    aspect = width / height
    _, w, h = min(
        (abs(aspect - w / h), w, h) for w, h in PREFERRED_KONTEXT_RESOLUTIONS
    )
    return w, h


def comfy_would_crop(width: int, height: int, target_w: int, target_h: int) -> tuple[int, int]:
    """How many pixels ``common_upscale(..., crop="center")`` would cut.

    A transcription of comfy/utils.py:1075-1086. Used to *verify* that padding
    actually removed the crop, rather than assuming that matching the aspect
    ratio to three decimal places is close enough for integer pixels.
    """
    old_aspect = width / height
    new_aspect = target_w / target_h
    x = y = 0
    if old_aspect > new_aspect:
        x = round((width - width * (new_aspect / old_aspect)) / 2)
    elif old_aspect < new_aspect:
        y = round((height - height * (old_aspect / new_aspect)) / 2)
    return x, y


def plan(
    source: tuple[int, int],
    bucket: tuple[int, int] | None = None,
    pad_mode: str = "black",
) -> Transform:
    """Work out the padding that makes the scaler stop cropping.

    Padding is split deterministically: the extra odd pixel always goes to the
    right (or the bottom). A rule that never varies matters more than a
    perfectly centred image, because the inverse has to agree with it exactly.
    """
    width, height = int(source[0]), int(source[1])
    if width <= 0 or height <= 0:
        raise GeometryError("Frame has no size.", "bad_source_size")
    if pad_mode not in PAD_MODES:
        raise GeometryError(f"Unknown padding mode: {pad_mode}", "bad_pad_mode")

    if min(width, height) < MIN_SOURCE_EDGE:
        raise GeometryError(
            f"The canvas is {width}x{height}. Each side must be at least "
            f"{MIN_SOURCE_EDGE} pixels.",
            "source_too_small",
        )
    if max(width, height) > MAX_SOURCE_EDGE or width * height > MAX_SOURCE_PIXELS:
        raise GeometryError(
            f"The canvas is {width}x{height}. The limit is {MAX_SOURCE_EDGE} pixels "
            f"per side and {MAX_SOURCE_PIXELS // 1_000_000} megapixels in total.",
            "source_too_large",
        )

    model_w, model_h = bucket or preferred_bucket(width, height)
    target_aspect = model_w / model_h
    source_aspect = width / height

    coverage = min(source_aspect, target_aspect) / max(source_aspect, target_aspect)
    if coverage < MIN_COVERAGE:
        low, high = supported_aspect_range()
        raise GeometryError(
            f"A {width}x{height} canvas is {source_aspect:.2f}:1, and the closest "
            f"shape this model supports is {target_aspect:.2f}:1 -- padding it "
            f"would leave {(1 - coverage) * 100:.0f}% of the picture as black "
            f"bars. Use an aspect ratio between {low:.2f}:1 and {high:.2f}:1.",
            "aspect_unsupported",
        )

    padding = {"left": 0, "right": 0, "top": 0, "bottom": 0}
    padded_w, padded_h = width, height

    if source_aspect < target_aspect:            # too tall -> bars at the sides
        padded_w = round(height * target_aspect)
        extra = padded_w - width
        padding["left"] = extra // 2
        padding["right"] = extra - extra // 2
    elif source_aspect > target_aspect:          # too wide -> bars top and bottom
        padded_h = round(width / target_aspect)
        extra = padded_h - height
        padding["top"] = extra // 2
        padding["bottom"] = extra - extra // 2

    transform = Transform(
        source=(width, height),
        padded=(padded_w, padded_h),
        model=(model_w, model_h),
        padding=padding,
        mode="fit_pad" if any(padding.values()) else "passthrough",
        pad_mode=pad_mode,
    )

    # Point 7: prove the padding did its job. Rounding to whole pixels can leave
    # the aspect ratio a hair off, and a hair is enough for the node to shave a
    # row. If it would still crop, say so loudly rather than compensate later --
    # compensating for a crop you cannot see is how alignment bugs are born.
    crop_x, crop_y = comfy_would_crop(padded_w, padded_h, model_w, model_h)
    if crop_x or crop_y:
        raise GeometryError(
            f"Padding to {padded_w}x{padded_h} still leaves FluxKontextImageScale "
            f"cropping {crop_x}px horizontally and {crop_y}px vertically. The "
            f"bridge must resize deterministically instead.",
            "still_crops",
        )
    return transform


# --------------------------------------------------------------------------
# Pixels
# --------------------------------------------------------------------------

def _pillow():
    try:
        from PIL import Image
        return Image
    except ImportError as exc:                    # pragma: no cover
        raise GeometryError(
            "Pillow is not available in ComfyUI's Python environment, so the "
            "bridge cannot pad or crop frames. Install it with: "
            "python -m pip install Pillow",
            "pillow_missing",
        ) from exc


def pillow_available() -> bool:
    try:
        _pillow()
        return True
    except GeometryError:
        return False


def pad(data: bytes, transform: Transform) -> bytes:
    """source -> padded. The clean frame and the annotated frame get the exact
    same call, which is what keeps the strokes over what they point at."""
    Image = _pillow()
    import io

    with Image.open(io.BytesIO(data)) as src:
        image = src.convert("RGB")
        if image.size != transform.source:
            raise GeometryError(
                f"Frame is {image.size[0]}x{image.size[1]} but the transform was "
                f"planned for {transform.source[0]}x{transform.source[1]}.",
                "size_mismatch",
            )
        if not transform.has_padding:
            return _encode(image)

        p = transform.padding
        if transform.pad_mode == "black":
            canvas = Image.new("RGB", transform.padded, (0, 0, 0))
            canvas.paste(image, (p["left"], p["top"]))
        else:
            # edge / reflect, kept behind the same contract so they can be
            # compared later without touching anything else.
            from PIL import ImageOps
            canvas = ImageOps.expand(image, border=(p["left"], p["top"], p["right"], p["bottom"]))
            if transform.pad_mode == "reflect":
                canvas = _reflect(image, transform)
        return _encode(canvas)


def _reflect(image, transform: Transform):
    Image = _pillow()
    p = transform.padding
    canvas = Image.new("RGB", transform.padded)
    canvas.paste(image, (p["left"], p["top"]))
    if p["left"]:
        strip = image.crop((0, 0, p["left"], image.height)).transpose(Image.FLIP_LEFT_RIGHT)
        canvas.paste(strip, (0, p["top"]))
    if p["right"]:
        strip = image.crop((image.width - p["right"], 0, image.width, image.height)).transpose(Image.FLIP_LEFT_RIGHT)
        canvas.paste(strip, (p["left"] + image.width, p["top"]))
    if p["top"]:
        strip = canvas.crop((0, p["top"], canvas.width, p["top"] * 2)).transpose(Image.FLIP_TOP_BOTTOM)
        canvas.paste(strip, (0, 0))
    if p["bottom"]:
        y = p["top"] + image.height
        strip = canvas.crop((0, y - p["bottom"], canvas.width, y)).transpose(Image.FLIP_TOP_BOTTOM)
        canvas.paste(strip, (0, y))
    return canvas


def fit_into(data: bytes, size: tuple[int, int], pad_mode: str = "black") -> bytes:
    """Letterbox an unrelated image into ``size``. No crop, no stretch.

    For the continuity board, which is a reference in its own right and may have
    any shape. It gets its own fit -- applying the frame's transform to it would
    be meaningless, since it is not the frame.
    """
    Image = _pillow()
    import io

    with Image.open(io.BytesIO(data)) as src:
        image = src.convert("RGB")
        scale = min(size[0] / image.width, size[1] / image.height)
        new = (max(1, round(image.width * scale)), max(1, round(image.height * scale)))
        resized = image.resize(new, Image.LANCZOS)
        canvas = Image.new("RGB", size, (0, 0, 0))
        canvas.paste(resized, ((size[0] - new[0]) // 2, (size[1] - new[1]) // 2))
        return _encode(canvas)


def unpad(data: bytes, transform: Transform) -> bytes:
    """model output -> exactly the source resolution.

    The crop box is computed in output coordinates from the integers stored on
    the way in, and rounded *inwards* -- ceil on the left/top edge, floor on the
    right/bottom -- so a rounding error can only ever cost a sliver of picture,
    never leave a line of black padding welded to the edge of the frame.
    """
    Image = _pillow()
    import io
    import math

    with Image.open(io.BytesIO(data)) as src:
        image = src.convert("RGB")
        if not transform.has_padding and image.size == transform.source:
            return _encode(image)

        scale_x = image.width / transform.padded[0]
        scale_y = image.height / transform.padded[1]
        p = transform.padding
        left = math.ceil(p["left"] * scale_x)
        top = math.ceil(p["top"] * scale_y)
        right = image.width - math.floor(p["right"] * scale_x)
        bottom = image.height - math.floor(p["bottom"] * scale_y)

        cropped = image.crop((left, top, max(left + 1, right), max(top + 1, bottom)))
        if cropped.size != transform.source:
            cropped = cropped.resize(transform.source, Image.LANCZOS)
        return _encode(cropped)


def _encode(image) -> bytes:
    import io

    buffer = io.BytesIO()
    image.save(buffer, format="PNG", compress_level=4)
    return buffer.getvalue()


def measure(data: bytes) -> tuple[int, int]:
    """The real size of the decoded image.

    Read here, never taken from the request: a client-declared resolution that
    disagrees with the actual file would put every later crop off by exactly the
    amount of the lie.
    """
    Image = _pillow()
    import io

    try:
        with Image.open(io.BytesIO(data)) as image:
            return int(image.width), int(image.height)
    except GeometryError:
        raise
    except Exception as exc:
        # Refusing here also stops non-images reaching ComfyUI's input folder,
        # and turns a raw PIL traceback into something the editor can show.
        raise GeometryError(
            "That upload is not a readable image.", "bad_image"
        ) from exc
