"""This package's nodes: MiniMax H3 keyframes at any frame, the canvas they
share, and MiniMax H3 from references (ReanimatorH3References, below).

ComfyUI's own MiniMaxH3AddGuide anchors ONE image at a frame, and the way to
anchor several is to chain several nodes. A template cannot do that for a
number of keys it only learns at run time, so this node does the chaining: it
takes the whole keyframe list (one MultiImageLoader batch) and the timings,
and calls MiniMaxH3AddGuide once per key.

Its widgets are laid out like LTXSequencer's -- num_images, insert_frame_N --
on purpose. The bridge already knows how to write a $SEQUENCER into that
shape, so the editor sends the same [{frame, strength}] list to LTX and to H3
and neither side learns which one it is talking to. H3 guides have no
strength, so there are no strength widgets and the binder skips them.
"""

from __future__ import annotations

MAX_KEYS = 32



def input_image_path(name: str) -> str:
    """A file directly inside ComfyUI's input folder, or an error.

    The names come from a text widget, so anyone building a workflow can type
    anything: "../../secrets.png", "C:/Users/...", "x.png [output]". Only a bare
    file name in input/ is read -- the same place LoadImage reads from, and
    where the bridge puts its uploads.
    """
    from pathlib import Path

    import folder_paths

    if not name or name != Path(name).name or name in (".", ".."):
        raise ValueError(f"Not a file in ComfyUI's input folder: {name!r}")
    base = Path(folder_paths.get_input_directory()).resolve()
    path = (base / name).resolve()
    if path.parent != base or not path.is_file():
        raise ValueError(f"Not a file in ComfyUI's input folder: {name!r}")
    return str(path)

class ReanimatorH3Sequencer:
    """Anchor every keyframe of a batch at its own frame of a MiniMax H3 video."""

    CATEGORY = "Reanimator"
    RETURN_TYPES = ("CONDITIONING",)
    RETURN_NAMES = ("positive",)
    FUNCTION = "apply"

    @classmethod
    def INPUT_TYPES(cls):
        required = {
            "positive": ("CONDITIONING",),
            "latent": ("LATENT",),
            "vae": ("VAE",),
            "images": ("IMAGE",),
            "num_images": ("INT", {"default": 2, "min": 1, "max": MAX_KEYS}),
        }
        for index in range(1, MAX_KEYS + 1):
            required[f"insert_frame_{index}"] = ("INT", {"default": 0, "min": 0, "max": 9999})
        return {"required": required}

    def apply(self, positive, latent, vae, images, num_images, **frames):
        from comfy_extras.nodes_minimax_h3 import MiniMaxH3AddGuide

        count = min(int(num_images), int(images.shape[0]))
        # A frame already anchored -- the first key, which MiniMaxH3ImageToVideo
        # takes as first_frame so the text encoder sees it too -- is not
        # anchored twice.
        anchored = {
            kf.get("resolved_frame_index")
            for kf in positive[0][1].get("minimax_keyframes", [])
        }
        for index in range(count):
            frame = int(frames.get(f"insert_frame_{index + 1}", 0))
            if frame in anchored:
                continue
            out = MiniMaxH3AddGuide.execute(
                positive=positive, latent=latent, frame_idx=frame,
                vae=vae, image=images[index:index + 1],
            )
            positive = out[0]
            anchored.add(frame)
        return (positive,)


def grid_size(width: int, height: int, megapixels: float, step: int = 32,
              tolerance: float = 0.005) -> tuple[int, int]:
    """The canvas closest to `megapixels` whose sides are multiples of `step`
    and whose shape matches width x height within `tolerance`.

    ImageScaleToTotalPixels rounds each side on its own, and a 1280x720 key
    came out 1280x736: 2.2 % taller than 16:9. H3 then stretched the first key
    to that shape and cover-cropped every other one, so the shot zoomed in the
    moment it left key 1 (measured: the guided frames matched their keys best
    at exactly 1.022x). Here the shape is kept -- 16:9 lands on 1312x736, off
    by 0.3 % -- and when no size is that close, the closest shape wins.
    """
    aspect = width / height
    target = megapixels * 1024 * 1024
    best = None
    for h in range(step, 4097, step):
        w = max(step, round(h * aspect / step) * step)
        error = abs((w / h) / aspect - 1)
        area_off = abs(w * h / target - 1)
        key = (error > tolerance, error if error > tolerance else area_off)
        if best is None or key < best[0]:
            best = (key, w, h)
    return best[1], best[2]


class ReanimatorFitToGrid:
    """Crop and scale every image of a batch to one grid-legal canvas."""

    CATEGORY = "Reanimator"
    RETURN_TYPES = ("IMAGE", "INT", "INT")
    RETURN_NAMES = ("images", "width", "height")
    FUNCTION = "fit"

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "images": ("IMAGE",),
            "megapixels": ("FLOAT", {"default": 0.9, "min": 0.1, "max": 8.0, "step": 0.05}),
            "step": ("INT", {"default": 32, "min": 8, "max": 128, "step": 8}),
        }}

    def fit(self, images, megapixels, step):
        import comfy.utils

        height, width = int(images.shape[1]), int(images.shape[2])
        w, h = grid_size(width, height, float(megapixels), int(step))
        # One crop for the whole batch: the same centre, the same scale, so no
        # key is framed differently from another.
        samples = images[..., :3].movedim(-1, 1)
        samples = comfy.utils.common_upscale(samples, w, h, "lanczos", "center")
        return (samples.movedim(1, -1), w, h)


MAX_REFERENCE_IMAGES = 9
REFERENCE_FPS = 24


def resample_indices(count: int, fps: float, target: float = REFERENCE_FPS) -> list[int]:
    """Frame indices that play `count` frames shot at `fps` back at `target`.
    H3 reads reference videos as 24 fps; a 30 fps phone clip handed over as is
    would play 25 % slow inside the model."""
    if count <= 0 or fps <= 0:
        return []
    out = int(count * target / fps)
    return [min(count - 1, int(i * fps / target)) for i in range(max(1, out))]


class ReanimatorH3References:
    """MiniMax H3 from reference images (and one reference video).

    ComfyUI's MiniMaxH3ReferenceToVideo takes each reference on its own input
    (ref_image_1..9, ref_video_1), which a template can only wire for a count
    it knows in advance. The bridge only learns the count at run time, so this
    node takes the whole list of uploaded file names -- the same $IMAGE_PATHS
    shape the keyframe templates use -- loads each image at its own size (a
    batch would force one size on all of them) and calls that node once.
    """

    CATEGORY = "Reanimator"
    RETURN_TYPES = ("CONDITIONING", "LATENT")
    RETURN_NAMES = ("positive", "latent")
    FUNCTION = "apply"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": ("CLIP",),
                "vae": ("VAE",),
                "audio_vae": ("VAE",),
                "prompt": ("STRING", {"multiline": True, "forceInput": True}),
                "width": ("INT", {"default": 1344, "min": 32, "max": 4096, "step": 32}),
                "height": ("INT", {"default": 768, "min": 32, "max": 4096, "step": 32}),
                "length": ("INT", {"default": 124, "min": 5, "max": 3600}),
                "image_paths": ("STRING", {"multiline": True, "default": ""}),
                "ref_image_size": (["match", "max"], {"default": "match"}),
            },
            "optional": {"video": ("VIDEO",)},
        }

    @staticmethod
    def _load_image(name: str):
        import numpy as np
        import torch
        from PIL import Image, ImageOps

        with Image.open(input_image_path(name)) as img:
            rgb = ImageOps.exif_transpose(img).convert("RGB")
            array = np.asarray(rgb, dtype=np.float32) / 255.0
        return torch.from_numpy(array)[None]

    def apply(self, clip, vae, audio_vae, prompt, width, height, length, image_paths,
              ref_image_size="match", video=None):
        from comfy_extras.nodes_minimax_h3 import MiniMaxH3ReferenceToVideo

        names = [n.strip() for n in str(image_paths or "").splitlines() if n.strip()]
        if len(names) > MAX_REFERENCE_IMAGES:
            raise ValueError(f"MiniMax H3 takes up to {MAX_REFERENCE_IMAGES} reference images")
        ref_images = {f"ref_image_{i + 1}": self._load_image(n) for i, n in enumerate(names)}

        ref_videos, ref_audios = {}, {}
        if video is not None:
            parts = video.get_components()
            frames = parts.images
            keep = resample_indices(int(frames.shape[0]), float(parts.frame_rate))
            ref_videos["ref_video_1"] = frames[keep]
            if parts.audio is not None:
                ref_audios["ref_video_audio_1"] = parts.audio

        out = MiniMaxH3ReferenceToVideo.execute(
            clip=clip, prompt=prompt, width=int(width), height=int(height), length=int(length),
            ref_image_size=ref_image_size, vae=vae, audio_vae=audio_vae,
            ref_images=ref_images or None, ref_videos=ref_videos or None,
            ref_video_audios=ref_audios or None,
        )
        return (out[0], out[1])


NODE_CLASS_MAPPINGS = {
    "ReanimatorH3Sequencer": ReanimatorH3Sequencer,
    "ReanimatorFitToGrid": ReanimatorFitToGrid,
    "ReanimatorH3References": ReanimatorH3References,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "ReanimatorH3Sequencer": "MiniMax H3 keyframes (Reanimator)",
    "ReanimatorFitToGrid": "Fit images to a grid canvas (Reanimator)",
    "ReanimatorH3References": "MiniMax H3 references (Reanimator)",
}
