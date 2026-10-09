"""Build minimax-h3-keyframes from minimax-h3-flf.

    python tools/build_h3_keyframes_template.py

Derived, not drawn: the first & last frame graph is the one measured on the
reference machine, and this changes only what has to change to place keys at
any frame --

  * the two LoadImage nodes become one MultiImageLoader ($IMAGE_PATHS), the
    same collected-keyframes loader LTX uses;
  * ReanimatorFitToGrid crops the whole batch, once, to a 32-px canvas of
    the keys' own shape. The flf graph's ImageScaleToTotalPixels made 16:9
    1280x736, and H3 stretches first_frame but cover-crops guides, so the
    shot zoomed 2.2 % when it left key 1;
  * ImageFromBatch takes key 1 out of the batch as first_frame, because
    MiniMaxH3ImageToVideo also hands the first frame to the text encoder,
    which is where identity comes from;
  * ReanimatorH3Sequencer ($SEQUENCER) anchors keys 2..N with ComfyUI's own
    MiniMaxH3AddGuide, at the frames the editor asked for.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

TEMPLATES = Path(__file__).resolve().parents[1] / "templates"
MAX_KEYS = 32


def build_graph(flf: dict) -> dict:
    g = copy.deepcopy(flf)
    del g["6"], g["7"]
    g["21"] = {
        "inputs": {
            "image_paths": "frame_0001.png\nframe_0002.png",
            "width": 0, "height": 0, "interpolation": "lanczos",
            "resize_method": "keep proportion", "multiple_of": 0, "img_compression": 18,
        },
        "class_type": "MultiImageLoader",
        "_meta": {"title": "$IMAGE_PATHS"},
    }
    del g["8"], g["9"]
    g["24"] = {
        "inputs": {"images": ["21", 0], "megapixels": 0.9, "step": 32},
        "class_type": "ReanimatorFitToGrid",
        "_meta": {"title": "Fit keys to one canvas"},
    }
    g["22"] = {
        "inputs": {"image": ["24", 0], "batch_index": 0, "length": 1},
        "class_type": "ImageFromBatch",
        "_meta": {"title": "Key 1"},
    }
    g["11"]["inputs"]["first_frame"] = ["22", 0]
    g["11"]["inputs"]["width"] = ["24", 1]
    g["11"]["inputs"]["height"] = ["24", 2]
    g["11"]["inputs"].pop("last_frame")
    seq = {
        "positive": ["11", 0], "latent": ["11", 1], "vae": ["4", 0], "images": ["24", 0],
        "num_images": 2,
    }
    for index in range(1, MAX_KEYS + 1):
        seq[f"insert_frame_{index}"] = 0
    g["23"] = {
        "inputs": seq,
        "class_type": "ReanimatorH3Sequencer",
        "_meta": {"title": "$SEQUENCER"},
    }
    g["15"]["inputs"]["conditioning"] = ["23", 0]
    g["20"]["inputs"]["filename_prefix"] = "video/Reanimator_H3_keys"
    return g


def build_manifest(flf: dict) -> dict:
    m = copy.deepcopy(flf)
    m["id"] = "minimax-h3-keyframes"
    m["name"] = "MiniMax H3 · keyframes at any frame"
    m["model"] = "minimax-h3-keys"
    m["priority"] = 60
    m["description"] = (
        "MiniMax H3 builds a video from your keyframes, each one pinned at its own frame, "
        "with sound."
    )
    m["images"] = MAX_KEYS
    m["intents"] = {
        "video_from_keyframes": {
            "requires": ["keyframes"],
            "note": (
                "Keys at arbitrary frames: key 1 is the first frame, and every other key is "
                "anchored where the timeline puts it with MiniMaxH3AddGuide."
            ),
        }
    }
    m["roles"] = {
        "keyframes": {
            "slot": "$IMAGE_PATHS",
            "required": True,
            "collect": True,
            "max": MAX_KEYS,
            "note": "One loader takes the whole list. Order matters: image N is timed by keyframe N.",
        }
    }
    m["frames"] = {
        "slot": "$FRAMES",
        "min": 39,
        "step": 17,
        "offset": 5,
        "note": (
            "H3 latents want 17k+5 frames at 24 fps; the bridge rounds the span up and says by "
            "how much. Floored at 39 (~1.6 s) rather than the first & last preset's 73: here the "
            "keys, not the clip length, carry the timing, and every frame past the last key is "
            "GPU time for footage the editor does not use."
        ),
    }
    m["sequencer"] = {
        "parameter": "$SEQUENCER",
        "note": (
            "The same [{frame, strength}] list LTX takes. H3 guides have no strength, so the "
            "node has no strength widgets and the binder writes frames only."
        ),
    }
    slots = m["slots"]
    slots.pop("$IMAGE_1")
    slots.pop("$IMAGE_2")
    slots["$IMAGE_PATHS"] = {"required": True, "note": "every keyframe image, one filename per line"}
    slots["$SEQUENCER"] = {"required": True, "note": "the frame of each key"}
    dropped = {"LoadImage", "ImageScaleToTotalPixels", "GetImageSize"}
    nodes = [n for n in m["requires"]["nodes"] if n not in dropped]
    nodes += ["MultiImageLoader", "ImageFromBatch", "ReanimatorFitToGrid", "ReanimatorH3Sequencer"]
    m["requires"]["nodes"] = nodes
    m["requires"]["nodePacks"] = [
        {"name": "WhatDreamsCost-ComfyUI", "provides": ["MultiImageLoader"]},
        {"name": "comfyui-reanimator", "provides": ["ReanimatorFitToGrid", "ReanimatorH3Sequencer"],
         "note": "This bridge's own nodes: one canvas for every key, and ComfyUI's "
                 "MiniMaxH3AddGuide once per key."},
    ]
    comp = m["requires"]["compatibility"]
    comp["note"] = (
        "Same models and sampler as minimax-h3-flf, whose numbers are the only measurement so "
        "far (RTX 3090, 5 s clip, 516 s cold). Keyframe guides add one VAE encode each."
    )
    comp.pop("measured", None)
    m["notes"] = [
        "Built by tools/build_h3_keyframes_template.py from minimax-h3-flf; edit that script, "
        "not this file.",
    ]
    return m


# The cloud variant: what a 96 GB GPU can afford and a 3090 cannot.
#   * no Turbo LoRA: the base model is already CFG-distilled, and the official
#     ComfyUI template runs it at 20 steps with the same BasicGuider. The
#     8-step LoRA was distilled at 544p, below what we render at;
#   * the unpruned checkpoint (int8_convrot, 34 GB) first, the pruned one as
#     the fallback while the volume does not have it;
#   * H3's own 768p budget (0.98 MP is the official 1344x768).
HQ_STEPS = 20
HQ_MEGAPIXELS = 0.98
FULL_CHECKPOINT = {
    "folder": "diffusion_models",
    "file": "minimax_h3_fl2va_int8_convrot.safetensors",
    "gb": 34.0,
    "url": "https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/diffusion_models/minimax_h3_fl2va_int8_convrot.safetensors",
    "requiresNativeOps": ["int8_tensorwise", "convrot_w4a4"],
    "note": "Unpruned. Cloud only: on the reference 3090 the pruned one is what fits.",
}


def build_hq_graph(graph: dict) -> dict:
    g = copy.deepcopy(graph)
    del g["2"]
    for node in g.values():
        for name, value in node["inputs"].items():
            if value == ["2", 0]:
                node["inputs"][name] = ["1", 0]
    g["14"]["inputs"]["steps"] = HQ_STEPS
    g["24"]["inputs"]["megapixels"] = HQ_MEGAPIXELS
    g["20"]["inputs"]["filename_prefix"] = "video/Reanimator_H3_keys_hq"
    return g


def build_hq_manifest(manifest: dict) -> dict:
    m = copy.deepcopy(manifest)
    m["id"] = "minimax-h3-keyframes-hq"
    m["name"] = "MiniMax H3 · keyframes, full quality"
    # Its own model id and a low priority: the local bridge must never pick
    # this one for minimax-h3-keys on a 24 GB card.
    m["model"] = "minimax-h3-keys-hq"
    m["priority"] = 10
    m["description"] = (
        "MiniMax H3 from your keyframes at full quality: the unpruned model, 20 steps, "
        "no Turbo LoRA. Made for the cloud GPU."
    )
    req = m["requires"]
    req["nodes"] = [n for n in req["nodes"] if n != "LoraLoaderModelOnly"]
    req["models"] = [x for x in req["models"] if x["folder"] != "loras"]
    ckpt = req["checkpoints"]
    ckpt["variants"] = {"int8_convrot_full": FULL_CHECKPOINT, **ckpt["variants"]}
    ckpt["preferred"] = "int8_convrot_full"
    ckpt["alternatives"] = ["int8_convrot"]
    comp = req["compatibility"]
    comp["minimumVramGb"] = 48
    comp["note"] = (
        f"{HQ_STEPS} steps instead of 8: about 2.5x the sampling time of minimax-h3-keyframes. "
        "Meant for RunPod (RTX PRO 6000 / H100)."
    )
    m["minimumVramGb"] = 48
    m["notes"] = [
        "Built by tools/build_h3_keyframes_template.py from minimax-h3-keyframes; edit that "
        "script, not this file.",
    ]
    return m


# References (fal's reference-to-video) on our GPU: the ref2va checkpoint and
# ComfyUI's MiniMaxH3ReferenceToVideo, reached through ReanimatorH3References
# because the number of references is only known at run time. Same sampler and
# step count as the HQ keyframes template; no Turbo LoRA exists for ref2va at
# this size.
MAX_REFERENCE_IMAGES = 9
REF_CHECKPOINTS = {
    "int8_convrot_full": {
        "folder": "diffusion_models",
        "file": "minimax_h3_ref2va_int8_convrot.safetensors",
        "gb": 34.04,
        "url": "https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/diffusion_models/minimax_h3_ref2va_int8_convrot.safetensors",
        "requiresNativeOps": ["int8_tensorwise", "convrot_w4a4"],
        "note": "Unpruned. Cloud only.",
    },
    "int8_convrot": {
        "folder": "diffusion_models",
        "file": "minimax_h3_ref2va_pruned_int8_convrot.safetensors",
        "gb": 20.97,
        "url": "https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/diffusion_models/minimax_h3_ref2va_pruned_int8_convrot.safetensors",
        "requiresNativeOps": ["int8_tensorwise", "convrot_w4a4"],
    },
}


def build_references_graph(flf: dict) -> dict:
    g = {k: copy.deepcopy(flf[k]) for k in ("1", "3", "4", "5", "12", "13", "14", "15",
                                             "16", "17", "18", "19", "20")}
    g["1"]["inputs"]["unet_name"] = REF_CHECKPOINTS["int8_convrot_full"]["file"]
    g["10"] = {"inputs": {"value": 124}, "class_type": "PrimitiveInt",
               "_meta": {"title": "$FRAMES:value"}}
    g["30"] = {"inputs": {"value": 1344}, "class_type": "PrimitiveInt",
               "_meta": {"title": "$WIDTH:value"}}
    g["31"] = {"inputs": {"value": 768}, "class_type": "PrimitiveInt",
               "_meta": {"title": "$HEIGHT:value"}}
    g["32"] = {"inputs": {"value": ""}, "class_type": "PrimitiveStringMultiline",
               "_meta": {"title": "$PROMPT:value"}}
    g["22"] = {"inputs": {"file": "example.mp4"}, "class_type": "LoadVideo",
               "_meta": {"title": "$VIDEO_1"}}
    g["21"] = {
        "inputs": {
            "clip": ["3", 0], "vae": ["4", 0], "audio_vae": ["5", 0], "prompt": ["32", 0],
            "width": ["30", 0], "height": ["31", 0], "length": ["10", 0],
            "image_paths": "", "ref_image_size": "match", "video": ["22", 0],
        },
        "class_type": "ReanimatorH3References",
        "_meta": {"title": "$IMAGE_PATHS"},
    }
    g["14"]["inputs"]["model"] = ["1", 0]
    g["14"]["inputs"]["steps"] = HQ_STEPS
    g["15"]["inputs"]["model"] = ["1", 0]
    g["15"]["inputs"]["conditioning"] = ["21", 0]
    g["16"]["inputs"]["latent_image"] = ["21", 1]
    g["20"]["inputs"]["filename_prefix"] = "video/Reanimator_H3_refs"
    return g


def build_references_manifest(flf: dict) -> dict:
    m = copy.deepcopy(flf)
    m["id"] = "minimax-h3-references-hq"
    m["name"] = "MiniMax H3 · references, full quality"
    m["model"] = "minimax-h3-refs-hq"
    m["priority"] = 10
    m["description"] = (
        "MiniMax H3 from reference images and a reference video, with sound: the unpruned "
        "ref2va model at 20 steps. Made for the cloud GPU."
    )
    m["images"] = MAX_REFERENCE_IMAGES
    m["intents"] = {
        "video_from_references": {
            "requires": [],
            "note": (
                "Reference-to-video: the images and the video are material (character, style, "
                "motion), not frames of the result. With neither it is text-to-video. The prompt "
                "names them <Picture N> and <Video 1>."
            ),
        }
    }
    m["roles"] = {
        "references": {
            "slot": "$IMAGE_PATHS",
            "required": False,
            "collect": True,
            "ownSize": True,
            "max": MAX_REFERENCE_IMAGES,
            "note": "Every reference image, one loader. Order matters: image N is <Picture N>.",
        },
        "referenceVideo": {
            "slot": "$VIDEO_1",
            "required": False,
            "max": 1,
            "detachWhenAbsent": [{
                "slot": "$IMAGE_PATHS", "input": "video",
                "note": "No reference video: LoadVideo is unplugged, so it never runs.",
            }],
            "note": "<Video 1>, with its own soundtrack when it has one.",
        },
    }
    m["frames"] = {
        "slot": "$FRAMES",
        "min": 39,
        "step": 17,
        "offset": 5,
        "note": "17k+5 frames at 24 fps, as every H3 template.",
    }
    m["slots"] = {
        "$IMAGE_PATHS": {"required": False, "note": "reference images, one filename per line"},
        "$VIDEO_1": {"required": False, "note": "reference video"},
        "$FRAMES": {"required": True, "note": "output length; the bridge rounds up to 17k+5"},
        "$WIDTH": {"required": True, "note": "output width, a multiple of 32"},
        "$HEIGHT": {"required": True, "note": "output height, a multiple of 32"},
        "$PROMPT": {"required": True},
        "$SEED": {"required": False, "randomizePerRun": True},
        "$CHECKPOINT": {"required": True,
                        "note": "chosen from requires.checkpoints, never by the caller"},
        "$OUTPUT": {"required": True},
    }
    req = m["requires"]
    dropped = {"LoadImage", "ImageScaleToTotalPixels", "GetImageSize", "LoraLoaderModelOnly",
               "MiniMaxH3ImageToVideo"}
    req["nodes"] = [n for n in req["nodes"] if n not in dropped] + [
        "PrimitiveStringMultiline", "LoadVideo", "ReanimatorH3References"]
    req["models"] = [x for x in req["models"] if x["folder"] != "loras"]
    req["checkpoints"] = {
        "slot": "$CHECKPOINT",
        "variants": copy.deepcopy(REF_CHECKPOINTS),
        "preferred": "int8_convrot_full",
        "alternatives": ["int8_convrot"],
    }
    req["nodePacks"] = [
        {"name": "comfyui-reanimator", "provides": ["ReanimatorH3References"],
         "note": "Loads a run-time list of references and calls ComfyUI's "
                 "MiniMaxH3ReferenceToVideo once."},
    ]
    comp = req["compatibility"]
    comp["minimumVramGb"] = 48
    comp["note"] = f"{HQ_STEPS} steps, unpruned ref2va. Meant for RunPod (RTX PRO 6000 / H100)."
    comp.pop("measured", None)
    m["minimumVramGb"] = 48
    m.pop("sequencer", None)
    m["notes"] = [
        "Built by tools/build_h3_keyframes_template.py from minimax-h3-flf; edit that script, "
        "not this file.",
    ]
    return m


# The same references on a 24 GB card: the pruned ref2va and Comfy-Org's
# ref2v Turbo LoRA at 4 steps (strength 1, as ComfyUI's own r2v template), so
# the 3090 does not pay for 20 steps. Its own model id: the hub lists it as a
# card of its own, next to the keyframe presets.
REF_TURBO_LORA = {
    "folder": "loras",
    "file": "minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors",
    "gb": 1.96,
    "url": "https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/loras/minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors",
}
REF_TURBO_STEPS = 4


def build_references_local_graph(flf: dict) -> dict:
    g = build_references_graph(flf)
    g["1"]["inputs"]["unet_name"] = REF_CHECKPOINTS["int8_convrot"]["file"]
    g["2"] = {"inputs": {"model": ["1", 0], "lora_name": REF_TURBO_LORA["file"], "strength_model": 1.0},
              "class_type": "LoraLoaderModelOnly", "_meta": {"title": "Turbo 4-step LoRA"}}
    g["14"]["inputs"]["model"] = ["2", 0]
    g["14"]["inputs"]["steps"] = REF_TURBO_STEPS
    g["15"]["inputs"]["model"] = ["2", 0]
    g["20"]["inputs"]["filename_prefix"] = "video/Reanimator_H3_refs_local"
    return g


def build_references_local_manifest(flf: dict) -> dict:
    m = build_references_manifest(flf)
    m["id"] = "minimax-h3-references"
    m["name"] = "MiniMax H3 · references"
    m["model"] = "minimax-h3-refs"
    m["priority"] = 50
    m["description"] = (
        "MiniMax H3 from reference images (characters, style) and a reference video, with sound. "
        "Runs on your GPU; nothing leaves this machine."
    )
    req = m["requires"]
    req["nodes"] = req["nodes"] + ["LoraLoaderModelOnly"]
    req["models"] = req["models"] + [dict(REF_TURBO_LORA)]
    req["checkpoints"] = {
        "slot": "$CHECKPOINT",
        "variants": {"int8_convrot": copy.deepcopy(REF_CHECKPOINTS["int8_convrot"])},
        "preferred": "int8_convrot",
        "alternatives": [],
    }
    comp = req["compatibility"]
    comp["minimumVramGb"] = 24
    comp["note"] = (f"Pruned ref2va with the Turbo LoRA (v0.1) at {REF_TURBO_STEPS} steps. "
                    "Not measured on the 3090 yet.")
    m["minimumVramGb"] = 24
    return m


def main() -> None:
    flf_graph = json.loads((TEMPLATES / "minimax-h3-flf.json").read_text(encoding="utf-8"))
    flf_manifest = json.loads((TEMPLATES / "minimax-h3-flf.manifest.json").read_text(encoding="utf-8"))
    graph = build_graph(flf_graph)
    manifest = build_manifest(flf_manifest)
    outputs = {
        "minimax-h3-keyframes": (graph, manifest),
        "minimax-h3-keyframes-hq": (build_hq_graph(graph), build_hq_manifest(manifest)),
        "minimax-h3-references-hq": (build_references_graph(flf_graph),
                                     build_references_manifest(flf_manifest)),
        "minimax-h3-references": (build_references_local_graph(flf_graph),
                                  build_references_local_manifest(flf_manifest)),
    }
    for name, (g, m) in outputs.items():
        (TEMPLATES / f"{name}.json").write_text(
            json.dumps(g, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
        (TEMPLATES / f"{name}.manifest.json").write_text(
            json.dumps(m, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"wrote {name}")


if __name__ == "__main__":
    main()
