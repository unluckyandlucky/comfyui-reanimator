"""Build the LTX Extend templates: continue a clip from its own last frames.

    python comfyui-reanimator/tools/build_ltx_extend_templates.py

Extend from the last frame alone is out (user's call, 2026-10-06): a single
still says nothing about where the camera and the people were going, and the
seam jumps. Here the tail of the clip itself goes in. LTXVImgToVideoInplace
encodes those frames and writes them over the first latent frames with a
noise mask of 0, so the sampler keeps them as they are and only denoises what
comes after: a continuation of real motion, not a guess from a picture. In a
multi-stage graph every stage pins the tail again on its own latent.

Three variants, to compare on the same shot:

* ltx-25-extend      LTX-2.5 distilled, one stage at the source's size.
* ltx-25-dev-extend  LTX-2.5 dev with CFG at half size, then the x2 latent
                     upscaler and a distilled refine (Lightricks' two-stage
                     recipe; the dev model is slower and better at detail).
* ltx-23-extend      the user's own LTX-2.3 three-stage graph (dev fp8 +
                     distilled LoRA, 360p -> 720p -> 1080p), the one behind
                     ltx-23-keyframes, with the tail pinned instead of keys.

Edit this file, not the JSON it writes.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

TEMPLATES = Path(__file__).resolve().parents[1] / "templates"
HF25 = "https://huggingface.co/Lightricks/LTX-2.5/resolve/main"
DISTILLED_SIGMAS = "1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0"
REFINE_SIGMAS = "0.85, 0.7250, 0.4219, 0.0"
NEGATIVE = ("pc game, console game, video game, cartoon, childish, ugly, deformed hands, extra fingers, "
            "fused fingers, distorted limbs, warped face, jump cut, scene change")
BASE25 = json.loads((TEMPLATES / "ltx-25-alpha-gen.manifest.json").read_text(encoding="utf-8"))
BASE23 = json.loads((TEMPLATES / "ltx-23-keyframes.manifest.json").read_text(encoding="utf-8"))
G23 = json.loads((TEMPLATES / "ltx-23-keyframes.json").read_text(encoding="utf-8"))

PROMPT_LINE = ("The video continues its opening frames without a cut: the same camera move at the same speed, "
               "the same action at the same pace, the same people, lighting and style. Nothing new happens.")


def node(class_type: str, inputs: dict, title: str | None = None) -> dict:
    n = {"class_type": class_type, "inputs": inputs}
    if title:
        n["_meta"] = {"title": title}
    return n


def math(a, b, op: str) -> dict:
    return node("easy mathInt", {"a": a, "b": b, "operation": op})


def source_tail() -> dict:
    """LoadVideo -> the last $CONTEXT frames (node 13), source size in 12."""
    return {
        "10": node("LoadVideo", {"file": "example.mp4"}, "$VIDEO_1"),
        "11": node("GetVideoComponents", {"video": ["10", 0]}),
        "12": node("GetImageSize", {"image": ["11", 0]}),
        "60": node("PrimitiveInt", {"value": 49}, "$CONTEXT:value"),
        "61": math(["12", 2], ["60", 0], "subtract"),
        "13": node("ImageFromBatch", {"image": ["11", 0], "batch_index": ["61", 0], "length": ["60", 0]}),
        "62": node("PrimitiveInt", {"value": 113}, "$FRAMES:value"),
    }


def output(decoded: list) -> dict:
    """Back to the source's exact size and fps, then saved."""
    return {
        "51": node("ImageScale", {"image": decoded, "upscale_method": "lanczos", "width": ["12", 0],
                                  "height": ["12", 1], "crop": "disabled"}),
        "52": node("CreateVideo", {"images": ["51", 0], "fps": ["11", 2]}),
        "53": node("SaveVideo", {"filename_prefix": "video/Reanimator_extend", "format": "auto", "codec": "auto",
                                 "video": ["52", 0]}, "$OUTPUT"),
    }


# --------------------------------------------------------------------------
# LTX-2.5 distilled, one stage
# --------------------------------------------------------------------------

def loaders25(unet: str, title: str = "$CHECKPOINT:unet_name") -> dict:
    return {
        "1": node("UNETLoader", {"unet_name": unet, "weight_dtype": "default"}, title),
        "3": node("CLIPLoader", {"clip_name": "gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot.safetensors",
                                 "type": "ltxv", "device": "default"}, "Gemma 4 text encoder"),
        "4": node("CLIPTextEncode", {"text": "", "clip": ["3", 0]}, "$PROMPT:text"),
        "5": node("VAELoader", {"vae_name": "ltx-2.5-video-vae-bf16.safetensors"}),
        "6": node("VAELoader", {"vae_name": "ltx-2.5-audio-vae-bf16.safetensors"}),
    }


def graph25() -> dict:
    g = {**loaders25("ltx-2.5-22b-distilled-transformer-comfy-int8-convrot.safetensors"), **source_tail()}
    g.update({
        # Pad the size up to LTX's 32-pixel blocks; the result is cropped back.
        "20": math(["12", 0], 32, "modulo"),
        "21": math(32, ["20", 0], "subtract"),
        "22": math(["21", 0], 32, "modulo"),
        "23": math(["12", 1], 32, "modulo"),
        "24": math(32, ["23", 0], "subtract"),
        "25": math(["24", 0], 32, "modulo"),
        "14": node("ImagePadForOutpaint", {"image": ["13", 0], "left": 0, "top": 0, "right": ["22", 0],
                                           "bottom": ["25", 0], "feathering": 0}),
        "15": node("GetImageSize", {"image": ["14", 0]}),
        "30": node("LTXVConditioning", {"positive": ["4", 0], "negative": ["4", 0], "frame_rate": ["11", 2]}),
        "31": node("EmptyLTXVLatentVideo", {"width": ["15", 0], "height": ["15", 1], "length": ["62", 0],
                                            "batch_size": 1}),
        "32": node("LTXVImgToVideoInplace", {"vae": ["5", 0], "image": ["14", 0], "latent": ["31", 0],
                                             "strength": 1.0, "bypass": False}, "Pin the clip's tail"),
        "33": node("LTXVEmptyLatentAudio", {"frames_number": ["62", 0], "frame_rate": ["11", 2],
                                            "batch_size": 1, "audio_vae": ["6", 0]}),
        "34": node("LTXVConcatAVLatent", {"video_latent": ["32", 0], "audio_latent": ["33", 0]}),
        "40": node("RandomNoise", {"noise_seed": 1234}, "$SEED:noise_seed"),
        "41": node("CFGGuider", {"model": ["1", 0], "positive": ["30", 0], "negative": ["30", 1], "cfg": 1.0}),
        "42": node("KSamplerSelect", {"sampler_name": "euler_ancestral"}),
        "43": node("ManualSigmas", {"sigmas": DISTILLED_SIGMAS}),
        "44": node("SamplerCustomAdvanced", {"noise": ["40", 0], "guider": ["41", 0], "sampler": ["42", 0],
                                             "sigmas": ["43", 0], "latent_image": ["34", 0]}),
        "45": node("LTXVSeparateAVLatent", {"av_latent": ["44", 0]}),
        "50": node("VAEDecodeTiled", {"samples": ["45", 0], "vae": ["5", 0], "tile_size": 512, "overlap": 64,
                                      "temporal_size": 512, "temporal_overlap": 8}),
    })
    # The padding is cropped off (not scaled): ImageCrop at the source size.
    g["51"] = node("ImageCrop", {"image": ["50", 0], "width": ["12", 0], "height": ["12", 1], "x": 0, "y": 0})
    g["52"] = node("CreateVideo", {"images": ["51", 0], "fps": ["11", 2]})
    g["53"] = node("SaveVideo", {"filename_prefix": "video/Reanimator_extend", "format": "auto", "codec": "auto",
                                 "video": ["52", 0]}, "$OUTPUT")
    return g


# --------------------------------------------------------------------------
# LTX-2.5 dev, two stages
# --------------------------------------------------------------------------

def graph25_dev(steps: int = 30, cfg: float = 3.5) -> dict:
    g = {**loaders25("ltx-2.5-22b-dev-transformer-comfy-int8-convrot.safetensors"), **source_tail()}
    g.update({
        "2": node("UNETLoader", {"unet_name": "ltx-2.5-22b-distilled-transformer-comfy-int8-convrot.safetensors",
                                 "weight_dtype": "default"}, "Distilled model for the refine stage"),
        "7": node("CLIPTextEncode", {"text": NEGATIVE, "clip": ["3", 0]}, "$NEGATIVE:text"),
        "8": node("LatentUpscaleModelLoader", {"model_name": "ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors"}),
        # Stage 1 at half the source size (in 32-pixel blocks: the latent rounds down).
        "16": math(["12", 0], 2, "divide"),
        "17": math(["12", 1], 2, "divide"),
        "30": node("LTXVConditioning", {"positive": ["4", 0], "negative": ["7", 0], "frame_rate": ["11", 2]}),
        "31": node("EmptyLTXVLatentVideo", {"width": ["16", 0], "height": ["17", 0], "length": ["62", 0],
                                            "batch_size": 1}),
        "32": node("LTXVImgToVideoInplace", {"vae": ["5", 0], "image": ["13", 0], "latent": ["31", 0],
                                             "strength": 1.0, "bypass": False}, "Pin the clip's tail (stage 1)"),
        "33": node("LTXVEmptyLatentAudio", {"frames_number": ["62", 0], "frame_rate": ["11", 2],
                                            "batch_size": 1, "audio_vae": ["6", 0]}),
        "34": node("LTXVConcatAVLatent", {"video_latent": ["32", 0], "audio_latent": ["33", 0]}),
        "40": node("RandomNoise", {"noise_seed": 1234}, "$SEED:noise_seed"),
        "41": node("CFGGuider", {"model": ["1", 0], "positive": ["30", 0], "negative": ["30", 1], "cfg": cfg},
                   "$CFG:cfg"),
        "42": node("KSamplerSelect", {"sampler_name": "euler"}),
        "43": node("LTXVScheduler", {"steps": steps, "max_shift": 2.05, "base_shift": 0.95, "stretch": True,
                                     "terminal": 0.1, "latent": ["34", 0]}, "$STEPS:steps"),
        "44": node("SamplerCustomAdvanced", {"noise": ["40", 0], "guider": ["41", 0], "sampler": ["42", 0],
                                             "sigmas": ["43", 0], "latent_image": ["34", 0]}),
        "45": node("LTXVSeparateAVLatent", {"av_latent": ["44", 0]}),
        # Stage 2: x2 in latent space, pin the tail again, distilled refine.
        "46": node("LTXVLatentUpsampler", {"samples": ["45", 0], "upscale_model": ["8", 0], "vae": ["5", 0]}),
        "47": node("LTXVImgToVideoInplace", {"vae": ["5", 0], "image": ["13", 0], "latent": ["46", 0],
                                             "strength": 1.0, "bypass": False}, "Pin the clip's tail (stage 2)"),
        "48": node("LTXVConcatAVLatent", {"video_latent": ["47", 0], "audio_latent": ["45", 1]}),
        "70": node("RandomNoise", {"noise_seed": 42}),
        "71": node("CFGGuider", {"model": ["2", 0], "positive": ["30", 0], "negative": ["30", 1], "cfg": 1.0}),
        "72": node("KSamplerSelect", {"sampler_name": "euler_ancestral"}),
        "73": node("ManualSigmas", {"sigmas": REFINE_SIGMAS}),
        "74": node("SamplerCustomAdvanced", {"noise": ["70", 0], "guider": ["71", 0], "sampler": ["72", 0],
                                             "sigmas": ["73", 0], "latent_image": ["48", 0]}),
        "75": node("LTXVSeparateAVLatent", {"av_latent": ["74", 0]}),
        "50": node("VAEDecodeTiled", {"samples": ["75", 0], "vae": ["5", 0], "tile_size": 512, "overlap": 64,
                                      "temporal_size": 512, "temporal_overlap": 8}),
    })
    g.update(output(["50", 0]))
    return g


# --------------------------------------------------------------------------
# LTX-2.3, the user's three stages
# --------------------------------------------------------------------------

def graph23() -> dict:
    keep = ("10:426", "10:338", "10:457", "10:429", "10:363", "10:364", "10:361", "10:430", "10:413",
            "10:419", "11:8:333")
    g = {k: copy.deepcopy(G23[k]) for k in keep}
    g["10:429"]["inputs"]["lora_name"] = "ltx2/ltx-2.3-22b-distilled-lora-dynamic_fro09_avg_rank_105_bf16.safetensors"
    g.update(source_tail())
    g["10:363"]["inputs"]["frame_rate"] = ["11", 2]
    model, vae, avae = ["10:429", 0], ["10:457", 0], ["10:338", 0]
    pos, neg = ["10:363", 0], ["10:363", 1]

    def stage(prefix: str, latent: list, audio: list, steps: int, denoise: float) -> None:
        g[f"{prefix}pin"] = node("LTXVImgToVideoInplace", {"vae": vae, "image": ["13", 0], "latent": latent,
                                                           "strength": 1.0, "bypass": False},
                                 f"Pin the clip's tail ({prefix})")
        g[f"{prefix}av"] = node("LTXVConcatAVLatent", {"video_latent": [f"{prefix}pin", 0], "audio_latent": audio})
        g[f"{prefix}sch"] = node("BasicScheduler", {"scheduler": "linear_quadratic", "steps": steps,
                                                    "denoise": denoise, "model": model})
        g[f"{prefix}smp"] = node("KSamplerSelect", {"sampler_name": "euler"})
        g[f"{prefix}gd"] = node("CFGGuider", {"cfg": 1, "model": model, "positive": pos, "negative": neg})
        g[f"{prefix}run"] = node("SamplerCustomAdvanced", {"noise": ["10:364", 0], "guider": [f"{prefix}gd", 0],
                                                           "sampler": [f"{prefix}smp", 0],
                                                           "sigmas": [f"{prefix}sch", 0],
                                                           "latent_image": [f"{prefix}av", 0]})
        g[f"{prefix}sep"] = node("LTXVSeparateAVLatent", {"av_latent": [f"{prefix}run", 0]})

    # Stage 1 at 360 tall, keeping the source's proportion; then x2 and x1.5.
    g["16"] = math(["12", 0], 360, "multiply")
    g["17"] = math(["16", 0], ["12", 1], "divide")
    g["s1lat"] = node("EmptyLTXVLatentVideo", {"width": ["17", 0], "height": 360, "length": ["62", 0],
                                               "batch_size": 1})
    g["s1aud"] = node("LTXVEmptyLatentAudio", {"frames_number": ["62", 0], "frame_rate": ["11", 2],
                                               "batch_size": 1, "audio_vae": avae})
    stage("s1", ["s1lat", 0], ["s1aud", 0], 8, 1)
    g["s2up"] = node("LTXVLatentUpsampler", {"samples": ["s1sep", 0], "upscale_model": ["10:419", 0], "vae": vae})
    stage("s2", ["s2up", 0], ["s1sep", 1], 6, 0.42)
    g["s3up"] = node("LTXVLatentUpsampler", {"samples": ["s2sep", 0], "upscale_model": ["11:8:333", 0], "vae": vae})
    stage("s3", ["s3up", 0], ["s2sep", 1], 4, 0.42)
    g["50"] = node("VAEDecodeTiled", {"samples": ["s3sep", 0], "vae": vae, "tile_size": 768, "overlap": 128,
                                      "temporal_size": 128, "temporal_overlap": 8})
    g.update(output(["50", 0]))
    return g


# --------------------------------------------------------------------------
# Manifests
# --------------------------------------------------------------------------

def common(m: dict, tid: str, name: str, description: str, notes: list[str]) -> dict:
    m.update({
        "id": tid,
        "name": name,
        "model": tid,
        "kind": "video",
        "version": 1,
        "priority": 10,
        "description": description,
        "inputStrategy": "roles",
        "images": 0,
        "intents": {"video_extend": {
            "requires": ["sourceVideo"],
            "note": "Send the end of the shot, not the whole of it: the template keeps its last $CONTEXT frames.",
        }},
        "roles": {"sourceVideo": {
            "slot": "$VIDEO_1", "required": True, "max": 1,
            "note": "The clip to continue. Only its last $CONTEXT frames reach the model.",
        }},
        "frames": {
            "slot": "$FRAMES", "min": 9, "step": 8, "offset": 1,
            "note": "Total length INCLUDING the $CONTEXT frames, which come back first. LTX wants 8n+1.",
        },
        "prompt": {"slot": "$PROMPT", "instructionParameter": "instruction",
                   "lines": [{"when": "always", "text": PROMPT_LINE}]},
        "notes": notes + [
            "The output starts with the $CONTEXT frames re-decoded; the caller drops them and keeps the rest.",
            "Write the prompt as a description of the shot as it already is. Verbs for new events (slams, "
            "bursts, spins faster) made the model invent action the shot never had (test 2026-10-06).",
        ],
    })
    m.pop("sequencer", None)
    return m


SLOTS = {
    "$VIDEO_1": {"required": True, "note": "source video"},
    "$CONTEXT": {"required": False, "note": "frames of the clip's tail kept as context; 8n+1 "
                                             "(49 = 2 s at 24 fps), no more than the clip has"},
    "$FRAMES": {"required": True, "note": "output length, context included; rounded up to 8n+1"},
    "$PROMPT": {"required": False},
    "$SEED": {"required": False, "randomizePerRun": True},
    "$CHECKPOINT": {"required": True, "note": "chosen from requires.checkpoints, never by the caller"},
    "$OUTPUT": {"required": True},
}


def manifest25() -> dict:
    m = copy.deepcopy(BASE25)
    nodes = [n for n in m["requires"]["nodes"]
             if n not in ("LTXICLoRALoaderModelOnly", "LTXAddVideoICLoRAGuide", "LTXVCropGuides",
                          "RepeatImageBatch", "ImageBatch")]
    m["requires"]["nodes"] = nodes + ["LTXVImgToVideoInplace", "PrimitiveInt"]
    m["requires"]["models"] = [x for x in m["requires"]["models"] if x["folder"] != "loras"]
    m["slots"] = SLOTS
    return common(m, "ltx-25-extend", "LTX-2.5 · Extend (continue a clip)",
                  "Continues a clip from its own last frames on LTX-2.5 distilled: the tail is pinned in the "
                  "latent and only what follows is generated, so camera and action carry on through the seam.",
                  ["Weights are gated on Hugging Face (accept the LTX-2.x Community License).",
                   "First test 2026-10-06 (LTX-2.3, last 16 frames as separate keys) joined without a jump but "
                   "moved 4-5x slower than the source. Pinning the tail is the answer to that."])


def manifest25_dev() -> dict:
    m = manifest25()
    m["requires"]["nodes"] += ["LTXVScheduler", "LTXVLatentUpsampler", "LatentUpscaleModelLoader", "ImageScale"]
    m["requires"]["checkpoints"] = {
        "slot": "$CHECKPOINT",
        "variants": {"int8_convrot": {
            "folder": "diffusion_models", "file": "ltx-2.5-22b-dev-transformer-comfy-int8-convrot.safetensors",
            "gb": 21.5, "url": f"{HF25}/diffusion_models/ltx-2.5-22b-dev-transformer-comfy-int8-convrot.safetensors",
            "requiresNativeOps": ["int8_tensorwise", "convrot_w4a4"]}},
        "preferred": "int8_convrot", "alternatives": [],
    }
    m["requires"]["models"] += [
        {"folder": "diffusion_models", "file": "ltx-2.5-22b-distilled-transformer-comfy-int8-convrot.safetensors",
         "gb": 21.5, "note": "the refine stage"},
        {"folder": "latent_upscale_models", "file": "ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors",
         "gb": 1.0, "url": f"{HF25}/latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors"},
    ]
    m["slots"] = {**SLOTS, "$NEGATIVE": {"required": False}, "$CFG": {"required": False},
                  "$STEPS": {"required": False}}
    # Pesos: dev + distilled (21,5 GB cada uno) y Gemma (15 GB); el muestreo
    # reserva ~9 GB (PRO 6000, 2026-10-06). Cabe en las H100 de 80 GB del pool.
    m["minimumVramGb"] = 64
    m["tier"] = "full"
    return common(m, "ltx-25-dev-extend", "LTX-2.5 dev · Extend (continue a clip)",
                  "Continues a clip from its own last frames with LTX-2.5 dev (CFG, more steps) at half size, "
                  "then the x2 latent upscaler and a distilled refine. Slower than ltx-25-extend, better detail.",
                  ["Both 2.5 transformers load (dev for stage 1, distilled for the refine): ~43 GB of weights, "
                   "made for the 80-96 GB cloud GPUs.",
                   "No official dev recipe in the model card; steps/CFG follow LTX-2's two-stage pipeline "
                   "(dev with CFG, then distilled refine with sigmas 0.85/0.725/0.4219)."])


def backwards(g: dict) -> dict:
    """Extend hacia atrás: LTX solo sabe seguir hacia delante, así que el clip
    entra al revés (su primer frame queda al final, como la cola de un plano),
    se continúa, y el resultado se vuelve a dar la vuelta. Sale lo nuevo en su
    orden y detrás los $CONTEXT primeros frames del clip. Vale para cualquier
    grafo de este fichero: todos leen el clip en "11" y decodifican en "50"."""
    g["19"] = node("ReverseImageBatch", {"images": ["11", 0]}, "Clip backwards")
    g["13"]["inputs"]["image"] = ["19", 0]
    g["55"] = node("ReverseImageBatch", {"images": ["50", 0]}, "Result forwards again")
    g["51"]["inputs"]["image"] = ["55", 0]
    return g


def backwards_manifest(m: dict, tid: str, name: str, description: str, pack_note: str) -> dict:
    if "ReverseImageBatch" not in m["requires"]["nodes"]:
        m["requires"]["nodes"] = m["requires"]["nodes"] + ["ReverseImageBatch"]
    packs = m["requires"].setdefault("nodePacks", [])
    kj = next((p for p in packs if p["name"] == "ComfyUI-KJNodes"), None)
    if kj is None:
        packs.append({"name": "ComfyUI-KJNodes", "provides": ["ReverseImageBatch"], "note": pack_note})
    elif "ReverseImageBatch" not in kj["provides"]:
        kj["provides"] = kj["provides"] + ["ReverseImageBatch"]
    m.update({
        "id": tid,
        "model": tid,
        "name": name,
        "description": description,
        "intents": {"video_extend_start": {
            "requires": ["sourceVideo"],
            "note": "Send the START of the shot: the template keeps its first $CONTEXT frames.",
        }},
        "roles": {"sourceVideo": {
            "slot": "$VIDEO_1", "required": True, "max": 1,
            "note": "The clip to lead into. Only its first $CONTEXT frames reach the model.",
        }},
    })
    m["frames"]["note"] = ("Total length INCLUDING the $CONTEXT frames, which come back LAST. "
                           "LTX wants 8n+1.")
    m["notes"] = [n.replace("starts with the $CONTEXT frames", "ends with the $CONTEXT frames")
                  for n in m["notes"]] + [
        "Motion that only runs one way (falling paper, smoke, water) runs backwards in what the model sees: "
        "it may straighten it out, which plays as the wrong way round once reversed."]
    return m


def graph25_dev_start() -> dict:
    return backwards(graph25_dev())


def manifest25_dev_start() -> dict:
    return backwards_manifest(
        manifest25_dev(), "ltx-25-dev-extend-start",
        "LTX-2.5 dev · Extend backwards (what came before a clip)",
        "The frames before a clip: its opening is reversed, continued with LTX-2.5 dev and turned "
        "forwards again, so the new part leads into the clip's first frame.",
        "Already in the cloud image (Dockerfile) for the LTX 2.3 graph.")


def manifest23() -> dict:
    m = copy.deepcopy(BASE23)
    m["requires"]["nodes"] = sorted(
        {n for n in m["requires"]["nodes"] if n not in ("LTXSequencer", "MultiImageLoader", "LTXVCropGuides",
                                                          "ImageScaleBy", "LazySwitchKJ", "VAEDecode",
                                                          "PrimitiveBoolean")}
        | {"LoadVideo", "GetVideoComponents", "ImageFromBatch", "LTXVImgToVideoInplace", "ImageScale",
           "SaveVideo"})
    m["requires"]["nodePacks"] = [p for p in m["requires"]["nodePacks"] if p["name"] != "WhatDreamsCost-ComfyUI"]
    m["slots"] = SLOTS
    for k in ("fps", "inputStrategy", "images", "roles"):
        m.pop(k, None)
    return common(m, "ltx-23-extend", "LTX 2.3 · Extend (continue a clip, three stages)",
                  "Continues a clip from its own last frames with the LTX 2.3 graph behind the keyframe preset: "
                  "360p, then x2 and x1.5 in latent space, the tail pinned again at every stage.",
                  ["Derived from ltx-23-keyframes: same models, schedulers and steps; LTXSequencer and the guide "
                   "crops replaced by LTXVImgToVideoInplace at each stage.",
                   "Renders near 1080p and is scaled back to the source's size on the way out."])


def graph23_start() -> dict:
    return backwards(graph23())


def manifest23_start() -> dict:
    """El "+" de delante en la 3090: el único Extend hacia atrás que cabe en
    24 GB (el de 2.5 dev pide 48+)."""
    return backwards_manifest(
        manifest23(), "ltx-23-extend-start",
        "LTX 2.3 · Extend backwards (what came before a clip, three stages)",
        "The frames before a clip: its opening is reversed, continued with the LTX 2.3 three-stage graph "
        "and turned forwards again, so the new part leads into the clip's first frame.",
        "Same pack the keyframe preset already needs.")


if __name__ == "__main__":
    for tid, g, m in (("ltx-25-extend", graph25(), manifest25()),
                      ("ltx-25-dev-extend", graph25_dev(), manifest25_dev()),
                      ("ltx-25-dev-extend-start", graph25_dev_start(), manifest25_dev_start()),
                      ("ltx-23-extend", graph23(), manifest23()),
                      ("ltx-23-extend-start", graph23_start(), manifest23_start())):
        (TEMPLATES / f"{tid}.json").write_text(json.dumps(g, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        (TEMPLATES / f"{tid}.manifest.json").write_text(json.dumps(m, indent=2, ensure_ascii=False) + "\n",
                                                        encoding="utf-8")
        print("wrote", tid)
