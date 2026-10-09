"""Build the LTX-2.5 IC-LoRA templates that are variations of one graph.

    python comfyui-reanimator/tools/build_ltx25_iclora_templates.py

Every Lightricks IC-LoRA for LTX-2.5 runs on the same distilled model and the
same text encoder; what changes is the LoRA, the prompt it was trained on and
how the canvas is laid out. Two graphs cover the five built here:

* single stage, at the source's own size (or twice it): the graph of
  ltx-25-alpha-gen.json, read from disk so a fix there reaches these too.
  Clean Plate, Water Simulation, Day to Night and the 2x Pixel Upscaler.
* tiled fusion: Lightricks' LTX-2.5_V2V_TiledFusion_Upscale workflow, for the
  LoRAs trained on small tiles that the model card says to run tiled at every
  output size. Refine Details and Restore.

Alpha Gen is the hand-written base and this script never touches it. Edit
this file, not the JSON it writes.

Every graph pads the clip with 8 copies of its last frame before trimming it
to LTX's 8n+1, and crops the result back to the source's frame count: a 120
frame clip trimmed straight to 113 came back 0.3 s short.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

TEMPLATES = Path(__file__).resolve().parents[1] / "templates"
HF = "https://huggingface.co/Lightricks"
SIGMAS = "1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0"

# The pieces every template shares, copied from ltx-25-alpha-gen.manifest.json.
BASE_MANIFEST = json.loads((TEMPLATES / "ltx-25-alpha-gen.manifest.json").read_text(encoding="utf-8"))


def lora_entry(repo: str, file: str, gb: float) -> dict:
    return {"folder": "loras", "file": file, "gb": gb, "url": f"{HF}/{repo}/resolve/main/{file}"}


def models(lora: dict, audio: bool) -> list[dict]:
    keep = [m for m in BASE_MANIFEST["requires"]["models"] if m["folder"] != "loras"]
    if not audio:
        keep = [m for m in keep if "audio" not in m["file"]]
    return keep + [lora]


# --------------------------------------------------------------------------
# Single-stage graph (from Alpha Gen)
# --------------------------------------------------------------------------

def single_stage(lora_file: str, prompt: str, negative: str, scale: int, prefix: str) -> dict:
    g = copy.deepcopy(json.loads((TEMPLATES / "ltx-25-alpha-gen.json").read_text(encoding="utf-8")))
    g["2"]["_meta"]["title"] = "IC-LoRA"
    g["2"]["inputs"]["lora_name"] = lora_file
    g["4"] = {"class_type": "CLIPTextEncode", "_meta": {"title": "$PROMPT:text"},
              "inputs": {"text": prompt, "clip": ["3", 0]}}
    g["7"] = {"class_type": "CLIPTextEncode", "_meta": {"title": "$NEGATIVE:text"},
              "inputs": {"text": negative, "clip": ["3", 0]}}
    g["30"]["inputs"]["negative"] = ["7", 0]
    g["53"]["inputs"]["filename_prefix"] = f"video/Reanimator_{prefix}"
    if scale != 1:
        # The guide node shrinks the reference by the LoRA's own
        # reference_downscale_factor, so a latent `scale` times the padded
        # source gets the source back as its reference, untouched.
        def times(node_id: str, src: list, title: str) -> list:
            g[node_id] = {"class_type": "easy mathInt", "_meta": {"title": title},
                          "inputs": {"a": src, "b": scale, "operation": "multiply"}}
            return [node_id, 0]
        g["31"]["inputs"]["width"] = times("70", ["15", 0], f"padded width x{scale}")
        g["31"]["inputs"]["height"] = times("71", ["15", 1], f"padded height x{scale}")
        g["51"]["inputs"]["width"] = times("72", ["12", 0], f"source width x{scale}")
        g["51"]["inputs"]["height"] = times("73", ["12", 1], f"source height x{scale}")
        g["51"]["_meta"]["title"] = "Crop the (scaled) padding back off"
    return g


# --------------------------------------------------------------------------
# Tiled-fusion graph (from LTX-2.5_V2V_TiledFusion_Upscale.json)
# --------------------------------------------------------------------------

def tiled(lora_file: str, prompt: str, negative: str, tile_size: str, prefix: str) -> dict:
    base = json.loads((TEMPLATES / "ltx-25-alpha-gen.json").read_text(encoding="utf-8"))
    # The source, padded with 8 copies of its last frame and trimmed to 8n+1:
    # the same frame handling as Alpha Gen, cropped back to the source's frame
    # count after decoding.
    keep = ["1", "3", "5", "10", "11", "12", "26", "80", "81", "82", "27", "28", "29", "35", "13"]
    g = {k: copy.deepcopy(base[k]) for k in keep}
    g["2"] = {"class_type": "LTXICLoRALoaderModelOnly", "_meta": {"title": "IC-LoRA"},
              "inputs": {"model": ["1", 0], "lora_name": lora_file, "strength_model": 1.0}}
    g["4"] = {"class_type": "CLIPTextEncode", "_meta": {"title": "$PROMPT:text"},
              "inputs": {"text": prompt, "clip": ["3", 0]}}
    g["7"] = {"class_type": "CLIPTextEncode", "_meta": {"title": "$NEGATIVE:text"},
              "inputs": {"text": negative, "clip": ["3", 0]}}
    g["60"] = {"class_type": "LTXVGetTilingSizes",
               "_meta": {"title": "$OUTPUT_SIZE:output_size"},
               "inputs": {"width": 1920, "height": 1080, "frame_count": 97,
                          "tile_size": tile_size, "initial_canvas_size": "FullHD",
                          "output_size": "FullHD", "image": ["13", 0]}}
    g["61"] = {"class_type": "ImageScale", "_meta": {"title": "Lanczos the clip to the output canvas"},
               "inputs": {"image": ["13", 0], "upscale_method": "lanczos",
                          "width": ["60", 4], "height": ["60", 5], "crop": "disabled"}}
    g["30"] = {"class_type": "LTXVConditioning", "_meta": {"title": "LTXVConditioning"},
               "inputs": {"positive": ["4", 0], "negative": ["7", 0], "frame_rate": ["11", 2]}}
    g["31"] = {"class_type": "EmptyLTXVLatentVideo", "_meta": {"title": "Latent at the output canvas"},
               "inputs": {"width": ["60", 4], "height": ["60", 5], "length": ["35", 0], "batch_size": 1}}
    g["32"] = {"class_type": "LTXAddVideoICLoRAGuide",
               "_meta": {"title": "Upscaled clip as the IC-LoRA reference, one guide per window"},
               "inputs": {"positive": ["30", 0], "negative": ["30", 1], "vae": ["5", 0],
                          "latent": ["31", 0], "image": ["61", 0], "frame_idx": 0, "strength": 1.0,
                          "latent_downscale_factor": ["2", 1], "crop": "disabled",
                          "use_tiled_encode": False, "tile_size": 256, "tile_overlap": 64,
                          "use_streaming": True, "tile_frames": ["60", 7]}}
    g["42"] = {"class_type": "KSamplerSelect", "_meta": {"title": "KSamplerSelect"},
               "inputs": {"sampler_name": "euler"}}
    g["43"] = {"class_type": "ManualSigmas", "_meta": {"title": "Distilled schedule, 8 steps"},
               "inputs": {"sigmas": SIGMAS}}
    g["44"] = {"class_type": "LTXVTiledFusionSampler", "_meta": {"title": "$SEED:seed"},
               "inputs": {"model": ["2", 0], "positive": ["32", 0], "negative": ["32", 1],
                          "latents": ["32", 2], "sigmas": ["43", 0], "sampler": ["42", 0],
                          "seed": 1234, "cfg": 1.0, "tile_width": ["60", 2], "tile_height": ["60", 3],
                          "overlap_frac": ["60", 6], "blend_var": 0.05, "grid_cycle": 1,
                          "tile_frames": ["60", 7], "vae": ["5", 0], "canvas_device": "auto"}}
    g["50"] = {"class_type": "VAEDecodeTiled", "_meta": {"title": "VAE Decode (Tiled)"},
               "inputs": {"samples": ["44", 0], "vae": ["5", 0], "tile_size": 512, "overlap": 64,
                          "temporal_size": 512, "temporal_overlap": 32}}
    g["54"] = {"class_type": "ImageFromBatch", "_meta": {"title": "Back to the source's frame count"},
               "inputs": {"image": ["50", 0], "batch_index": 0, "length": ["12", 2]}}
    g["52"] = {"class_type": "CreateVideo", "_meta": {"title": "Create Video"},
               "inputs": {"images": ["54", 0], "fps": ["11", 2]}}
    g["53"] = {"class_type": "SaveVideo", "_meta": {"title": "$OUTPUT"},
               "inputs": {"filename_prefix": f"video/Reanimator_{prefix}", "format": "auto",
                          "codec": "auto", "video": ["52", 0]}}
    return g


TILED_NODES = ["UNETLoader", "CLIPLoader", "VAELoader", "CLIPTextEncode", "LoadVideo",
               "GetVideoComponents", "GetImageSize", "ImageFromBatch", "RepeatImageBatch", "ImageBatch",
               "ImageScale", "easy mathInt",
               "LTXVGetTilingSizes", "LTXVConditioning", "EmptyLTXVLatentVideo",
               "LTXICLoRALoaderModelOnly", "LTXAddVideoICLoRAGuide", "KSamplerSelect",
               "ManualSigmas", "LTXVTiledFusionSampler", "VAEDecodeTiled", "CreateVideo", "SaveVideo"]


# --------------------------------------------------------------------------
# Manifests
# --------------------------------------------------------------------------

def manifest(tid: str, name: str, intent: str, intent_note: str, description: str,
             source_note: str, prompt_lines: list[str], prompt_note: str, negative_note: str,
             lora: dict, graph: dict, tiled_graph: bool) -> dict:
    m = copy.deepcopy(BASE_MANIFEST)
    m.update({"id": tid, "name": name, "model": tid, "description": description})
    m["intents"] = {intent: {"requires": ["sourceVideo"], "note": intent_note}}
    m["roles"]["sourceVideo"]["note"] = source_note
    m["prompt"] = {"slot": "$PROMPT", "instructionParameter": "instruction",
                   "lines": [{"text": t} for t in prompt_lines], "note": prompt_note}
    slots = {
        "$VIDEO_1": m["slots"]["$VIDEO_1"],
        "$PROMPT": {"required": True},
        "$NEGATIVE": {"required": False, "note": negative_note},
        "$SEED": m["slots"]["$SEED"],
        "$CHECKPOINT": m["slots"]["$CHECKPOINT"],
        "$OUTPUT": m["slots"]["$OUTPUT"],
    }
    if tiled_graph:
        slots["$OUTPUT_SIZE"] = {"required": False,
                                 "note": "FullHD (default), 4K or 8K: the canvas the clip is lanczos-resized to and refined at."}
    m["slots"] = slots
    m["requires"]["nodes"] = sorted({n["class_type"] for n in graph.values()} |
                                    (set(TILED_NODES) if tiled_graph else set()))
    m["requires"]["models"] = models(lora, audio=not tiled_graph)
    m["requires"]["nodePacks"][0]["provides"] = sorted(
        {n["class_type"] for n in graph.values()} &
        {"LTXICLoRALoaderModelOnly", "LTXAddVideoICLoRAGuide", "LTXVGetTilingSizes", "LTXVTiledFusionSampler"})
    m["notes"] = [
        "Built by tools/build_ltx25_iclora_templates.py; edit that script, not this file.",
        "Weights are gated on Hugging Face: each IC-LoRA repo has its own license gate to accept.",
    ]
    return m


NEG_GENERIC = "blurry, soft, plastic, smeared detail, oversharpened halos, warped faces"

SPECS = []

# Clean Plate: single stage at native resolution. The LoRA was trained on
# captions describing the EMPTY scene, so a generic one always goes first and
# the user's words (e.g. "no bicycle anywhere in the frame") follow it.
CLEAN_PRE = ("An empty clean plate of the exact same location: identical background, environment, structures "
             "and lighting as the source video, with no people, no humans, no figures, and no body parts such as "
             "arms, hands or legs anywhere in the frame. Static photorealistic footage, natural light, high detail.")
SPECS.append(dict(
    tid="ltx-25-clean-plate", name="LTX-2.5 · Clean Plate (remove people)", intent="video_clean_plate",
    intent_note="Video in, the same shot with people and other moving subjects removed and the background rebuilt, frame-aligned. No mask: the model removes dynamic subjects globally.",
    description="Clean background plate with Lightricks' Clean Plate IC-LoRA on LTX-2.5 distilled: removes people, pedestrians and vehicles and rebuilds the static background, one stage at the source's own size. Works when the subject leaves background around it; not when it fills the frame. Made for the cloud GPU.",
    source_note="The clip to clean. Trained at 1024x576 and 49 frames, validated at 1920x1088. The workflow pads and trims to 8n+1 frames and to a multiple of 32, then crops back to the source.",
    lines=[CLEAN_PRE],
    prompt_note="The generic empty-scene description always goes first; the user's words follow it. Name an object that must go in both the prompt and the negative.",
    negative_note="Things that must not appear. Has no effect at cfg 1 (the distilled schedule); kept for a future cfg > 1 run.",
    lora=lora_entry("LTX-2.5-22b-IC-LoRA-Clean-Plate", "ltx-2.5-22b-ic-lora-clean-plate-1.0.safetensors", 0.33),
    graph=lambda l: single_stage(l, CLEAN_PRE,
                                 "person, people, human, figure, limb, arm, hand, leg, body part, backpack, shadow of a person, "
                                 "leftover people, leftover figures, ghosting, semi-transparent remnants, blurry, soft, distorted, "
                                 "inconsistent motion, jitter, worst quality, cartoon, video game, ugly",
                                 1, "cleanplate"),
    tiled=False))

# Water Simulation: single stage at native resolution, cfg 1 (the card's own
# distilled recipe), prompt in the "Reference / Edited" form it was trained on
# with the literal ADD WATER trigger.
WATER_PRE = ("Reference shows the scene. Edited shows the same scene with water added. "
             "Subject identity, clothing, framing, and background geometry are identical to the reference; "
             "only water-related elements differ between reference and edited. ADD WATER")
SPECS.append(dict(
    tid="ltx-25-water", name="LTX-2.5 · Water Simulation (add water)", intent="video_add_water",
    intent_note="Video in, the same shot with water added: rivers, surf, rain, floods, splashes, wet surfaces. The user's words say what water.",
    description="Adds believable moving water to a shot with Lightricks' Water Simulation IC-LoRA on LTX-2.5 distilled, one stage at the source's own size so identity holds. Made for the cloud GPU.",
    source_note="The dry clip, 24 fps. Trained at 1920x1088; up to 185 frames. The workflow trims to 8n+1 frames and pads to a multiple of 32, then crops the padding back off.",
    lines=[WATER_PRE],
    prompt_note="The trigger and the Reference/Edited frame always go first; the user's description of the water follows ADD WATER. Be concrete: the amount of water follows the wording.",
    negative_note="Not used by the card's recipe (cfg 1, no negative). Has no effect at cfg 1.",
    lora=lora_entry("LTX-2.5-22b-IC-LoRA-Water-Simulation", "ltx-2.5-22b-ic-lora-water-simulation-0.9.safetensors", 0.91),
    graph=lambda l: single_stage(l, WATER_PRE + " shallow clear water flooding the ground, rippling around everything it touches and reflecting the light.",
                                 "", 1, "water"),
    tiled=False))

# Day to Night: the card's best recipe is the dev model at cfg 3-4 with STG;
# this runs it on the distilled single-stage graph the card also points to.
NIGHT_PRE = ("A realistic nighttime scene. Only the lighting changes from day to night; identical composition, "
             "framing, camera movement and motion.")
SPECS.append(dict(
    tid="ltx-25-day-to-night", name="LTX-2.5 · Day to Night (relight)", intent="video_day_to_night",
    intent_note="Video in, the same shot at night, frame for frame. The user's words set the night look (moonlight, tungsten, LED...).",
    description="Re-renders a daytime shot as night with Lightricks' Day-to-Night IC-LoRA on LTX-2.5 distilled, one stage at the source's own size. Trained on 768x448 exteriors at 97 frames. Made for the cloud GPU.",
    source_note="The daytime clip, 24 fps. Best around 97 frames; longer clips drift toward the end.",
    lines=[NIGHT_PRE],
    prompt_note="The night frame always goes first; the user's description of the light follows. The reference drives structure, so the words mainly set brightness and colour temperature.",
    negative_note="The card's negative. Has no effect at cfg 1 (the distilled schedule).",
    lora=lora_entry("LTX-2.5-22b-IC-LoRA-Day-To-Night", "ltx-2.5-22b-ic-lora-day-to-night-0.9.safetensors", 0.33),
    graph=lambda l: single_stage(l, NIGHT_PRE + " Photorealistic moonlight, deep natural shadows, warm practical lights.",
                                 "daytime, bright sunlight, blue sky, overexposed, worst quality, inconsistent motion, blurry, jittery, distorted",
                                 1, "night"),
    tiled=False))

# 2x Pixel Spatial Upscaler: a generative upscaler for low-resolution drafts.
UP_PRE = "sharp photographic detail, crisp natural texture, clean edges, high resolution footage"
SPECS.append(dict(
    tid="ltx-25-upscale-x2", name="LTX-2.5 · Pixel Upscaler x2 (generative)", intent="video_upscale_x2",
    intent_note="Video in, the same shot at twice the width and height with synthesized detail. Meant for low-resolution drafts, not for faithful upscaling of live action.",
    description="Generative 2x upscale with Lightricks' Pixel Spatial Upscaler IC-LoRA on LTX-2.5 distilled: the clip is the reference at half the output size (the LoRA's reference_downscale_factor). Made for the cloud GPU.",
    source_note="A clean low-resolution clip (about 280p to 540p). The output is twice its size; the workflow pads to a multiple of 32 and crops the doubled padding back off.",
    lines=[UP_PRE],
    prompt_note="A generic rendering description; the user's words follow it.",
    negative_note="Has no effect at cfg 1 (the distilled schedule).",
    lora=lora_entry("LTX-2.5-22b-IC-LoRA-Pixel-Spatial-Upscaler", "ltx-2.5-22b-ic-lora-pixel-spatial-upscaler-x2-1.0.safetensors", 0.33),
    graph=lambda l: single_stage(l, UP_PRE, NEG_GENERIC, 2, "upscale2x"),
    tiled=False))

# Refine Details: tiled fusion, HD tiles, output FullHD by default.
REFINE_PRE = "sharp photographic detail, crisp natural texture, fine surface detail, clean edges, natural film grain, high resolution footage"
SPECS.append(dict(
    tid="ltx-25-refine-details", name="LTX-2.5 · Refine Details (tiled upscale)", intent="video_refine",
    intent_note="Video in, the same shot lanczos-resized to the output canvas and re-rendered with native-resolution texture. Generative: it rebuilds detail rather than preserving pixels.",
    description="Detail refinement and upscale with Lightricks' Refine Details IC-LoRA on LTX-2.5 distilled through the tiled fusion sampler (HD tiles, 50% overlap), as the model card asks for every output size. Made for the cloud GPU.",
    source_note="Any resolution. The output canvas is FullHD, 4K or 8K in the source's aspect, a multiple of 32.",
    lines=[REFINE_PRE],
    prompt_note="Keep it about the rendering, not the subjects: every tile gets the whole prompt, so a named subject can be painted into tiles that do not hold it.",
    negative_note="Has no effect at cfg 1 (the distilled schedule).",
    lora=lora_entry("LTX-2.5-22b-IC-LoRA-Refine-Details", "ltx-2.5-22b-ic-lora-refine-details-1.0.safetensors", 1.31),
    graph=lambda l: tiled(l, REFINE_PRE, NEG_GENERIC, "HD", "refine"),
    tiled=True))

# Restore: tiled fusion on qHD tiles (the card's 960x544 training tile).
RESTORE_PRE = "natural colour, daylight, sharp photographic detail, crisp faces and clothing texture, natural grain, high resolution footage"
SPECS.append(dict(
    tid="ltx-25-restore", name="LTX-2.5 · Restore (archive footage)", intent="video_restore",
    intent_note="Video in, the same shot as a clean, colour, higher-resolution capture: removes compression damage, tape and sepia casts, flicker. Colour is inferred from content; the user's words steer it.",
    description="Restoration and colourisation of archive footage with Lightricks' Restore IC-LoRA on LTX-2.5 distilled through the tiled fusion sampler (qHD tiles). Made for the cloud GPU.",
    source_note="Archive or low-bitrate footage, progressive. 49 or 97 frames per pass are best. The output canvas is FullHD by default in the source's aspect.",
    lines=[RESTORE_PRE],
    prompt_note="Describe period, place, light and clothing: colour is a semantic decision the prompt steers. Keep it generic across the frame.",
    negative_note="Things that must not be invented. Has no effect at cfg 1 (the distilled schedule).",
    lora=lora_entry("LTX-2.5-22b-IC-LoRA-Restore", "ltx-2.5-22b-ic-lora-restore-1.0.safetensors", 1.71),
    graph=lambda l: tiled(l, RESTORE_PRE,
                          "logos, signage, lettering, modern vehicles, graffiti, modern buildings, " + NEG_GENERIC,
                          "qHD", "restore"),
    tiled=True))


def write_graph(path: Path, g: dict) -> None:
    body = ",\n".join(f'  "{k}": ' + json.dumps(v, ensure_ascii=False) for k, v in g.items())
    path.write_text("{\n" + body + "\n}\n", encoding="utf-8", newline="\n")


def main() -> None:
    for s in SPECS:
        graph = s["graph"](s["lora"]["file"])
        m = manifest(s["tid"], s["name"], s["intent"], s["intent_note"], s["description"],
                     s["source_note"], s["lines"], s["prompt_note"], s["negative_note"],
                     s["lora"], graph, s["tiled"])
        write_graph(TEMPLATES / f"{s['tid']}.json", graph)
        (TEMPLATES / f"{s['tid']}.manifest.json").write_text(
            json.dumps(m, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
        print("wrote", s["tid"])


if __name__ == "__main__":
    main()
