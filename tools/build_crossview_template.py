"""Builds templates/ltx-23-crossview-recamera.json.

Written as code rather than exported from the ComfyUI canvas because the graph
was assembled and measured by hand (docs/crossview-warp-ab.md): there is no
canvas workflow it was exported from, and a script says why each wire exists.
Run from the comfyui-reanimator folder: python tools/build_crossview_template.py
"""
import json
from pathlib import Path

DISTILLED = "ltx2/ltx-2.3-22b-distilled-lora-dynamic_fro09_avg_rank_105_bf16.safetensors".replace("/", chr(92))
CKPT = "ltx-2.3-22b-dev-fp8.safetensors"


def node(cls, title, **inputs):
    return {"class_type": cls, "inputs": inputs, "_meta": {"title": title}}


g = {
    # ── the clip, at a size LTX can take ─────────────────────────────
    "1": node("VHS_LoadVideo", "$VIDEO_1", video="example.mp4", force_rate=24,
              custom_width=0, custom_height=0, frame_load_cap=["2", 0],
              skip_first_frames=0, select_every_nth=1),
    "2": node("PrimitiveInt", "$FRAMES:value", value=49),
    "3": node("ImageScaleToTotalPixels", "Stage 1 size", image=["1", 0],
              upscale_method="lanczos", megapixels=0.34, resolution_steps=32),
    "4": node("GetImageSizeAndCount", "Stage 1 size read", image=["3", 0]),
    # ── depth, and the warp into the new camera ──────────────────────
    "10": node("LoadMoGeModel", "MoGe", model_name="moge_2_vitl_normal_fp16.safetensors"),
    "11": node("MoGeInference", "MoGe inference", moge_model=["10", 0], image=["3", 0],
               resolution_level=9, fov_x_degrees=0, batch_size=4,
               force_projection=True, apply_mask=True),
    "12": node("PrimitiveFloat", "$AZIMUTH:value", value=25.0),
    "13": node("PrimitiveFloat", "$ELEVATION:value", value=0.0),
    "14": node("CrossViewWarp", "CrossView Warp", frames=["3", 0], moge_geometry=["11", 0],
               azimuth=["12", 0], elevation=["13", 0], distance=1.0, hfov=0,
               vertical_shift=0, depth_ratio=6, smooth_depth=False, invert_depth=False,
               pivot_override=False, keep_source_aim=True),
    # ── model ────────────────────────────────────────────────────────
    "20": node("CheckpointLoaderSimple", "$CHECKPOINT:ckpt_name", ckpt_name=CKPT),
    "21": node("LoraLoaderModelOnly", "Distilled LoRA", model=["20", 0],
               lora_name=DISTILLED, strength_model=0.5),
    "22": node("LoraLoaderModelOnly", "CrossView-Warp IC-LoRA", model=["21", 0],
               lora_name="LTX2.3-22B_IC-LoRA-CrossView-Warp_v2_6000.safetensors",
               strength_model=1.0),
    "23": node("GetICLoRAParameters", "IC-LoRA parameters", iclora_model=["22", 0]),
    "24": node("LTXAVTextEncoderLoader", "Text encoder",
               text_encoder="gemma_3_12B_it_fp4_mixed.safetensors", ckpt_name=CKPT,
               device="default"),
    "25": node("CLIPTextEncode", "$PROMPT:text", text="Crossview.", clip=["24", 0]),
    "26": node("ConditioningZeroOut", "Negative", conditioning=["25", 0]),
    "27": node("LTXVConditioning", "Conditioning", frame_rate=24,
               positive=["25", 0], negative=["26", 0]),
    "28": node("VAELoaderKJ", "Video VAE", vae_name="LTX23_video_vae_bf16.safetensors",
               device="main_device", weight_dtype="bf16"),
    "29": node("VAELoaderKJ", "Audio VAE", vae_name="LTX23_audio_vae_bf16.safetensors",
               device="main_device", weight_dtype="bf16"),
    "30": node("RandomNoise", "$SEED:noise_seed", noise_seed=11),
    # ── stage 1: the new camera, at low resolution ───────────────────
    "40": node("EmptyLTXVLatentVideo", "Stage 1 latent", width=["4", 1], height=["4", 2],
               length=["2", 0], batch_size=1),
    "41": node("LTXVEmptyLatentAudio", "Audio latent", frames_number=["2", 0],
               frame_rate=24, batch_size=1, audio_vae=["29", 0]),
    # Order matters and is the training order: the warp first, then the source.
    "42": node("LTXVAddGuide", "Guide: warp", positive=["27", 0], negative=["27", 1],
               vae=["28", 0], latent=["40", 0], image=["14", 0], frame_idx=0,
               strength=1.0, iclora_parameters=["23", 0]),
    "43": node("LTXVAddGuide", "Guide: source", positive=["42", 0], negative=["42", 1],
               vae=["28", 0], latent=["42", 2], image=["3", 0], frame_idx=0,
               strength=1.0, iclora_parameters=["23", 0]),
    "44": node("LTXVConcatAVLatent", "Stage 1 AV", video_latent=["43", 2], audio_latent=["41", 0]),
    "45": node("KSamplerSelect", "Stage 1 sampler", sampler_name="euler"),
    "46": node("BasicScheduler", "Stage 1 schedule", scheduler="linear_quadratic",
               steps=8, denoise=1.0, model=["22", 0]),
    "47": node("CFGGuider", "Stage 1 guider", cfg=1, model=["22", 0],
               positive=["43", 0], negative=["43", 1]),
    "48": node("SamplerCustomAdvanced", "Stage 1 sample", noise=["30", 0], guider=["47", 0],
               sampler=["45", 0], sigmas=["46", 0], latent_image=["44", 0]),
    "49": node("LTXVSeparateAVLatent", "Stage 1 split", av_latent=["48", 0]),
    "50": node("LTXVCropGuides", "Stage 1 crop guides", positive=["43", 0],
               negative=["43", 1], latent=["49", 0]),
    # ── stage 2: x2, guided by the source only (the author's recipe) ──
    "60": node("LatentUpscaleModelLoader", "Upscaler",
               model_name="ltx-2.3-spatial-upscaler-x2-1.1.safetensors"),
    "61": node("LTXVLatentUpsampler", "Stage 2 upsample", samples=["50", 2],
               upscale_model=["60", 0], vae=["28", 0]),
    "62": node("ImageScaleBy", "Source at stage 2 size", image=["3", 0],
               upscale_method="lanczos", scale_by=2.0),
    "63": node("LTXVAddGuide", "Stage 2 guide: source", positive=["27", 0],
               negative=["27", 1], vae=["28", 0], latent=["61", 0], image=["62", 0],
               frame_idx=0, strength=1.0, iclora_parameters=["23", 0]),
    "64": node("LTXVConcatAVLatent", "Stage 2 AV", video_latent=["63", 2], audio_latent=["49", 1]),
    "65": node("ManualSigmas", "Stage 2 sigmas", sigmas="0.85, 0.7250, 0.4219, 0.0"),
    "66": node("CFGGuider", "Stage 2 guider", cfg=1, model=["22", 0],
               positive=["63", 0], negative=["63", 1]),
    "67": node("SamplerCustomAdvanced", "Stage 2 sample", noise=["30", 0], guider=["66", 0],
               sampler=["45", 0], sigmas=["65", 0], latent_image=["64", 0]),
    "68": node("LTXVSeparateAVLatent", "Stage 2 split", av_latent=["67", 0]),
    "69": node("LTXVCropGuides", "Stage 2 crop guides", positive=["63", 0],
               negative=["63", 1], latent=["68", 0]),
    "70": node("VAEDecodeTiled", "Decode", samples=["69", 2], vae=["28", 0],
               tile_size=768, overlap=128, temporal_size=128, temporal_overlap=8),
    # No audio. Copying the source's track was the plan, but VHS_LoadVideo
    # raises when asked for the audio of a clip that has none -- and the
    # clips LTX generates here have none. The editor plays a generated video
    # muted anyway; the project's sound comes from the source video.
    "71": node("CreateVideo", "Create video", images=["70", 0], fps=24),
    "72": node("SaveVideo", "$OUTPUT", video=["71", 0],
               filename_prefix="video/Reanimator_recamera", format="auto", codec="auto"),
}

out = Path(__file__).resolve().parents[1] / "templates" / "ltx-23-crossview-recamera.json"
out.write_text(json.dumps(g, indent=1, ensure_ascii=False), encoding="utf-8")
print("wrote", out)
