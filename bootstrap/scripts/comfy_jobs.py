#!/usr/bin/env python3
"""Headless ComfyUI job runner for the rent session (MiniMax-H3, Wan2.1-VACE-14B, Wan2.2-T2V-A14B, LTX-2.3, HunyuanVideo-1.5).

Builds API-format graphs directly (no UI templates needed), submits them to a running ComfyUI server,
waits, and writes <out>.metrics.json with wall seconds and peak VRAM (nvidia-smi poll).

Usage:
  comfy_jobs.py <job> --name NAME [--prompt A|B|<free text>] [--strength 0.6] [--steps N] [--seed S]
                [--width W --height H --length L] [--lora 0|1] [--cfg C] [--server http://127.0.0.1:8188]
Jobs: h3_t2v, h3_ctrl, wan_vace, wan22_t2v, ltx_t2v, hy15_t2v
Control inputs (put into ComfyUI/input/): depth16.mp4 (81f@16fps 480x832), depth24.mp4 (124f@24fps 480x832).
"""
import argparse, json, os, subprocess, threading, time, urllib.request, urllib.error, uuid

PROMPTS = {
    "A": ("Cinematic photoreal shot of a silver hatchback driving on a wet mountain road at golden hour. "
          "Warm low sun flares across the glossy wet asphalt, sunlight glints on the car's metallic silver paint, "
          "fine water spray from the tires, pine trees and misty mountain ridges in the background. "
          "The camera tracks the car in a smooth three-quarter orbit. Shot on 35mm film, film grain, "
          "shallow depth of field, natural color grade, realistic reflections, ultra detailed."),
    "B": ("The same silver hatchback driving on a snowy city street at night. Pink and cyan neon signs reflect "
          "on wet snow and on the car's glossy silver paint, snowflakes falling, headlights and red taillights glowing, "
          "steam rising from street vents, bokeh of city lights. The camera tracks the car in a smooth three-quarter orbit. "
          "Cinematic, photoreal, 35mm film grain, moody high-contrast night lighting, ultra detailed."),
}
AUDIO = {
    "A": " Audio: a smooth engine hum rising as the car accelerates, tires hissing on wet asphalt, light wind, no music, no speech.",
    "B": " Audio: a soft engine purr, tires crunching on wet snow, distant city traffic ambience, no music, no speech.",
}
WAN_NEG = ("色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，"
           "残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，"
           "三条腿，背景人很多，倒着走, cartoon, clay render, low poly, untextured 3d blockout, plastic, CGI, video game")
LTX_NEG = "pc game, console game, video game, cartoon, childish, ugly, clay render, low poly, blurry, distorted car"


class G:
    """Tiny API-graph builder: node() returns an id; out(id, slot) is a link."""
    def __init__(self):
        self.g, self.n = {}, 0

    def node(self, cls, **inputs):
        self.n += 1
        nid = str(self.n)
        self.g[nid] = {"class_type": cls, "inputs": inputs}
        return nid

    @staticmethod
    def o(nid, slot=0):
        return [nid, slot]


def text(args, audio=False):
    p = PROMPTS.get(args.prompt, args.prompt)
    if audio and args.prompt in AUDIO:
        p += AUDIO[args.prompt]
    return p


# ---------------------------------------------------------------- MiniMax H3 (fl2va pruned + optional Fun ControlNet Union 2.0)
def build_h3(args, control):
    g = G(); o = G.o
    unet = g.node("UNETLoader", unet_name=args.h3_unet, weight_dtype="default")
    model = unet
    if args.lora:
        model = g.node("LoraLoaderModelOnly", model=o(unet), lora_name="minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors",
                       strength_model=1.0)
    clip = g.node("CLIPLoader", clip_name=args.h3_te, type="minimax", device="default")
    vae = g.node("VAELoader", vae_name="minimax_h3_video_vae_fp16.safetensors")
    avae = g.node("VAELoader", vae_name="minimax_h3_audio_vae_fp32.safetensors")
    if control:
        patch = g.node("ModelPatchLoader", name="minimax_h3_fun_controlnet_union_2.0_pruned_bf16.safetensors")
        lv = g.node("LoadVideo", file=args.control or "depth24.mp4")
        comp = g.node("GetVideoComponents", video=o(lv))
        model = g.node("MiniMaxH3FunControlNetApply", model=o(model), model_patch=o(patch), vae=o(vae),
                       strength=args.strength, start_percent=0.0, end_percent=args.end_percent, control_video=o(comp, 0))
    i2v = g.node("MiniMaxH3ImageToVideo", clip=o(clip), vae=o(vae), prompt=text(args, audio=True),
                 width=args.width or 480, height=args.height or 832, length=args.length or 124)
    steps = args.steps or (8 if args.lora else 20)
    sched = g.node("BasicScheduler", model=o(model), scheduler="simple", steps=steps, denoise=1.0)
    guider = g.node("BasicGuider", model=o(model), conditioning=o(i2v, 0))
    sampler = g.node("KSamplerSelect", sampler_name="res_multistep")
    noise = g.node("RandomNoise", noise_seed=args.seed)
    s = g.node("SamplerCustomAdvanced", noise=o(noise), guider=o(guider), sampler=o(sampler), sigmas=o(sched), latent_image=o(i2v, 1))
    img = g.node("VAEDecode", samples=o(s, 0), vae=o(vae))
    aud = g.node("VAEDecodeAudio", samples=o(s, 0), vae=o(avae))
    g.node("SaveRawAV", images=o(img), fps=24.0, filename=args.name + ".mp4", audio=o(aud))
    return g.g


# ---------------------------------------------------------------- Wan2.1 VACE 14B with depth control
def build_wan_vace(args):
    g = G(); o = G.o
    unet = g.node("UNETLoader", unet_name="wan2.1_vace_14B_fp16.safetensors", weight_dtype="default")
    model = unet
    if args.lora:
        model = g.node("LoraLoaderModelOnly", model=o(unet), lora_name="Wan21_T2V_14B_lightx2v_cfg_step_distill_lora_rank32.safetensors",
                       strength_model=1.0)
    model = g.node("ModelSamplingSD3", model=o(model), shift=8.0 if args.lora else 5.0)
    clip = g.node("CLIPLoader", clip_name="umt5_xxl_fp8_e4m3fn_scaled.safetensors", type="wan", device="default")
    vae = g.node("VAELoader", vae_name="wan_2.1_vae.safetensors")
    pos = g.node("CLIPTextEncode", clip=o(clip), text=text(args))
    neg = g.node("CLIPTextEncode", clip=o(clip), text=WAN_NEG)
    lv = g.node("LoadVideo", file=args.control or "depth16.mp4")
    comp = g.node("GetVideoComponents", video=o(lv))
    vace = g.node("WanVaceToVideo", positive=o(pos), negative=o(neg), vae=o(vae), width=args.width or 480, height=args.height or 832,
                  length=args.length or 81, batch_size=1, strength=args.strength, control_video=o(comp, 0))
    steps = args.steps or (8 if args.lora else 25)
    cfg = args.cfg or (1.0 if args.lora else 5.0)
    ks = g.node("KSampler", model=o(model), seed=args.seed, steps=steps, cfg=cfg, sampler_name="euler" if args.lora else "uni_pc",
                scheduler="simple", positive=o(vace, 0), negative=o(vace, 1), latent_image=o(vace, 2), denoise=1.0)
    trim = g.node("TrimVideoLatent", samples=o(ks), trim_amount=o(vace, 3))
    img = g.node("VAEDecode", samples=o(trim), vae=o(vae))
    g.node("SaveRawAV", images=o(img), fps=16.0, filename=args.name + ".mp4")
    return g.g


# ---------------------------------------------------------------- Wan2.2 T2V A14B (MoE high/low), no control
def build_wan22(args):
    g = G(); o = G.o
    hi = g.node("UNETLoader", unet_name="wan2.2_t2v_high_noise_14B_fp8_scaled.safetensors", weight_dtype="default")
    lo = g.node("UNETLoader", unet_name="wan2.2_t2v_low_noise_14B_fp8_scaled.safetensors", weight_dtype="default")
    hi_m, lo_m = hi, lo
    if args.lora:
        hi_m = g.node("LoraLoaderModelOnly", model=o(hi), lora_name="wan2.2_t2v_lightx2v_4steps_lora_v1.1_high_noise.safetensors", strength_model=1.0)
        lo_m = g.node("LoraLoaderModelOnly", model=o(lo), lora_name="wan2.2_t2v_lightx2v_4steps_lora_v1.1_low_noise.safetensors", strength_model=1.0)
    hi_m = g.node("ModelSamplingSD3", model=o(hi_m), shift=8.0)
    lo_m = g.node("ModelSamplingSD3", model=o(lo_m), shift=8.0)
    clip = g.node("CLIPLoader", clip_name="umt5_xxl_fp8_e4m3fn_scaled.safetensors", type="wan", device="default")
    vae = g.node("VAELoader", vae_name="wan_2.1_vae.safetensors")
    pos = g.node("CLIPTextEncode", clip=o(clip), text=text(args))
    neg = g.node("CLIPTextEncode", clip=o(clip), text=WAN_NEG)
    lat = g.node("EmptyHunyuanLatentVideo", width=args.width or 480, height=args.height or 832, length=args.length or 81, batch_size=1)
    steps = args.steps or (4 if args.lora else 20)
    cfg = args.cfg or (1.0 if args.lora else 3.5)
    half = steps // 2
    k1 = g.node("KSamplerAdvanced", model=o(hi_m), add_noise="enable", noise_seed=args.seed, steps=steps, cfg=cfg, sampler_name="euler",
                scheduler="simple", positive=o(pos), negative=o(neg), latent_image=o(lat), start_at_step=0, end_at_step=half,
                return_with_leftover_noise="enable")
    k2 = g.node("KSamplerAdvanced", model=o(lo_m), add_noise="disable", noise_seed=0, steps=steps, cfg=cfg, sampler_name="euler",
                scheduler="simple", positive=o(pos), negative=o(neg), latent_image=o(k1), start_at_step=half, end_at_step=10000,
                return_with_leftover_noise="disable")
    img = g.node("VAEDecode", samples=o(k2), vae=o(vae))
    g.node("SaveRawAV", images=o(img), fps=16.0, filename=args.name + ".mp4")
    return g.g


# ---------------------------------------------------------------- LTX-2.3 T2V with native audio (two-stage, per official template)
def build_ltx(args):
    g = G(); o = G.o
    W, H = args.width or 704, args.height or 1280
    L, fps = args.length or 145, 24
    ck = "ltx-2.3-22b-dev-fp8.safetensors"
    ckpt = g.node("CheckpointLoaderSimple", ckpt_name=ck)
    model = g.node("LoraLoaderModelOnly", model=o(ckpt, 0), lora_name="ltx_2.3_22b_distilled_1.1_lora_dynamic_fro09_avg_rank_111_bf16.safetensors",
                   strength_model=0.5)
    te = g.node("LTXAVTextEncoderLoader", text_encoder=args.ltx_te, ckpt_name=ck, device="default")
    avae = g.node("LTXVAudioVAELoader", ckpt_name=ck)
    pos = g.node("CLIPTextEncode", clip=o(te), text=text(args, audio=True))
    neg = g.node("CLIPTextEncode", clip=o(te), text=LTX_NEG)
    cond = g.node("LTXVConditioning", positive=o(pos), negative=o(neg), frame_rate=float(fps))
    v1 = g.node("EmptyLTXVLatentVideo", width=W // 2, height=H // 2, length=L, batch_size=1)
    a1 = g.node("LTXVEmptyLatentAudio", audio_vae=o(avae), frames_number=L, frame_rate=fps, batch_size=1)
    av1 = g.node("LTXVConcatAVLatent", video_latent=o(v1), audio_latent=o(a1))
    ksel = g.node("KSamplerSelect", sampler_name="euler")
    sig1 = g.node("ManualSigmas", sigmas="1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0")
    gd1 = g.node("CFGGuider", model=o(model), positive=o(cond, 0), negative=o(cond, 1), cfg=1.0)
    n1 = g.node("RandomNoise", noise_seed=args.seed)
    s1 = g.node("SamplerCustomAdvanced", noise=o(n1), guider=o(gd1), sampler=o(ksel), sigmas=o(sig1), latent_image=o(av1))
    sep1 = g.node("LTXVSeparateAVLatent", av_latent=o(s1, 0))
    ups = g.node("LatentUpscaleModelLoader", model_name="ltx-2.3-spatial-upscaler-x2-1.1.safetensors")
    up = g.node("LTXVLatentUpsampler", samples=o(sep1, 0), upscale_model=o(ups), vae=o(ckpt, 2))
    av2 = g.node("LTXVConcatAVLatent", video_latent=o(up), audio_latent=o(sep1, 1))
    crop = g.node("LTXVCropGuides", positive=o(cond, 0), negative=o(cond, 1), latent=o(sep1, 0))
    gd2 = g.node("CFGGuider", model=o(model), positive=o(crop, 0), negative=o(crop, 1), cfg=1.0)
    sig2 = g.node("ManualSigmas", sigmas="0.85, 0.7250, 0.4219, 0.0")
    n2 = g.node("RandomNoise", noise_seed=42)
    s2 = g.node("SamplerCustomAdvanced", noise=o(n2), guider=o(gd2), sampler=o(ksel), sigmas=o(sig2), latent_image=o(av2))
    sep2 = g.node("LTXVSeparateAVLatent", av_latent=o(s2, 0))
    img = g.node("VAEDecodeTiled", samples=o(sep2, 0), vae=o(ckpt, 2), tile_size=768, overlap=64, temporal_size=4096, temporal_overlap=4)
    aud = g.node("LTXVAudioVAEDecode", samples=o(sep2, 1), audio_vae=o(avae))
    g.node("SaveRawAV", images=o(img), fps=float(fps), filename=args.name + ".mp4", audio=o(aud))
    return g.g


# ---------------------------------------------------------------- HunyuanVideo 1.5 T2V (480p cfg-distilled)
def build_hy15(args):
    g = G(); o = G.o
    unet = g.node("UNETLoader", unet_name="hunyuanvideo1.5_480p_t2v_cfg_distilled_fp16.safetensors", weight_dtype="default")
    model = g.node("ModelSamplingSD3", model=o(unet), shift=5.0)
    clip = g.node("DualCLIPLoader", clip_name1="qwen_2.5_vl_7b_fp8_scaled.safetensors", clip_name2="byt5_small_glyphxl_fp16.safetensors",
                  type="hunyuan_video_15", device="default")
    vae = g.node("VAELoader", vae_name="hunyuanvideo15_vae_fp16.safetensors")
    pos = g.node("CLIPTextEncode", clip=o(clip), text=text(args))
    neg = g.node("CLIPTextEncode", clip=o(clip), text="")
    lat = g.node("EmptyHunyuanVideo15Latent", width=args.width or 480, height=args.height or 848, length=args.length or 121, batch_size=1)
    sched = g.node("BasicScheduler", model=o(model), scheduler="simple", steps=args.steps or 30, denoise=1.0)
    guider = g.node("CFGGuider", model=o(model), positive=o(pos), negative=o(neg), cfg=args.cfg or 1.0)
    sampler = g.node("KSamplerSelect", sampler_name="euler")
    noise = g.node("RandomNoise", noise_seed=args.seed)
    s = g.node("SamplerCustomAdvanced", noise=o(noise), guider=o(guider), sampler=o(sampler), sigmas=o(sched), latent_image=o(lat))
    img = g.node("VAEDecodeTiled", samples=o(s, 0), vae=o(vae), tile_size=512, overlap=64, temporal_size=64, temporal_overlap=8)
    g.node("SaveRawAV", images=o(img), fps=24.0, filename=args.name + ".mp4")
    return g.g


class VramPoll(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True); self.peak = 0; self.stop = False

    def run(self):
        while not self.stop:
            try:
                v = int(subprocess.check_output(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"]).split()[0])
                self.peak = max(self.peak, v)
            except Exception:
                pass
            time.sleep(1)


def post(server, path, obj=None):
    req = urllib.request.Request(server + path, data=json.dumps(obj).encode() if obj is not None else None,
                                 headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=60).read())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("job"); ap.add_argument("--name", required=True)
    ap.add_argument("--prompt", default="A"); ap.add_argument("--strength", type=float, default=0.6)
    ap.add_argument("--end-percent", dest="end_percent", type=float, default=1.0)
    ap.add_argument("--steps", type=int, default=0); ap.add_argument("--seed", type=int, default=20260925)
    ap.add_argument("--width", type=int, default=0); ap.add_argument("--height", type=int, default=0)
    ap.add_argument("--length", type=int, default=0); ap.add_argument("--lora", type=int, default=1)
    ap.add_argument("--cfg", type=float, default=0.0); ap.add_argument("--control", default="")
    ap.add_argument("--h3-unet", dest="h3_unet", default="minimax_h3_fl2va_pruned_bf16.safetensors")
    ap.add_argument("--h3-te", dest="h3_te", default="qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors")
    ap.add_argument("--ltx-te", dest="ltx_te", default="gemma_3_12B_it_fp4_mixed.safetensors")
    ap.add_argument("--server", default="http://127.0.0.1:8188"); ap.add_argument("--dump", action="store_true")
    args = ap.parse_args()
    graph = {"h3_t2v": lambda: build_h3(args, False), "h3_ctrl": lambda: build_h3(args, True), "wan_vace": lambda: build_wan_vace(args),
             "wan22_t2v": lambda: build_wan22(args), "ltx_t2v": lambda: build_ltx(args), "hy15_t2v": lambda: build_hy15(args)}[args.job]()
    if args.dump:
        print(json.dumps(graph, indent=1, ensure_ascii=False)); return
    vp = VramPoll(); vp.start(); t0 = time.time()
    try:
        r = post(args.server, "/prompt", {"prompt": graph, "client_id": str(uuid.uuid4())})
    except urllib.error.HTTPError as e:
        print("VALIDATION FAILED", e.code, e.read().decode()[:4000], flush=True); raise SystemExit(2)
    pid = r["prompt_id"]; print("queued", args.name, pid, flush=True)
    while True:
        time.sleep(3)
        h = post(args.server, "/history/" + pid)
        if pid in h:
            st = h[pid].get("status", {}); break
    dt = time.time() - t0; vp.stop = True
    ok = st.get("status_str") == "success"
    msgs = [m for m in st.get("messages", []) if m[0] in ("execution_error", "execution_interrupted")]
    meta = {"name": args.name, "job": args.job, "ok": ok, "seconds": round(dt, 1), "peak_vram_mib": vp.peak, "args": vars(args),
            "errors": [m[1].get("exception_message", "")[:2000] for m in msgs]}
    os.makedirs("/workspace/out", exist_ok=True)
    json.dump(meta, open(f"/workspace/out/{args.name}.metrics.json", "w"), indent=1, ensure_ascii=False)
    print(json.dumps({k: meta[k] for k in ("name", "ok", "seconds", "peak_vram_mib", "errors")}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
