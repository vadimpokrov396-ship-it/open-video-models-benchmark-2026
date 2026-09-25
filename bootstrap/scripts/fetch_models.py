#!/usr/bin/env python3
"""Download model weights from Hugging Face into /workspace/models/<comfy_type>/<basename> (+ ACE-Step checkpoints).
Usage: fetch_models.py <group> [<group> ...]   groups: h3 wan wan22 ltx ace hy   (order = priority)
Parallel (4 workers) via huggingface_hub + hf_xet. Logs per-file seconds to /workspace/dl_times.jsonl."""
import os, sys, time, json
from concurrent.futures import ThreadPoolExecutor
from huggingface_hub import hf_hub_download, snapshot_download

M = os.environ.get("MODELS", "/workspace/models")
ACE = os.environ.get("ACE_ROOT", "/workspace/ACE-Step-1.5")
CACHE = os.environ.get("HF_DL", "/workspace/hf")

GROUPS = {  # (comfy dir, repo, path-in-repo, size GB)
    "h3": [("diffusion_models", "Comfy-Org/MiniMax-H3", "diffusion_models/minimax_h3_fl2va_pruned_bf16.safetensors", 40.2),
           ("text_encoders", "Comfy-Org/MiniMax-H3", "text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors", 15.7),
           ("vae", "Comfy-Org/MiniMax-H3", "vae/minimax_h3_video_vae_fp16.safetensors", 5.2),
           ("vae", "Comfy-Org/MiniMax-H3", "vae/minimax_h3_audio_vae_fp32.safetensors", 0.6),
           ("model_patches", "Comfy-Org/MiniMax-H3", "model_patches/minimax_h3_fun_controlnet_union_2.0_pruned_bf16.safetensors", 8.4),
           ("loras", "Comfy-Org/MiniMax-H3", "loras/minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors", 2.0)],
    "h3int8te": [("text_encoders", "Comfy-Org/MiniMax-H3", "text_encoders/qwen3vl_32b_minimax_h3_int8_convrot.safetensors", 27.1)],
    "wan": [("diffusion_models", "Comfy-Org/Wan_2.1_ComfyUI_repackaged", "split_files/diffusion_models/wan2.1_vace_14B_fp16.safetensors", 34.7),
            ("text_encoders", "Comfy-Org/Wan_2.1_ComfyUI_repackaged", "split_files/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors", 6.7),
            ("vae", "Comfy-Org/Wan_2.1_ComfyUI_repackaged", "split_files/vae/wan_2.1_vae.safetensors", 0.25),
            ("loras", "Kijai/WanVideo_comfy", "Wan21_T2V_14B_lightx2v_cfg_step_distill_lora_rank32.safetensors", 0.3)],
    "wan22": [("diffusion_models", "Comfy-Org/Wan_2.2_ComfyUI_Repackaged", "split_files/diffusion_models/wan2.2_t2v_high_noise_14B_fp8_scaled.safetensors", 14.3),
              ("diffusion_models", "Comfy-Org/Wan_2.2_ComfyUI_Repackaged", "split_files/diffusion_models/wan2.2_t2v_low_noise_14B_fp8_scaled.safetensors", 14.3),
              ("loras", "Comfy-Org/Wan_2.2_ComfyUI_Repackaged", "split_files/loras/wan2.2_t2v_lightx2v_4steps_lora_v1.1_high_noise.safetensors", 1.2),
              ("loras", "Comfy-Org/Wan_2.2_ComfyUI_Repackaged", "split_files/loras/wan2.2_t2v_lightx2v_4steps_lora_v1.1_low_noise.safetensors", 1.2)],
    "ltx": [("checkpoints", "Lightricks/LTX-2.3-fp8", "ltx-2.3-22b-dev-fp8.safetensors", 29.1),
            ("text_encoders", "Comfy-Org/ltx-2", "split_files/text_encoders/gemma_3_12B_it_fp4_mixed.safetensors", 9.4),
            ("loras", "Comfy-Org/ltx-2.3", "split_files/loras/ltx_2.3_22b_distilled_1.1_lora_dynamic_fro09_avg_rank_111_bf16.safetensors", 2.7),
            ("latent_upscale_models", "Lightricks/LTX-2.3", "ltx-2.3-spatial-upscaler-x2-1.1.safetensors", 1.0)],
    "hy": [("diffusion_models", "Comfy-Org/HunyuanVideo_1.5_repackaged", "split_files/diffusion_models/hunyuanvideo1.5_480p_t2v_cfg_distilled_fp16.safetensors", 16.7),
           ("text_encoders", "Comfy-Org/HunyuanVideo_1.5_repackaged", "split_files/text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors", 9.4),
           ("text_encoders", "Comfy-Org/HunyuanVideo_1.5_repackaged", "split_files/text_encoders/byt5_small_glyphxl_fp16.safetensors", 0.4),
           ("vae", "Comfy-Org/HunyuanVideo_1.5_repackaged", "split_files/vae/hunyuanvideo15_vae_fp16.safetensors", 2.5)],
}


def log(rec):
    with open("/workspace/dl_times.jsonl", "a") as f:
        f.write(json.dumps(rec) + "\n")


def one(item):
    d, repo, path, gb = item
    dst = os.path.join(M, d, os.path.basename(path))
    if os.path.exists(dst):
        return
    t = time.time()
    for attempt in range(3):
        try:
            p = hf_hub_download(repo, path, local_dir=os.path.join(CACHE, repo))
            break
        except Exception as e:
            print("retry", path, e, flush=True); time.sleep(5)
    else:
        log({"file": path, "error": "failed"}); return
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    os.symlink(p, dst)
    dt = time.time() - t
    log({"file": os.path.basename(path), "gb": gb, "sec": round(dt, 1), "MBps": round(gb * 1024 / max(dt, 0.1))})
    print("done %-70s %5.1f GB %6.0fs" % (os.path.basename(path), gb, dt), flush=True)


def ace():
    t = time.time()
    ck = os.path.join(ACE, "checkpoints")
    snapshot_download("ACE-Step/acestep-v15-xl-sft", local_dir=os.path.join(ck, "acestep-v15-xl-sft"))
    snapshot_download("ACE-Step/Ace-Step1.5", local_dir=ck, allow_patterns=["vae/*", "Qwen3-Embedding-0.6B/*", "config.json"])
    log({"file": "ace_xl_sft+base", "gb": 21.5, "sec": round(time.time() - t, 1)})
    print("done ACE %.0fs" % (time.time() - t), flush=True)


if __name__ == "__main__":
    t0 = time.time()
    for grp in sys.argv[1:]:
        tg = time.time()
        if grp == "ace":
            ace()
        else:
            with ThreadPoolExecutor(4) as ex:
                list(ex.map(one, GROUPS[grp]))
        print("GROUP_DONE", grp, "%.0fs" % (time.time() - tg), flush=True)
        log({"group": grp, "sec": round(time.time() - tg, 1)})
    print("ALL_DL_DONE %.0fs" % (time.time() - t0), flush=True)
