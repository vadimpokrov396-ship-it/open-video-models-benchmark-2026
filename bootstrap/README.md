# Rent-session bootstrap (tested 25/09/2026 on Vast.ai RTX PRO 6000 WS 96 GB, driver 595.71, Japan, $1.625/h incl. 280 GB disk)

Goal: next rental starts generating in ~2-3 min (weights only) instead of ~9 min (install + weights).

## Files
| file | what |
|---|---|
| `Dockerfile` | base `pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime` + ComfyUI @ `88ab4a06566454ad89db8f0bedb970d6c08cd1b7` (requirements minus `comfyui-workflow-templates`) + ACE-Step-1.5 @ `ca1e85fe9430179831e6bc6be790c332190a3866` (`uv sync`, own `.venv`, torch 2.10+cu128) + our scripts. **No weights.** |
| `onstart.sh` | downloads weights (background) -> writes `extra_model_paths.yaml` -> starts ComfyUI on 127.0.0.1:8188. `INSTALL=1` = also do the full install on a plain pytorch image (what we did this time). `DL_GROUPS="h3 wan wan22 ltx ace hy"` picks model groups. |
| `scripts/fetch_models.py` | exact HF repo/file list + sizes per group (below); parallel 4, hf_xet; logs `/workspace/dl_times.jsonl` |
| `scripts/comfy_jobs.py` | headless API-graph builder + runner for `h3_t2v`, `h3_ctrl`, `wan_vace`, `wan22_t2v`, `ltx_t2v`, `hy15_t2v`; writes `<name>.metrics.json` (wall s, peak VRAM) |
| `scripts/ace_gen.py` | ACE-Step 1.5 XL-sft, 3 captions x 2 seeds, 34 s, 50 steps, CFG 7, FLAC |
| `scripts/depth_da2.py` | Depth-Anything-V2 depth video from the previz (ran on our CPU machine CPU, 4 min, before renting) |
| `scripts/sbs.py` | labeled side-by-side comparison video |
| `comfy_nodes/raw_av_save.py` | tiny output node IMAGE(+AUDIO) -> mp4 (avoids SaveVideo DynamicCombo quirks in API mode) |
| `inputs/depth16.mp4`, `inputs/depth24.mp4` | DA2 depth of `auto_drive_color.mp4`: 81 f @16 fps (Wan) and 124 f @24 fps (H3, mci-interpolated) |

## Weights (Hugging Face, all ungated; into /workspace/models/<dir>/)
| group | dir | repo | file | GB |
|---|---|---|---|---|
| h3 | diffusion_models | Comfy-Org/MiniMax-H3 | diffusion_models/minimax_h3_fl2va_pruned_bf16.safetensors | 40.2 |
| h3 | text_encoders | Comfy-Org/MiniMax-H3 | text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors | 15.7 |
| h3 | vae | Comfy-Org/MiniMax-H3 | vae/minimax_h3_video_vae_fp16.safetensors, vae/minimax_h3_audio_vae_fp32.safetensors | 5.2+0.6 |
| h3 | model_patches | Comfy-Org/MiniMax-H3 | model_patches/minimax_h3_fun_controlnet_union_2.0_pruned_bf16.safetensors | 8.4 |
| h3 | loras | Comfy-Org/MiniMax-H3 | loras/minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors | 2.0 |
| wan | diffusion_models | Comfy-Org/Wan_2.1_ComfyUI_repackaged | split_files/diffusion_models/wan2.1_vace_14B_fp16.safetensors | 34.7 |
| wan | text_encoders / vae | Comfy-Org/Wan_2.1_ComfyUI_repackaged | split_files/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors, split_files/vae/wan_2.1_vae.safetensors | 6.7+0.25 |
| wan | loras | Kijai/WanVideo_comfy | Wan21_T2V_14B_lightx2v_cfg_step_distill_lora_rank32.safetensors | 0.3 |
| wan22 | diffusion_models | Comfy-Org/Wan_2.2_ComfyUI_Repackaged | split_files/diffusion_models/wan2.2_t2v_{high,low}_noise_14B_fp8_scaled.safetensors | 2x14.3 |
| wan22 | loras | Comfy-Org/Wan_2.2_ComfyUI_Repackaged | split_files/loras/wan2.2_t2v_lightx2v_4steps_lora_v1.1_{high,low}_noise.safetensors (not used this time) | 2x1.2 |
| ltx | checkpoints | Lightricks/LTX-2.3-fp8 | ltx-2.3-22b-dev-fp8.safetensors | 29.1 |
| ltx | text_encoders | Comfy-Org/ltx-2 | split_files/text_encoders/gemma_3_12B_it_fp4_mixed.safetensors | 9.4 |
| ltx | loras | Comfy-Org/ltx-2.3 | split_files/loras/ltx_2.3_22b_distilled_1.1_lora_dynamic_fro09_avg_rank_111_bf16.safetensors | 2.7 |
| ltx | latent_upscale_models | Lightricks/LTX-2.3 | ltx-2.3-spatial-upscaler-x2-1.1.safetensors | 1.0 |
| ace | ACE-Step-1.5/checkpoints/acestep-v15-xl-sft | ACE-Step/acestep-v15-xl-sft | whole repo | 20.0 |
| ace | ACE-Step-1.5/checkpoints | ACE-Step/Ace-Step1.5 | vae/*, Qwen3-Embedding-0.6B/*, config.json | 1.5 |
| hy | diffusion_models | Comfy-Org/HunyuanVideo_1.5_repackaged | split_files/diffusion_models/hunyuanvideo1.5_480p_t2v_cfg_distilled_fp16.safetensors | 16.7 |
| hy | text_encoders / vae | Comfy-Org/HunyuanVideo_1.5_repackaged | split_files/text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors, byt5_small_glyphxl_fp16.safetensors, split_files/vae/hunyuanvideo15_vae_fp16.safetensors | 9.4+0.4+2.5 |

LTX-2.5 (`Lightricks/LTX-2.5`) is **gated** (our HF account is not authorized) -> LTX-2.3 used. To use 2.5 next time: accept the licence on HF with our account, export `HF_TOKEN`, add a group.

## Measured timings this session (box clock)
- instance create -> SSH ready: < 1 min (image was cached on host).
- weights: h3 72 GB 272 s · wan 42 GB 183 s · wan22 31 GB 138 s · ltx 42 GB 167 s · ace 22 GB 107 s · hy 29 GB 127 s  => **~240 GB in ~16.5 min, ~250 MB/s** (Japan host, unauthenticated HF).
- installs (with the downloads competing for bandwidth): ComfyUI clone 8 s; ComfyUI requirements **~7.5 min** (of which ~5 min wasted on `comfyui-workflow-templates*` wheels from PyPI -> now skipped); ACE `uv sync` **~15.7 min** (background, not on the critical path).
- ComfyUI up and first H3 clip rendered at **~+9 min** after start (H3 weights were first in the queue).
- Pre-built image would remove ~8 min of installs; the first H3 clip could then start at ~+5 min (72 GB of H3 weights at ~250 MB/s).

## Launch on Vast with a template (after the lead builds/pushes the image)
1. `docker build -t <registry>/rent-session:2509 bootstrap/ && docker push <registry>/rent-session:2509`
2. Vast template: image `<registry>/rent-session:2509`, launch mode SSH, disk **>= 280 GB** (all groups ~240 GB), on-start `bash /workspace/bootstrap/onstart.sh` (env `DL_GROUPS="h3 ltx ace"` etc. to fetch only what the job needs).
3. Offer filter: `gpu_ram>=80 reliability>0.97 inet_down>1000 disk_space>=280 cpu_ram>=120`. RTX PRO 6000 (Blackwell, 96 GB) was ~1.6 $/h; A100 80 GB in Sweden was 1.15 $/h but driver 535 / CUDA 12.2 (would need a cu121/cu124 torch build - untested).
4. Run jobs: `python3 /workspace/bootstrap/scripts/comfy_jobs.py h3_ctrl --name X --prompt A --strength 1.0 --width 704 --height 1280 [--lora 0 --steps 20]`; music: `cd /workspace/ACE-Step-1.5 && .venv/bin/python /workspace/bootstrap/scripts/ace_gen.py /workspace/ACE-Step-1.5 /workspace/out/music`. Free ComfyUI VRAM before ACE: `curl -X POST 127.0.0.1:8188/free -d '{"unload_models":true,"free_memory":true}'`.

## Cost lesson
Vast bills **download traffic** on this host: 245 GB -> **$0.64** (almost half of the GPU bill of $1.47 for 56 min). Fetch only the groups the job needs (`DL_GROUPS`), or pick offers with free/low `inet_down_cost`.

## Gotchas found
- `GROUPS` is a bash builtin array -> the env var is `DL_GROUPS`.
- `pgrep -f "<text>"` inside `ssh '...'` matches its own shell -> wait on a PID instead.
- `pip install torch --index-url .../cu130` is a no-op when torch is already installed (needs `-U`); we stayed on the tested cu128 build.
- ComfyUI fills the whole 96 GB with dynamic VRAM loading (peak 96 GB reported) - that is caching, not a requirement; H3 alone peaked ~59-72 GB at 480x832.
