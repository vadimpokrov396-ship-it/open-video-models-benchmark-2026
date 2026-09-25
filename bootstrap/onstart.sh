#!/usr/bin/env bash
# Rent-session bootstrap. On a PRE-BUILT image (see Dockerfile) only the model download + server start run.
# On a plain pytorch image run with INSTALL=1 to do the full install (what we did on 25/09).
# Usage: INSTALL=1 DL_GROUPS="h3 wan wan22 ltx ace" bash onstart.sh
set -u
W=/workspace; B=$W/bootstrap; mkdir -p $W/models $W/out $W/logs
COMFY_REV=${COMFY_REV:-88ab4a06566454ad89db8f0bedb970d6c08cd1b7}
ACE_REV=${ACE_REV:-ca1e85fe9430179831e6bc6be790c332190a3866}
DL_GROUPS=${DL_GROUPS:-"h3 wan wan22 ltx ace"}
ts(){ echo "$(date +%s) $*" | tee -a $W/logs/setup_times.log; }
export HF_XET_HIGH_PERFORMANCE=1 HF_HUB_DISABLE_PROGRESS_BARS=1

ts start
if [ "${INSTALL:-0}" = 1 ]; then
  pip install -q -U "huggingface_hub[hf_xet]" imageio-ffmpeg uv && ts hf_tools
fi
# 1) weights in the background (priority order)
nohup python3 $B/scripts/fetch_models.py $DL_GROUPS > $W/logs/dl.log 2>&1 &
echo $! > $W/logs/dl.pid

if [ "${INSTALL:-0}" = 1 ]; then
  # 2) ACE-Step (own uv venv, pinned lock) -- in background, independent of ComfyUI
  ( cd $W && git clone -q https://github.com/ace-step/ACE-Step-1.5 && cd ACE-Step-1.5 && git checkout -q $ACE_REV \
      && uv sync --quiet > $W/logs/ace_install.log 2>&1 && ts ace_installed || ts ace_install_FAILED ) &
  # 3) ComfyUI (system python of the image, its torch 2.8.0+cu128 -- tested config)
  cd $W && git clone -q https://github.com/comfyanonymous/ComfyUI.git && cd ComfyUI && git checkout -q $COMFY_REV && ts comfy_cloned
  grep -v workflow-templates requirements.txt > $W/req_min.txt   # UI templates: ~1 GB, not needed headless
  pip install -q -r $W/req_min.txt imageio-ffmpeg > $W/logs/comfy_req.log 2>&1 && ts comfy_reqs
fi
cp $B/comfy_nodes/raw_av_save.py $W/ComfyUI/custom_nodes/
cat > $W/ComfyUI/extra_model_paths.yaml <<YAML
rent:
  base_path: $W/models
  diffusion_models: diffusion_models
  text_encoders: text_encoders
  vae: vae
  loras: loras
  checkpoints: checkpoints
  model_patches: model_patches
  latent_upscale_models: latent_upscale_models
YAML
cp $B/inputs/*.mp4 $W/ComfyUI/input/ 2>/dev/null
cd $W/ComfyUI && nohup python3 main.py --listen 127.0.0.1 --port 8188 --output-directory $W/out --disable-auto-launch > $W/logs/comfy.log 2>&1 &
echo $! > $W/logs/comfy.pid
for i in $(seq 1 120); do curl -sf http://127.0.0.1:8188/system_stats >/dev/null && break; sleep 2; done
ts comfy_up
