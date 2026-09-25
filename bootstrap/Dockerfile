# Rent-session image: ComfyUI (MiniMax-H3 / Wan / LTX-2.3 / HunyuanVideo-1.5 nodes) + ACE-Step 1.5 XL.
# NO model weights inside -- onstart.sh pulls them from Hugging Face into /workspace/models.
# Base = exactly what ran on the rented RTX PRO 6000 (driver 595, CUDA 13.2) on 25/09/2026.
# TESTED config = the image's torch 2.8.0+cu128 (ComfyUI then warns "need pytorch cu130 for optimized CUDA ops" and
# comfy-kitchen falls back to its eager backend -- everything still ran, see README timings).
# Optional speed-up (NOT tested on 25/09): build with --build-arg TORCH_CU130=1 (needs host driver >= 580).
FROM pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime

ARG COMFY_REV=88ab4a06566454ad89db8f0bedb970d6c08cd1b7
ARG ACE_REV=ca1e85fe9430179831e6bc6be790c332190a3866
ARG TORCH_CU130=0
ENV DEBIAN_FRONTEND=noninteractive HF_XET_HIGH_PERFORMANCE=1 HF_HUB_DISABLE_PROGRESS_BARS=1 PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends git curl ffmpeg openssh-server && rm -rf /var/lib/apt/lists/*

# tools used by the scripts
RUN pip install --no-cache-dir -U "huggingface_hub[hf_xet]" imageio-ffmpeg uv

# ComfyUI pinned
RUN git clone https://github.com/comfyanonymous/ComfyUI.git /workspace/ComfyUI \
 && cd /workspace/ComfyUI && git checkout ${COMFY_REV} \
 && grep -v workflow-templates requirements.txt > /tmp/req.txt \
 && pip install --no-cache-dir -r /tmp/req.txt \
 && if [ "$TORCH_CU130" = 1 ]; then pip install --no-cache-dir -U torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130; fi
# (comfyui-workflow-templates* is skipped on purpose: ~1 GB of UI templates, useless headless, took >5 min from PyPI)

# ACE-Step 1.5 pinned, own uv venv from its lockfile (torch 2.10 cu128 inside .venv)
RUN git clone https://github.com/ace-step/ACE-Step-1.5 /workspace/ACE-Step-1.5 \
 && cd /workspace/ACE-Step-1.5 && git checkout ${ACE_REV} && uv sync

# our scripts + custom output node + control inputs
COPY . /workspace/bootstrap
RUN cp /workspace/bootstrap/comfy_nodes/raw_av_save.py /workspace/ComfyUI/custom_nodes/

WORKDIR /workspace
# Vast: set "on-start script" to:  bash /workspace/bootstrap/onstart.sh   (INSTALL unset -> skips installs)
CMD ["bash", "-lc", "bash /workspace/bootstrap/onstart.sh; sleep infinity"]
