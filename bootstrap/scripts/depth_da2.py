#!/usr/bin/env python3
"""Per-frame Depth-Anything-V2 depth for a previz clip -> grayscale depth mp4 (near=white).
Usage: depth_da2.py <in.mp4> <out.mp4> [model_id]  (default depth-anything/Depth-Anything-V2-Base-hf)
Normalizes with global (clip-wide) percentiles so depth doesn't flicker frame to frame."""
import sys, subprocess, json, numpy as np, torch
from PIL import Image
from transformers import pipeline

src, dst = sys.argv[1], sys.argv[2]
mid = sys.argv[3] if len(sys.argv) > 3 else "depth-anything/Depth-Anything-V2-Base-hf"
info = json.loads(subprocess.check_output(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                                           "stream=width,height,r_frame_rate", "-of", "json", src]))["streams"][0]
w, h, fps = info["width"], info["height"], info["r_frame_rate"]
raw = subprocess.check_output(["ffmpeg", "-v", "error", "-i", src, "-f", "rawvideo", "-pix_fmt", "rgb24", "-"])
frames = np.frombuffer(raw, np.uint8).reshape(-1, h, w, 3)
dev = 0 if torch.cuda.is_available() else -1
pipe = pipeline("depth-estimation", model=mid, device=dev)
preds = []
for i, f in enumerate(frames):
    d = pipe(Image.fromarray(f))["predicted_depth"]
    d = torch.nn.functional.interpolate(d[None, None] if d.dim() == 2 else d[None], size=(h, w), mode="bicubic")[0, 0]
    preds.append(d.float().cpu().numpy())
    if i % 10 == 0: print("frame", i, flush=True)
arr = np.stack(preds)
lo, hi = np.percentile(arr, 1), np.percentile(arr, 99.5)
out = (np.clip((arr - lo) / (hi - lo + 1e-6), 0, 1) * 255).astype(np.uint8)  # DA2 = relative inverse depth: near is bright
p = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "gray", "-s", f"{w}x{h}", "-r", fps, "-i", "-",
                      "-c:v", "libx264", "-crf", "12", "-pix_fmt", "yuv420p", dst], stdin=subprocess.PIPE)
p.stdin.write(out.tobytes()); p.stdin.close(); p.wait()
print("done", dst, arr.shape)
