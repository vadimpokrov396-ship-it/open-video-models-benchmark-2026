#!/usr/bin/env python3
"""ACE-Step 1.5 XL-sft instrumental beds (4B DiT, 50 steps, CFG 7), full bf16 on GPU, FLAC out.
Usage: ace_gen.py <ACE_ROOT> <out_dir> [spec.json]
Checkpoints expected in <ACE_ROOT>/checkpoints/{acestep-v15-xl-sft,vae,Qwen3-Embedding-0.6B} (auto-download otherwise)."""
import os, sys, time, json, glob, shutil
ROOT, OUT = sys.argv[1], sys.argv[2]
sys.path.insert(0, ROOT)
from acestep.inference import GenerationParams, GenerationConfig, generate_music
from acestep.handler import AceStepHandler

SPECS = [
    {"name": "kubyshka", "bpm": 118, "key": "C Major", "seeds": [1101, 1102],
     "caption": "upbeat modern corporate pop, bright acoustic guitar strums, punchy claps, warm electric piano, catchy whistled hook, radio-ready studio mix"},
    {"name": "auto", "bpm": 128, "key": "E Minor", "seeds": [2201, 2202],
     "caption": "driving modern rock instrumental, big distorted guitar riff, live drums, punchy bass, energetic trailer build, polished loud mix"},
    {"name": "posle40", "bpm": 84, "key": "D Major", "seeds": [3301, 3302],
     "caption": "warm emotional cinematic pop, felt piano, soft strings, heartbeat kick, hopeful and intimate, studio quality"},
]
if len(sys.argv) > 3:
    SPECS = json.load(open(sys.argv[3]))
os.makedirs(OUT, exist_ok=True)
t0 = time.time(); dit = AceStepHandler()
msg = dit.initialize_service(project_root=ROOT, config_path="acestep-v15-xl-sft", device="auto", offload_to_cpu=False)
print("init", msg, "%.0fs" % (time.time() - t0), flush=True)
metrics = []
for sp in SPECS:
    d = os.path.join(OUT, "_raw_" + sp["name"]); os.makedirs(d, exist_ok=True)
    t1 = time.time()
    p = GenerationParams(caption=sp["caption"], lyrics="[Instrumental]", instrumental=True, bpm=sp["bpm"], duration=sp.get("dur", 34),
                         keyscale=sp["key"], timesignature="4", inference_steps=sp.get("steps", 50), guidance_scale=sp.get("cfg", 7.0),
                         seed=sp["seeds"][0], thinking=False, use_cot_metas=False, use_cot_caption=False, use_cot_language=False)
    cfg = GenerationConfig(batch_size=len(sp["seeds"]), use_random_seed=False, seeds=sp["seeds"], audio_format="flac")
    r = generate_music(dit, None, params=p, config=cfg, save_dir=d)
    dt = time.time() - t1
    files = sorted(glob.glob(os.path.join(d, "**", "*.flac"), recursive=True), key=os.path.getmtime)
    for i, f in enumerate(files[-len(sp["seeds"]):]):
        dst = os.path.join(OUT, "%s_%d.flac" % (sp["name"], i + 1)); shutil.copy(f, dst)
    print(sp["name"], "%.0fs" % dt, len(files), "files", getattr(r, "error", None) or "", flush=True)
    metrics.append({"name": sp["name"], "seconds": round(dt, 1), "files": len(files)})
json.dump(metrics, open(os.path.join(OUT, "ace_metrics.json"), "w"), indent=1)
print("ALLDONE %.0fs" % (time.time() - t0), flush=True)
