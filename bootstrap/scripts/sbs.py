#!/usr/bin/env python3
"""Side-by-side comparison: sbs.py out.mp4 "Label=path.mp4" ... [--audio N] (N = index of panel whose audio to keep).
Each panel -> 24 fps, first 5.0 s, 360x640 letterboxed, label burned on top."""
import subprocess, sys
args = [a for a in sys.argv[2:] if not a.startswith("--")]
aud = None
if "--audio" in sys.argv:
    aud = int(sys.argv[sys.argv.index("--audio") + 1]); args = [a for a in args if a != str(aud)]
out, W, H, D = sys.argv[1], 360, 640, 5.0
font = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
cmd, fl = ["ffmpeg", "-v", "error", "-y"], []
for i, a in enumerate(args):
    lab, path = a.split("=", 1)
    cmd += ["-i", path]
    lab = lab.replace(":", r"\:").replace("'", "")
    fl.append(f"[{i}:v]fps=24,trim=0:{D},setpts=PTS-STARTPTS,scale={W}:{H}:force_original_aspect_ratio=decrease,"
              f"pad={W}:{H}:(ow-iw)/2:(oh-ih)/2,drawbox=y=0:w=iw:h=34:color=black@0.55:t=fill,"
              f"drawtext=fontfile={font}:text='{lab}':x=8:y=8:fontsize=17:fontcolor=white[v{i}]")
fl.append("".join(f"[v{i}]" for i in range(len(args))) + f"hstack=inputs={len(args)}[v]")
cmd += ["-filter_complex", ";".join(fl), "-map", "[v]"]
if aud is not None:
    cmd += ["-map", f"{aud}:a", "-c:a", "aac", "-b:a", "192k", "-t", str(D)]
cmd += ["-c:v", "libx264", "-crf", "18", "-preset", "medium", "-pix_fmt", "yuv420p", "-movflags", "+faststart", out]
subprocess.run(cmd, check=True); print("ok", out)
