"""Minimal, dependency-light output node: IMAGE (+ optional AUDIO) -> mp4 via ffmpeg (imageio-ffmpeg binary).
Avoids SaveVideo's DynamicCombo API quirks when driving ComfyUI headless."""
import os, subprocess, tempfile, wave
import numpy as np
import folder_paths

try:
    import imageio_ffmpeg
    FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()
except Exception:
    FFMPEG = "ffmpeg"


class SaveRawAV:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"images": ("IMAGE",), "fps": ("FLOAT", {"default": 16.0, "min": 1.0, "max": 120.0}),
                             "filename": ("STRING", {"default": "out.mp4"})},
                "optional": {"audio": ("AUDIO",)}}
    RETURN_TYPES = ()
    FUNCTION = "save"
    OUTPUT_NODE = True
    CATEGORY = "rent_session"

    def save(self, images, fps, filename, audio=None):
        out_dir = folder_paths.get_output_directory()
        path = os.path.join(out_dir, filename)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        frames = (images.clamp(0, 1).cpu().numpy() * 255).round().astype(np.uint8)
        n, h, w, _ = frames.shape
        cmd = [FFMPEG, "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", str(fps), "-i", "-"]
        wav = None
        if audio is not None:
            wf = audio["waveform"][0].float().cpu().numpy()  # [C, T]
            sr = int(audio["sample_rate"])
            wf = np.clip(wf, -1, 1)
            pcm = (wf.T * 32767).astype(np.int16)
            wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
            with wave.open(wav, "wb") as wv:
                wv.setnchannels(pcm.shape[1]); wv.setsampwidth(2); wv.setframerate(sr); wv.writeframes(pcm.tobytes())
            cmd += ["-i", wav, "-c:a", "aac", "-b:a", "256k", "-shortest"]
        cmd += ["-c:v", "libx264", "-crf", "14", "-preset", "slow", "-pix_fmt", "yuv420p", "-movflags", "+faststart", path]
        p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        p.stdin.write(frames.tobytes()); p.stdin.close(); rc = p.wait()
        if wav:
            os.unlink(wav)
        if rc != 0:
            raise RuntimeError(f"ffmpeg failed rc={rc}")
        # also keep a lossless-ish frame dump for review is overkill; mp4 crf14 is enough
        return {"ui": {"text": [path]}}


NODE_CLASS_MAPPINGS = {"SaveRawAV": SaveRawAV}
