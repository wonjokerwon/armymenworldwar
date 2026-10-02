"""STR video / XA audio conversion via ffmpeg (psxstr demuxer, mdec and
adpcm_xa decoders). jPSXdec is more thorough if ffmpeg chokes on a file."""

import json
import os
import shutil
import subprocess


def convert_stream(raw_path, out_base):
    """Convert a raw-sector STR/XA file. Returns (outputs, note)."""
    ffmpeg, ffprobe = shutil.which("ffmpeg"), shutil.which("ffprobe")
    if not ffmpeg or not ffprobe:
        return [], "ffmpeg not installed; raw sectors kept for jPSXdec"
    probe = subprocess.run(
        [ffprobe, "-v", "error", "-f", "psxstr", "-show_streams", "-of", "json", raw_path],
        capture_output=True, text=True)
    try:
        streams = json.loads(probe.stdout or "{}").get("streams", [])
    except ValueError:
        streams = []
    if not streams:
        return [], "ffmpeg found no STR/XA streams; try jPSXdec"
    os.makedirs(os.path.dirname(out_base) or ".", exist_ok=True)
    outputs = []
    video = [s for s in streams if s.get("codec_type") == "video"]
    audio = [s for s in streams if s.get("codec_type") == "audio"]
    if video:
        dest = out_base + ".mp4"
        cmd = [ffmpeg, "-v", "error", "-y", "-f", "psxstr", "-i", raw_path,
               "-map", "0:v:0", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "16"]
        if audio:
            cmd += ["-map", "0:a:0", "-c:a", "aac", "-b:a", "192k"]
        if subprocess.run(cmd + [dest], capture_output=True).returncode == 0:
            outputs.append(dest)
    else:
        # XA files interleave several independent audio channels (music
        # tracks, voice lines); each one becomes its own WAV.
        for i in range(len(audio)):
            dest = "%s_ch%02d.wav" % (out_base, i)
            cmd = [ffmpeg, "-v", "error", "-y", "-f", "psxstr", "-i", raw_path,
                   "-map", "0:a:%d" % i, dest]
            if subprocess.run(cmd, capture_output=True).returncode == 0:
                outputs.append(dest)
    note = None if outputs else "ffmpeg failed to decode; try jPSXdec"
    return outputs, note
