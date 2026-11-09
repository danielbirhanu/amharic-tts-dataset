import argparse, json, subprocess, sys
from pathlib import Path
import numpy as np
import soundfile as sf
import torch
from silero_vad import load_silero_vad, get_speech_timestamps

ap = argparse.ArgumentParser()
ap.add_argument("--input", required=True)                      # any audio file, e.g. data/raw/book02.mp3
ap.add_argument("--source", required=True)                     # short unique name, e.g. book02 (no spaces)
ap.add_argument("--start", type=float, default=0)              # optional crop: start second in the input
ap.add_argument("--duration", type=float, default=0)           # optional crop: length in seconds (0 = to the end)
ap.add_argument("--force", action="store_true")                # redo the ffmpeg conversion
ap.add_argument("--threshold", type=float, default=0.5)
ap.add_argument("--min_silence_ms", type=int, default=300)
ap.add_argument("--pad_ms", type=int, default=150)
ap.add_argument("--max_gap", type=float, default=0.6)
ap.add_argument("--target_max", type=float, default=12.0)
ap.add_argument("--min_len", type=float, default=1.5)
args = ap.parse_args()

if " " in args.source:
    sys.exit("--source must not contain spaces")
master = Path(f"data/interim/master/{args.source}.wav")
asr_wav = Path(f"data/interim/asr_audio/{args.source}_asr.wav")
for p in (master, asr_wav):
    p.parent.mkdir(parents=True, exist_ok=True)


def run(cmd):
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        sys.exit("ffmpeg failed:\n" + r.stderr[-800:])


if args.force or not master.exists():
    cmd = ["ffmpeg", "-y", "-loglevel", "error"]
    if args.start:
        cmd += ["-ss", str(args.start)]
    if args.duration:
        cmd += ["-t", str(args.duration)]
    run(cmd + ["-i", args.input, "-vn", "-ac", "1", "-c:a", "pcm_s16le", str(master)])
    print("wrote master:", master)
if args.force or not asr_wav.exists():
    run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(master), "-ac", "1", "-ar", "16000",
         "-c:a", "pcm_s16le", str(asr_wav)])
    print("wrote 16 kHz copy:", asr_wav)

m_info = sf.info(str(master))
m_sr = m_info.samplerate
asr_audio, asr_sr = sf.read(str(asr_wav), dtype="float32")
if asr_audio.ndim > 1:
    asr_audio = asr_audio.mean(axis=1)
assert asr_sr == 16000
print(f"duration: master {m_info.frames / m_sr / 60:.1f} min | asr copy {len(asr_audio) / asr_sr / 60:.1f} min (should match)")

print("finding speech (this can take a few minutes for long audio)...")
model = load_silero_vad()
ts = get_speech_timestamps(torch.from_numpy(asr_audio), model, sampling_rate=16000,
                           threshold=args.threshold, min_speech_duration_ms=500,
                           min_silence_duration_ms=args.min_silence_ms, speech_pad_ms=args.pad_ms,
                           max_speech_duration_s=15, return_seconds=True)
print("speech regions found by VAD:", len(ts))

segs, cur = [], None
for t in ts:
    s, e = t["start"], t["end"]
    if cur is None:
        cur = [s, e]
    elif (s - cur[1]) <= args.max_gap and (e - cur[0]) <= args.target_max:
        cur[1] = e
    else:
        segs.append(cur)
        cur = [s, e]
if cur:
    segs.append(cur)
segs = [s for s in segs if (s[1] - s[0]) >= args.min_len]

asr_dir = Path("data/interim/asr_audio/segments")
tts_dir = Path("data/interim/tts_audio/segments")
asr_dir.mkdir(parents=True, exist_ok=True)
tts_dir.mkdir(parents=True, exist_ok=True)
for d in (asr_dir, tts_dir):                       # remove only THIS source's old clips
    for p in d.glob(f"{args.source}_*.wav"):
        p.unlink()

records = []
for i, (s, e) in enumerate(segs, 1):
    sid = f"{args.source}_{i:06d}"
    sf.write(asr_dir / f"{sid}.wav", asr_audio[int(s * asr_sr):int(e * asr_sr)], asr_sr, subtype="PCM_16")
    clip, _ = sf.read(str(master), start=int(s * m_sr), stop=int(e * m_sr), dtype="float32")
    sf.write(tts_dir / f"{sid}.wav", clip, m_sr, subtype="PCM_16")
    records.append({"id": sid, "source": args.source, "start": round(s, 3), "end": round(e, 3),
                    "duration": round(e - s, 3), "crop_start": args.start})

seg_path = Path("data/transcripts/segments.jsonl")
seg_path.parent.mkdir(parents=True, exist_ok=True)
keep = []
if seg_path.exists():
    for l in open(seg_path, encoding="utf-8"):
        if l.strip():
            r = json.loads(l)
            if r.get("source") != args.source:     # keep every other source untouched
                keep.append(r)
with open(seg_path, "w", encoding="utf-8") as f:
    for r in sorted(keep + records, key=lambda r: r["id"]):
        f.write(json.dumps(r) + "\n")

d = np.array([e - s for s, e in segs])
print(f"\nsource {args.source}: {len(segs)} clips, {d.sum() / 60:.1f} min")
print(f"shorter than 3 s: {(d < 3).sum()}   3-12 s: {((d >= 3) & (d <= 12)).sum()}   longer than 12 s: {(d > 12).sum()}")
print(f"shortest {d.min():.1f} s   longest {d.max():.1f} s   median {np.median(d):.1f} s")
print(f"segments.jsonl now has {len(keep) + len(records)} clips in total")