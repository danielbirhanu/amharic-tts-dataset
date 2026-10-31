import argparse, json
from pathlib import Path
import numpy as np
import soundfile as sf
import torch
from silero_vad import load_silero_vad, get_speech_timestamps

ap = argparse.ArgumentParser()
ap.add_argument("--source", default="book01")
ap.add_argument("--master", default="data/interim/master/book01_15min.wav")
ap.add_argument("--asr", default="data/interim/asr_audio/book01_15min_asr.wav")
ap.add_argument("--threshold", type=float, default=0.5)       # lower = more sensitive to speech
ap.add_argument("--min_silence_ms", type=int, default=300)    # pause length that splits speech
ap.add_argument("--pad_ms", type=int, default=150)            # safety padding around speech
ap.add_argument("--max_gap", type=float, default=0.6)         # merge pieces separated by <= this (seconds)
ap.add_argument("--target_max", type=float, default=12.0)     # preferred maximum clip length (seconds)
ap.add_argument("--min_len", type=float, default=1.5)         # drop clips shorter than this (seconds)
args = ap.parse_args()

# 1) read both files (soundfile, so we don't depend on torchaudio's loader)
asr_audio, asr_sr = sf.read(args.asr, dtype="float32")
master, m_sr = sf.read(args.master, dtype="float32")
if asr_audio.ndim > 1: asr_audio = asr_audio.mean(axis=1)
if master.ndim > 1: master = master.mean(axis=1)
assert asr_sr == 16000, f"ASR file must be 16 kHz, got {asr_sr}"
print(f"durations: asr={len(asr_audio)/asr_sr:.2f}s  master={len(master)/m_sr:.2f}s (should match)")

# 2) find speech regions in the 16 kHz copy
model = load_silero_vad()
ts = get_speech_timestamps(
    torch.from_numpy(asr_audio), model, sampling_rate=16000,
    threshold=args.threshold,
    min_speech_duration_ms=500,
    min_silence_duration_ms=args.min_silence_ms,
    speech_pad_ms=args.pad_ms,
    max_speech_duration_s=15,
    return_seconds=True,
)
print("speech regions found by VAD:", len(ts))

# 3) merge neighbouring regions into clips of about 3-12 s
segs, cur = [], None
for t in ts:
    s, e = t["start"], t["end"]
    if cur is None:
        cur = [s, e]
    elif (s - cur[1]) <= args.max_gap and (e - cur[0]) <= args.target_max:
        cur[1] = e
    else:
        segs.append(cur); cur = [s, e]
if cur: segs.append(cur)
segs = [s for s in segs if (s[1] - s[0]) >= args.min_len]

# 4) cut BOTH files with exactly the same times
asr_dir = Path("data/interim/asr_audio/segments"); asr_dir.mkdir(parents=True, exist_ok=True)
tts_dir = Path("data/interim/tts_audio/segments"); tts_dir.mkdir(parents=True, exist_ok=True)
for p in list(asr_dir.glob("*.wav")) + list(tts_dir.glob("*.wav")):   # remove clips from earlier runs
    p.unlink()
out = Path("data/transcripts/segments.jsonl"); out.parent.mkdir(parents=True, exist_ok=True)

with out.open("w", encoding="utf-8") as f:
    for i, (s, e) in enumerate(segs, 1):
        sid = f"{args.source}_{i:06d}"
        sf.write(asr_dir / f"{sid}.wav", asr_audio[int(s*asr_sr):int(e*asr_sr)], asr_sr, subtype="PCM_16")
        sf.write(tts_dir / f"{sid}.wav", master[int(s*m_sr):int(e*m_sr)], m_sr, subtype="PCM_16")
        f.write(json.dumps({"id": sid, "source": args.source, "start": round(s, 3),
                            "end": round(e, 3), "duration": round(e - s, 3)}) + "\n")

# 5) summary
d = np.array([e - s for s, e in segs])
print(f"\nclips written: {len(segs)}   total: {d.sum()/60:.1f} min")
print(f"shorter than 3 s: {(d < 3).sum()}   3-12 s: {((d >= 3) & (d <= 12)).sum()}   longer than 12 s: {(d > 12).sum()}")
print(f"shortest: {d.min():.1f} s   longest: {d.max():.1f} s   median: {np.median(d):.1f} s")
print("saved list to", out)