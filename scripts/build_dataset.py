import argparse, collections, csv, json, shutil, unicodedata
from pathlib import Path
import re
import soundfile as sf
from textutil import HOMO

ap = argparse.ArgumentParser()
ap.add_argument("--segments", default="data/transcripts/segments.jsonl")
ap.add_argument("--asr", default="data/transcripts/asr.jsonl")
ap.add_argument("--decisions", default="data/transcripts/suggestions_decisions.jsonl")
ap.add_argument("--reviewed", default="data/transcripts/reviewed.jsonl")
ap.add_argument("--audio_dir", default="data/interim/tts_audio/segments")
ap.add_argument("--out_dir", default="data/final")
ap.add_argument("--speaker", default="narrator_01")
ap.add_argument("--sr", type=int, default=0)                # 0 = keep the original sample rate
ap.add_argument("--min_dur", type=float, default=1.5)
ap.add_argument("--max_dur", type=float, default=15.0)
ap.add_argument("--keep_digits", action="store_true")       # by default clips with digits are dropped
args = ap.parse_args()


def load(path):
    p = Path(path)
    if not p.exists():
        print("(not found, skipped):", path)
        return {}
    return {r["id"]: r for r in map(json.loads, open(p, encoding="utf-8"))}


def clean(t):
    return re.sub(r"\s+", " ", unicodedata.normalize("NFC", str(t))).strip()


seg, asr = load(args.segments), load(args.asr)
decisions, reviewed = load(args.decisions), load(args.reviewed)
if not seg or not asr:
    raise SystemExit("segments.jsonl or asr.jsonl is missing.")

out = Path(args.out_dir)
(out / "audio").mkdir(parents=True, exist_ok=True)
for old in (out / "audio").glob("*.wav"):        # remove files from earlier builds
    old.unlink()

DIGITS = re.compile("[0-9\u1369-\u137c]")
rows, dropped = [], []
sources, conflicts = collections.Counter(), 0

for sid, s in seg.items():
    a = asr.get(sid)
    if a is None:
        dropped.append((sid, "no ASR transcript"))
        continue
    rv, dc = reviewed.get(sid), decisions.get(sid)
    if rv and rv.get("status") in ("rejected", "unsure"):
        dropped.append((sid, "full review: " + rv["status"]))
        continue
    if dc and dc.get("status") in ("rejected", "unsure"):
        dropped.append((sid, "suggestion review: " + dc["status"]))
        continue
    if rv and rv.get("status") == "accepted" and clean(rv.get("text", "")):
        text, src = clean(rv["text"]), "human_review"
        if dc and dc.get("status") == "accepted" and clean(dc.get("text", "")) != text:
            conflicts += 1
    elif dc and dc.get("status") == "accepted" and clean(dc.get("text", "")):
        text, src = clean(dc["text"]), "suggestion_review"
    elif dc:
        text, src = clean(a["asr_text"]), "asr_suggestion_not_reviewed"
    else:
        text, src = clean(a["asr_text"]), "asr_only"
    if not text:
        dropped.append((sid, "empty text"))
        continue
    if DIGITS.search(text) and not args.keep_digits:
        dropped.append((sid, "contains digits (convert numbers to words first)"))
        continue
    wav = Path(args.audio_dir) / f"{sid}.wav"
    if not wav.exists():
        dropped.append((sid, "audio file missing"))
        continue
    info = sf.info(str(wav))
    dur = info.frames / info.samplerate
    if not (args.min_dur <= dur <= args.max_dur):
        dropped.append((sid, f"duration {dur:.1f}s out of range"))
        continue
    dst = out / "audio" / f"{sid}.wav"
    if args.sr and args.sr != info.samplerate:
        import librosa
        y, orig = sf.read(str(wav), dtype="float32")
        if y.ndim > 1:
            y = y.mean(axis=1)
        sf.write(str(dst), librosa.resample(y, orig_sr=orig, target_sr=args.sr), args.sr, subtype="PCM_16")
    else:
        shutil.copy2(wav, dst)
    sources[src] += 1
    rows.append({"file_name": f"audio/{sid}.wav", "text": text,
                 "text_normalized": unicodedata.normalize("NFC", text).translate(HOMO),
                 "speaker_id": args.speaker, "source_id": s.get("source", ""),
                 "start": s.get("start", ""), "end": s.get("end", ""),
                 "duration": round(dur, 3), "text_source": src})

fields = ["file_name", "text", "text_normalized", "speaker_id", "source_id", "start", "end", "duration", "text_source"]
with open(out / "metadata.csv", "w", encoding="utf-8", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    w.writerows(rows)
with open(out / "dropped.csv", "w", encoding="utf-8", newline="") as f:
    w = csv.writer(f)
    w.writerow(["id", "reason"])
    w.writerows(dropped)

hours = sum(r["duration"] for r in rows) / 3600
print(f"kept {len(rows)} clips ({hours * 60:.1f} min) | dropped {len(dropped)}")
print("where the text came from:")
for k, v in sources.most_common():
    print(f"  {k}: {v}")
if dropped:
    print("dropped because:")
    for k, v in collections.Counter(r for _, r in dropped).most_common():
        print(f"  {k}: {v}")
if sources["asr_suggestion_not_reviewed"]:
    print(f"\nWARNING: {sources['asr_suggestion_not_reviewed']} clips had Gemma suggestions you did not review (ASR text kept).")
if conflicts:
    print(f"NOTE: {conflicts} clips had different texts in your full review and suggestions review; the full review won.")
print("wrote", out / "metadata.csv", "and", out / "dropped.csv")