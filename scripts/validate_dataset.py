import argparse, collections, csv, hashlib, re, sys
from pathlib import Path
import numpy as np
import soundfile as sf

ap = argparse.ArgumentParser()
ap.add_argument("--dir", default="data/final")
ap.add_argument("--min_dur", type=float, default=1.5)
ap.add_argument("--max_dur", type=float, default=15.0)
args = ap.parse_args()

root = Path(args.dir)
meta = root / "metadata.csv"
if not meta.exists():
    sys.exit(f"{meta} not found. Run scripts/build_dataset.py first.")
rows = list(csv.DictReader(open(meta, encoding="utf-8", newline="")))
print("rows in metadata.csv:", len(rows))

hard, warn = collections.defaultdict(list), collections.defaultdict(list)
rates, durs, cps, hashes = collections.Counter(), [], {}, {}
name_count = collections.Counter(r["file_name"] for r in rows)
texts = collections.Counter(r["text"].strip() for r in rows)
used_chars = set()
ALLOWED_PUNCT = set(".,?!:;-\"'“”‘’«»() ")

for r in rows:
    fn, text = r["file_name"], r["text"].strip()
    if name_count[fn] > 1:
        hard["duplicate file name"].append(fn)
    if not text:
        hard["empty text"].append(fn)
    if re.search("[0-9\u1369-\u137c]", text):
        warn["digits or Ge'ez numerals in text"].append(fn)
    if re.search("[A-Za-z]", text):
        warn["Latin letters in text"].append(fn)
    odd = {c for c in text if not ("\u1200" <= c <= "\u137f" or c in ALLOWED_PUNCT)}
    if odd:
        warn["unexpected characters: " + "".join(sorted(odd))[:10]].append(fn)
    used_chars |= {c for c in text if "\u1200" <= c <= "\u1360"}
    if texts[text] > 1:
        warn["duplicate text (check it is not a duplicate clip)"].append(fn)
    p = root / fn
    if not p.exists():
        hard["missing audio file"].append(fn)
        continue
    try:
        info = sf.info(str(p))
        y, _ = sf.read(str(p), dtype="float32")
    except Exception:
        hard["unreadable audio"].append(fn)
        continue
    if info.channels != 1:
        hard["not mono"].append(fn)
    rates[info.samplerate] += 1
    dur = info.frames / info.samplerate
    durs.append(dur)
    cps[fn] = len(text.replace(" ", "")) / dur if dur else 0
    if not (args.min_dur <= dur <= args.max_dur):
        warn[f"duration outside {args.min_dur}-{args.max_dur} s"].append(fn)
    if y.ndim > 1:
        y = y.mean(axis=1)
    if np.mean(np.abs(y) > 0.99) > 0.001:
        warn["clipped samples"].append(fn)
    if np.abs(y).max() < 0.1:
        warn["very quiet (peak below 0.1)"].append(fn)
    h = hashlib.md5(p.read_bytes()).hexdigest()
    if h in hashes:
        hard["duplicate audio"].append(f"{fn} = {hashes[h]}")
    hashes[h] = fn

if len(rates) > 1:
    hard["mixed sample rates"].append(str(dict(rates)))
if cps:
    med = float(np.median(list(cps.values())))
    for fn, c in cps.items():
        if c < med * 0.5 or c > med * 1.6:
            warn["speaking rate far from the median (text may not match the audio)"].append(fn)

d = np.array(durs) if durs else np.array([0.0])
print(f"sample rate(s): {dict(rates)} | total: {d.sum() / 60:.1f} min ({d.sum() / 3600:.2f} h)")
print(f"duration  <3 s: {(d < 3).sum()}  3-12 s: {((d >= 3) & (d <= 12)).sum()}  >12 s: {(d > 12).sum()}"
      f"  | median {np.median(d):.1f} s  longest {d.max():.1f} s")
print("distinct Ethiopic characters used:", len(used_chars))
print("text sources:", dict(collections.Counter(r["text_source"] for r in rows)))

print("\nHARD PROBLEMS (must fix):")
for k, v in hard.items():
    print(f"  {k}: {len(v)}  e.g. {v[:5]}")
print("WARNINGS (look at them):")
for k, v in warn.items():
    print(f"  {k}: {len(v)}  e.g. {v[:5]}")
print("\nRESULT:", "FAIL" if hard else "PASS (no hard problems)")
sys.exit(1 if hard else 0)