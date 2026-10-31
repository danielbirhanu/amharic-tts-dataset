import json, re

print("check_asr.py started")

seg = {}
for l in open("data/transcripts/segments.jsonl", encoding="utf-8"):
    r = json.loads(l); seg[r["id"]] = r
asr = [json.loads(l) for l in open("data/transcripts/asr.jsonl", encoding="utf-8")]
print("segments:", len(seg), "  transcripts:", len(asr))
missing = set(seg) - {a["id"] for a in asr}
print("clips without a transcript:", len(missing))

def repeats(t):
    w = t.split()
    return any(w[i:i+3] == w[i+3:i+6] for i in range(max(0, len(w) - 5)))

rows = []
for a in asr:
    t, d = a["asr_text"], seg[a["id"]]["duration"]
    rows.append([a["id"], d, len(t.replace(" ", "")) / d, [], t])
cps = sorted(r[2] for r in rows)
med = cps[len(cps) // 2]
print(f"characters per second: min {cps[0]:.1f}  median {med:.1f}  max {cps[-1]:.1f}")

flagged = 0
for r in rows:
    id_, d, c, flags, t = r
    if not t.strip(): flags.append("EMPTY")
    if repeats(t): flags.append("REPEATED PHRASE")
    if re.search(r"[A-Za-z0-9]", t): flags.append("LATIN/DIGITS")
    if c < med * 0.5: flags.append("TOO FEW CHARS")
    if c > med * 1.6: flags.append("TOO MANY CHARS")
    if flags:
        flagged += 1
        print(id_, f"{d:.1f}s", f"cps={c:.1f}", flags, t[:70])
print("flagged clips:", flagged)
