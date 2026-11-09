import argparse, json
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("target")
ap.add_argument("new", nargs="+")
args = ap.parse_args()

rows = {}
t = Path(args.target)
if t.exists():
    for l in open(t, encoding="utf-8"):
        if l.strip():
            r = json.loads(l)
            rows[r["id"]] = r
before = len(rows)
for p in args.new:
    for l in open(p, encoding="utf-8"):
        if l.strip():
            r = json.loads(l)
            rows[r["id"]] = r                  # newer entries win
with open(t, "w", encoding="utf-8") as f:
    for k in sorted(rows):
        f.write(json.dumps(rows[k], ensure_ascii=False) + "\n")
print(f"{t}: {before} -> {len(rows)} entries")