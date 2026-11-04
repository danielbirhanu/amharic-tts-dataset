import argparse, csv, hashlib, json, os, random, re, sys, time, unicodedata
from pathlib import Path
import requests
from dotenv import load_dotenv

load_dotenv()
ap = argparse.ArgumentParser()
ap.add_argument("--meta", default="data/final/metadata.csv")
ap.add_argument("--segments", default="data/transcripts/segments.jsonl")
ap.add_argument("--cache", default="data/transcripts/punctuated.jsonl")
ap.add_argument("--model", default="")
ap.add_argument("--batch", type=int, default=10)
ap.add_argument("--limit", type=int, default=0)         # 0 = all clips; use 20 for a first test
ap.add_argument("--sleep", type=float, default=2.0)
ap.add_argument("--max_retries", type=int, default=4)
args = ap.parse_args()

API_KEY = os.environ.get("OPENROUTER_API_KEY")
MODEL = args.model or os.environ.get("OPENROUTER_MODEL", "google/gemma-4-26b-a4b-it")
if not API_KEY:
    raise SystemExit("OPENROUTER_API_KEY is missing in .env")
URL = "https://openrouter.ai/api/v1/chat/completions"
FENCE = "`" * 3

SYSTEM = """You add punctuation to Amharic transcripts of narrated speech, for a speech-synthesis dataset.

Rules:
1. Do NOT change, add, remove, reorder or respell any word or letter. Only insert punctuation marks and keep the words and the spaces between them.
2. Allowed marks: ። (end of a sentence), ፣ (comma: a short pause inside a sentence), ? (a question). Use no other symbols.
3. Each text is one audio clip of continuous narration. It may start or end in the middle of a sentence. previous_context and next_context are the neighbouring clips, only for understanding. Never copy them.
4. pause_after_seconds is the measured silence after the clip. A longer pause makes the end of a sentence more likely. A very short pause means the sentence probably continues, so then do not end the clip with ።.
5. Be conservative: add a mark only when it is clearly warranted. Use ? only for a clear question. When unsure, add less.
6. Return valid JSON only: {"segments":[{"id":"...","punctuated":"..."}]}
Return one entry for every input id."""

PUNCT = "።፣፤፥፦፧፠.,!?:;\"'“”‘’«»()-"
SPLIT = re.compile("[" + re.escape(PUNCT) + "]")


def words(s):
    return SPLIT.sub(" ", s).split()


def tidy(s):
    s = unicodedata.normalize("NFC", s).strip()
    s = re.sub(r"\s+([።፣?])", r"\1", s)       # no space before a mark
    s = re.sub(r"([።፣?])\1+", r"\1", s)       # no repeated marks
    return re.sub(r"\s+", " ", s)


def check(orig, new):
    if not new:
        return ["empty"]
    if words(new) != words(orig):
        return ["words were changed"]
    for c in new:
        if c.isspace() or c in "።፣?" or "\u1200" <= c <= "\u135f":
            continue
        return ["unexpected character: " + c]
    if new[0] in "።፣?":
        return ["starts with a mark"]
    return []


def call_llm(items):
    body = {"model": MODEL, "temperature": 0,
            "messages": [{"role": "system", "content": SYSTEM},
                         {"role": "user", "content": json.dumps({"segments": items}, ensure_ascii=False)}],
            "response_format": {"type": "json_object"}}
    for attempt in range(args.max_retries):
        try:
            r = requests.post(URL, headers={"Authorization": f"Bearer {API_KEY}"}, json=body, timeout=180)
        except requests.RequestException:
            time.sleep(5 * (attempt + 1))
            continue
        if r.status_code == 200:
            content = ((r.json().get("choices") or [{}])[0].get("message") or {}).get("content")
            if content:
                return content
            time.sleep(5)
            continue
        print(f"  HTTP {r.status_code}: {r.text[:200]}")
        if r.status_code == 400 and "response_format" in body:
            body.pop("response_format")
            continue
        if r.status_code in (401, 402, 403, 404):
            raise SystemExit("Cannot continue: " + r.text[:200])
        time.sleep(15 * (attempt + 1))
    return None


def parse(content):
    c = re.sub(r"^" + FENCE + r"(?:json)?\s*|\s*" + FENCE + r"$", "", content.strip())
    try:
        obj = json.loads(c)
    except Exception:
        m = re.search(r"\{.*\}", c, re.S)
        if not m:
            return None
        try:
            obj = json.loads(m.group(0))
        except Exception:
            return None
    return {s.get("id"): s for s in obj.get("segments", []) if isinstance(s, dict)}


def sha(t):
    return hashlib.sha1(t.encode("utf-8")).hexdigest()[:12]


meta_path = Path(args.meta)
with open(meta_path, encoding="utf-8", newline="") as f:
    rows = list(csv.DictReader(f))
fields = list(rows[0].keys())
text_of = {Path(r["file_name"]).stem: r["text"].strip() for r in rows}

seg = {}
for l in open(args.segments, encoding="utf-8"):
    if l.strip():
        r = json.loads(l)
        seg[r["id"]] = r
order = sorted(seg)
pos = {cid: i for i, cid in enumerate(order)}


def neighbour(cid, d):
    i = pos.get(cid)
    if i is None or not (0 <= i + d < len(order)):
        return None
    nid = order[i + d]
    return nid if seg[nid].get("source") == seg[cid].get("source") else None


def pause_after(cid):
    n = neighbour(cid, 1)
    return round(seg[n]["start"] - seg[cid]["end"], 2) if n else None


cache = {}
cp = Path(args.cache)
if cp.exists():
    for l in open(cp, encoding="utf-8"):
        if l.strip():
            r = json.loads(l)
            cache[r["id"]] = r

todo = [cid for cid in text_of if cid in seg and not (cid in cache and cache[cid]["text_in"] == sha(text_of[cid]))]
if args.limit:
    todo = todo[:args.limit]
print(f"{len(text_of)} clips | {len(todo)} to punctuate | model {MODEL}")

failed = 0
for k in range(0, len(todo), args.batch):
    ids = todo[k:k + args.batch]
    items = []
    for cid in ids:
        p, n = neighbour(cid, -1), neighbour(cid, 1)
        items.append({"id": cid, "previous_context": text_of.get(p, "") if p else "",
                      "text": text_of[cid], "next_context": text_of.get(n, "") if n else "",
                      "pause_after_seconds": pause_after(cid)})
    content = call_llm(items)
    parsed = parse(content) if content else None
    if not parsed:
        failed += 1
        print("  no usable answer for this batch (will retry on the next run)")
        if failed >= 2:
            print("Stopping early after two failed batches in a row.")
            break
        continue
    failed = 0
    with cp.open("a", encoding="utf-8") as f:
        for cid in ids:
            s = parsed.get(cid)
            if s is None:
                continue
            new = tidy(str(s.get("punctuated", "")))
            reasons = check(text_of[cid], new)
            f.write(json.dumps({"id": cid, "text_in": sha(text_of[cid]), "punctuated": new,
                                "ok": not reasons, "reasons": reasons, "model": MODEL}, ensure_ascii=False) + "\n")
            cache[cid] = {"id": cid, "text_in": sha(text_of[cid]), "punctuated": new, "ok": not reasons}
    print(f"batch {k // args.batch + 1} saved")
    time.sleep(args.sleep)

# write the text_punct column into metadata.csv (the verbatim `text` column is not touched)
if "text_punct" not in fields:
    fields.append("text_punct")
used = bad = 0
for r in rows:
    cid = Path(r["file_name"]).stem
    c = cache.get(cid)
    if c and c["text_in"] == sha(r["text"].strip()) and c["ok"]:
        r["text_punct"] = c["punctuated"]
        used += 1
    else:
        r["text_punct"] = r["text"]
        bad += 1
with open(meta_path, "w", encoding="utf-8", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    w.writerows(rows)

print(f"\ntext_punct filled for {used} clips | kept unpunctuated (rejected or not run yet): {bad}")
print("clips ending with ።:", sum(r["text_punct"].endswith("።") for r in rows),
      "| with ?:", sum(r["text_punct"].endswith("?") for r in rows),
      "| commas ፣ total:", sum(r["text_punct"].count("፣") for r in rows))
print("\nrandom examples (listen to these!):")
for r in random.sample(rows, min(8, len(rows))):
    cid = Path(r["file_name"]).stem
    print(f"{cid}  pause_after={pause_after(cid)}\n   {r['text_punct']}")