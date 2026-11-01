import argparse, json, os, re, sys, time, unicodedata
from pathlib import Path
import requests, jiwer
from dotenv import load_dotenv
from textutil import norm

load_dotenv()
ap = argparse.ArgumentParser()
ap.add_argument("--asr", default="data/transcripts/asr.jsonl")
ap.add_argument("--out", default="data/transcripts/llm_cleaned.jsonl")
ap.add_argument("--model", default="")                       # overrides OPENROUTER_MODEL from .env
ap.add_argument("--batch", type=int, default=10)
ap.add_argument("--limit", type=int, default=0)              # 0 = all clips; use 20 for a first test
ap.add_argument("--max_cer", type=float, default=0.15)       # reject edits that change more than this
ap.add_argument("--max_word_delta", type=int, default=2)     # reject edits that add/remove more words than this
ap.add_argument("--sleep", type=float, default=2.0)          # pause between batches
ap.add_argument("--max_retries", type=int, default=4)        # tries per batch
ap.add_argument("--max_failed_batches", type=int, default=2) # stop after this many failed batches in a row
args = ap.parse_args()

API_KEY = os.environ.get("OPENROUTER_API_KEY")
MODEL = args.model or os.environ.get("OPENROUTER_MODEL", "google/gemma-4-26b-a4b-it")
if not API_KEY:
    raise SystemExit("OPENROUTER_API_KEY is missing. Create .env first (step 6.1).")
URL = "https://openrouter.ai/api/v1/chat/completions"
FENCE = "`" * 3

SYSTEM = """You correct Amharic automatic speech recognition (ASR) transcripts for a speech-synthesis dataset.
Each text is what a narrator SAID. You cannot hear the audio.

Rules:
1. The corrected text must contain exactly what was spoken. Never paraphrase, summarize, translate, modernize, or improve style or grammar.
2. Fix only errors you are highly confident are recognition mistakes: a wrong letter (a similar-sounding syllable), wrong word boundaries (split or merged words), or an obviously misrecognized word given the surrounding context.
3. Do not add words that seem to be missing and do not remove words. Do not change names, loanwords, dialect forms, or unusual but plausible words.
4. Keep the existing punctuation style (Amharic marks such as ።). Only fix punctuation that is clearly wrong.
5. Write numbers exactly as in the input. Do not convert between digits and words.
6. previous_context and next_context are only for understanding. Never copy text from them into the output.
7. If you are not sure whether something is an error, set needs_review to true and make clean_text IDENTICAL to the input asr_text. Put your suspicion in notes. Whenever needs_review is true, clean_text must not differ from asr_text.
8. Return valid JSON only, in this shape:
{"segments":[{"id":"...","clean_text":"...","needs_review":false,"notes":"short reason, or empty"}]}
Return one entry for every input id."""


class Stop(Exception):
    """Raised for errors where retrying cannot help."""


def error_text(r):
    try:
        e = r.json().get("error", {})
        msg = str(e.get("message", ""))
        meta = e.get("metadata", {}) or {}
        if meta.get("provider_name"):
            msg += f" [provider: {meta['provider_name']}]"
        if meta.get("raw"):
            msg += f" raw: {str(meta['raw'])[:200]}"
        return msg.strip() or r.text[:300]
    except Exception:
        return r.text[:300]


def call_llm(items):
    body = {"model": MODEL, "temperature": 0,
            "messages": [{"role": "system", "content": SYSTEM},
                         {"role": "user", "content": json.dumps({"segments": items}, ensure_ascii=False)}],
            "response_format": {"type": "json_object"}}
    for attempt in range(args.max_retries):
        try:
            r = requests.post(URL, headers={"Authorization": f"Bearer {API_KEY}"}, json=body, timeout=180)
        except requests.RequestException as e:
            wait = 5 * (attempt + 1)
            print(f"  network error ({str(e)[:60]}), retrying in {wait}s")
            time.sleep(wait)
            continue
        if r.status_code == 200:
            content = ((r.json().get("choices") or [{}])[0].get("message") or {}).get("content")
            if content:
                return content
            print("  empty answer from the model, retrying")
            time.sleep(5)
            continue
        msg = error_text(r)
        print(f"  HTTP {r.status_code}: {msg}")
        if r.status_code == 400 and "response_format" in body:
            body.pop("response_format")
            print("  retrying without JSON mode")
            continue
        if r.status_code in (401, 402, 403, 404):
            raise Stop(f"HTTP {r.status_code} cannot be fixed by retrying: {msg}")
        if r.status_code == 429 or r.status_code >= 500:
            try:
                wait = float(r.headers.get("Retry-After", 0))
            except ValueError:
                wait = 0
            wait = wait or 15 * (attempt + 1)
            print(f"  waiting {wait:.0f}s (try {attempt + 1} of {args.max_retries})")
            time.sleep(wait)
            continue
        raise Stop(f"Unexpected HTTP {r.status_code}: {msg}")
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


ETH_NUM = re.compile("[\u1369-\u137c]")


def guard(asr_text, new):
    """Return a list of reasons to REJECT the LLM's version (empty list = accepted)."""
    if not new or not new.strip():
        return ["empty"]
    reasons = []
    bad = {ch for ch in new if not (ch.isspace() or "\u1200" <= ch <= "\u137f" or ch in ".,?!:;-\"'“”‘’«»()")}
    if bad:
        reasons.append("unexpected characters: " + "".join(sorted(bad))[:10])
    if ETH_NUM.search(new) and not ETH_NUM.search(asr_text):
        reasons.append("introduced Ge'ez numerals")
    a, b = norm(asr_text), norm(new)
    if a:
        c = jiwer.cer(a, b)
        if c > args.max_cer:
            reasons.append(f"too many changes (cer {c:.2f})")
    if abs(len(a.split()) - len(b.split())) > args.max_word_delta:
        reasons.append("word count changed")
    return reasons


asr = sorted((json.loads(l) for l in open(args.asr, encoding="utf-8")), key=lambda r: r["id"])
if args.limit:
    asr = asr[:args.limit]
out = Path(args.out)
out.parent.mkdir(parents=True, exist_ok=True)
done = {}
if out.exists():
    for l in open(out, encoding="utf-8"):
        r = json.loads(l)
        done[r["id"]] = r
todo = [i for i, r in enumerate(asr) if r["id"] not in done]
print(f"{len(asr)} clips | {len(done)} already done | {len(todo)} to do | model {MODEL}")

failed_in_a_row = 0
try:
    for k in range(0, len(todo), args.batch):
        idxs = todo[k:k + args.batch]
        items = [{"id": asr[i]["id"],
                  "previous_context": asr[i-1]["asr_text"] if i > 0 else "",
                  "asr_text": asr[i]["asr_text"],
                  "next_context": asr[i+1]["asr_text"] if i + 1 < len(asr) else ""} for i in idxs]
        content = call_llm(items)
        parsed = parse(content) if content else None
        if not parsed:
            failed_in_a_row += 1
            print(f"batch starting {items[0]['id']}: no usable answer, will be retried on the next run")
            if failed_in_a_row >= args.max_failed_batches:
                print("\nStopping early: several batches failed in a row. Run scripts/check_openrouter.py to see why.")
                break
            continue
        failed_in_a_row = 0
        with out.open("a", encoding="utf-8") as f:
            for i in idxs:
                rec_asr = asr[i]["asr_text"]
                s = parsed.get(asr[i]["id"])
                if s is None:
                    print("  missing in answer:", asr[i]["id"])
                    continue
                new = unicodedata.normalize("NFC", str(s.get("clean_text", "")).strip())
                llm_unsure = bool(s.get("needs_review"))
                reasons = guard(rec_asr, new)
                guard_ok = not reasons
                apply_change = guard_ok and not llm_unsure          # unsure => keep the ASR text
                clean = new if apply_change else rec_asr
                f.write(json.dumps({"id": asr[i]["id"], "asr_text": rec_asr, "llm_text": new, "clean_text": clean,
                                    "changed": clean != rec_asr,
                                    "guard_ok": guard_ok, "guard_reasons": reasons,
                                    "llm_unsure": llm_unsure,
                                    "pending_suggestion": bool(new) and new != rec_asr and not apply_change,
                                    "needs_review": llm_unsure or not guard_ok,
                                    "notes": str(s.get("notes", "")), "model": MODEL}, ensure_ascii=False) + "\n")
        print(f"batch {k // args.batch + 1} saved ({idxs[0] + 1}-{idxs[-1] + 1} of {len(asr)})")
        time.sleep(args.sleep)
except Stop as e:
    print("\nSTOPPED:", e)

rows = [json.loads(l) for l in open(out, encoding="utf-8")] if out.exists() else []
if not rows:
    print("\nNo clips were processed, so there is nothing to summarize.")
    sys.exit(1)
print(f"\nclips processed: {len(rows)}")
print("changes applied (confident + passed guardrails):", sum(r["changed"] for r in rows))
print("rejected by guardrails:", sum(not r["guard_ok"] for r in rows))
print("Gemma unsure (ASR text kept):", sum(r["llm_unsure"] for r in rows))
print("suggestions waiting for a human:", sum(r["pending_suggestion"] for r in rows))
print("flagged needs_review (total):", sum(r["needs_review"] for r in rows))