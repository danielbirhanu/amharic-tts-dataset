import json
import jiwer
from textutil import norm

rows = [json.loads(l) for l in open("data/transcripts/reviewed.jsonl", encoding="utf-8")]
blind = [r for r in rows
         if r.get("blind") and r["status"] == "accepted" and norm(r["text"])
         and "seen" not in r.get("notes", "").lower()]
print("blind clips usable:", len(blind), "(typed from scratch, status accepted)")
if not blind:
    raise SystemExit("No usable blind clips yet.")

ref = [norm(r["text"]) for r in blind]
hyp = [norm(r["asr_text"]) for r in blind]
print("raw ASR CER:", round(jiwer.cer(ref, hyp), 4))
print("raw ASR WER:", round(jiwer.wer(ref, hyp), 4))
for r in blind:
    c = jiwer.cer(norm(r["text"]), norm(r["asr_text"]))
    if c > 0:
        print(f"\n{r['id']}  cer {c:.3f}\n  you: {r['text']}\n  ASR: {r['asr_text']}")