import argparse, difflib, hashlib, json
from pathlib import Path
from textutil import norm

ap = argparse.ArgumentParser()
ap.add_argument("--asr", default="data/transcripts/asr.jsonl")
ap.add_argument("--segments", default="data/transcripts/segments.jsonl")
ap.add_argument("--proposals", nargs="+",
                default=["data/transcripts/llm_cleaned.jsonl", "data/transcripts/llm_cleaned_31b.jsonl"])
ap.add_argument("--audio_dir", default="data/interim/tts_audio/segments")
ap.add_argument("--out", default="suggestions.html")
args = ap.parse_args()


def load(path):
    p = Path(path)
    if not p.exists():
        print("(missing, skipped):", path)
        return {}
    return {r["id"]: r for r in map(json.loads, open(p, encoding="utf-8"))}


asr = load(args.asr)
seg = load(args.segments)
if not asr:
    raise SystemExit("asr.jsonl not found or empty")
sources = {Path(p).stem: load(p) for p in args.proposals}
sources = {k: v for k, v in sources.items() if v}
if not sources:
    raise SystemExit("No Gemma output found. Run scripts/llm_clean.py first.")

items, skipped_same_sound, total_edits = [], 0, 0
for cid in sorted(asr):
    words = asr[cid]["asr_text"].split()
    edits, notes = {}, []
    for sname, recs in sources.items():
        r = recs.get(cid)
        if not r or not r.get("llm_text"):
            continue
        new_words = r["llm_text"].split()
        found = False
        for op, i1, i2, j1, j2 in difflib.SequenceMatcher(None, words, new_words, autojunk=False).get_opcodes():
            if op == "equal":
                continue
            new = new_words[j1:j2]
            if norm(" ".join(words[i1:i2])) == norm(" ".join(new)):
                skipped_same_sound += 1          # only same-sounding letters or punctuation differ
                continue
            edits.setdefault((i1, i2, tuple(new)), []).append(sname)
            found = True
        if found and r.get("notes"):
            notes.append({"source": sname, "note": r["notes"]})
    if not edits:
        continue
    total_edits += len(edits)
    items.append({
        "id": cid, "duration": seg.get(cid, {}).get("duration"),
        "audio": f"{args.audio_dir}/{cid}.wav", "asr": words,
        "edits": [{"i1": i1, "i2": i2, "from": " ".join(words[i1:i2]), "to": " ".join(new), "sources": srcs}
                  for (i1, i2, new), srcs in sorted(edits.items())],
        "notes": notes,
    })

print(f"Gemma files used: {', '.join(sources)}")
print(f"clips with suggestions: {len(items)} | edits to review: {total_edits} | "
      f"ignored same-sound/punctuation-only edits: {skipped_same_sound}")
if not items:
    raise SystemExit("Nothing to review: Gemma proposed no real changes.")

items_json = json.dumps(items, ensure_ascii=False).replace("</", "<\\/")
short_hash = hashlib.md5(items_json.encode("utf-8")).hexdigest()[:8]

TEMPLATE = r"""<!doctype html><html><head><meta charset="utf-8"><title>Gemma suggestions review</title>
<style>
body{font-family:sans-serif;max-width:900px;margin:20px auto;padding:0 12px}
.item{border:1px solid #ccc;border-radius:8px;padding:10px;margin:14px 0}
.meta{color:#555;font-size:13px;margin-bottom:4px}
.line{font-size:20px;margin:6px 0;line-height:1.6}
.del{background:#ffd6d6;border-radius:3px;padding:0 2px}
.add{background:#d4f5d4;border-radius:3px;padding:0 2px}
.edit{margin:6px 0;padding:6px;background:#f6f6f6;border-radius:6px;font-size:18px}
.edit small{color:#666;font-size:12px;display:block;margin-left:26px}
.note{color:#555;font-size:12px;margin:4px 0}
textarea{width:100%;font-size:20px;min-height:60px}
button{font-size:14px;margin:6px 6px 0 0;padding:4px 10px}
input[type=text]{font-size:14px;margin-left:8px}
.on-accepted{background:#b8e6b8}.on-rejected{background:#f2b8b8}.on-unsure{background:#f5e3a1}
</style></head><body>
<h2>Gemma suggestions: review</h2>
<p>For each clip, <b>listen</b> and compare. <span class="del">Red</span> = the ASR word, <span class="add">green</span> = Gemma's suggestion.
Tick an edit <b>only if the narrator really said Gemma's word</b>. Leave it unticked to keep the ASR word.
You can also type directly in the text box (the final text for the clip is whatever is in the box).
Then press <b>Accept text</b>, <b>Reject clip</b> (bad audio, cut-off word) or <b>Unsure</b>.
Clips you never decide keep the ASR text. Progress saves in this browser. Press Download every 10 clips.</p>
<p id="prog"></p><button onclick="dl()">Download suggestions_decisions.jsonl</button>
<div id="list"></div>
<script>
const ITEMS = __ITEMS__;
const KEY = "amharic_suggestions_" + "__HASH__";
let state = {};
try { state = JSON.parse(localStorage.getItem(KEY) || "{}"); } catch (e) { state = {}; }
const $ = id => document.getElementById(id);
function mk(tag, cls, text) {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (text !== undefined) e.textContent = text;
  return e;
}
function st(it) {
  if (!state[it.id]) state[it.id] = {sel: [], text: it.asr.join(" "), status: "unreviewed", notes: "", manual: false};
  return state[it.id];
}
function save() { localStorage.setItem(KEY, JSON.stringify(state)); progress(); }
function progress() {
  const n = ITEMS.filter(i => st(i).status !== "unreviewed").length;
  $("prog").textContent = "Decided: " + n + " / " + ITEMS.length;
}
function overlap(a, b) {
  if (a.i1 === b.i1 && (a.i1 === a.i2 || b.i1 === b.i2)) return true;
  return Math.max(a.i1, b.i1) < Math.min(a.i2, b.i2);
}
function applied(it, sel) {
  const w = it.asr.slice();
  sel.map(k => it.edits[k]).sort((a, b) => b.i1 - a.i1).forEach(e => {
    w.splice(e.i1, e.i2 - e.i1, ...(e.to ? e.to.split(" ") : []));
  });
  return w.join(" ");
}
function asrLine(it) {
  const d = mk("div", "line");
  const ins = {}, marked = new Set();
  it.edits.forEach(e => {
    if (e.i1 === e.i2) ins[e.i1] = true;
    for (let i = e.i1; i < e.i2; i++) marked.add(i);
  });
  for (let i = 0; i <= it.asr.length; i++) {
    if (ins[i]) d.append(mk("span", "add", "＋"), " ");
    if (i < it.asr.length) d.append(mk("span", marked.has(i) ? "del" : "", it.asr[i]), " ");
  }
  return d;
}
const list = $("list");
ITEMS.forEach(it => {
  const s = st(it);
  const box = mk("div", "item");
  box.append(mk("div", "meta", it.id + " · " + (it.duration || "?") + " s"));
  const audio = document.createElement("audio");
  audio.controls = true; audio.preload = "none"; audio.src = it.audio;
  box.append(audio, asrLine(it));
  const ta = document.createElement("textarea");
  ta.lang = "am"; ta.value = s.text;
  const cbs = [];
  it.edits.forEach((e, k) => {
    const row = mk("div", "edit");
    const cb = document.createElement("input");
    cb.type = "checkbox"; cb.checked = s.sel.includes(k);
    cbs.push(cb);
    cb.onchange = () => {
      if (cb.checked) { s.sel = s.sel.filter(j => !overlap(it.edits[j], e)); s.sel.push(k); }
      else { s.sel = s.sel.filter(j => j !== k); }
      cbs.forEach((c, j) => { c.checked = s.sel.includes(j); });
      s.text = applied(it, s.sel); s.manual = false; ta.value = s.text; save();
    };
    row.append(cb, " ", mk("span", "del", e.from || "(nothing)"), "  →  ", mk("span", "add", e.to || "(delete)"),
               mk("small", "", "proposed by: " + e.sources.join(", ")));
    box.append(row);
  });
  it.notes.forEach(n => box.append(mk("div", "note", "Gemma note (" + n.source + "): " + n.note)));
  ta.oninput = () => { s.text = ta.value; s.manual = true; save(); };
  box.append(ta);
  const bar = mk("div");
  const label = mk("span", "meta", "");
  const btns = {};
  const names = {accepted: "✔ Accept text", rejected: "✖ Reject clip", unsure: "? Unsure"};
  const paint = () => {
    Object.keys(btns).forEach(v => btns[v].classList.toggle("on-" + v, s.status === v));
    label.textContent = " status: " + s.status;
  };
  Object.keys(names).forEach(v => {
    const b = mk("button", "", names[v]);
    b.onclick = () => { s.status = v; paint(); save(); };
    btns[v] = b; bar.append(b);
  });
  const note = document.createElement("input");
  note.type = "text"; note.placeholder = "notes"; note.value = s.notes;
  note.oninput = () => { s.notes = note.value; save(); };
  bar.append(label, note);
  paint();
  box.append(bar);
  list.append(box);
});
function dl() {
  const lines = ITEMS.map(it => {
    const s = st(it);
    return JSON.stringify({id: it.id, asr_text: it.asr.join(" "), text: s.text, status: s.status,
      applied_edits: s.sel.map(k => ({from: it.edits[k].from, to: it.edits[k].to, sources: it.edits[k].sources})),
      manual: s.manual, notes: s.notes});
  });
  const blob = new Blob([lines.join("\n") + "\n"], {type: "application/json"});
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob); a.download = "suggestions_decisions.jsonl"; a.click();
}
progress();
</script></body></html>"""

html = TEMPLATE.replace("__ITEMS__", items_json).replace("__HASH__", short_hash)
Path(args.out).write_text(html, encoding="utf-8")
print("wrote", args.out)