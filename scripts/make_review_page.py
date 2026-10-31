import json, random
from pathlib import Path

seg = [json.loads(l) for l in open("data/transcripts/segments.jsonl", encoding="utf-8")]
asr = {}
for l in open("data/transcripts/asr.jsonl", encoding="utf-8"):
    r = json.loads(l); asr[r["id"]] = r["asr_text"]

ids = [s["id"] for s in seg]
random.seed(0)
blind = set(random.sample(ids, 20))
items = [{"id": s["id"], "duration": s["duration"], "asr": asr.get(s["id"], ""),
          "blind": s["id"] in blind,
          "audio": f"data/interim/tts_audio/segments/{s['id']}.wav"} for s in seg]

TEMPLATE = """<!doctype html><html><head><meta charset="utf-8"><title>Transcript review</title>
<style>
body{font-family:sans-serif;max-width:900px;margin:20px auto;padding:0 12px}
.item{border:1px solid #ccc;border-radius:8px;padding:10px;margin:12px 0}
.blind{border-color:#d9822b;background:#fff7ee}
textarea{width:100%;font-size:20px;min-height:60px;margin-top:6px}
.meta{color:#555;font-size:13px;margin-bottom:4px}
.asr{background:#eef;padding:6px;margin-top:6px;font-size:18px}
button,select,input{font-size:14px;margin:4px 6px 0 0}
</style></head><body>
<h2>Transcript review</h2>
<p><b>Orange clips are BLIND:</b> listen and type the transcript from scratch <b>before</b> pressing "Show ASR text".
For all other clips, listen and fix the pre-filled text. Set a status for each clip:
<b>accepted</b> (text matches the audio exactly), <b>rejected</b> (bad clip: cut-off word, noise, non-narration), or <b>unsure</b>.
Progress saves in this browser automatically. Press Download now and then to keep a copy.</p>
<p id="prog"></p><button onclick="dl()">Download reviewed.jsonl</button>
<div id="list"></div>
<script>
const ITEMS = __ITEMS__;
const KEY = "amharic_review_v1";
let state = JSON.parse(localStorage.getItem(KEY) || "{}");
function get(it){ return state[it.id] || (state[it.id] = {text: it.blind ? "" : it.asr, status: "unreviewed", notes: "", revealed: false}); }
function prog(){ const n = ITEMS.filter(i => get(i).status !== "unreviewed").length;
  document.getElementById("prog").textContent = "Reviewed: " + n + " / " + ITEMS.length; }
function save(){ localStorage.setItem(KEY, JSON.stringify(state)); prog(); }
const list = document.getElementById("list");
ITEMS.forEach(it => {
  const s = get(it);
  const d = document.createElement("div"); d.className = "item" + (it.blind ? " blind" : "");
  d.innerHTML = '<div class="meta"><b>' + it.id + '</b> · ' + it.duration + ' s' + (it.blind ? ' · BLIND: type from scratch' : '') + '</div>' +
                '<audio controls preload="none" src="' + it.audio + '"></audio>';
  const ta = document.createElement("textarea"); ta.lang = "am"; ta.value = s.text;
  ta.oninput = () => { s.text = ta.value; save(); };
  const sel = document.createElement("select");
  ["unreviewed","accepted","rejected","unsure"].forEach(v => { const o = document.createElement("option"); o.value = v; o.textContent = v; sel.appendChild(o); });
  sel.value = s.status; sel.onchange = () => { s.status = sel.value; save(); };
  const notes = document.createElement("input"); notes.placeholder = "notes"; notes.value = s.notes;
  notes.oninput = () => { s.notes = notes.value; save(); };
  d.append(ta, sel, notes);
  if (it.blind) {
    const b = document.createElement("button"); b.textContent = "Show ASR text";
    const a = document.createElement("div"); a.className = "asr"; a.textContent = it.asr; a.style.display = s.revealed ? "block" : "none";
    b.onclick = () => { s.revealed = true; a.style.display = "block"; save(); };
    d.append(b, a);
  }
  list.appendChild(d);
});
function dl(){
  const lines = ITEMS.map(it => { const s = get(it);
    return JSON.stringify({id: it.id, duration: it.duration, asr_text: it.asr, text: s.text, status: s.status, notes: s.notes, blind: it.blind}); });
  const blob = new Blob([lines.join("\\n") + "\\n"], {type: "application/json"});
  const a = document.createElement("a"); a.href = URL.createObjectURL(blob); a.download = "reviewed.jsonl"; a.click();
}
prog();
</script></body></html>"""

html = TEMPLATE.replace("__ITEMS__", json.dumps(items, ensure_ascii=False).replace("</", "<\\/"))
Path("review.html").write_text(html, encoding="utf-8")
print("wrote review.html with", len(items), "clips,", len(blind), "of them blind")