import os
import requests
from dotenv import load_dotenv

load_dotenv()
key = os.environ.get("OPENROUTER_API_KEY")
model = os.environ.get("OPENROUTER_MODEL", "google/gemma-4-26b-a4b-it:free")
if not key:
    raise SystemExit("OPENROUTER_API_KEY is missing in .env")
H = {"Authorization": f"Bearer {key}"}

print("== account / key status ==")
r = requests.get("https://openrouter.ai/api/v1/key", headers=H, timeout=60)
print("HTTP", r.status_code)
try:
    d = r.json().get("data", r.json())
    for k in ["is_free_tier", "limit", "limit_remaining", "usage", "usage_daily", "free_model_daily_requests"]:
        if k in d:
            print(f"  {k}: {d[k]}")
except Exception:
    print(r.text[:300])

print("\n== one tiny test request to", model, "==")
r = requests.post(
    "https://openrouter.ai/api/v1/chat/completions", headers=H, timeout=120,
    json={"model": model, "max_tokens": 20,
          "messages": [{"role": "user", "content": "Reply with the word OK."}]},
)
print("HTTP", r.status_code)
for h in ["X-RateLimit-Limit", "X-RateLimit-Remaining", "X-RateLimit-Reset", "Retry-After"]:
    if h in r.headers:
        print(f"  {h}: {r.headers[h]}")
print("  body:", r.text[:600])