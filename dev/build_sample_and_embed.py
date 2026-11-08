"""Dev script: build the stratified sample and embed it (full 2560 dims) for the dimension ablation."""
import os, time, random, numpy as np, pandas as pd
from pathlib import Path
RAW = Path("data/raw/glassdoor_reviews.csv"); N = 20_000; SEED = 42
raw = pd.read_csv(RAW)
raw["review_id"] = np.arange(len(raw))
d = raw[raw["recommend"].isin(["v", "x"])].copy()
for c in ["headline", "pros", "cons"]:
    d[c] = d[c].fillna("").str.strip()
d = d.drop_duplicates(subset=["headline", "pros", "cons"])
sample = (d.groupby("recommend", group_keys=False)
           .sample(frac=N / len(d), random_state=SEED)
           .sort_values("review_id").reset_index(drop=True))
print(len(sample), sample.recommend.value_counts(normalize=True).round(3).to_dict())
texts = ("Headline: " + sample.headline + "\nPros: " + sample.pros + "\nCons: " + sample.cons).tolist()
for line in open(".env"):
    if "=" in line and not line.lstrip().startswith("#"):
        k, v = line.strip().split("=", 1); os.environ.setdefault(k.strip(), v.strip().strip("'\""))
from openai import OpenAI
c = OpenAI(base_url="https://litellm.vse.cz/v1/", api_key=os.environ["LLM_API_KEY"], timeout=120, max_retries=0)
out, B, t0, req, tok = [], 64, time.perf_counter(), 0, 0
for i in range(0, len(texts), B):
    for a in range(6):
        try:
            r = c.embeddings.create(model="qwen3-embedding:4b", input=texts[i:i+B]); break
        except Exception as e:
            if a == 5: raise
            time.sleep(min(30, 2 ** a * (0.5 + random.random()))); print("retry", type(e).__name__)
    req += 1; tok += r.usage.prompt_tokens
    out.append(np.array([x.embedding for x in r.data], dtype=np.float32))
    if req % 50 == 0: print(i + B, round(time.perf_counter() - t0, 1), "s", flush=True)
E = np.vstack(out)
Path("dev/cache").mkdir(exist_ok=True)
np.save("dev/cache/emb_full.npy", E); sample[["review_id"]].to_csv("dev/cache/emb_ids.csv", index=False)
print("done", E.shape, "requests", req, "tokens", tok, "seconds", round(time.perf_counter() - t0, 1))
