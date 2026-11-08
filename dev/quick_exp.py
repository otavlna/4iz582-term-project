import time, numpy as np, pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
raw = pd.read_csv("data/raw/glassdoor_reviews.csv", usecols=["firm","recommend"])
ids = pd.read_csv("dev/cache/emb_ids.csv")["review_id"].to_numpy()
E = np.load("dev/cache/emb_full.npy")
y = (raw.recommend.to_numpy()[ids]=="v").astype(int); g = raw.firm.to_numpy()[ids]
tr, te = next(StratifiedGroupKFold(5, shuffle=True, random_state=42).split(E, y, g))
cv = StratifiedGroupKFold(3, shuffle=True, random_state=42)
def nrm(M): return M/np.linalg.norm(M,axis=1,keepdims=True)
for k in [128, 512, 1024, 2560]:
    X = nrm(E[tr,:k])
    for C in [0.01, 0.05]:
        t=time.perf_counter()
        s = cross_val_score(Pipeline([("s",StandardScaler()),("c",LogisticRegression(C=C,max_iter=3000))]), X, y[tr], groups=g[tr], cv=cv, scoring="roc_auc", n_jobs=-1)
        print(k, C, round(s.mean(),4), round(time.perf_counter()-t,1),"s", flush=True)
X = nrm(E[tr,:1024]); t=time.perf_counter()
h = HistGradientBoostingClassifier(max_iter=400, early_stopping=True, n_iter_no_change=20, learning_rate=0.1, random_state=42).fit(X[:int(len(X)*.66)], y[tr][:int(len(X)*.66)])
print("HGB 1024 single fit", round(time.perf_counter()-t,1), "s, iters", h.n_iter_)
