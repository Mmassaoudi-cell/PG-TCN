"""Paired detection delay on commonly detected attack episodes at a matched test FAR (1%), PG-TCN vs. strong
benchmarks; Poisson CIs of stream alarm rates and IEEE-118 alarms by stream class.
Outputs results/lib_final/paired_delay.csv, stream_ci.csv
"""
import os
import glob
import json
import numpy as np
import pandas as pd
from scipy import stats as st

import metrics as M

ROOT = os.path.join(os.path.dirname(__file__), "..")
OUT = os.path.join(ROOT, "results", "lib_final")
P = "bus_graph_tcn"
HS = np.concatenate([np.linspace(0.2, 20, 60), np.linspace(21, 400, 80)])


def alarms_at_far(f, target=0.01):
    j = json.load(open(f.replace("_testprob.npz", ".json")))
    d = np.load(f)
    llr = M.calibrated_llr(d["prob"].astype(np.float32), tuple(j["llr_cal"]))
    for h in HS:
        al = M.cusum_alarms(llr, h, k=j["cusum_k2"])
        if M.episode_metrics(al, d["cls"], d["tau"], llr.shape[1])["far"] <= target:
            return al, d["cls"], d["tau"]
    return al, d["cls"], d["tau"]


rows = []
for s in ["ieee14", "ieee30", "ieee57", "ieee118"]:
    for m, tag in [("lgbm_tok", "default"), ("gcn_gru", "matched"), ("rf", "default"), ("xgb", "default")]:
        for k in range(5):
            fp = os.path.join(ROOT, "results", "lib", s, f"{P}__final__seed{k}_testprob.npz")
            fb = os.path.join(ROOT, "results", "lib", s, f"{m}__{tag}__seed{k}_testprob.npz")
            if not (os.path.exists(fp) and os.path.exists(fb)):
                continue
            a1, cls, tau = alarms_at_far(fp)
            a2, _, _ = alarms_at_far(fb)
            att = np.isin(cls, M.ATT)
            both = att & (a1 >= tau) & (a1 < 200) & (a2 >= tau) & (a2 < 200)
            rows.append(dict(system=s, model=m if tag == "default" else "gcn_gru_matched", seed=k, n_common=int(both.sum()),
                             delay_prop=float((a1 - tau)[both].mean()), delay_base=float((a2 - tau)[both].mean()),
                             median_prop=float(np.median((a1 - tau)[both])), median_base=float(np.median((a2 - tau)[both]))))
d = pd.DataFrame(rows)
d.to_csv(os.path.join(OUT, "paired_delay.csv"), index=False)
print(d.groupby("model")[["n_common", "delay_prop", "delay_base", "median_prop", "median_base"]].mean().round(2))

# stream alarm rates with exact Poisson CIs (pooled over seeds per system)
srows = []
for f in glob.glob(os.path.join(ROOT, "results", "eval_lib", "*", f"{P}__final__seed*.json")):
    j = json.load(open(f))["stream"]
    srows.append(dict(system=os.path.basename(os.path.dirname(f)), alarms=j["alarms_k"], hours=j["hours"],
                      **{f"cls{c}": v for c, v in j.get("alarms_by_class_k", {}).items()}))
S = pd.DataFrame(srows).groupby("system").sum(numeric_only=True)
S["per_day"] = S.alarms / S.hours * 24
S["lo"] = [st.chi2.ppf(0.025, 2 * a) / 2 / h * 24 if a > 0 else 0.0 for a, h in zip(S.alarms, S.hours)]
S["hi"] = [st.chi2.ppf(0.975, 2 * a + 2) / 2 / h * 24 for a, h in zip(S.alarms, S.hours)]
S.to_csv(os.path.join(OUT, "stream_ci.csv"))
print(S.round(3))
