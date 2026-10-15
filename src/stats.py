"""Aggregate seed-level JSON results -> CSVs; paired statistics (Wilcoxon, paired t, Holm, effect size)."""
import os
import glob
import json
import numpy as np
import pandas as pd
from scipy import stats as st

ROOT = os.path.join(os.path.dirname(__file__), "..")
KEYS = ["macro_f1", "weighted_f1", "balanced_acc", "bin_f1", "pr_auc_att", "pr_auc_macro", "roc_auc_att", "step_fpr",
        "step_fnr", "minority_recall", "ep_far", "ep_far_benign", "ep_det_rate", "ep_det_rate_step", "ep_det_rate_ramp",
        "ep_delay", "ep_delay_step", "ep_delay_ramp", "ep_timely_rate", "ep_class_macro_f1", "loc_f1",
        "rec_0", "rec_1", "rec_2", "rec_3", "rec_4", "f1_0", "f1_1", "f1_2", "f1_3", "f1_4",
        "prec_0", "prec_1", "prec_2", "prec_3", "prec_4", "loc_precision", "loc_recall",
        "epk_far", "epk_far_benign", "epk_det_rate", "epk_det_rate_step", "epk_det_rate_ramp", "epk_delay",
        "epk_delay_step", "epk_delay_ramp", "epcal_det_rate", "epcal_far", "epcal_delay"]


def collect(stage, split="test"):
    rows = []
    for f in glob.glob(os.path.join(ROOT, "results", stage, "*", "*.json")):
        r = json.load(open(f))
        if split not in r:
            continue
        row = dict(system=r["system"], model=r["model"], seed=r["seed"], tag=r.get("tag", "default"),
                   params=r.get("params", np.nan), train_time_s=r.get("train_time_s", np.nan),
                   latency_ms=r.get("latency_ms", np.nan), throughput_steps_s=r.get("throughput_steps_s", np.nan),
                   peak_mem_mb=r.get("peak_mem_mb", np.nan), model_size_mb=r.get("model_size_mb", np.nan),
                   frac=r.get("frac", 1.0))
        for k in KEYS:
            row[k] = r[split].get(k, np.nan)
        rows.append(row)
    return pd.DataFrame(rows)


def ci95(x):
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    if len(x) < 2:
        return np.nan, np.nan
    h = st.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))
    return x.mean() - h, x.mean() + h


def aggregate(df, by=("system", "model", "tag"), keys=None):
    keys = keys or [k for k in KEYS + ["params", "train_time_s", "latency_ms", "throughput_steps_s", "peak_mem_mb",
                                       "model_size_mb"] if k in df]
    out = []
    for g, d in df.groupby(list(by)):
        row = dict(zip(by, g))
        row["n_seeds"] = len(d)
        for k in keys:
            v = d[k].astype(float)
            row[f"{k}_mean"] = v.mean()
            row[f"{k}_std"] = v.std(ddof=1) if len(v) > 1 else 0.0
            row[f"{k}_median"] = v.median()
            lo, hi = ci95(v)
            row[f"{k}_ci_lo"], row[f"{k}_ci_hi"] = lo, hi
        out.append(row)
    return pd.DataFrame(out)


def holm(p):
    p = np.asarray(p, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    m = len(p)
    run = 0
    for i, j in enumerate(order):
        run = max(run, min(1, (m - i) * p[j]))
        adj[j] = run
    return adj


def paired_tests(df, proposed, metric="macro_f1", higher_better=True, practical=0.005, by_system=True):
    """Compare proposed vs every other model on the same seeds. Returns per (system, baseline) rows."""
    rows = []
    systems = sorted(df.system.unique()) if by_system else [None]
    for s in systems:
        d = df if s is None else df[df.system == s]
        P = d[d.model == proposed].set_index("seed")[metric]
        for m in sorted(d.model.unique()):
            if m == proposed:
                continue
            B = d[d.model == m].set_index("seed")[metric]
            common = P.index.intersection(B.index)
            if len(common) >= 2:
                a, b = P[common].values.astype(float), B[common].values.astype(float)
                paired = True
            else:  # deterministic baseline (single run): compare against its value
                a = P.values.astype(float)
                b = np.full_like(a, float(B.mean()))
                paired = False
            diff = (a - b) if higher_better else (b - a)
            try:
                pw = st.wilcoxon(diff, zero_method="zsplit").pvalue if np.any(diff != 0) else 1.0
            except ValueError:
                pw = 1.0
            pt = st.ttest_1samp(diff, 0).pvalue if diff.std() > 0 else (0.0 if diff.mean() != 0 else 1.0)
            dz = diff.mean() / diff.std(ddof=1) if diff.std(ddof=1) > 0 else np.sign(diff.mean()) * np.inf
            rows.append(dict(system=s, baseline=m, metric=metric, n=len(diff), paired=paired, proposed_mean=a.mean(),
                             baseline_mean=b.mean(), diff_mean=diff.mean(), p_wilcoxon=pw, p_ttest=pt, cohen_dz=dz))
    out = pd.DataFrame(rows)
    if len(out):
        out["p_holm"] = holm(out["p_wilcoxon"].values)
        out["p_holm_t"] = holm(out["p_ttest"].values)

        def verdict(r):
            sig = r.p_holm_t < 0.05  # paired t-test (Holm); Wilcoxon reported as a robustness check
            if sig and r.diff_mean > 0:
                return "statistically superior"
            if sig and r.diff_mean < 0:
                return "inferior"
            if r.diff_mean > practical:
                return "practically superior"
            if r.diff_mean < -practical:
                return "inferior (not significant)"
            return "statistically indistinguishable"
        out["verdict"] = out.apply(verdict, axis=1)
    return out
