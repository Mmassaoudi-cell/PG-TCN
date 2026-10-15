"""Reproduce the source paper (SP) Tables I-IV with the coupled CKF + DQN detector.
Resumable: every unit of work is cached as JSON in results/reproduction/units/.

Outputs -> results/reproduction/*.csv
"""
import os
import sys
import json
import time
import numpy as np
import pandas as pd
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
sys.path.insert(0, HERE)
from dqn_detector import DQNDetector, detection_metrics, T0  # noqa: E402

DATA = os.path.join(ROOT, "data")
OUT = os.path.join(ROOT, "results", "reproduction")
UNITS = os.path.join(OUT, "units_v2")  # v2 = counterfactual-replay DQN
os.makedirs(UNITS, exist_ok=True)
os.makedirs(os.path.join(OUT, "models_v2"), exist_ok=True)
C_BETAS = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5]
SEEDS = [0, 1, 2, 3, 4]


def load(name):
    d = np.load(os.path.join(DATA, name + ".npz"))
    ok = d["conv"].astype(bool)
    return dict(eta=d["eta"][ok], tau=d["tau"][ok], cls=d["cls"][ok], T=d["eta"].shape[1])


def is_att(d):
    return np.isin(d["cls"], [3, 4])


def evaluate(det, d):
    t0 = time.time()
    al = det.alarm_times(d["eta"])
    lat = (time.time() - t0) / d["eta"].size
    m = detection_metrics(al, d["tau"], is_att(d), d["T"])
    m["dqn_latency_us_per_step"] = lat * 1e6
    return {k: float(v) for k, v in m.items()}


def unit(name, fn):
    p = os.path.join(UNITS, name + ".json")
    if os.path.exists(p):
        return json.load(open(p))
    r = fn()
    json.dump(r, open(p, "w"))
    return r


def model_path(tag, seed):
    return os.path.join(OUT, "models_v2", f"dqn_{tag}_cb{CB[tag]}_seed{seed}.pt")


CB = {}


def trained(tag, cb, seed, tr):
    CB[tag] = cb
    p = model_path(tag, seed)
    det = DQNDetector(c_beta=cb, seed=seed)
    if os.path.exists(p):
        st = torch.load(p)
        det.q.load_state_dict(st["q"]); det.ref = st["ref"]; det.train_time = st["train_time"]
        return det
    det.fit(tr["eta"], tr["tau"], is_att(tr))
    torch.save(dict(q=det.q.state_dict(), ref=det.ref, train_time=det.train_time, c_beta=cb), p)
    return det


def select_cbeta(tag, tr, va, seed=0):
    rows = []
    for cb in C_BETAS:
        def f():
            det = DQNDetector(c_beta=cb, seed=seed).fit(tr["eta"], tr["tau"], is_att(tr))
            m = evaluate(det, va)
            m.update(c_beta=cb, train_time_s=det.train_time)
            return m
        m = unit(f"cbeta_{tag}_{cb}", f)
        rows.append(m)
        print(f"   [{tag}] c_beta={cb}: val timely={m['timely_rate']:.4f} FAR={m['far']:.4f} delay={m['delay_s']:.3f}", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, f"cbeta_selection_{tag}.csv"), index=False)
    # SP Pareto rule: FAR < 5% and delay < 5 s; if no c_beta meets both, keep FAR < 5% and minimise delay;
    # if none has FAR < 5%, minimise FAR.
    feas = df[(df.far < 0.05) & (df.delay_s < 5.0)]
    if not len(feas):
        feas = df[df.far < 0.05]
    if len(feas):
        best = feas.sort_values(["delay_s", "far", "timely_rate"], ascending=[True, True, False]).iloc[0]
    else:
        best = df.sort_values(["far", "delay_s"]).iloc[0]
    return float(best.c_beta)


def threshold_rule(tr, va, te):
    """Diagnostic: one-parameter Shewhart rule on eta (threshold chosen on VAL: FAR<5%, min delay)."""
    def alarms(eta, h):
        s = eta >= h
        s[:, :T0] = False
        return np.where(s.any(1), s.argmax(1), eta.shape[1])
    ref = np.median(tr["eta"][:, T0:60])
    best = None
    for k in np.linspace(1.05, 6, 100):
        m = detection_metrics(alarms(va["eta"], k * ref), va["tau"], is_att(va), va["T"])
        if m["far"] < 0.05 and (best is None or m["delay_s"] < best[1]["delay_s"] - 1e-9):
            best = (k, m)
    k = best[0] if best else 6
    m = detection_metrics(alarms(te["eta"], k * ref), te["tau"], is_att(te), te["T"])
    return {**{a: float(b) for a, b in m.items()}, "threshold_k": float(k)}


def main(systems):
    rows = []
    for s in systems:
        tr, va, te = load(f"{s}_src_train"), load(f"{s}_src_val"), load(f"{s}_src_test")
        cb = select_cbeta(s, tr, va)
        print(f"[{s}] selected c_beta={cb}", flush=True)
        for seed in SEEDS:
            def f():
                det = trained(s, cb, seed, tr)
                m = evaluate(det, te)
                m.update(system=s, seed=seed, c_beta=cb, train_time_s=det.train_time, method="SOURCE_METHOD_REPRODUCTION")
                return m
            m = unit(f"test_{s}_cb{cb}_seed{seed}", f)
            rows.append(m)
            print(f"[{s}] seed {seed}: timely={m['timely_rate']:.4f} FAR={m['far']:.4f} delay={m['delay_s']:.4f} "
                  f"P={m['precision']:.4f} R={m['recall']:.4f}", flush=True)
        m = unit(f"thr_{s}", lambda: {**threshold_rule(tr, va, te), "system": s, "seed": -1, "method": "eta_threshold_rule"})
        rows.append(m)
        pd.DataFrame(rows).to_csv(os.path.join(OUT, "table_I_IV_seeds.csv"), index=False)

    if "ieee14" not in systems:
        return
    # ---------------- SP Table II: correlation train/test matrix (IEEE 14) -----------------
    wc_tr = load("ieee14_src_train")
    wo_tr, wo_va = load("ieee14_srcwocwg_train"), load("ieee14_srcwocwg_val")
    tests = {("WCWG", "WCWG"): "ieee14_src_test", ("WCWG", "WOCWG"): "ieee14_src_on_wocwg_test",
             ("WOCWG", "WOCWG"): "ieee14_srcwocwg_test", ("WOCWG", "WCWG"): "ieee14_srcwocwg_on_wcwg_test"}
    cb_wc = select_cbeta("ieee14", wc_tr, load("ieee14_src_val"))
    cb_wo = select_cbeta("ieee14wocwg", wo_tr, wo_va)
    rows = []
    for seed in SEEDS:
        for (trn, tst), name in tests.items():
            def f():
                det = trained("ieee14", cb_wc, seed, wc_tr) if trn == "WCWG" else trained("ieee14wocwg", cb_wo, seed, wo_tr)
                m = evaluate(det, load(name))
                m.update(train=trn, test=tst, seed=seed)
                return m
            rows.append(unit(f"tab2_{trn}_{tst}_cb{cb_wc}_{cb_wo}_{seed}", f))
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "table_II_seeds.csv"), index=False)

    # ---------------- SP Table III: noise x load grid (IEEE 14, WCWG detectors) -------
    rows = []
    for ns in [0.5, 1.0, 1.5]:
        for ls in [0.7, 1.0, 1.3]:
            for seed in SEEDS:
                def f():
                    det = trained("ieee14", cb_wc, seed, wc_tr)
                    m = evaluate(det, load(f"ieee14_src_n{ns}_l{ls}_test"))
                    m.update(noise=ns, load=ls, seed=seed)
                    return m
                rows.append(unit(f"tab3_{ns}_{ls}_cb{cb_wc}_{seed}", f))
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "table_III_seeds.csv"), index=False)


def fixed_op_diag():
    """Diagnostic: single fixed operating point (IEEE 14) -> is the published ~100% attainable?"""
    tr, va, te = load("ieee14_srcfixed_train"), load("ieee14_srcfixed_val"), load("ieee14_srcfixed_test")
    cb = select_cbeta("ieee14fixed", tr, va)
    rows = []
    for seed in SEEDS:
        def f():
            det = trained("ieee14fixed", cb, seed, tr)
            m = evaluate(det, te)
            m.update(seed=seed, c_beta=cb, method="SOURCE_METHOD_REPRODUCTION (fixed operating point)")
            return m
        rows.append(unit(f"fixed_test_cb{cb}_{seed}", f))
    rows.append(unit("fixed_thr", lambda: {**threshold_rule(tr, va, te), "seed": -1, "method": "eta_threshold_rule (fixed op)"}))
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "fixed_op_diagnostic.csv"), index=False)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "fixed":
        fixed_op_diag()
        sys.exit(0)
    systems = sys.argv[1].split(",") if len(sys.argv) > 1 else ["ieee14", "ieee30", "ieee57", "ieee118"]
    main(systems)
