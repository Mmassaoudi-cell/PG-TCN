"""Validation-only comparison of the raw log-odds CUSUM and the calibrated-LLR CUSUM (split-half protocol)."""
import glob, sys, numpy as np, metrics as M
rows = []
for f in sorted(glob.glob("../results/rev_screen/*/bus_graph_tcn__*_valprob.npz")):
    d = np.load(f); p = d["prob"].astype(np.float32); cls, tau = d["cls"], d["tau"]
    rng = np.random.default_rng(0); idx = rng.permutation(len(cls)); A, B = idx[: len(idx) // 2], idx[len(idx) // 2:]
    out = {}
    for name in ["raw", "cal"]:
        if name == "raw":
            la, lb = M.attack_llr(p[A]), M.attack_llr(p[B])
        else:
            cal = M.fit_llr_calibration(p[A], cls[A], tau[A]); la, lb = M.calibrated_llr(p[A], cal), M.calibrated_llr(p[B], cal)
        h = M.calibrate_h(la, cls[A], tau[A])
        e = M.episode_metrics(M.cusum_alarms(lb, h), cls[B], tau[B], p.shape[1])
        out[name] = (round(e["det_rate"], 4), round(e["far"], 4), round(e["delay"], 1))
    print(f.split("rev_screen")[1], out)
