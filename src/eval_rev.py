"""Evaluate trained checkpoints (neural .pt or tabular .pkl) on the additional test sets:
attack continuum, unseen benign events, unseen real drivers, distribution shifts and hour-long streams.
Thresholds (raw and calibrated-LLR CUSUM) are the ones calibrated on validation during training.

python eval_rev.py --stage rev --system ieee14 --model bus_graph_tcn --tag final --seed 0
Output: results/rev_eval/<system>/<model>__<tag>__seed<k>.json
"""
import os
import json
import pickle
import argparse
import numpy as np
import torch

import metrics as M
from features import region_of, DATA
from train_nn import Runner, load_split, _summ, ROOT

FEAT = os.path.join(DATA, "feat")
SETS = ["ood_mimic", "ood_mix25", "ood_mix50", "ood_mix75", "ood_wocwg", "ood_noise1.5", "ood_load0.7", "ood_load1.3",
        "ood_bn_multi_step", "ood_bn_ext_gen_trip", "ood_bn_ext_line_out",
        "ood_zones58", "ood_pjm"]


def cusum_restart_count(llr, h, t0=M.T_EVAL):
    """Number of alarms per stream for a CUSUM that restarts after every alarm."""
    E, T = llr.shape
    S = np.zeros(E)
    n = np.zeros(E, int)
    for t in range(t0, T):
        S = np.maximum(0, S + llr[:, t])
        hit = S >= h
        n += hit
        S[hit] = 0
    return n


class Predictor:
    def __init__(self, system, res, path):
        self.system = system
        if path.endswith(".pt"):
            self.kind = "nn"
            self.r = Runner(system, res["model"], res["hp"], res["seed"])
            st = torch.load(path, map_location="cuda")
            self.r.net.load_state_dict(st)
        else:
            import tabular as TB
            import __main__
            __main__.Std = TB.Std          # tabular models were pickled from tabular.py run as __main__
            self.kind = "tab"
            d = pickle.load(open(path, "rb"))
            self.clf, self.std = d["clf"], d["std"]
            self.TB = TB
            TB.TOK["fz"] = None
            if d["feat"] == "tok":
                fz = pickle.load(open(os.path.join(FEAT, f"{system}_featurizer.pkl"), "rb"))
                fz.r = region_of(system)
                TB.TOK["fz"] = fz

    def __call__(self, X, long=False):
        if self.kind == "nn":
            return self.r.predict(X, bs=2 if long else 128)
        S = self.TB.summaries(X, bs=4 if long else 100)
        return self.TB.predict_episodes(self.clf, S, self.std, bs=2 if long else 100)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="lib")
    ap.add_argument("--system", default="ieee14")
    ap.add_argument("--model", default="bus_graph_tcn")
    ap.add_argument("--tag", default="final")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sets", default=",".join(SETS))
    a = ap.parse_args()
    base = os.path.join(ROOT, "results", a.stage, a.system, f"{a.model}__{a.tag}__seed{a.seed}")
    out = os.path.join(ROOT, "results", f"eval_{a.stage}", a.system, f"{a.model}__{a.tag}__seed{a.seed}.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    done = json.load(open(out)) if os.path.exists(out) else {}
    res = json.load(open(base + ".json"))
    path = base + ".pt" if os.path.exists(base + ".pt") else base + ".pkl"
    pred = Predictor(a.system, res, path)
    h, h_cal, cal = res["cusum_h"], res["cusum_h_cal"], tuple(res["llr_cal"])
    D = {k: res[k] for k in ("cusum_h", "llr_cal", "cusum_h_cal", "cusum_k2", "cusum_h2")}
    for name in [s for s in a.sets.split(",") if s]:
        if name in done or not os.path.exists(os.path.join(FEAT, f"{a.system}_{name}_test.npy")):
            continue
        X, m = load_split(a.system, name, "test")
        p = pred(X)
        o = _summ(p, m, h, cal, h_cal, D)
        if "ev_kind" in m:   # false-alarm rate per benign event kind (episodes without attack)
            al = M.cusum_alarms(M.calibrated_llr(p, cal), D["cusum_h2"], k=D["cusum_k2"])
            for kd in np.unique(m["ev_kind"]):
                sel = (m["ev_kind"] == kd) & ~np.isin(m["cls"], M.ATT)
                if sel.any():
                    o[f"far_kind{int(kd)}"] = float((al[sel] < p.shape[1]).mean())
        o.pop("cm", None)
        done[name] = o
        json.dump(done, open(out, "w"))
        print(a.system, a.model, a.seed, name, "det", round(o["ep_det_rate"], 4), "far", round(o["ep_far"], 4),
              "F1", round(o["macro_f1"], 4), flush=True)
    if "stream" not in done and os.path.exists(os.path.join(FEAT, f"{a.system}_ood_stream_test.npy")):
        X, m = load_split(a.system, "ood_stream", "test")
        p = pred(X, long=True)
        hours = X.shape[0] * (X.shape[1] - M.T_EVAL) / 3600.0
        n_raw = cusum_restart_count(M.attack_llr(p), h)
        n_cal = cusum_restart_count(M.calibrated_llr(p, cal), h_cal)
        n_k = M.stream_alarms(M.calibrated_llr(p, cal), D["cusum_h2"], D["cusum_k2"])
        done["stream"] = dict(hours=hours, alarms_raw=int(n_raw.sum()), alarms_cal=int(n_cal.sum()),
                              alarms_per_day_raw=float(n_raw.sum() / hours * 24), alarms_per_day_cal=float(n_cal.sum() / hours * 24),
                              frac_streams_alarm_raw=float((n_raw > 0).mean()), frac_streams_alarm_cal=float((n_cal > 0).mean()),
                              alarms_k=int(n_k.sum()), alarms_per_day_k=float(n_k.sum() / hours * 24),
                              frac_streams_alarm_k=float((n_k > 0).mean()),
                              alarms_by_class_k={int(c): int(n_k[m["cls"] == c].sum()) for c in np.unique(m["cls"])},
                              step_attack_frac=float(np.isin(p[:, M.T_EVAL:].argmax(-1), M.ATT).mean()))
        json.dump(done, open(out, "w"))
        print(a.system, a.model, a.seed, "stream", done["stream"], flush=True)


if __name__ == "__main__":
    main()
