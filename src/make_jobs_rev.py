"""Job lists for the revision campaign (stage 'rev'). Writes jobs_rev_<group>.txt in the project root.

python make_jobs_rev.py <horizons e.g. 60,120>
"""
import os
import sys
import json

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
SYS = ["ieee14", "ieee30", "ieee57", "ieee118"]
HOR = [int(x) for x in sys.argv[1].split(",")]
BASE = dict(lr=0.0035998822266397157, hid=64, drop=0.003914683519333911, wd=1.0412648986213771e-05, cw_pow=0.5,
            loc_w=0.5, bs=64, amp=True, epochs=30, patience=6)
FINAL = {**BASE, "horizons": HOR}


def q(d):
    return json.dumps(json.dumps(d))


def nn(system, model, seed, tag, hp, extra=""):
    out = f"results/lib/{system}/{model}__{tag}__seed{seed}.json"
    return f"{out} ||| python -u train_nn.py --system {system} --model {model} --seed {seed} --stage lib --tag {tag} --hp {q(hp)} --save_all {extra}"


def tab(system, model, seed, feat="flat"):
    name = model + ("_tok" if feat == "tok" else "")
    out = f"results/lib/{system}/{name}__default__seed{seed}.json"
    return f"{out} ||| python -u tabular.py --system {system} --model {model} --seed {seed} --stage lib --feat {feat} --save_all"


def write(name, lines):
    open(os.path.join(ROOT, f"jobs_rev_{name}.txt"), "w", encoding="utf-8").write("\n".join(lines) + "\n")
    print(name, len(lines))


# 1. proposed model, 10 seeds on every system
write("final", [nn(s, "bus_graph_tcn", k, "final", FINAL) for s in SYS for k in range(10)])
# 2. ablations on IEEE-14 and IEEE-57, 5 seeds (paired with seeds 0-4 of the full model)
abl = {"abl_no_horizon": (BASE, ""), "abl_no_graph": ({**FINAL, "use_graph": False}, ""),
       "abl_no_velocity": (FINAL, "--ablate no_vel"), "abl_no_eta": (FINAL, "--ablate no_eta"),
       "abl_no_innovation": (FINAL, "--ablate no_innov"), "abl_no_wind_residual": (FINAL, "--ablate no_wres"),
       "abl_no_localisation_loss": ({**FINAL, "loc_w": 0.0}, "")}
write("ablation", [nn(s, "bus_graph_tcn", k, t, hp, ex) for s in ["ieee14", "ieee57"] for t, (hp, ex) in abl.items()
                   for k in range(5)])
# 3. deep baselines with the tuned hyper-parameters (12 trials each, IEEE-14 validation), 5 seeds, all systems
deep = []
for m in ["flat_tcn", "gru", "lstm", "transformer", "gcn_gru", "mlp", "cnn1d", "stdcnn", "mamba"]:
    tf = os.path.join(ROOT, "results", "tuning", "ieee14", f"nn_{m}{'_tok' if m == 'gcn_gru' else ''}.json")
    best = json.load(open(tf))["best"] if os.path.exists(tf) else {}
    hp = {k: v for k, v in best.items()}
    hp.update(epochs=30, patience=6)
    if m == "gcn_gru":
        hp["horizons"] = HOR                       # identical multi-horizon bus tokens as PG-TCN
    deep += [nn(s, m, k, "default", hp) for s in SYS for k in range(5)]
    if m == "gcn_gru":                             # capacity-matched graph baseline (about the PG-TCN parameter count)
        deep += [nn(s, m, k, "matched", {**hp, "hid": 112}) for s in SYS for k in range(5)]
write("deep", deep)
# 4. tabular baselines (flat window statistics) and a tree on the multi-horizon bus tokens
write("tab", [tab(s, m, k) for m in ["rf", "lgbm", "xgb", "et", "cat", "lr", "svm", "knn", "hgb"] for s in SYS for k in range(5)]
      + [tab(s, "lgbm", k, "tok") for s in SYS for k in range(5)])

# 5. source DQN (5 seeds) and deterministic sequential tests, refitted on the same data
TESTS = "ood_mimic,ood_mix25,ood_mix50,ood_mix75,ood_wocwg,ood_noise1.5,ood_load0.7,ood_load1.3,ood_bn_multi_step,ood_bn_ext_gen_trip,ood_bn_ext_line_out,ood_zones58,ood_pjm"
write("classical", [f"results/lib/{s}/source_dqn__default__seed{k}.json ||| python -u classical.py --system {s} --seed {k} --stage lib --tests {TESTS}"
                    for s in SYS for k in range(5)])
