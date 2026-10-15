"""Write FINAL_MODEL_CONFIG.yaml from VALIDATION-only tuning records (run once, before any test evaluation)."""
import os
import json
import hashlib
import datetime
import yaml

ROOT = os.path.join(os.path.dirname(__file__), "..")
T = os.path.join(ROOT, "results", "tuning", "ieee14")
out = os.path.join(ROOT, "FINAL_MODEL_CONFIG.yaml")
assert not os.path.exists(out), "config already frozen"


def best(m):
    f = os.path.join(T, f"nn_{m}.json")
    return json.load(open(f))["best"] if os.path.exists(f) else {}


c1 = best("bus_graph_tcn")
deep = ["flat_tcn", "gcn_gru", "stdcnn", "transformer", "gru", "lstm", "cnn1d", "mlp", "mamba"]
deep_hp = {}
for m in deep:
    b = best(m)
    hp = {k: v for k, v in b.items() if k in ("lr", "hid", "drop", "wd", "cw_pow")}
    if m == "mamba":
        hp["amp"] = False
    deep_hp[m] = hp
cfg = {
    "frozen_at": datetime.datetime.now().isoformat(timespec="seconds"),
    "model": {"name": "bus_graph_tcn", "display_name": "PG-TCN (Physics-structured Graph TCN + learned-LLR CUSUM)",
              "architecture": {"token_features": 24, "temporal_blocks": 4, "kernel": 3, "dilations": [1, 2, 4, 8],
                               "graph_propagation": "h + GELU(A_hat h W) after every temporal block",
                               "pooling": "attention + max over buses", "heads": ["5-class softmax", "bus-level sigmoid (localisation)"],
                               "receptive_field_scans": 31},
              "hyperparameters": {"lr": c1["lr"], "hid": c1["hid"], "drop": c1["drop"], "wd": c1["wd"],
                                  "cw_pow": c1["cw_pow"], "loc_w": c1["loc_w"], "bs": 64, "amp": True},
              "selection_metric": "validation step-level 5-class macro-F1"},
    "data": {"systems": ["ieee14", "ieee30", "ieee57", "ieee118"],
             "split": "temporal: train anchors 2012, val Jan-May 2013, test Jun-Dec 2013",
             "features": ["standardised whitened CKF innovation (M)", "posterior measurement velocity (M)",
                          "wind correlation-consistency residual (n_w)", "standardised log eta (1)"],
             "preprocessing": "channel standardisation, wind-residual regression and eta scaling fitted on TRAIN normal episodes",
             "class_balancing": "class-weighted cross-entropy, weights (N/(K N_c))^cw_pow",
             "augmentation": "none"},
    "training": {"optimizer": "AdamW", "scheduler": "OneCycleLR (pct_start 0.1)", "loss": "weighted CE + loc_w * BCE (bus head)",
                 "max_epochs": 30, "patience": 6, "early_stopping": "validation macro-F1", "grad_clip": 1.0,
                 "mixed_precision": "bf16 autocast"},
    "decision": {"statistic": "CUSUM on clipped attack log-odds log(p3+p4)/(1-p3-p4)",
                 "threshold": "smallest h with validation episode FAR <= 1% (calibrated per trained model)",
                 "far_target": 0.01},
    "evaluation": {"seeds": list(range(10)), "benchmark_seeds": list(range(5)),
                   "ood_sets": ["ood_mimic", "ood_wocwg", "ood_noise1.5", "ood_load0.7", "ood_load1.3"],
                   "primary_metric": "test step-level 5-class macro-F1 (scans 52..199)"},
    "benchmarks": {"deep": deep, "deep_hp": "read from results/tuning/ieee14/nn_<model>.json (validation-only Optuna, 6 trials; flat_tcn 12)",
                   "tabular": ["lr", "svm", "knn", "rf", "et", "xgb", "lgbm", "cat", "hgb"],
                   "classical": ["source_dqn", "eta_shewhart", "eta_cusum", "chi2_cusum"]},
    "ablations": {"no_graph": {"hp": {"use_graph": False}}, "no_velocity": {"ablate": "no_vel"},
                  "no_wind_residual": {"ablate": "no_wres"}, "no_eta": {"ablate": "no_eta"},
                  "no_innovation": {"ablate": "no_innov"}, "no_localisation_loss": {"hp": {"loc_w": 0.0}},
                  "velocity_and_residual_removed": {"ablate": "no_vel+no_wres"}},
    "low_data": {"systems": ["ieee14", "ieee118"], "fractions": [0.01, 0.05, 0.1, 0.25, 0.5], "seeds": [0, 1, 2],
                 "baselines_nn": ["flat_tcn", "gcn_gru"], "baselines_tab": ["lgbm"]},
}
s = yaml.safe_dump(cfg, sort_keys=False)
cfg_hash = hashlib.sha256(s.encode()).hexdigest()[:16]
open(out, "w").write(f"# FROZEN before test evaluation. sha256[:16]={cfg_hash}\n" + s)
print(s)
print("hash", cfg_hash)
