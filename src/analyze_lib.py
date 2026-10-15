"""Analysis of the final campaign (stage 'lib'): aggregates, paired statistics, WTL, LaTeX tables and figures.

python analyze_lib.py
Outputs: results/lib_final/*.csv, BENCHMARK_WTL.csv, paper/tables/*.tex, figures/*.pdf|png
"""
import os
import glob
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import stats as S
import metrics as M

ROOT = os.path.join(os.path.dirname(__file__), "..")
OUT = os.path.join(ROOT, "results", "lib_final")
TAB = os.path.join(ROOT, "paper", "tables")
FIG = os.path.join(ROOT, "figures")
for d in (OUT, TAB, FIG):
    os.makedirs(d, exist_ok=True)
P = "bus_graph_tcn"
SYS = ["ieee14", "ieee30", "ieee57", "ieee118"]
SYSN = {"ieee14": "IEEE-14", "ieee30": "IEEE-30", "ieee57": "IEEE-57", "ieee118": "IEEE-118"}
BINARY = ["source_dqn", "eta_shewhart", "eta_cusum", "chi2_cusum"]
NAME = {P: "PG-TCN (proposed)", "flat_tcn": "TCN", "gcn_gru": "GCN-GRU", "gcn_gru_matched": "GCN-GRU (matched)",
        "gru": "GRU", "lstm": "LSTM", "transformer": "Transformer", "mlp": "MLP", "cnn1d": "1D-CNN", "stdcnn": "ST-DCNN",
        "mamba": "Mamba-SSM", "rf": "Random forest", "et": "Extra trees", "xgb": "XGBoost", "lgbm": "LightGBM",
        "lgbm_tok": "LightGBM (bus tokens)", "cat": "CatBoost", "hgb": "HistGB", "lr": "Logistic reg.", "svm": "RBF-SVM",
        "knn": "$k$-NN", "source_dqn": "Source DQN~\\cite{ran2026}", "eta_shewhart": "$\\eta$-Shewhart",
        "eta_cusum": "$\\eta$-CUSUM", "chi2_cusum": "$\\chi^2$-CUSUM"}
PNAME = {k: v.replace("~\\cite{ran2026}", "").replace("$k$-NN", "k-NN").replace("$\\eta$", "η").replace("$\\chi^2$", "χ²")
         for k, v in NAME.items()}
FAM = {P: "Proposed", "flat_tcn": "Deep", "gru": "Deep", "lstm": "Deep", "transformer": "Deep", "mlp": "Deep",
       "cnn1d": "Deep", "stdcnn": "Deep", "mamba": "Deep", "gcn_gru": "Graph", "gcn_gru_matched": "Graph",
       "rf": "Tabular", "et": "Tabular", "xgb": "Tabular", "lgbm": "Tabular", "lgbm_tok": "Tabular", "cat": "Tabular",
       "hgb": "Tabular", "lr": "Tabular", "svm": "Tabular", "knn": "Tabular", "source_dqn": "Source/QCD",
       "eta_shewhart": "Source/QCD", "eta_cusum": "Source/QCD", "chi2_cusum": "Source/QCD"}
ORDER = [P, "flat_tcn", "gru", "lstm", "transformer", "mamba", "stdcnn", "cnn1d", "mlp", "gcn_gru", "gcn_gru_matched",
         "lgbm_tok", "lgbm", "xgb", "rf", "et", "cat", "hgb", "lr", "svm", "knn", "source_dqn", "eta_shewhart", "eta_cusum",
         "chi2_cusum"]
plt.rcParams.update({"font.family": "serif", "font.size": 8, "axes.linewidth": 0.6, "figure.dpi": 200,
                     "axes.grid": True, "grid.alpha": 0.3, "grid.linewidth": 0.4})


def save(fig, name):
    fig.savefig(os.path.join(FIG, name + ".pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(FIG, name + ".png"), bbox_inches="tight", dpi=250)
    plt.close(fig)


def load(stage="lib", split="test"):
    df = S.collect(stage, split)
    df.loc[(df.model == "gcn_gru") & (df.tag == "matched"), "model"] = "gcn_gru_matched"
    main = df[((df.model == P) & (df.tag == "final")) | ((df.model != P) & df.tag.isin(["default", "matched"]))].copy()
    # unified decision metrics: learned models use the validation-calibrated two-parameter LLR-CUSUM (epk_*),
    # binary detectors their own validation-calibrated rule (ep_*)
    for k in ["det_rate", "far", "delay", "det_rate_step", "det_rate_ramp", "far_benign"]:
        main[k] = np.where(main.model.isin(BINARY), main[f"ep_{k}"], main[f"epk_{k}"])
    abl = df[(df.model == P) & (df.tag.str.startswith("abl_") | (df.tag == "final"))].copy()
    return main, abl


def tests_and_wtl(df):
    mc = df[~df.model.isin(BINARY)]
    rows = []
    for metric, hb, d in [("macro_f1", True, mc), ("balanced_acc", True, mc), ("det_rate", True, df), ("far", False, df),
                          ("delay", False, df), ("bin_f1", True, df)]:
        for s in SYS:
            t = S.paired_tests(d[d.system == s], P, metric, higher_better=hb)
            if len(t):
                rows.append(t)
        dp = d.copy()
        dp["seed"] = dp.system + "_" + dp.seed.astype(str)
        t = S.paired_tests(dp.assign(system="pooled"), P, metric, higher_better=hb)
        if len(t):
            rows.append(t)
    st = pd.concat(rows, ignore_index=True)
    st.to_csv(os.path.join(OUT, "statistical_tests.csv"), index=False)
    w = []
    for m in [x for x in ORDER if x in df.model.unique() and x != P]:
        metric = "bin_f1" if m in BINARY else "macro_f1"
        sub = st[(st.baseline == m) & (st.metric == metric)]
        per = sub[sub.system != "pooled"]
        pooled = sub[sub.system == "pooled"]
        won = per.verdict.isin(["statistically superior", "practically superior"]).sum()
        lost = per.verdict.str.startswith("inferior").sum()
        w.append(dict(benchmark=PNAME.get(m, m), model=m, metric=metric, n_systems=len(per), systems_won=int(won),
                      systems_tied=int(len(per) - won - lost), systems_lost=int(lost),
                      pooled_diff=float(pooled.diff_mean.iloc[0]), pooled_p_holm_t=float(pooled.p_holm_t.iloc[0]),
                      pooled_p_holm_wilcoxon=float(pooled.p_holm.iloc[0]), pooled_cohen_dz=float(pooled.cohen_dz.iloc[0]),
                      pooled_verdict=pooled.verdict.iloc[0]))
    wtl = pd.DataFrame(w)
    wtl.to_csv(os.path.join(ROOT, "BENCHMARK_WTL.csv"), index=False)
    n_w = int(wtl.pooled_verdict.isin(["statistically superior", "practically superior"]).sum())
    n_l = int(wtl.pooled_verdict.str.startswith("inferior").sum())
    summ = dict(wins=n_w, ties=len(wtl) - n_w - n_l, losses=n_l, n_benchmarks=len(wtl),
                sig_wins=int((wtl.pooled_verdict == "statistically superior").sum()),
                per_system_wins=int(wtl.systems_won.sum()), per_system_comparisons=int(wtl.n_systems.sum()),
                per_system_losses=int(wtl.systems_lost.sum()))
    json.dump(summ, open(os.path.join(OUT, "wtl_summary.json"), "w"), indent=1)
    print("WTL", summ)
    return st, wtl


def fmt(m, s, nd=3):
    return f"{m:.{nd}f}$\\pm${s:.{nd}f}"


def main_table(agg):
    L = [r"\begin{table*}[!t]\centering",
         r"\caption{Test results (mean$\pm$std over training seeds; ten seeds for PG-TCN, five for the benchmarks). "
         r"Macro-F1: five-class step-level; $\dagger$: binary detector, binary step F1. Det., FAR and delay: episode detection rate, "
         r"episode false-alarm rate and mean detection delay of the validation-calibrated sequential test, averaged over the four systems.}\label{tab:main}",
         r"\setlength{\tabcolsep}{3.6pt}\begin{tabular}{llccccccrr}\toprule",
         r"Family & Method & IEEE-14 & IEEE-30 & IEEE-57 & IEEE-118 & Det.\ (\%) & FAR (\%) & Delay (s) & Params \\ \midrule"]
    best = {s: agg[(agg.system == s) & (agg.model != P) & ~agg.model.isin(BINARY)].macro_f1_mean.max() for s in SYS}
    for m in [x for x in ORDER if x in agg.model.unique()]:
        cells = []
        for s in SYS:
            x = agg[(agg.system == s) & (agg.model == m)]
            if not len(x):
                cells.append("--"); continue
            key = "bin_f1" if m in BINARY else "macro_f1"
            c = fmt(x[f"{key}_mean"].iloc[0], x[f"{key}_std"].iloc[0])
            if m == P:
                c = r"\textbf{" + c + "}"
            cells.append(c + (r"$^\dagger$" if m in BINARY else ""))
        a = agg[agg.model == m]
        par = a.params_mean.mean()
        L.append(f"{FAM[m]} & {NAME[m]} & " + " & ".join(cells) +
                 f" & {100 * a.det_rate_mean.mean():.1f} & {100 * a.far_mean.mean():.2f} & " +
                 ("--" if np.isnan(a.delay_mean.mean()) else f"{a.delay_mean.mean():.1f}") + " & " +
                 (f"{int(par):,}".replace(",", "{,}") if not np.isnan(par) else "--") + r" \\")
        if m == P:
            L.append(r"\midrule")
    L.append(r"\bottomrule\end{tabular}\end{table*}")
    open(os.path.join(TAB, "main_table_ieee.tex"), "w", encoding="utf-8").write("\n".join(L) + "\n")


def fig_main(agg):
    ms = [m for m in ORDER if m in agg.model.unique() and m not in BINARY][::-1]
    fig, ax = plt.subplots(figsize=(7.0, 2.6))
    x = np.arange(len(ms))
    for j, s in enumerate(SYS):
        v = [agg[(agg.system == s) & (agg.model == m)].macro_f1_mean.mean() for m in ms]
        e = [agg[(agg.system == s) & (agg.model == m)].macro_f1_std.mean() for m in ms]
        ax.bar(x + (j - 1.5) * 0.2, v, 0.2, yerr=e, label=SYSN[s], error_kw=dict(lw=0.5, capsize=1))
    ax.set_xticks(x); ax.set_xticklabels([PNAME[m] for m in ms], rotation=35, ha="right", fontsize=6.5)
    ax.set_ylabel("Test macro-F1 (5 classes)"); ax.set_ylim(0, 1.02)
    ax.legend(ncol=4, frameon=False, fontsize=7, loc="upper left")
    save(fig, "fig3_main_benchmark")


def amoc_curves(models=(P, "flat_tcn", "gru", "gcn_gru_matched", "lgbm_tok", "rf", "xgb")):
    """Detection rate and delay vs test episode FAR (threshold swept on TEST, each model's own calibrated LLR and offset)."""
    hs = np.concatenate([np.linspace(0.2, 20, 50), np.linspace(21, 400, 70)])
    rows = []
    for s in SYS:
        for m in models:
            tag = "final" if m == P else ("matched" if m == "gcn_gru_matched" else "default")
            mm = "gcn_gru" if m == "gcn_gru_matched" else m
            for f in sorted(glob.glob(os.path.join(ROOT, "results", "lib", s, f"{mm}__{tag}__seed*_testprob.npz")))[:5]:
                j = json.load(open(f.replace("_testprob.npz", ".json")))
                d = np.load(f)
                llr = M.calibrated_llr(d["prob"].astype(np.float32), tuple(j["llr_cal"]))
                for h in hs:
                    e = M.episode_metrics(M.cusum_alarms(llr, h, k=j["cusum_k2"]), d["cls"], d["tau"], llr.shape[1])
                    rows.append(dict(system=s, model=m, seed=j["seed"], h=h, far=e["far"], det=e["det_rate"], delay=e["delay"]))
    c = pd.DataFrame(rows)
    c.to_csv(os.path.join(OUT, "amoc_curves.csv"), index=False)
    pts = []
    for (s, m, k), g in c.groupby(["system", "model", "seed"]):
        g = g.sort_values("h")
        for f in (0.005, 0.01, 0.02, 0.05):
            ok = g[g.far <= f]
            r = ok.iloc[0] if len(ok) else g.iloc[-1]
            pts.append(dict(system=s, model=m, seed=k, far_target=f, det=r.det, delay=r.delay))
    pts = pd.DataFrame(pts)
    pts.to_csv(os.path.join(OUT, "matched_far.csv"), index=False)
    fig, axs = plt.subplots(1, 4, figsize=(7.16, 1.9), sharey=True)
    for ax, s in zip(axs, SYS):
        for m in models:
            g = c[(c.system == s) & (c.model == m)]
            if not len(g):
                continue
            bins = np.logspace(-3, np.log10(0.3), 25)
            g = g.assign(b=np.digitize(g.far, bins))
            curve = g.groupby("b").agg(far=("far", "mean"), det=("det", "max"))
            ax.plot(curve.far, curve.det, lw=1.4 if m == P else 0.8, color="k" if m == P else None, label=PNAME[m])
        ax.set_xscale("log"); ax.set_title(SYSN[s], fontsize=7.5); ax.set_xlabel("Test episode FAR", fontsize=7)
        ax.axvline(0.01, color="0.5", lw=0.5, ls="--")
    axs[0].set_ylabel("Detection rate")
    axs[-1].legend(frameon=False, fontsize=5.5, loc="lower right")
    save(fig, "fig_amoc")
    return pts


def ablation(abl):
    lab = {"final": "Full PG-TCN", "abl_no_horizon": "-- multi-horizon drift", "abl_no_velocity": "-- posterior velocity (and drift)",
           "abl_no_graph": "-- graph propagation", "abl_no_eta": r"-- $\eta$ channel", "abl_no_innovation": "-- innovation",
           "abl_no_wind_residual": "-- wind residual", "abl_no_localisation_loss": "-- localisation loss"}
    abl = abl[abl.system.isin(["ieee14", "ieee57"]) & (abl.seed < 5)]
    a = S.aggregate(abl, by=("system", "tag"))
    a.to_csv(os.path.join(OUT, "ablation_aggregate.csv"), index=False)
    x = abl.copy(); x["model"] = x.tag
    tests = pd.concat([S.paired_tests(x[x.system == s], "final", "macro_f1") for s in ["ieee14", "ieee57"]])
    tests.to_csv(os.path.join(OUT, "ablation_tests.csv"), index=False)
    L = [r"\begin{table}[!t]\centering\caption{Ablation on IEEE-14 and IEEE-57 (test macro-F1 and localisation F1, mean over five "
         r"paired seeds; $p$: Holm-adjusted paired $t$-test of macro-F1 against the full model, capped at 1 by the adjustment).}\label{tab:abl}",
         r"\setlength{\tabcolsep}{2.2pt}\begin{tabular}{lcccccc}\toprule",
         r" & \multicolumn{3}{c}{IEEE-14} & \multicolumn{3}{c}{IEEE-57} \\ \cmidrule(lr){2-4}\cmidrule(lr){5-7}",
         r"Variant & F1 & Loc. & $p$ & F1 & Loc. & $p$ \\ \midrule"]
    for t in lab:
        cells = []
        for s in ["ieee14", "ieee57"]:
            r = a[(a.system == s) & (a.tag == t)]
            p = tests[(tests.system == s) & (tests.baseline == t)]
            if not len(r):
                cells += ["--"] * 3; continue
            cells += [f"{r.macro_f1_mean.iloc[0]:.3f}", f"{r.loc_f1_mean.iloc[0]:.3f}",
                      "--" if t == "final" or not len(p) else
                      (f"{p.p_holm_t.iloc[0]:.3f}" if p.p_holm_t.iloc[0] >= 0.001 else "$<$0.001")]
        L.append(lab[t] + " & " + " & ".join(cells) + r" \\")
    L.append(r"\bottomrule\end{tabular}\end{table}")
    open(os.path.join(TAB, "ablation_table.tex"), "w", encoding="utf-8").write("\n".join(L) + "\n")
    # figure
    fig, axs = plt.subplots(1, 2, figsize=(3.5, 2.2), sharey=True)
    for ax, s in zip(axs, ["ieee14", "ieee57"]):
        b = a[a.system == s].set_index("tag").reindex([t for t in lab if t in a[a.system == s].tag.values])
        y = np.arange(len(b))[::-1]
        ax.barh(y, b.macro_f1_mean, xerr=b.macro_f1_std, color=["k" if t == "final" else "#4C72B0" for t in b.index],
                height=0.6, error_kw=dict(lw=0.6, capsize=1.2))
        ax.set_yticks(y); ax.set_yticklabels([lab[t].replace("-- ", "w/o ").replace(r"$\eta$", "η") for t in b.index], fontsize=6)
        ax.set_xlim(max(0.0, b.macro_f1_mean.min() - 0.1), 1.0); ax.set_title(SYSN[s], fontsize=7.5)
        ax.set_xlabel("Test macro-F1", fontsize=7)
    save(fig, "fig5a_ablation")
    return a, tests


def ood(models=(P, "flat_tcn", "gru", "gcn_gru_matched", "lgbm_tok", "rf", "xgb", "source_dqn")):
    rows = []
    for f in glob.glob(os.path.join(ROOT, "results", "eval_lib", "*", "*.json")):
        system = os.path.basename(os.path.dirname(f))
        model, tag, seed = os.path.basename(f)[:-5].split("__")
        if model == "gcn_gru" and tag == "matched":
            model = "gcn_gru_matched"
        d = json.load(open(f))
        for name, o in d.items():
            if name == "stream":
                rows.append(dict(system=system, model=model, seed=seed, set="stream", alarms_per_day=o["alarms_per_day_k"],
                                 frac_streams=o["frac_streams_alarm_k"]))
            else:
                rows.append(dict(system=system, model=model, seed=seed, set=name, macro_f1=o["macro_f1"],
                                 det=o.get("epk_det_rate"), far=o.get("epk_far")))
    # binary detectors (source DQN) evaluated inside classical.py
    for f in glob.glob(os.path.join(ROOT, "results", "lib", "*", "source_dqn__default__seed*.json")):
        d = json.load(open(f))
        for name, o in d.items():
            if isinstance(o, dict) and name.startswith("ood"):
                rows.append(dict(system=d["system"], model="source_dqn", seed=str(d["seed"]), set=name,
                                 det=o["ep_det_rate"], far=o["ep_far"], macro_f1=np.nan))
    r = pd.DataFrame(rows)
    r.to_csv(os.path.join(OUT, "robustness_raw.csv"), index=False)
    return r


def strength_analysis(models=(P, "flat_tcn", "gru", "gcn_gru_matched", "lgbm_tok", "rf", "xgb")):
    """Detection rate per attack-strength quintile at each model's validation-calibrated operating point."""
    rows = []
    for f in glob.glob(os.path.join(ROOT, "results", "lib", "*", "*_testprob.npz")):
        model, tag, seed = os.path.basename(f).replace("_testprob.npz", "").split("__")
        if model == "gcn_gru" and tag == "matched":
            model = "gcn_gru_matched"
        elif (model == P and tag != "final") or (model != P and tag != "default"):
            continue
        if model not in models or not os.path.exists(f.replace("_testprob.npz", ".json")):
            continue
        r = json.load(open(f.replace("_testprob.npz", ".json")))
        z = np.load(f)
        p = z["prob"].astype(np.float32)
        al = M.cusum_alarms(M.calibrated_llr(p, tuple(r["llr_cal"])), r["cusum_h2"], k=r["cusum_k2"])
        cls, tau, kap = z["cls"], z["tau"], z["kappa"]
        att = np.isin(cls, M.ATT)
        det = att & (al >= tau) & (al < p.shape[1])
        bins = np.quantile(kap[att], [0, 0.2, 0.4, 0.6, 0.8, 1.0])
        for c, nm in [(3, "step"), (4, "ramp")]:
            for i in range(5):
                m = (cls == c) & (kap >= bins[i]) & (kap <= bins[i + 1])
                rows.append(dict(system=os.path.basename(os.path.dirname(f)), model=model, seed=int(seed[4:]), attack=nm,
                                 kappa_bin=i, det_rate=det[m].mean() if m.any() else np.nan))
    d = pd.DataFrame(rows)
    d.to_csv(os.path.join(OUT, "strength_analysis.csv"), index=False)
    fig, axs = plt.subplots(1, 2, figsize=(7, 2.2), sharey=True)
    for ax, nm in zip(axs, ["step", "ramp"]):
        for m in models:
            g = d[(d.attack == nm) & (d.model == m)]
            if not len(g):
                continue
            x = g.groupby("kappa_bin").det_rate.mean()
            ax.plot(x.index + 1, x.values, marker="o", ms=3, lw=1.4 if m == P else 0.8, color="k" if m == P else None,
                    label=PNAME[m])
        ax.set_title(f"FDIA {nm}", fontsize=8); ax.set_xlabel("Attack-strength quintile"); ax.set_xticks(range(1, 6))
    axs[0].set_ylabel("Detection rate")
    axs[1].legend(frameon=False, fontsize=6, loc="lower right")
    save(fig, "fig4_strength")


def continuum_fig(r):
    sets = ["test", "ood_mix25", "ood_mix50", "ood_mix75", "ood_mimic"]
    beta = [0, 0.25, 0.5, 0.75, 1.0]
    raw = pd.read_csv(os.path.join(OUT, "raw_seed_results_test.csv"))
    fig, ax = plt.subplots(figsize=(3.5, 2.3))
    for m in [P, "flat_tcn", "gru", "gcn_gru_matched", "lgbm_tok", "rf", "source_dqn"]:
        y = [raw[raw.model == m].det_rate.mean()] + [r[(r.model == m) & (r.set == s)].det.mean() for s in sets[1:]]
        if np.all(np.isnan(y)):
            continue
        ax.plot(beta, y, marker="o", ms=3, lw=1.4 if m == P else 0.8, color="k" if m == P else None, label=PNAME[m])
    ax.set_xlabel(r"Mimicry blend $\beta$"); ax.set_ylabel("Detection rate")
    ax.legend(frameon=False, fontsize=5.5, loc="lower left")
    save(fig, "fig5b_continuum")


def confusion_fig():
    cms = {}
    for s in SYS:
        acc = None
        for f in glob.glob(os.path.join(ROOT, "results", "lib", s, f"{P}__final__seed*.json")):
            c = np.array(json.load(open(f))["test"]["cm"], float)
            acc = c if acc is None else acc + c
        if acc is not None:
            cms[s] = acc / acc.sum(1, keepdims=True)
    fig, axs = plt.subplots(1, len(cms), figsize=(1.75 * len(cms), 1.9), squeeze=False)
    lab = ["N", "WE", "LE", "FS", "FR"]
    for ax, (s, c) in zip(axs[0], cms.items()):
        ax.imshow(c, cmap="Blues", vmin=0, vmax=1)
        for i in range(5):
            for j in range(5):
                ax.text(j, i, f"{c[i, j]:.2f}", ha="center", va="center", fontsize=4.6, color="w" if c[i, j] > 0.5 else "k")
        ax.set_xticks(range(5)); ax.set_xticklabels(lab, fontsize=5.5); ax.set_yticks(range(5))
        ax.set_yticklabels(lab, fontsize=5.5); ax.set_title(SYSN[s], fontsize=7); ax.grid(False)
    save(fig, "fig_confusion")


def efficiency_fig(agg):
    ef = os.path.join(OUT, "efficiency_controlled.csv")
    if not os.path.exists(ef):
        return
    e = pd.read_csv(ef)
    cpu = e[e.device == "cpu"].groupby("model").latency_ms.mean()
    g = agg.groupby("model")[["macro_f1_mean", "train_time_s_mean", "params_mean"]].mean()
    fig, axs = plt.subplots(1, 2, figsize=(7, 2.5))
    for ax, col, lab in [(axs[0], "cpu", "CPU decision latency (ms)"), (axs[1], "train_time_s_mean", "Training time (s)")]:
        for m, r in g.iterrows():
            if m in BINARY:
                continue
            x = cpu.get(m, np.nan) if col == "cpu" else r[col]
            if np.isnan(x):
                continue
            ax.scatter(x, r.macro_f1_mean, s=55 if m == P else 16, marker="*" if m == P else "o",
                       c="k" if m == P else ("#55A868" if FAM[m] == "Tabular" else "#4C72B0"))
            if FAM[m] != "Tabular" or col == "cpu":
                ax.annotate(PNAME[m], (x, r.macro_f1_mean), fontsize=5.5, xytext=(3, 2), textcoords="offset points")
        ax.set_xscale("log"); ax.set_xlabel(lab); ax.set_ylabel("Test macro-F1")
    save(fig, "fig6_pareto")
    # efficiency table
    gpu = e[e.device == "cuda"].groupby("model")[["params", "latency_ms", "throughput_steps_s", "peak_infer_mem_mb",
                                                  "gflops_per_decision"]].mean()
    L = [r"\begin{table}[!t]\centering\caption{Efficiency averaged over the four systems: parameters, batch-1 decision latency "
         r"on four CPU threads and on the GPU, GPU throughput and FLOPs per decision (160-scan input).}\label{tab:eff}",
         r"\setlength{\tabcolsep}{2.4pt}\begin{tabular}{lrrrrr}\toprule",
         r"Model & Params & CPU ms & GPU ms & kscans/s & MFLOP \\ \midrule"]
    for m in [x for x in ORDER if x in gpu.index]:
        r = gpu.loc[m]
        L.append(f"{PNAME[m]} & {int(r.params):,}".replace(",", "{,}") +
                 f" & {cpu.get(m, np.nan):.2f} & {r.latency_ms:.2f} & {r.throughput_steps_s / 1e3:.0f} & {1e3 * r.gflops_per_decision:.1f} \\\\")
    L.append(r"\bottomrule\end{tabular}\end{table}")
    open(os.path.join(TAB, "efficiency_table.tex"), "w", encoding="utf-8").write("\n".join(L) + "\n")


def robustness_table(r):
    sets = [("ood_noise1.5", r"Noise $\times1.5$"), ("ood_wocwg", "Uncorr. wind"), ("ood_load0.7", r"Load $0.7\times$"),
            ("ood_load1.3", r"Load $1.3\times$"), ("ood_zones58", "Wind zones 5--8"), ("ood_pjm", "PJM load"),
            ("ood_mix25", r"Blend $\beta{=}0.25$"), ("ood_mix50", r"$\beta{=}0.50$"), ("ood_mix75", r"$\beta{=}0.75$"),
            ("ood_mimic", r"$\beta{=}1$ (Prop.~1)")]
    cols = [P, "gcn_gru_matched", "lgbm_tok", "flat_tcn", "rf"]
    L = [r"\begin{table}[!t]\centering\caption{Held-out evaluation (mean over systems and seeds): detection rate / episode FAR "
         r"(\%) under distribution shifts, unseen real drivers and the attack continuum; FAR (\%) for benign event types never "
         r"seen in training; alarms per day on hour-long test streams.}\label{tab:rob}",
         r"\footnotesize\setlength{\tabcolsep}{1.5pt}\begin{tabular}{l" + "c" * len(cols) + r"}\toprule",
         "Set & " + " & ".join({P: "PG-TCN", "gcn_gru_matched": "GCN-GRU$^m$", "lgbm_tok": "LGBM$^{t}$", "flat_tcn": "TCN",
                                 "rf": "RF"}[c] for c in cols) + r" \\ \midrule"]
    for s, lab in sets:
        cells = []
        for c in cols:
            x = r[(r.model == c) & (r.set == s)]
            cells.append(f"{100 * x.det.mean():.1f}/{100 * x.far.mean():.1f}" if len(x) else "--")
        L.append(lab + " & " + " & ".join(cells) + r" \\")
    L.append(r"\midrule")
    for s, lab in [("ood_bn_multi_step", "Two-bus steps"), ("ood_bn_ext_gen_trip", "Ext. unit trip"),
                   ("ood_bn_ext_line_out", "Ext. line outage")]:
        cells = [f"{100 * r[(r.model == c) & (r.set == s)].far.mean():.1f}" if len(r[(r.model == c) & (r.set == s)]) else "--" for c in cols]
        L.append(lab + " & " + " & ".join(cells) + r" \\")
    cells = [f"{r[(r.model == c) & (r.set == 'stream')].alarms_per_day.mean():.2f}" if len(r[(r.model == c) & (r.set == 'stream')]) else "--" for c in cols]
    L.append(r"Alarms/day & " + " & ".join(cells) + r" \\")
    L.append(r"\bottomrule\end{tabular}\end{table}")
    open(os.path.join(TAB, "robustness_table.tex"), "w", encoding="utf-8").write("\n".join(L) + "\n")


def main():
    df, abl = load()
    df.to_csv(os.path.join(OUT, "raw_seed_results_test.csv"), index=False)
    agg = S.aggregate(df, by=("system", "model"), keys=[k for k in S.KEYS + ["det_rate", "far", "delay", "det_rate_ramp",
                      "det_rate_step", "far_benign", "params", "train_time_s", "latency_ms"] if k in df])
    agg.to_csv(os.path.join(OUT, "aggregate_test.csv"), index=False)
    tests_and_wtl(df)
    main_table(agg)
    fig_main(agg)
    if len(abl):
        ablation(abl)
    amoc_curves()
    strength_analysis()
    confusion_fig()
    r = ood()
    continuum_fig(r)
    robustness_table(r)
    efficiency_fig(agg)


if __name__ == "__main__":
    main()
