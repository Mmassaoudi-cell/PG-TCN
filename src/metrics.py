"""Evaluation: step-level multiclass metrics, learned-LLR CUSUM stopping, episode detection metrics."""
import numpy as np
from sklearn.metrics import (f1_score, balanced_accuracy_score, precision_recall_fscore_support,
                             confusion_matrix, average_precision_score, roc_auc_score)

T_EVAL = 52            # first evaluated step (CKF burn-in 20 + 32-step context)
K = 5
ATT = [3, 4]
FAR_TARGET = 0.01


def step_labels(cls, tau, T):
    t = np.arange(T)[None]
    return np.where(t >= tau[:, None], cls[:, None], 0)


def step_metrics(prob, cls, tau):
    """prob (E,T,K) per-step class probabilities."""
    E, T, _ = prob.shape
    y = step_labels(cls, tau, T)[:, T_EVAL:].ravel()
    p = prob[:, T_EVAL:].reshape(-1, K)
    yh = p.argmax(1)
    pr, rc, f1, sup = precision_recall_fscore_support(y, yh, labels=range(K), zero_division=0)
    yb = np.isin(y, ATT)
    pb = p[:, ATT].sum(1)
    ybh = np.isin(yh, ATT)
    tn = ((~yb) & (~ybh)).sum(); fp = ((~yb) & ybh).sum(); fn = (yb & ~ybh).sum(); tp = (yb & ybh).sum()
    out = dict(macro_f1=f1_score(y, yh, average="macro", labels=range(K), zero_division=0),
               weighted_f1=f1_score(y, yh, average="weighted", labels=range(K), zero_division=0),
               balanced_acc=balanced_accuracy_score(y, yh),
               bin_f1=2 * tp / max(2 * tp + fp + fn, 1),
               step_fpr=fp / max(fp + tn, 1), step_fnr=fn / max(fn + tp, 1),
               pr_auc_att=average_precision_score(yb, pb), roc_auc_att=roc_auc_score(yb, pb))
    try:
        out["pr_auc_macro"] = np.mean([average_precision_score(y == k, p[:, k]) for k in range(K)])
    except Exception:
        out["pr_auc_macro"] = np.nan
    for k in range(K):
        out[f"prec_{k}"], out[f"rec_{k}"], out[f"f1_{k}"] = pr[k], rc[k], f1[k]
    out["minority_recall"] = min(rc[1:])
    out["cm"] = confusion_matrix(y, yh, labels=range(K))
    return out


def attack_llr(prob, eps=1e-6):
    pa = np.clip(prob[..., ATT].sum(-1), eps, 1 - eps)
    return np.clip(np.log(pa) - np.log(1 - pa), -10, 10)


def attack_logit(prob, eps=1e-6):
    pa = np.clip(prob[..., ATT].sum(-1), eps, 1 - eps)
    return np.log(pa) - np.log(1 - pa)


def fit_llr_calibration(prob_val, cls_val, tau_val):
    """Platt-calibrate the attack logit on VALIDATION steps and remove the validation prior log-odds,
    so that the CUSUM increment estimates log p(x|attack)/p(x|no attack). Returns (a, c): llr = a*s + c."""
    from sklearn.linear_model import LogisticRegression
    E, T, _ = prob_val.shape
    s = attack_logit(prob_val)[:, T_EVAL:].reshape(-1, 1)
    y = np.isin(step_labels(cls_val, tau_val, T)[:, T_EVAL:], ATT).ravel()
    lr = LogisticRegression(C=1e4, max_iter=1000).fit(np.clip(s, -30, 30), y)
    pi = y.mean()
    return float(lr.coef_[0, 0]), float(lr.intercept_[0] - np.log(pi / (1 - pi)))


def calibrated_llr(prob, cal):
    a, c = cal
    return np.clip(a * np.clip(attack_logit(prob), -30, 30) + c, -10, 10)


def cusum_alarms(llr, h, t0=T_EVAL, k=0.0):
    E, T = llr.shape
    S = np.zeros(E)
    alarm = np.full(E, T)
    for t in range(t0, T):
        S = np.maximum(0, S + llr[:, t] - k)
        new = (S >= h) & (alarm == T)
        alarm[new] = t
    return alarm


HS = np.concatenate([np.linspace(0.5, 20, 40), np.linspace(21, 200, 60)])
K_GRID = [0.0, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0]
HS_WIDE = np.concatenate([HS, np.linspace(220, 1000, 40)])


def stream_alarms(llr, h, k=0.0, t0=T_EVAL):
    """Alarms per stream of a CUSUM that restarts after every alarm."""
    E, T = llr.shape
    S = np.zeros(E)
    n = np.zeros(E, int)
    for t in range(t0, T):
        S = np.maximum(0, S + llr[:, t] - k)
        hit = S >= h
        n += hit
        S[hit] = 0
    return n


def calibrate_hk(llr_val, cls_val, tau_val, llr_stream, far=FAR_TARGET, per_day=1.0):
    """Two-parameter CUSUM (drift offset k, threshold h) chosen on VALIDATION data only: episode FAR <= far and
    at most `per_day` false alarms per day on validation streams of normal operation; maximise detection."""
    hours = llr_stream.shape[0] * (llr_stream.shape[1] - T_EVAL) / 3600.0
    best = None
    for k in K_GRID:
        h0 = calibrate_h(llr_val - k, cls_val, tau_val, far)
        rate = np.inf
        for h in [x for x in HS_WIDE if x >= h0] or [HS_WIDE[-1]]:
            rate = stream_alarms(llr_stream, h, k).sum() / hours * 24
            if rate <= per_day:
                break
        e = episode_metrics(cusum_alarms(llr_val, h, k=k), cls_val, tau_val, llr_val.shape[1])
        # feasible pairs first (both constraints met), then detection, then delay; infeasible: fewest stream alarms
        key = (rate <= per_day, e["det_rate"] if rate <= per_day else -rate, -np.nan_to_num(e["delay"], nan=1e9))
        if best is None or key > best[0]:
            best = (key, float(k), float(h))
    return best[1], best[2]


def calibrate_h(llr_val, cls_val, tau_val, far=FAR_TARGET):
    """Smallest CUSUM threshold whose VALIDATION episode false-alarm rate <= far."""
    hs = HS
    best = hs[-1]
    for h in hs:
        al = cusum_alarms(llr_val, h)
        m = episode_metrics(al, cls_val, tau_val, llr_val.shape[1])
        if m["far"] <= far:
            best = h
            break
    return float(best)


def episode_metrics(alarm, cls, tau, T):
    is_att = np.isin(cls, ATT)
    fa = (alarm < T) & (~is_att | (alarm < tau))
    det = is_att & (alarm >= tau) & (alarm < T)
    out = dict(far=fa.mean(), far_benign=(fa & np.isin(cls, [1, 2])).sum() / max(np.isin(cls, [1, 2]).sum(), 1),
               det_rate=det.sum() / max(is_att.sum(), 1),
               timely_rate=(is_att & (alarm == tau)).sum() / max(is_att.sum(), 1),
               delay=float((alarm - tau)[det].mean()) if det.any() else np.nan)
    for c, nm in [(3, "step"), (4, "ramp")]:
        m = cls == c
        out[f"det_rate_{nm}"] = (det & m).sum() / max(m.sum(), 1)
        out[f"delay_{nm}"] = float((alarm - tau)[det & m].mean()) if (det & m).any() else np.nan
    return out


def episode_class_f1(prob, cls, last=30):
    yh = prob[:, -last:].mean(1).argmax(1)
    return f1_score(cls, yh, average="macro", labels=range(K), zero_division=0)
