"""Feature pipeline for the new study (all statistics fitted on TRAIN only).

Per step t the defender has: standardized whitened CKF innovation nun_t (M), posterior estimated
measurements zpost_t = h(xhat_t) (M) and eta_t. We derive:
  * vel_t   = (zpost_t - zpost_{t-1}) / sigma_v           posterior measurement velocity (M)
  * wres_t  = wind correlation-consistency residual       (n_wind_internal)
       r_j = v_j - E[v_j | v_-j] under the TRAIN normal-episode covariance of wind-bus injection velocities
  * bus tokens: measurement-to-bus typed aggregation -> (nR, F_node) per step (graph input)
"""
import os
import numpy as np
import torch

from grid import build, DEV, DT
from sim import SIG_V

DATA = os.path.join(os.path.dirname(__file__), "..", "data")
T0 = 20
CLASSES = ["normal", "wind_ramp", "load_step", "fdia_step", "fdia_ramp"]
_REG = {}


def region_of(system):
    if system not in _REG:
        _REG[system] = build(system)[1]
    return _REG[system]


def load_raw(system, name, split):
    d = np.load(os.path.join(DATA, f"{system}_{name}_{split}.npz"))
    ok = d["conv"].astype(bool)
    out = {k: d[k][ok] for k in ["nun", "xhat", "eta", "cls", "tau", "att_bus", "kappa", "ramp_L", "scale", "amp", "ev_kind"] if k in d}
    return out


@torch.no_grad()
def posterior_velocity(region, xhat, chunk=500):
    E, T, N = xhat.shape
    out = np.zeros((E, T, region.M), np.float32)
    for s in range(0, E, chunk):
        x = torch.tensor(xhat[s:s + chunk], dtype=DT, device=DEV)
        z = region.h(x)
        v = torch.zeros_like(z)
        v[:, 1:] = (z[:, 1:] - z[:, :-1]) / SIG_V
        out[s:s + chunk] = v.float().cpu().numpy()
    return out


class Featurizer:
    """Fits normalization + wind-residual regression on TRAIN; transforms any split."""

    def __init__(self, system):
        self.system = system
        self.r = region_of(system)
        r = self.r
        # bus-typed aggregation matrices (nR x M) for each measurement group
        nR = r.nR
        groups = {"V": r.mV, "P": r.mP, "Q": r.mQ, "Pf": np.concatenate([r.mPf, r.mPt]),
                  "Qf": np.concatenate([r.mQf, r.mQt])}
        self.groups = groups
        mats = []
        for g, idx in groups.items():
            A = np.zeros((nR, r.M), np.float32)
            for m in idx:
                A[r.meas_bus[m], m] = 1.0
            deg = A.sum(1, keepdims=True)
            mats.append(A / np.maximum(deg, 1))
        self.Agg = np.stack(mats, 0)                                    # (G,nR,M) mean-aggregation
        self.is_wind = np.isin(r.R, r.wind_int).astype(np.float32)
        self.is_bnd = np.isin(r.R, r.B).astype(np.float32)
        self.wpos = np.array([r.pos[w] for w in r.wind_int], int)

    # ------------------------------------------------------------------ wind residual
    def _wind_vel(self, vel):
        return vel[..., self.r.mPw]

    def fit(self, d):
        vel = posterior_velocity(self.r, d["xhat"])
        d["vel"] = vel
        norm = d["cls"] == 0
        vw = self._wind_vel(vel[norm][:, T0:]).reshape(-1, len(self.r.mPw)).astype(np.float64)
        C = np.cov(vw.T) if vw.shape[1] > 1 else np.eye(1)
        self.Cw = np.atleast_2d(C)
        nw = self.Cw.shape[0]
        B = np.zeros((nw, nw))                                         # r = v - B v  (conditional mean)
        for j in range(nw):
            o = [k for k in range(nw) if k != j]
            if o:
                B[j, o] = self.Cw[j, o] @ np.linalg.inv(self.Cw[np.ix_(o, o)])
        self.Bw = B
        rs = (vw - vw @ B.T)
        self.wres_std = rs.std(0) + 1e-6
        # per-channel scaling for flat inputs
        nun = d["nun"][norm][:, T0:].astype(np.float32)
        self.nun_mu, self.nun_sd = nun.reshape(-1, nun.shape[-1]).mean(0), nun.reshape(-1, nun.shape[-1]).std(0) + 1e-6
        v = vel[norm][:, T0:]
        self.vel_sd = v.reshape(-1, v.shape[-1]).std(0) + 1e-6
        le = np.log(d["eta"][norm][:, T0:])
        self.eta_mu, self.eta_sd = le.mean(), le.std() + 1e-6
        return self

    def transform(self, d):
        if "vel" not in d:
            d["vel"] = posterior_velocity(self.r, d["xhat"])
        E, T, M = d["nun"].shape
        nun = ((d["nun"].astype(np.float32) - self.nun_mu) / self.nun_sd).astype(np.float32)
        vel = (d["vel"] / self.vel_sd).astype(np.float32)
        vw = self._wind_vel(d["vel"]).astype(np.float64)
        wres = ((vw - vw @ self.Bw.T) / self.wres_std).astype(np.float32)
        eta = ((np.log(d["eta"]) - self.eta_mu) / self.eta_sd).astype(np.float32)
        d["X_flat"] = np.concatenate([nun, vel, wres, eta[..., None]], -1)  # (E,T,2M+nw+1)
        return d

    def node_tokens(self, X_flat, horizons=()):
        """X_flat (...,T,2M+nw+1) torch -> node tokens (...,T,nR,F).

        horizons: optional window lengths W of the causal multi-horizon drift
        d^W_t = sum_{j=t-W+1}^{t} vel_j / sqrt(W), aggregated like the other channels (F += 10 per W).
        """
        r = self.r
        M = r.M
        nun, vel = X_flat[..., :M], X_flat[..., M:2 * M]
        nw = len(r.mPw)
        wres = X_flat[..., 2 * M:2 * M + nw]
        eta = X_flat[..., -1:]
        Agg = torch.as_tensor(self.Agg, device=X_flat.device)
        srcs = [nun, vel]
        if horizons:
            cs = torch.cumsum(vel, dim=-2)
            for W in horizons:
                lag = torch.zeros_like(cs)
                lag[..., W:, :] = cs[..., :-W, :]
                srcs.append((cs - lag) / float(np.sqrt(W)))
        feats = []
        for src in srcs:
            for g in range(Agg.shape[0]):
                feats.append(src @ Agg[g].T)                          # mean
                feats.append((src.abs()) @ Agg[g].T)                  # mean |.|
        F = torch.stack(feats, -1)                                     # (...,T,nR,20)
        wr = torch.zeros(F.shape[:-1] + (1,), device=F.device)
        if nw:
            wr[..., self.wpos, 0] = wres
        stat = torch.as_tensor(np.stack([self.is_wind, self.is_bnd], -1), device=F.device)
        stat = stat.expand(F.shape[:-1] + (2,))
        et = eta[..., None, :].expand(F.shape[:-1] + (1,))
        return torch.cat([F, wr, et, stat], -1)                        # F_node = 24


def step_labels(cls, tau, T):
    t = np.arange(T)[None]
    return np.where(t >= tau[:, None], cls[:, None], 0)


def sample_windows(E_idx, tau, T, W, per_ep, rng, post_frac=0.5):
    """Random window end-times; half of attack/event windows drawn after onset for balance."""
    ends = []
    for i in E_idx:
        lo = T0 + W - 1
        n_post = int(per_ep * post_frac)
        a = rng.integers(lo, T, size=per_ep - n_post)
        b = rng.integers(max(lo, tau[i]), T, size=n_post)
        ends.append(np.stack([np.full(per_ep, i), np.concatenate([a, b])], 1))
    return np.concatenate(ends)
