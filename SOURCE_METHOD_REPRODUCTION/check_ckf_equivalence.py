"""Numerical check: with f = identity and the linearised measurement map Z = H chi (SP Eqs. 15, 27),
the coupled CKF equals an (iterated-linearisation) extended KF step-by-step. Also times both."""
import os
import sys
import time
import json
import numpy as np
import torch

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(ROOT, "src"))
from grid import build, DEV, DT  # noqa
import sim  # noqa


def ekf_run(ckf, z, x0):
    r = ckf.r
    B, T, M = z.shape
    N = r.N
    x = x0.clone()
    P = 1e-4 * torch.eye(N, dtype=DT, device=DEV).expand(B, N, N).clone()
    eta = torch.zeros(B, T, dtype=DT, device=DEV)
    xs = torch.zeros(B, T, N, dtype=DT, device=DEV)
    I_M = torch.eye(M, dtype=DT, device=DEV)
    for k in range(T):
        xp, Pp = x, P + ckf.Q
        H = r.jac(xp)
        Hc = ckf.Psi_inv @ H
        nu = (ckf.Psi_inv @ (z[:, k] - r.h(xp))[..., None])[..., 0]
        S = Hc @ Pp @ Hc.transpose(1, 2) + I_M
        K = Pp @ Hc.transpose(1, 2) @ torch.linalg.inv(S)
        x = xp + (K @ nu[..., None])[..., 0]
        IKH = torch.eye(N, dtype=DT, device=DEV) - K @ Hc
        P = IKH @ Pp @ IKH.transpose(1, 2) + K @ K.transpose(1, 2)
        zh = (Hc @ xp[..., None])[..., 0]
        zt = nu + zh
        eta[:, k] = nu.norm(dim=1) / (zt.norm(dim=1) * zh.norm(dim=1))
        xs[:, k] = x
    return eta, xs


if __name__ == "__main__":
    out = {}
    for system in ["ieee14", "ieee118"]:
        g, r = build(system)
        rng = np.random.default_rng(0)
        B, T = 64, 60
        w, l, pf, _ = sim.sample_profiles(g, r, "train", B, T, rng)
        Vm, Va, c = sim.run_pf(g, w, l, pf, np.zeros_like(l), r)
        x = r.state_from_full(torch.tensor(Vm, dtype=DT, device=DEV), torch.tensor(Va, dtype=DT, device=DEV))
        z = r.h(x) + sim.SIG_V * torch.randn(B, T, r.M, dtype=DT, device=DEV)
        ckf = sim.CoupledCKF(r, np.eye(2 * len(r.mPw)) * 1e-5)
        x0 = x[:, 0]
        torch.cuda.synchronize(); t0 = time.time()
        res = ckf.run(z, x0)
        torch.cuda.synchronize(); t_ckf = time.time() - t0
        t0 = time.time()
        eta2, xs2 = ekf_run(ckf, z, x0)
        torch.cuda.synchronize(); t_ekf = time.time() - t0
        rel_eta = float(((res["eta"].double() - eta2).abs() / eta2.abs()).max())
        dx = float((res["xhat"].double() - xs2).abs().max())
        out[system] = dict(max_rel_diff_eta=rel_eta, max_abs_diff_xhat=dx, t_ckf_s=t_ckf, t_ekf_s=t_ekf,
                           ckf_ms_per_step_per_ep=1e3 * t_ckf / (B * T), ekf_ms_per_step_per_ep=1e3 * t_ekf / (B * T))
        print(system, out[system])
    os.makedirs(os.path.join(ROOT, "results", "reproduction"), exist_ok=True)
    json.dump(out, open(os.path.join(ROOT, "results", "reproduction", "ckf_vs_ekf.json"), "w"), indent=1)
