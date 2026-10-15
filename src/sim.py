"""Episode simulator: correlated wind/load -> batched AC power flow -> events/FDIAs -> noisy
reduced-order measurements -> coupled CKF (source paper, Eqs. 13-33) -> saved features.

Episode classes (new study):
  0 normal, 1 benign wind ramp, 2 benign load step, 3 FDIA step (source-paper attack), 4 FDIA ramp.
The source-paper reproduction uses only class 3 (every trial attacked at tau, as in SP Alg. 1-2).
"""
import os
import time
import argparse
import numpy as np
import torch

from grid import build, DEV, DT
import profiles

T_DEFAULT = 200
SIG_V = 1e-2          # measurement noise std (sigma_v^2 = 1e-4)
SIG_W2 = 2e-4         # process noise variance (random-walk DSE)
RATED = 1.0           # 100 MW on 100 MVA base
ATT_EST_STD = 3e-3    # attacker's local state-estimate error (A8)
D_PER_RAD = 4.0       # wind-bus correlated injection target std per rad of angle increment (A7)
V_TO_TH = 0.2         # voltage-increment std relative to angle-increment std (A7)
WIND_Q_SIGN = -1.0    # wind farms absorb reactive power (voltage support), A15
PF_LO = 0.9           # wind power-factor range [PF_LO, 1.0], A15
FLOW_FLOOR = 0.2      # line-flow limit floor (p.u.) for lightly loaded lines, A9

CLASSES = ["normal", "wind_ramp", "load_step", "fdia_step", "fdia_ramp"]
KINDS = ["", "wind_ramp", "wind_trip", "load_step", "q_step", "load_ramp", "sensor_drift", "multi_step", "ext_gen_trip",
         "ext_line_out"]


# ----------------------------------------------------------------------------------- profiles
def load_map(split_df, train_df):
    lo, hi = np.quantile(train_df["load"], [0.01, 0.99])
    return np.clip(0.7 + 0.6 * (split_df["load"].values - lo) / (hi - lo), 0.65, 1.35)


def sample_profiles(grid, region, split, B, T, rng, corr_wind=True, noise_scale=1.0, load_scale=None, fixed_anchor=None,
                    ext=None):
    """Return wind (B,T,nw) in p.u., load multiplier (B,T,n), pf (B,nw).
    ext: optional (zones, load_src) of unseen real drivers (profiles.ext_frame)."""
    if ext is not None:
        df, tr = profiles.ext_frame(split, *ext), profiles.ext_frame("train", *ext)
    else:
        df = profiles.split_frame(split if fixed_anchor is None else "train")
        tr = profiles.split_frame("train")
    W = df[[f"w{z}" for z in profiles.ZONES]].values
    Lm = load_map(df, tr)
    nw = len(region.wind)
    C = np.corrcoef(tr[[f"w{z}" for z in profiles.ZONES]].values.T)[:nw, :nw]
    if not corr_wind:
        C = np.eye(nw)
    Lc = np.linalg.cholesky(C)
    idx = rng.integers(0, len(df) - 2, size=B)
    if fixed_anchor is not None:  # single operating point for all episodes (diagnostic)
        idx = np.full(B, fixed_anchor)
    if corr_wind:
        lvl = W[idx, :nw]
    else:  # independent anchors per farm destroy spatial correlation (WOCWG)
        lvl = np.stack([W[rng.integers(0, len(df) - 2, size=B), j] for j in range(nw)], 1)
    a, sig = 0.95, 0.006
    u = np.zeros((B, T, nw))
    u[:, 0] = (rng.standard_normal((B, nw)) @ Lc.T) * sig / np.sqrt(1 - a * a)
    e = rng.standard_normal((B, T, nw)) @ Lc.T * sig
    for k in range(1, T):
        u[:, k] = a * u[:, k - 1] + e[:, k]
    wind = np.clip(lvl[:, None, :] + u, 0, 1) * RATED
    L0 = Lm[idx] if load_scale is None else np.full(B, load_scale)
    n = grid.n
    ar = np.zeros((B, T, n))
    eps = rng.standard_normal((B, T, n)) * 0.002
    for k in range(1, T):
        ar[:, k] = 0.97 * ar[:, k - 1] + eps[:, k]
    load = L0[:, None, None] * (1 + ar)
    pf = rng.uniform(PF_LO, 1.0, size=(B, nw))
    return wind, load, pf, idx


# ----------------------------------------------------------------------------------- events
def apply_wind_ramp(wind, tau, rng):
    B, T, nw = wind.shape
    s = rng.choice([-1, 1], size=B)
    d = rng.uniform(0.15, 0.5, size=B)
    delta = s[:, None] * (d[:, None] + 0.1 * rng.standard_normal((B, nw)))
    L = rng.integers(5, 60, size=B)
    t = np.arange(T)[None]
    r = np.clip((t - tau[:, None]) / L[:, None], 0, 1)
    return np.clip(wind + r[:, :, None] * delta[:, None, :] * RATED, 0, RATED), dict(L=L, delta=delta)


def apply_load_step(grid, region, dPload, tau, rng):
    """Add an absolute load step at a random internal non-wind region bus. dPload: (B,T,n) extra load."""
    B, T, n = dPload.shape
    cand = [b for b in region.I if b not in set(region.wind)]
    bus = rng.choice(cand, size=B)
    mag = rng.choice([-1, 1], size=B) * rng.uniform(0.05, 0.25, size=B)
    t = np.arange(T)[None]
    on = (t >= tau[:, None]).astype(float)
    dP = np.zeros((B, T, n))
    dP[np.arange(B), :, bus] = on * mag[:, None]
    return dPload + dP, dict(bus=bus, mag=mag)


# ----------------------------------------------------------------------------------- AC power flow over episode
def run_pf(grid, wind, load, pf, extra_load, region, chunk=1000, extra_q=None, Y_after=None, tau_sw=None):
    B, T, _ = wind.shape
    n = grid.n
    Pg0, Pd0, Qd0 = grid.Pg0, grid.Pd0, grid.Qd0
    Vm_all = np.zeros((B, T, n), np.float64)
    Va_all = np.zeros((B, T, n), np.float64)
    conv_all = np.ones(B, bool)
    tanphi = WIND_Q_SIGN * np.tan(np.arccos(pf))
    for s in range(0, B, chunk):
        e = min(B, s + chunk)
        Vm = Va = None
        for k in range(T):
            L = load[s:e, k]                                      # (b,n)
            P = Pg0[None] * L.mean(1, keepdims=True) - Pd0[None] * L - extra_load[s:e, k]
            Q = -Qd0[None] * L - 0.3 * extra_load[s:e, k]
            if extra_q is not None:
                Q = Q - extra_q[s:e, k]
            np.add.at(P, (slice(None), region.wind), wind[s:e, k])
            np.add.at(Q, (slice(None), region.wind), wind[s:e, k] * tanphi[s:e])
            Pt = torch.tensor(P, dtype=DT, device=DEV)
            Qt = torch.tensor(Q, dtype=DT, device=DEV)
            Yk = None
            if Y_after is not None:
                sw = torch.tensor(k >= tau_sw[s:e], device=DEV)[:, None, None]
                Yk = torch.where(sw, Y_after[s:e], grid.Ybus[None])
            Vm, Va, conv = grid.pf(Pt, Qt, Vm, Va, iters=10 if k else 15, Y=Yk)
            conv_all[s:e] &= conv.cpu().numpy()
            Vm_all[s:e, k] = Vm.cpu().numpy()
            Va_all[s:e, k] = Va.cpu().numpy()
    return Vm_all, Va_all, conv_all


# ----------------------------------------------------------------------------------- attacks
def attack_direction(region, x_tau, C_w, kappa, rng, mimic=False):
    """Correlation-aware FDIA direction (SP Eqs. 5-7, assumption A7).

    x_tau: (B,N) torch state at onset. Returns dx (B,N) with nonzero entries only on internal-bus states.
    """
    B = x_tau.shape[0]
    J = region.jac(x_tau)                                          # (B,M,N)
    att = torch.tensor(region.att_idx, device=DEV)
    nwi = len(region.mPw)
    kap = torch.tensor(kappa, dtype=DT, device=DEV)[:, None]
    Lc = np.linalg.cholesky(C_w + 1e-9 * np.eye(len(C_w)))
    d = (rng.standard_normal((B, nwi)) @ Lc.T) * (D_PER_RAD * kappa[:, None])  # correlated wind-bus target
    if mimic:
        rows = np.concatenate([region.mP, region.mQ])
        tgt = np.zeros((B, len(rows)))
        pos = {m: i for i, m in enumerate(rows)}
        for j, m in enumerate(region.mPw):
            tgt[:, pos[m]] = d[:, j]
    else:
        rows = region.mPw
        tgt = d
    A = J[:, rows][:, :, att]                                      # (B,r,na)
    tgt = torch.tensor(tgt, dtype=DT, device=DEV)
    lam = 1e-6
    AAt = A @ A.transpose(1, 2) + lam * torch.eye(A.shape[1], dtype=DT, device=DEV)
    sol = A.transpose(1, 2) @ torch.linalg.solve(AAt, tgt[..., None])
    dx = torch.zeros_like(x_tau)
    if not mimic:
        # SP Step 1: generic state-increment FDIA vector, projected onto the null space of the
        # wind-bus injection rows so that wind-bus injections follow the correlated target exactly.
        na = len(region.att_idx)
        nth_a = len(region.att_th)
        r0 = torch.randn(B, na, dtype=DT, device=DEV)
        r0[:, nth_a:] *= V_TO_TH
        r0 = r0 * kap
        proj = A.transpose(1, 2) @ torch.linalg.solve(AAt, A @ r0[..., None])
        sol = sol + (r0[..., None] - proj)
    dx[:, att] = sol[..., 0]
    # one Gauss-Newton correction on the nonlinear model (SP iterative Steps 1-4)
    for _ in range(2):
        z1 = region.h(x_tau + dx)[:, rows] - region.h(x_tau)[:, rows]
        J1 = region.jac(x_tau + dx)[:, rows][:, :, att]
        r = tgt - z1
        AAt = J1 @ J1.transpose(1, 2) + lam * torch.eye(J1.shape[1], dtype=DT, device=DEV)
        dx[:, att] += (J1.transpose(1, 2) @ torch.linalg.solve(AAt, r[..., None]))[..., 0]
    return dx


_ZC = {}


def case_base_measurements(region):
    """Reduced measurements at the unmodified IEEE base-case power flow (no wind)."""
    key = region.grid.name
    if key not in _ZC:
        g = region.grid
        P = torch.tensor(g.Pg0 - g.Pd0, dtype=DT, device=DEV)[None]
        Q = torch.tensor(-g.Qd0, dtype=DT, device=DEV)[None]
        Vm, Va, _ = g.pf(P, Q)
        _ZC[key] = region.h(region.state_from_full(Vm, Va))[0]
    return _ZC[key]


def feasibility_scale(region, x_tau, dx, z_base):
    """Largest s in (0,1] such that the attacked operating point satisfies SP limits (slacks u_i = 0)."""
    nth = len(region.th_idx)
    Lf, Lt = region.Lf, region.Lt
    base = z_base

    th0, V0 = region.unpack(x_tau)
    Vlo = torch.clamp(V0, max=0.8)            # slack semantics: never worsen a pre-existing violation
    Vhi = torch.clamp(V0, min=1.1)
    dth0 = torch.clamp((th0[:, Lf] - th0[:, Lt]).abs(), min=np.deg2rad(15))
    # SP Step 2 limits relative to the current operating point (assumption A9)
    inj = torch.tensor(np.concatenate([region.mP, region.mQ]), device=DEV)
    fl = torch.tensor(np.concatenate([region.mPf, region.mQf, region.mPt, region.mQt]), device=DEV)
    b = base[:, inj]
    inj_lo = torch.minimum(0.5 * b, 1.5 * b) - 0.1
    inj_hi = torch.maximum(0.5 * b, 1.5 * b) + 0.1
    fl_hi = 1.5 * torch.clamp(base[:, fl].abs(), min=FLOW_FLOOR)

    def ok(s):
        xa = x_tau + s[:, None] * dx
        za = region.h(xa)
        th, V = region.unpack(xa)
        c1 = ((V >= Vlo - 1e-9) & (V <= Vhi + 1e-9)).all(1)
        c2 = ((th[:, Lf] - th[:, Lt]).abs() <= dth0 + 1e-9).all(1)
        c3 = ((za[:, inj] >= inj_lo - 1e-9) & (za[:, inj] <= inj_hi + 1e-9)).all(1)
        c4 = (za[:, fl].abs() <= fl_hi + 1e-9).all(1)
        return c1 & c2 & c3 & c4

    lo = torch.zeros(dx.shape[0], dtype=DT, device=DEV)
    hi = torch.ones_like(lo)
    good = ok(hi)
    s = torch.where(good, hi, lo)
    for _ in range(12):
        mid = (lo + hi) / 2
        g = ok(mid)
        lo = torch.where(g & ~good, mid, lo)
        hi = torch.where(~g & ~good, mid, hi)
    s = torch.where(good, hi, lo)
    return s


# ----------------------------------------------------------------------------------- coupled CKF
def _chol(P):
    L, info = torch.linalg.cholesky_ex(P)
    j = 1e-12
    while bool((info > 0).any()):
        j *= 100
        L, info = torch.linalg.cholesky_ex(P + j * torch.eye(P.shape[-1], dtype=P.dtype, device=P.device))
    return L

class CoupledCKF:
    """Coupled cubature Kalman filter on the reduced-order system (SP Sec. IV-B, Eqs. 13-31)."""

    def __init__(self, region, S_hat, sig_v=SIG_V, sig_w2=SIG_W2, use_corr=True):
        self.r = region
        M, N = region.M, region.N
        R = np.eye(M) * sig_v ** 2
        if use_corr and S_hat is not None:
            idx = np.concatenate([region.mPw, region.mQw]).astype(int)
            R[np.ix_(idx, idx)] += S_hat
        self.Psi = torch.tensor(np.linalg.cholesky(R), dtype=DT, device=DEV)
        self.Psi_inv = torch.linalg.inv(self.Psi)
        self.Q = sig_w2 * torch.eye(N, dtype=DT, device=DEV)
        xi = np.sqrt(N) * np.concatenate([np.eye(N), -np.eye(N)], 0)  # (2N,N) Eq. 23
        self.xi = torch.tensor(xi, dtype=DT, device=DEV)

    def run(self, z, x0):
        """z: (B,T,M) measurements; x0: (B,N) initial state. Returns dict of per-step outputs."""
        r = self.r
        B, T, M = z.shape
        N = r.N
        x = x0.clone()
        P = 1e-4 * torch.eye(N, dtype=DT, device=DEV).expand(B, N, N).clone()
        eta = torch.zeros(B, T, dtype=DT, device=DEV)
        nu = torch.zeros(B, T, M, dtype=torch.float32, device=DEV)
        nun = torch.zeros(B, T, M, dtype=torch.float32, device=DEV)
        xs = torch.zeros(B, T, N, dtype=torch.float32, device=DEV)
        I_M = torch.eye(M, dtype=DT, device=DEV)
        for k in range(T):
            # --- time prediction (Eqs. 21-25), f = identity (A6)
            S = _chol(P)
            chi = x[:, None, :] + (S @ self.xi.T).transpose(1, 2)        # (B,2N,N)
            xp = chi.mean(1)
            dchi = chi - xp[:, None]
            Pp = dchi.transpose(1, 2) @ dchi / (2 * N) + self.Q
            # --- measurement update (Eqs. 26-31) on whitened linearised model (Eqs. 15, 20)
            H = r.jac(xp)                                                  # (B,M,N)
            hx = r.h(xp)
            Hc = self.Psi_inv @ H
            zt = (self.Psi_inv @ (z[:, k] - hx + (H @ xp[..., None])[..., 0])[..., None])[..., 0]
            Sp = _chol(Pp)
            chi = xp[:, None, :] + (Sp @ self.xi.T).transpose(1, 2)
            Zc = chi @ Hc.transpose(1, 2)                                  # (B,2N,M)
            zh = Zc.mean(1)
            dZ = Zc - zh[:, None]
            dX = chi - xp[:, None]
            Pzz = dZ.transpose(1, 2) @ dZ / (2 * N) + I_M                  # whitened noise cov = I
            Pxz = dX.transpose(1, 2) @ dZ / (2 * N)
            Lz = torch.linalg.cholesky(Pzz)
            K = torch.cholesky_solve(Pxz.transpose(1, 2), Lz).transpose(1, 2)   # Eq. 29
            innov = zt - zh
            x = xp + (K @ innov[..., None])[..., 0]                       # Eq. 30
            # Eq. 31 evaluated in the (algebraically identical) Joseph form for numerical PD-ness
            IKH = torch.eye(N, dtype=DT, device=DEV) - K @ Hc
            P = IKH @ Pp @ IKH.transpose(1, 2) + K @ K.transpose(1, 2)
            P = (P + P.transpose(1, 2)) / 2
            eta[:, k] = innov.norm(dim=1) / (zt.norm(dim=1) * zh.norm(dim=1))   # Eq. 33
            nu[:, k] = innov.float()
            nun[:, k] = (innov / torch.diagonal(Pzz, dim1=1, dim2=2).sqrt()).float()
            xs[:, k] = x.float()
        return dict(eta=eta.float(), nu=nu, nun=nun, xhat=xs)


# ----------------------------------------------------------------------------------- VAR(1) Yule-Walker (SP Eqs. 8-11)
def fit_var1(Xs):
    """Xs: list/array (E,T,d) of training series. Returns phi (d,d), S_hat (d,d)."""
    X = np.concatenate([x for x in Xs], 0) if isinstance(Xs, list) else Xs.reshape(-1, Xs.shape[-1])
    zeta = Xs.reshape(-1, Xs.shape[-1]).mean(0)
    Xc = Xs - zeta
    d = Xs.shape[-1]
    P0 = np.einsum("etd,etf->df", Xc, Xc) / Xc[:, :].shape[0] / Xc.shape[1]
    P1 = np.einsum("etd,etf->df", Xc[:, 1:], Xc[:, :-1]) / Xc.shape[0] / (Xc.shape[1] - 1)
    phi = P1 @ np.linalg.pinv(P0)
    S = P0 - phi @ P1.T
    S = (S + S.T) / 2
    w, V = np.linalg.eigh(S)
    S = V @ np.diag(np.maximum(w, 1e-10)) @ V.T
    return phi, S


# ----------------------------------------------------------------------------------- main episode generator
def generate(system, split, n_ep, mode="hard", seed=0, T=T_DEFAULT, corr_wind=True, noise_scale=1.0,
             load_scale=None, mimic=False, kappa_range=None, S_hat=None, use_corr_filter=True,
             class_probs=None, save_nu=True, chunk=1000, verbose=True, fixed_anchor=None, mimic_blend=None,
             benign_ood=None, ext=None, tau_range=(60, 141), library=False):
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    grid, region = build(system)
    wind, load, pf, anchors = sample_profiles(grid, region, split, n_ep, T, rng, corr_wind, noise_scale, load_scale,
                                              fixed_anchor, ext)
    if fixed_anchor is not None:
        pf[:] = 0.95
    tau = rng.integers(tau_range[0], tau_range[1], size=n_ep)
    if mode == "src":
        cls = np.full(n_ep, 3)
    elif mode == "normal":
        cls = np.zeros(n_ep, int)
    else:
        p = class_probs or [0.4, 0.15, 0.15, 0.15, 0.15]
        cls = rng.choice(5, size=n_ep, p=p)
    extra = np.zeros_like(load)
    # ---- benign events. kind per episode: library events (training) and held-out events (benign_ood tests)
    kind = np.array([""] * n_ep, dtype=object)
    kind[cls == 1], kind[cls == 2] = "wind_ramp", "load_step"
    if library and mode == "hard":
        u = rng.random(n_ep)
        kind[(cls == 1) & (u < 0.3)] = "wind_trip"
        kind[(cls == 2) & (u < 0.25)] = "q_step"
        kind[(cls == 2) & (u >= 0.25) & (u < 0.5)] = "load_ramp"
        kind[(cls == 0) & (rng.random(n_ep) < 0.2)] = "sensor_drift"     # benign meter drift, labelled non-attack
    if benign_ood is not None:
        kind[:] = benign_ood
    wr, wr_info = apply_wind_ramp(wind, tau, rng)
    m1 = kind == "wind_ramp"
    wind[m1] = wr[m1]
    ex2, ls_info = apply_load_step(grid, region, extra, tau, rng)
    m2 = kind == "load_step"
    extra[m2] = ex2[m2]
    extra_q, Y_after = None, None
    cand = [b_ for b_ in region.I if b_ not in set(region.wind)]
    t_ = np.arange(T)[None]
    ar = np.arange(n_ep)
    m = kind == "load_ramp"
    if m.any():
        bus = rng.choice(cand, size=n_ep)
        mag = rng.choice([-1, 1], size=n_ep) * rng.uniform(0.05, 0.25, size=n_ep)
        Lr = rng.integers(30, 121, size=n_ep)
        prof = np.clip((t_ - tau[:, None] + 1) / Lr[:, None], 0, 1)
        extra[ar[m], :, bus[m]] += (prof * mag[:, None])[m]
    m = kind == "multi_step"                 # simultaneous steps at two buses (held out)
    if m.any():
        for _ in range(2):
            bus = rng.choice(cand, size=n_ep)
            mag = rng.choice([-1, 1], size=n_ep) * rng.uniform(0.05, 0.2, size=n_ep)
            extra[ar[m], :, bus[m]] += ((t_ >= tau[:, None]) * mag[:, None])[m]
    m = kind == "q_step"                     # capacitor / reactor switching
    if m.any():
        extra_q = np.zeros_like(load)
        bus = rng.choice(cand, size=n_ep)
        mag = rng.choice([-1, 1], size=n_ep) * rng.uniform(0.05, 0.2, size=n_ep)
        extra_q[ar[m], :, bus[m]] = ((t_ >= tau[:, None]) * mag[:, None])[m]
    m = kind == "wind_trip"                  # one farm disconnects within 1-5 s
    if m.any():
        j = rng.integers(0, wind.shape[2], size=n_ep)
        Lt = rng.integers(1, 6, size=n_ep)
        prof = np.clip((t_ - tau[:, None] + 1) / Lt[:, None], 0, 1)
        wind[ar[m], :, j[m]] *= (1 - prof)[m]
    Rset = set(region.R.tolist())
    m = kind == "ext_gen_trip"               # generator outage in the unobserved external network (held out)
    if m.any():
        gb = [b_ for b_ in np.where(grid.Pg0 > 0.05)[0] if b_ not in Rset and b_ != grid.slack]
        if gb:
            amt = grid.Pg0
        else:  # no external generator: trip of the largest external loads instead (sign reversed)
            gb = [b_ for b_ in np.argsort(-grid.Pd0) if b_ not in Rset and grid.Pd0[b_] > 0][:5]
            amt = -grid.Pd0
        bus = rng.choice(gb, size=n_ep)
        Lt = rng.integers(1, 6, size=n_ep)
        prof = np.clip((t_ - tau[:, None] + 1) / Lt[:, None], 0, 1)
        extra[ar[m], :, bus[m]] += (prof * amt[bus][:, None])[m]
    m = kind == "ext_line_out"               # line outage in the unobserved external network (held out)
    if m.any():
        ext_lines = [l for l in range(len(grid.br_f)) if grid.br_f[l] not in Rset and grid.br_t[l] not in Rset]
        ok_lines = []
        for l in ext_lines:  # keep outages that leave a convergent power flow at the base case
            Y = torch.tensor(grid.ybus_without(l), dtype=torch.complex128, device=DEV)
            P0 = torch.tensor(grid.Pg0 - grid.Pd0, dtype=DT, device=DEV)[None]
            Q0 = torch.tensor(-grid.Qd0, dtype=DT, device=DEV)[None]
            try:
                _, _, cv = grid.pf(P0, Q0, Y=Y)
            except Exception:   # islanding outage (singular Jacobian)
                continue
            if bool(cv.all()):
                ok_lines.append(l)
        line = rng.choice(ok_lines, size=n_ep)
        Ys = {l: grid.ybus_without(l) for l in ok_lines}
        Y_after = torch.tensor(np.stack([Ys[l] if m[i] else grid.Ybus_np for i, l in enumerate(line)]),
                               dtype=torch.complex128, device=DEV)
    drift = kind == "sensor_drift"
    drift_m = rng.integers(0, region.M, size=n_ep)
    drift_A = rng.choice([-1, 1], size=n_ep) * rng.uniform(3, 10, size=n_ep) * drift
    drift_L = rng.integers(30, 121, size=n_ep)
    t0 = time.time()
    Vm, Va, conv = run_pf(grid, wind, load, pf, extra, region, chunk=chunk, extra_q=extra_q, Y_after=Y_after, tau_sw=tau)
    t_pf = time.time() - t0
    # non-converged / non-finite episodes are replaced by a converged one and flagged (dropped at load time)
    conv &= np.isfinite(Vm).all((1, 2)) & np.isfinite(Va).all((1, 2)) & (Vm > 0.5).all((1, 2)) & (Vm < 1.5).all((1, 2))
    if not conv.all():
        g0 = int(np.argmax(conv))
        Vm[~conv] = Vm[g0]
        Va[~conv] = Va[g0]
    # ---- wind-injection statistics for the coupled filter (P_w, Q_w at internal wind buses)
    wi = [list(region.wind).index(w) for w in region.wind_int]
    tanphi = WIND_Q_SIGN * np.tan(np.arccos(pf))
    PQw = np.concatenate([wind[:, :, wi], wind[:, :, wi] * tanphi[:, None, wi]], -1)
    phi_w, S_emp = fit_var1(PQw[cls == 0] if (cls == 0).sum() > 10 else PQw)
    if S_hat is None:
        S_hat = S_emp
    # ---- measurements + attacks + CKF
    kr = kappa_range or ((0.01, 0.1) if mode == "src" else (0.001, 0.1))
    kappa = np.exp(rng.uniform(np.log(kr[0]), np.log(kr[1]), size=n_ep))
    ramp_L = rng.integers(30, 121, size=n_ep)
    _trw = profiles.ext_frame("train", *ext) if ext is not None else profiles.split_frame("train")
    C_w = np.corrcoef(_trw[[f"w{z}" for z in profiles.ZONES]].values.T)[:len(region.wind), :len(region.wind)]
    C_w = C_w[np.ix_(wi, wi)] if len(wi) else np.eye(1)
    ckf = CoupledCKF(region, S_hat, use_corr=use_corr_filter)
    out = {k: [] for k in ["eta", "nu", "nun", "xhat", "amp", "att_bus", "scale"]}
    t0 = time.time()
    for s in range(0, n_ep, chunk):
        e = min(n_ep, s + chunk)
        b = e - s
        Vm_t = torch.tensor(Vm[s:e], dtype=DT, device=DEV)
        Va_t = torch.tensor(Va[s:e], dtype=DT, device=DEV)
        x = region.state_from_full(Vm_t, Va_t)                             # (b,T,N)
        z = region.h(x)                                                     # clean
        a = torch.zeros_like(z)
        att = np.isin(cls[s:e], [3, 4])
        scale = np.zeros(b)
        if att.any():
            ia = np.where(att)[0]
            xt = x[ia, tau[s:e][ia]]
            dx = attack_direction(region, xt, C_w, kappa[s:e][ia], rng, mimic=mimic)
            if mimic_blend is not None:  # continuum between the standard (0) and the injection-mimicry (1) attack
                dxm = attack_direction(region, xt, C_w, kappa[s:e][ia], rng, mimic=True)
                dx = (1 - mimic_blend) * dx + mimic_blend * dxm
            sc = feasibility_scale(region, xt, dx, z[ia, tau[s:e][ia]])
            dx = dx * sc[:, None]
            scale[ia] = sc.cpu().numpy()
            tt = torch.arange(T, device=DEV)[None].to(DT)
            ta = torch.tensor(tau[s:e][ia], device=DEV, dtype=DT)[:, None]
            Ls = torch.tensor(ramp_L[s:e][ia], device=DEV, dtype=DT)[:, None]
            step = (tt >= ta).to(DT)
            ramp = torch.clamp((tt - ta + 1) / Ls, 0, 1)
            is_ramp = torch.tensor(cls[s:e][ia] == 4, device=DEV)[:, None]
            prof = torch.where(is_ramp, ramp, step)                        # (na,T)
            xa = x[ia] + ATT_EST_STD * torch.randn_like(x[ia])
            a_ia = region.h(xa + prof[..., None] * dx[:, None, :]) - region.h(xa)
            a[ia] = a_ia
        noise = SIG_V * noise_scale * torch.randn_like(z)
        zm = z + a + noise
        if drift[s:e].any():  # slowly drifting bias on one meter (benign meter fault)
            tt = torch.arange(T, device=DEV)[None].to(DT)
            mi = torch.tensor(drift_m[s:e], device=DEV)
            prof = torch.clamp((tt - torch.tensor(tau[s:e], device=DEV, dtype=DT)[:, None] + 1)
                               / torch.tensor(drift_L[s:e], device=DEV, dtype=DT)[:, None], 0, 1)
            bias = prof * torch.tensor(drift_A[s:e], device=DEV, dtype=DT)[:, None] * SIG_V
            zm[torch.arange(b, device=DEV), :, mi] += bias
        x0 = x[:, 0] + 1e-3 * torch.randn_like(x[:, 0])
        res = ckf.run(zm, x0)
        out["eta"].append(res["eta"].cpu().numpy())
        if save_nu:
            out["nun"].append(res["nun"].cpu().numpy().astype(np.float16))
            out["xhat"].append(res["xhat"].cpu().numpy().astype(np.float32))
        # attack amplitude (full-strength) and bus-level localisation labels
        amp = a.abs().amax(1).cpu().numpy()                                # (b,M)
        out["amp"].append(amp.astype(np.float32))
        ab = np.zeros((b, region.nR), bool)
        for m in range(region.M):
            ab[:, region.meas_bus[m]] |= amp[:, m] > 3 * SIG_V
        out["att_bus"].append(ab)
        out["scale"].append(scale)
        if verbose:
            print(f"  [{system}/{split}/{mode}] chunk {s}-{e} done ({time.time()-t0:.1f}s)", flush=True)
    t_ckf = time.time() - t0
    res = dict(eta=np.concatenate(out["eta"]), amp=np.concatenate(out["amp"]),
               att_bus=np.concatenate(out["att_bus"]), scale=np.concatenate(out["scale"]),
               cls=cls, tau=tau, kappa=kappa, ramp_L=ramp_L, conv=conv, anchors=anchors,
               S_hat=S_hat, S_emp=S_emp, t_pf=t_pf, t_ckf=t_ckf,
               ls_bus=ls_info["bus"], ls_mag=ls_info["mag"], wr_L=wr_info["L"],
               ev_kind=np.array([KINDS.index(k) if k in KINDS else -1 for k in kind]))
    if save_nu:
        res["nun"] = np.concatenate(out["nun"])
        res["xhat"] = np.concatenate(out["xhat"])
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--system", default="ieee14")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--mode", default="hard")
    a = ap.parse_args()
    t0 = time.time()
    r = generate(a.system, "train", a.n, mode=a.mode, seed=1)
    print("time", time.time() - t0, "pf", r["t_pf"], "ckf", r["t_ckf"], "conv", r["conv"].mean())
    cls, tau, eta = r["cls"], r["tau"], r["eta"]
    for c in range(5):
        m = cls == c
        if m.sum() == 0:
            continue
        pre = np.mean([eta[i, 20:tau[i]].mean() for i in np.where(m)[0]])
        post = np.mean([eta[i, tau[i]:].mean() for i in np.where(m)[0]])
        jump = np.mean([eta[i, tau[i]] for i in np.where(m)[0]])
        print(f"class {c} n={m.sum()} eta pre {pre:.3e} at-tau {jump:.3e} post {post:.3e}")
    am = r["amp"][np.isin(cls, [3, 4])]
    print("attack amp (mean over meas) quantiles", np.quantile(am.mean(1), [0.1, 0.5, 0.9]).round(4),
          "max", am.max().round(3), "scale mean", r["scale"][np.isin(cls, [3, 4])].mean().round(3))
    print("S_hat diag", np.diag(r["S_hat"]).round(6))
