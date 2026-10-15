"""Grid models: IEEE cases, batched AC power flow (GPU), reduced-order region measurement model.

All quantities are in per unit on a 100 MVA base. Bus numbers in public APIs are 1-based
(as in the IEEE case files); internal arrays are 0-based.
"""
import warnings
import numpy as np
import torch

warnings.filterwarnings("ignore")

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DT = torch.float64

# ---------------------------------------------------------------------------
# Source-paper region definitions (SP Sec. VI-A). Buses are 1-based.
# ---------------------------------------------------------------------------
REGIONS = {
    "ieee14": dict(region=[6, 9, 10, 11, 12, 13, 14], boundary=[6, 9], wind=[9, 10, 11]),
    "ieee30": dict(region=[15, 18, 19, 20, 23, 24, 25, 26, 27, 28, 29, 30],
                   boundary=[15, 20, 24, 28], wind=[15, 18, 19]),
    "ieee57": dict(region=list(range(1, 19)) + [27, 28, 29, 43, 45, 46] + list(range(49, 56)),
                   boundary=[11, 18, 27, 43, 49], wind=[5, 6, 7, 8]),
    "ieee118": dict(region=list(range(1, 24)) + list(range(25, 33)) + [113, 114, 115, 117],
                    boundary=[15, 19, 23, 30], wind=[5, 6, 7, 16]),
}


def load_case(name):
    """Return a MATPOWER-style dict with numpy arrays and consecutive 1-based bus numbers."""
    if name == "ieee30":
        import pandapower as pp
        import pandapower.networks as pn
        net = pn.case_ieee30()
        pp.runpp(net)
        p = net._ppc
        bus = np.real(p["bus"][:, :13]).astype(float).copy()
        bus[:, 0] += 1
        branch = np.real(p["branch"][:, :13]).astype(float).copy()
        branch[:, 0] += 1
        branch[:, 1] += 1
        gen = np.real(p["gen"][:, :10]).astype(float).copy()
        gen[:, 0] += 1
        gen[:, 7] = 1.0
        return dict(baseMVA=float(p["baseMVA"]), bus=bus, branch=branch, gen=gen)
    mod = {"ieee14": "case14", "ieee57": "case57", "ieee118": "case118"}[name]
    m = __import__(f"pypower.{mod}", fromlist=[mod])
    c = getattr(m, mod)()
    return dict(baseMVA=float(c["baseMVA"]), bus=c["bus"].astype(float), branch=c["branch"].astype(float),
                gen=c["gen"].astype(float))


def branch_admittances(case):
    """Per-branch (Yff, Yft, Ytf, Ytt) following MATPOWER makeYbus."""
    br = case["branch"]
    st = br[:, 10] > 0
    br = br[st]
    r, x, b = br[:, 2], br[:, 3], br[:, 4]
    ys = 1.0 / (r + 1j * x)
    tap = np.where(br[:, 8] == 0, 1.0, br[:, 8]) * np.exp(1j * np.pi / 180 * br[:, 9])
    Ytt = ys + 1j * b / 2
    Yff = Ytt / (tap * np.conj(tap))
    Yft = -ys / np.conj(tap)
    Ytf = -ys / tap
    f = br[:, 0].astype(int) - 1
    t = br[:, 1].astype(int) - 1
    return f, t, Yff, Yft, Ytf, Ytt


class Grid:
    def __init__(self, name):
        self.name = name
        c = load_case(name)
        self.case = c
        base = c["baseMVA"]
        bus, gen = c["bus"], c["gen"]
        n = bus.shape[0]
        self.n = n
        f, t, Yff, Yft, Ytf, Ytt = branch_admittances(c)
        Y = np.zeros((n, n), complex)
        np.add.at(Y, (f, f), Yff)
        np.add.at(Y, (f, t), Yft)
        np.add.at(Y, (t, f), Ytf)
        np.add.at(Y, (t, t), Ytt)
        Y[np.arange(n), np.arange(n)] += (bus[:, 4] + 1j * bus[:, 5]) / base
        self.Ybus_np = Y
        self.Ybus = torch.tensor(Y, dtype=torch.complex128, device=DEV)
        self.base_bus = bus
        self.br_f, self.br_t = f, t
        self.br_Y = (Yff, Yft, Ytf, Ytt)
        btype = bus[:, 1].astype(int)
        self.slack = int(np.where(btype == 3)[0][0])
        on = gen[:, 7] > 0
        gbus = gen[on, 0].astype(int) - 1
        self.Pd0 = bus[:, 2] / base
        self.Qd0 = bus[:, 3] / base
        self.Pg0 = np.zeros(n)
        np.add.at(self.Pg0, gbus, gen[on, 1] / base)
        self.Vset = np.ones(n)
        self.Vset[gbus] = gen[on, 5]
        pv = np.zeros(n, bool)
        pv[gbus] = True
        pv[self.slack] = False
        self.pv = np.where(pv)[0]
        self.pq = np.array([i for i in range(n) if i not in set(self.pv) and i != self.slack])
        self.nonslack = np.array([i for i in range(n) if i != self.slack])
        self.Va0 = np.deg2rad(bus[:, 8])
        self.Vm0 = bus[:, 7]

    def ybus_without(self, line):
        """Bus admittance matrix with branch `line` removed (numpy)."""
        Yff, Yft, Ytf, Ytt = self.br_Y
        f, t = self.br_f[line], self.br_t[line]
        Y = self.Ybus_np.copy()
        Y[f, f] -= Yff[line]; Y[f, t] -= Yft[line]; Y[t, f] -= Ytf[line]; Y[t, t] -= Ytt[line]
        return Y

    # ------------------------------------------------------------------ power flow
    def pf(self, Pinj, Qinj, Vm0=None, Va0=None, iters=12, tol=1e-9, Y=None):
        """Batched Newton-Raphson power flow.

        Pinj, Qinj: (B, n) torch net injections (gen - load) for non-slack (P) / PQ (Q) buses.
        Generator reactive limits are ignored (documented assumption).
        Returns Vm, Va (B, n) and a convergence mask (B,).
        """
        B, n = Pinj.shape
        Vset = torch.tensor(self.Vset, dtype=DT, device=DEV)
        Vm = (torch.tensor(self.Vm0, dtype=DT, device=DEV).expand(B, n).clone() if Vm0 is None else Vm0.clone())
        Va = (torch.tensor(self.Va0, dtype=DT, device=DEV).expand(B, n).clone() if Va0 is None else Va0.clone())
        pvsl = np.concatenate([self.pv, [self.slack]])
        Vm[:, pvsl] = Vset[pvsl]
        ns = torch.tensor(self.nonslack, device=DEV)
        pq = torch.tensor(self.pq, device=DEV)
        Y = self.Ybus if Y is None else Y          # optional per-sample admittance (B,n,n)
        Yb = Y[None] if Y.dim() == 2 else Y
        Ssp = torch.complex(Pinj, Qinj)
        for it in range(iters):
            V = torch.polar(Vm, Va)
            I = (Yb @ V[..., None])[..., 0]
            S = V * I.conj()
            mis = S - Ssp
            F = torch.cat([mis.real[:, ns], mis.imag[:, pq]], 1)
            err = F.abs().amax(1)
            if err.max() < tol:
                break
            Vn = V / Vm
            # dS/dVm = diag(V) conj(Y diag(Vn)) + conj(diag(I)) diag(Vn)
            dS_dVm = V[:, :, None] * (Yb * Vn[:, None, :]).conj()
            dS_dVm = dS_dVm + torch.diag_embed(I.conj() * Vn)
            # dS/dVa = j diag(V) conj(diag(I) - Y diag(V))
            dS_dVa = 1j * V[:, :, None] * (torch.diag_embed(I) - Yb * V[:, None, :]).conj()
            J11 = dS_dVa.real[:, ns][:, :, ns]
            J12 = dS_dVm.real[:, ns][:, :, pq]
            J21 = dS_dVa.imag[:, pq][:, :, ns]
            J22 = dS_dVm.imag[:, pq][:, :, pq]
            J = torch.cat([torch.cat([J11, J12], 2), torch.cat([J21, J22], 2)], 1)
            dx = torch.linalg.solve(J, -F)
            Va[:, ns] += dx[:, : len(ns)]
            Vm[:, pq] += dx[:, len(ns):]
        conv = err < 1e-6
        return Vm, Va, conv


class Region:
    """Reduced-order (partially observable) region model used by attacker and defender.

    State x = [theta_{R \\ ref}, V_R] with the first listed boundary bus as local angle reference.
    Measurements z = [V_R, P_I, Q_I, Pf_L, Qf_L, Pt_L, Qt_L] where I = internal (non-boundary)
    buses and L = lines with both terminals in the region. Boundary-bus injections are excluded
    because they contain flows to the unknown external network (assumption A14).
    """

    def __init__(self, grid: Grid, spec):
        self.grid = grid
        R = np.array(sorted(spec["region"])) - 1
        self.R = R
        inR = np.zeros(grid.n, bool)
        inR[R] = True
        f, t = grid.br_f, grid.br_t
        ext = [b for b in R if np.any((f == b) & ~inR[t]) or np.any((t == b) & ~inR[f])]
        self.boundary_computed = np.array(sorted(ext)) + 1
        self.boundary = np.array(spec["boundary"]) - 1
        # Use topology-consistent boundary: union of stated and computed (any bus with external ties)
        bset = sorted(set(self.boundary.tolist()) | set(ext))
        self.B = np.array(bset)
        self.I = np.array([b for b in R if b not in set(bset)])
        self.ref = self.boundary[0]
        Lmask = inR[f] & inR[t]
        self.L = np.where(Lmask)[0]
        self.wind = np.array(spec["wind"]) - 1
        # local indices
        self.pos = {b: k for k, b in enumerate(R)}
        self.nR = len(R)
        self.th_idx = np.array([k for k, b in enumerate(R) if b != self.ref])  # positions in R carrying an angle state
        self.N = len(self.th_idx) + self.nR
        self.nI = len(self.I)
        self.nL = len(self.L)
        self.M = self.nR + 2 * self.nI + 4 * self.nL
        # measurement index maps
        o = 0
        self.mV = np.arange(o, o + self.nR); o += self.nR
        self.mP = np.arange(o, o + self.nI); o += self.nI
        self.mQ = np.arange(o, o + self.nI); o += self.nI
        self.mPf = np.arange(o, o + self.nL); o += self.nL
        self.mQf = np.arange(o, o + self.nL); o += self.nL
        self.mPt = np.arange(o, o + self.nL); o += self.nL
        self.mQt = np.arange(o, o + self.nL); o += self.nL
        # wind buses: internal ones have P/Q injection measurements
        self.wind_int = np.array([w for w in self.wind if w in set(self.I)])
        self.I_pos = {b: k for k, b in enumerate(self.I)}
        self.mPw = np.array([self.mP[self.I_pos[w]] for w in self.wind_int], dtype=int)
        self.mQw = np.array([self.mQ[self.I_pos[w]] for w in self.wind_int], dtype=int)
        # tensors
        Y = grid.Ybus_np
        self.Y_IR = torch.tensor(Y[np.ix_(self.I, R)], dtype=torch.complex128, device=DEV)
        Yff, Yft, Ytf, Ytt = grid.br_Y
        L = self.L
        self.Lf = torch.tensor([self.pos[b] for b in f[L]], device=DEV)
        self.Lt = torch.tensor([self.pos[b] for b in t[L]], device=DEV)
        self.Iloc = torch.tensor([self.pos[b] for b in self.I], device=DEV)
        self.bY = [torch.tensor(a[L], dtype=torch.complex128, device=DEV) for a in (Yff, Yft, Ytf, Ytt)]
        # adjacency among region buses (for graph models) and bus index of each measurement
        A = np.zeros((self.nR, self.nR))
        for lf, lt in zip(self.Lf.cpu().numpy(), self.Lt.cpu().numpy()):
            A[lf, lt] = A[lt, lf] = 1
        self.A = A
        mbus = np.zeros(self.M, int)
        mbus[self.mV] = np.arange(self.nR)
        mbus[self.mP] = [self.pos[b] for b in self.I]
        mbus[self.mQ] = [self.pos[b] for b in self.I]
        mbus[self.mPf] = self.Lf.cpu().numpy(); mbus[self.mQf] = self.Lf.cpu().numpy()
        mbus[self.mPt] = self.Lt.cpu().numpy(); mbus[self.mQt] = self.Lt.cpu().numpy()
        self.meas_bus = mbus
        mtype = np.zeros(self.M, int)
        for k, arr in enumerate([self.mV, self.mP, self.mQ, self.mPf, self.mQf, self.mPt, self.mQt]):
            mtype[arr] = k
        self.meas_type = mtype
        # attackable states: internal buses (boundary states untouched, Liu & Li 2017)
        self.att_th = np.array([j for j, k in enumerate(self.th_idx) if R[k] in set(self.I)])
        self.att_v = len(self.th_idx) + np.array([k for k, b in enumerate(R) if b in set(self.I)])
        self.att_idx = np.concatenate([self.att_th, self.att_v])

    # ---- state <-> full grid
    def state_from_full(self, Vm, Va):
        """Vm, Va: (..., n) full-grid; returns x (..., N)."""
        th = Va[..., self.R] - Va[..., [self.ref]]
        return torch.cat([th[..., self.th_idx], Vm[..., self.R]], -1)

    def unpack(self, x):
        nth = len(self.th_idx)
        if not hasattr(self, "_thgather"):
            g = np.zeros(self.nR, int)
            g[self.th_idx] = np.arange(1, nth + 1)
            self._thgather = torch.tensor(g, device=DEV)
        z0 = torch.zeros_like(x[..., :1])
        th = torch.cat([z0, x[..., :nth]], -1)[..., self._thgather]
        V = x[..., nth:]
        return th, V

    def h(self, x):
        """Reduced measurement function, x (..., N) -> z (..., M)."""
        th, Vm = self.unpack(x)
        V = torch.polar(Vm, th)
        S_I = V[..., self.Iloc] * (V @ self.Y_IR.T).conj()
        Vf, Vt = V[..., self.Lf], V[..., self.Lt]
        Yff, Yft, Ytf, Ytt = self.bY
        Sf = Vf * (Yff * Vf + Yft * Vt).conj()
        St = Vt * (Ytf * Vf + Ytt * Vt).conj()
        return torch.cat([Vm, S_I.real, S_I.imag, Sf.real, Sf.imag, St.real, St.imag], -1)

    def jac(self, x):
        """Batched Jacobian dh/dx: x (B, N) -> (B, M, N)."""
        return torch.func.vmap(torch.func.jacfwd(self.h))(x)


def build(name):
    g = Grid(name)
    r = Region(g, REGIONS[name])
    return g, r


if __name__ == "__main__":
    for name in REGIONS:
        g, r = build(name)
        B = 4
        P = torch.tensor(g.Pg0 - g.Pd0, dtype=DT, device=DEV).expand(B, -1).clone()
        Q = torch.tensor(-g.Qd0, dtype=DT, device=DEV).expand(B, -1).clone()
        Vm, Va, conv = g.pf(P, Q)
        err = np.abs(Vm[0].cpu().numpy() - g.Vm0).max()
        x = r.state_from_full(Vm, Va)
        z = r.h(x)
        J = r.jac(x)
        print(f"{name}: n={g.n} conv={bool(conv.all())} |Vm-case|max={err:.2e} region={r.nR} "
              f"stated boundary={sorted((r.boundary+1).tolist())} computed={r.boundary_computed.tolist()} "
              f"I={r.nI} L={r.nL} N={r.N} M={r.M} wind_int={(r.wind_int+1).tolist()} J={tuple(J.shape)}")
