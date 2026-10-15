"""SOURCE_METHOD_REPRODUCTION: DQN-based FDIA detector on the scalar coupled-CKF metric eta (SP Sec. V).

POMDP (SP Fig. 4, Eq. 38): state s_t = eta_t (scaled by the TRAIN pre-attack mean, A11b);
actions {0: continue, 1: stop}; costs: stop before tau -> 1 (false alarm, terminal); continue after
tau -> c_beta per step; stop at/after tau -> 0 (terminal). Rewards are negative costs.
DQN with evaluation/target networks and experience replay (SP Eqs. 36-37).
"""
import time
import numpy as np
import torch
import torch.nn as nn

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
T0 = 20  # CKF burn-in steps excluded from the decision horizon


class QNet(nn.Module):
    def __init__(self, d_in=1, h=64):
        super().__init__()
        self.f = nn.Sequential(nn.Linear(d_in, h), nn.ReLU(), nn.Linear(h, h), nn.ReLU(), nn.Linear(h, 2))

    def forward(self, s):
        return self.f(s)


class DQNDetector:
    def __init__(self, c_beta=0.05, gamma=0.99, eps=0.1, lr=1e-3, optim="adam", n_episodes=10000,
                 n_envs=500, batch=1024, updates_per_step=1, buffer=200_000, target_sync=500, seed=0,
                 hidden=64, counterfactual=True):
        self.__dict__.update(locals())
        del self.__dict__["self"]
        torch.manual_seed(seed)
        self.rng = np.random.default_rng(seed)
        self.d_in = 1
        self.q = QNet(1, hidden).to(DEV)
        self.qt = QNet(1, hidden).to(DEV)
        self.qt.load_state_dict(self.q.state_dict())
        self.ref = 1.0

    def set_vector_input(self, d_in):
        """Candidate C5: vector-state DQN (observations passed as (E,T,D) arrays, already standardized)."""
        self.d_in = d_in
        self.q = QNet(d_in, self.hidden).to(DEV)
        self.qt = QNet(d_in, self.hidden).to(DEV)
        self.qt.load_state_dict(self.q.state_dict())
        return self

    def _obs(self, eta):
        if self.d_in > 1:
            return torch.as_tensor(eta, dtype=torch.float32, device=DEV)
        return torch.as_tensor(np.log(eta / self.ref), dtype=torch.float32, device=DEV)[..., None]

    def fit(self, eta, tau, is_attack):
        """eta (E,T); tau (E,) onset; is_attack (E,) bool (non-attack episodes: any stop = false alarm)."""
        E, T = eta.shape[:2]
        if self.d_in == 1:
            pre = np.concatenate([eta[i, T0:(tau[i] if is_attack[i] else T)] for i in range(E)])
            self.ref = float(np.exp(np.mean(np.log(pre))))
        opt = (torch.optim.Adam(self.q.parameters(), lr=self.lr) if self.optim == "adam"
               else torch.optim.SGD(self.q.parameters(), lr=self.lr))
        cap = self.buffer
        D_ = self.d_in
        bs = torch.zeros(cap, D_, device=DEV); ba = torch.zeros(cap, dtype=torch.long, device=DEV)
        br = torch.zeros(cap, device=DEV); bs2 = torch.zeros(cap, D_, device=DEV); bd = torch.zeros(cap, device=DEV)
        ptr, full, n_upd = 0, False, 0
        t0 = time.time()
        order = self.rng.choice(E, size=self.n_episodes, replace=self.n_episodes > E)
        tau_eff = np.where(is_attack, tau, 10 ** 9)
        self.losses = []
        for b0 in range(0, self.n_episodes, self.n_envs):
            idx = order[b0:b0 + self.n_envs]
            P = len(idx)
            obs_all = self._obs(eta[idx])                                   # (P,T,D)
            ta = torch.as_tensor(tau_eff[idx], device=DEV)
            alive = torch.ones(P, dtype=torch.bool, device=DEV)
            for t in range(T0, T):
                if not alive.any():
                    break
                s = obs_all[:, t]
                with torch.no_grad():
                    greedy = self.q(s).argmax(1)
                rnd = torch.rand(P, device=DEV) < self.eps
                a = torch.where(rnd, torch.randint(0, 2, (P,), device=DEV), greedy)
                post = t >= ta
                cost = torch.where(a == 1, torch.where(post, 0.0, 1.0), torch.where(post, self.c_beta, 0.0))
                done = (a == 1) | (t == T - 1)
                s2 = obs_all[:, min(t + 1, T - 1)]
                if self.counterfactual:
                    # observations are exogenous (eta does not depend on the action): store the outcome of BOTH
                    # actions at every step of every episode (documented deviation D2)
                    cs = torch.cat([s, s]); ca = torch.cat([torch.zeros_like(a), torch.ones_like(a)])
                    cc = torch.cat([torch.where(post, self.c_beta, 0.0), torch.where(post, 0.0, 1.0)])
                    cd = torch.cat([torch.full_like(post, t == T - 1), torch.ones_like(post)])
                    cs2 = torch.cat([s2, s2])
                    m = torch.ones(2 * P, dtype=torch.bool, device=DEV)
                else:
                    cs, ca, cc, cd, cs2, m = s, a, cost, done, s2, alive
                k = int(m.sum())
                ii = (torch.arange(k, device=DEV) + ptr) % cap
                bs[ii] = cs[m]; ba[ii] = ca[m]; br[ii] = -cc[m].float(); bs2[ii] = cs2[m]; bd[ii] = cd[m].float()
                ptr = (ptr + k) % cap
                full = full or ptr < k
                if not self.counterfactual:
                    alive = alive & ~done
                size = cap if full else ptr
                if size < 2000:
                    continue
                for _ in range(self.updates_per_step):
                    j = torch.randint(0, size, (self.batch,), device=DEV)
                    q = self.q(bs[j]).gather(1, ba[j][:, None])[:, 0]
                    with torch.no_grad():
                        y = br[j] + self.gamma * (1 - bd[j]) * self.qt(bs2[j]).max(1).values   # Eq. 37
                    loss = ((y - q) ** 2).mean()
                    opt.zero_grad(); loss.backward(); opt.step()
                    n_upd += 1
                    if n_upd % self.target_sync == 0:
                        self.qt.load_state_dict(self.q.state_dict())
                    self.losses.append(loss.item()) if n_upd % 200 == 0 else None
        self.train_time = time.time() - t0
        return self

    @torch.no_grad()
    def alarm_times(self, eta):
        """Greedy policy; returns first stop time per episode (T if never)."""
        E, T = eta.shape[:2]
        obs = self._obs(eta)
        q = self.q(obs.reshape(-1, self.d_in)).reshape(E, T, 2)
        stop = (q[..., 1] > q[..., 0]).cpu().numpy()
        stop[:, :T0] = False
        first = np.where(stop.any(1), stop.argmax(1), T)
        return first

    @torch.no_grad()
    def stop_margin(self, eta):
        E, T = eta.shape
        q = self.q(self._obs(eta).reshape(-1, 1)).reshape(E, T, 2)
        return (q[..., 1] - q[..., 0]).cpu().numpy()


def detection_metrics(alarm, tau, is_attack, T):
    """SP trial-level metrics. alarm: first alarm time (T = never)."""
    alarm = np.asarray(alarm); tau = np.asarray(tau); is_attack = np.asarray(is_attack, bool)
    fa = (alarm < T) & (~is_attack | (alarm < tau))           # false alarm
    tp = is_attack & (alarm >= tau) & (alarm < T)
    miss = is_attack & (alarm >= T)
    timely = is_attack & (alarm == tau)
    delay = np.where(tp, alarm - tau, np.where(miss, T - tau, 0))
    n_att = max(is_attack.sum(), 1)
    n_all = len(alarm)
    prec = tp.sum() / max(tp.sum() + fa.sum(), 1)
    rec = tp.sum() / n_att
    f1 = 2 * prec * rec / max(prec + rec, 1e-12)
    return dict(timely_rate=timely.sum() / n_att, far=fa.sum() / n_all, far_nonattack=(fa & ~is_attack).sum() / max((~is_attack).sum(), 1),
                delay_s=float(delay[is_attack & ~(alarm < tau)].mean()) if (is_attack & ~(alarm < tau)).any() else np.nan,
                miss_ratio=miss.sum() / n_att, precision=prec, recall=rec, f_score=f1)
