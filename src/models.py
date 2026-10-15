"""Causal per-step sequence models. Every model maps an episode (B,T,C) of flat defender features
(or node tokens) to per-step class logits (B,T,K). Graph models additionally emit per-bus attack
logits (B,T,nR) for localisation.
"""
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

K = 5


class CausalConv(nn.Module):
    def __init__(self, cin, cout, k, d=1, groups=1):
        super().__init__()
        self.pad = (k - 1) * d
        self.conv = nn.Conv1d(cin, cout, k, dilation=d, groups=groups)

    def forward(self, x):          # x (B,C,T)
        return self.conv(F.pad(x, (self.pad, 0)))


class TCNBlock(nn.Module):
    def __init__(self, c, k=3, d=1, drop=0.1):
        super().__init__()
        self.c1 = CausalConv(c, c, k, d)
        self.c2 = CausalConv(c, c, k, d)
        self.n1 = nn.GroupNorm(1, c)
        self.n2 = nn.GroupNorm(1, c)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        h = self.drop(F.gelu(self.n1(self.c1(x))))
        h = self.drop(F.gelu(self.n2(self.c2(h))))
        return x + h


# ----------------------------------------------------------------------------------- flat baselines
class FlatTCN(nn.Module):
    """C3 / benchmark: whitened-innovation TCN on flat channels (receptive field 31 for k=3, d=1..8)."""

    def __init__(self, cin, hid=64, dil=(1, 2, 4, 8), drop=0.1, **_):
        super().__init__()
        self.inp = nn.Conv1d(cin, hid, 1)
        self.blocks = nn.Sequential(*[TCNBlock(hid, 3, d, drop) for d in dil])
        self.out = nn.Linear(hid, K)

    def forward(self, x, **_):
        h = self.blocks(self.inp(x.transpose(1, 2)))
        return self.out(h.transpose(1, 2))


class CNN1D(nn.Module):
    """Benchmark: plain causal 1-D CNN (no dilation, no residual)."""

    def __init__(self, cin, hid=64, drop=0.1, **_):
        super().__init__()
        self.net = nn.Sequential(CausalConv(cin, hid, 5), nn.GELU(), nn.Dropout(drop),
                                 CausalConv(hid, hid, 5), nn.GELU(), nn.Dropout(drop),
                                 CausalConv(hid, hid, 5), nn.GELU())
        self.out = nn.Linear(hid, K)

    def forward(self, x, **_):
        return self.out(self.net(x.transpose(1, 2)).transpose(1, 2))


class RNN(nn.Module):
    def __init__(self, cin, hid=64, cell="gru", layers=2, drop=0.1, **_):
        super().__init__()
        self.inp = nn.Linear(cin, hid)
        R = nn.GRU if cell == "gru" else nn.LSTM
        self.rnn = R(hid, hid, layers, batch_first=True, dropout=drop)
        self.out = nn.Linear(hid, K)

    def forward(self, x, **_):
        h, _ = self.rnn(F.gelu(self.inp(x)))
        return self.out(h)


class MLPStep(nn.Module):
    """Benchmark: per-step MLP on causal window summaries (computed inside, window W)."""

    def __init__(self, cin, hid=256, W=32, drop=0.1, **_):
        super().__init__()
        self.W = W
        self.net = nn.Sequential(nn.Linear(3 * cin, hid), nn.GELU(), nn.Dropout(drop),
                                 nn.Linear(hid, hid), nn.GELU(), nn.Dropout(drop), nn.Linear(hid, K))

    def forward(self, x, **_):
        return self.net(window_summary(x, self.W))


def window_summary(x, W):
    """Causal trailing-window mean, mean|.| and last value: (B,T,C)->(B,T,3C)."""
    c = torch.cumsum(F.pad(x, (0, 0, W, 0)), 1)
    ca = torch.cumsum(F.pad(x.abs(), (0, 0, W, 0)), 1)
    m = (c[:, W:] - c[:, :-W]) / W
    ma = (ca[:, W:] - ca[:, :-W]) / W
    return torch.cat([m, ma, x], -1)


class CausalTransformer(nn.Module):
    """Benchmark: causal Transformer encoder with local attention window W."""

    def __init__(self, cin, hid=64, layers=2, heads=4, W=32, drop=0.1, **_):
        super().__init__()
        self.inp = nn.Linear(cin, hid)
        self.pos = nn.Parameter(torch.zeros(1, 512, hid))
        nn.init.normal_(self.pos, std=0.02)
        el = nn.TransformerEncoderLayer(hid, heads, 2 * hid, drop, batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(el, layers)
        self.out = nn.Linear(hid, K)
        self.W = W

    def _fwd(self, x):
        B, T, _ = x.shape
        h = self.inp(x) + self.pos[:, :T]
        i = torch.arange(T, device=x.device)
        mask = (i[None, :] > i[:, None]) | (i[:, None] - i[None, :] >= self.W)
        return self.out(self.enc(h, mask=mask))

    def forward(self, x, chunk=200, ctx=64, **_):
        T = x.shape[1]
        if T <= chunk:
            return self._fwd(x)
        # long streams: overlapping causal chunks keep positions inside the trained range (ctx >= layers * W)
        outs = [self._fwd(x[:, :chunk])]
        s = chunk
        while s < T:
            a = s - ctx
            y = self._fwd(x[:, a:min(a + chunk, T)])
            outs.append(y[:, ctx:])
            s = a + chunk
        return torch.cat(outs, 1)[:, :T]


class STDCNN(nn.Module):
    """Benchmark in the spirit of the spatio-temporal DCNN detector [SP ref. 38]: 2-D convolutions on the
    (time x measurement) image of innovations, causal in time."""

    def __init__(self, cin, hid=32, drop=0.1, **_):
        super().__init__()
        self.c1 = nn.Conv2d(1, hid, (5, 5), padding=(0, 2))
        self.c2 = nn.Conv2d(hid, hid, (5, 5), padding=(0, 2), stride=(1, 2))
        self.pool = nn.AdaptiveAvgPool2d((None, 8))
        self.out = nn.Sequential(nn.Linear(hid * 8, 64), nn.GELU(), nn.Dropout(drop), nn.Linear(64, K))

    def forward(self, x, **_):
        h = x[:, None]                               # (B,1,T,C)
        h = F.gelu(self.c1(F.pad(h, (0, 0, 4, 0))))
        h = F.gelu(self.c2(F.pad(h, (0, 0, 4, 0))))
        h = self.pool(h)                             # (B,hid,T,8)
        B, Cc, T, _ = h.shape
        return self.out(h.permute(0, 2, 1, 3).reshape(B, T, -1))


class SelectiveSSM(nn.Module):
    """C4 / benchmark: Mamba-style selective state-space layer (diagonal, input-dependent step size),
    implemented as an explicit causal scan."""

    def __init__(self, d, n=16):
        super().__init__()
        self.inx = nn.Linear(d, 2 * d)
        self.conv = CausalConv(d, d, 4, groups=d)
        self.dt = nn.Linear(d, d)
        self.Bp = nn.Linear(d, n)
        self.Cp = nn.Linear(d, n)
        self.A_log = nn.Parameter(torch.log(torch.arange(1, n + 1).float()).repeat(d, 1))
        self.D = nn.Parameter(torch.ones(d))
        self.out = nn.Linear(d, d)

    def forward(self, x):                             # (B,T,d)
        u, g = self.inx(x).chunk(2, -1)
        u = F.silu(self.conv(u.transpose(1, 2)).transpose(1, 2))
        dt = F.softplus(self.dt(u))                   # (B,T,d)
        A = -torch.exp(self.A_log)                    # (d,n)
        Bm, Cm = self.Bp(u), self.Cp(u)               # (B,T,n)
        dA = torch.exp(dt[..., None] * A)             # (B,T,d,n)
        dBu = dt[..., None] * Bm[:, :, None, :] * u[..., None]
        h = torch.zeros(x.shape[0], x.shape[2], A.shape[1], device=x.device)
        ys = []
        for t in range(x.shape[1]):
            h = dA[:, t] * h + dBu[:, t]
            ys.append((h * Cm[:, t, None, :]).sum(-1))
        y = torch.stack(ys, 1) + u * self.D
        return self.out(y * F.silu(g))


class Mamba(nn.Module):
    def __init__(self, cin, hid=64, layers=2, drop=0.1, **_):
        super().__init__()
        self.inp = nn.Linear(cin, hid)
        self.layers = nn.ModuleList([SelectiveSSM(hid) for _ in range(layers)])
        self.norms = nn.ModuleList([nn.LayerNorm(hid) for _ in range(layers)])
        self.drop = nn.Dropout(drop)
        self.out = nn.Linear(hid, K)

    def forward(self, x, **_):
        h = self.inp(x)
        for l, n in zip(self.layers, self.norms):
            h = h + self.drop(l(n(h)))
        return self.out(h)


# ----------------------------------------------------------------------------------- graph models
def norm_adj(A):
    A = A + np.eye(len(A))
    d = A.sum(1)
    return (A / np.sqrt(d[:, None] * d[None, :])).astype(np.float32)


class GCNGRU(nn.Module):
    """Benchmark: temporal GNN (2-layer GCN per step on bus tokens, then GRU on pooled embedding)."""

    def __init__(self, fnode, A, hid=64, drop=0.1, **_):
        super().__init__()
        self.register_buffer("A", torch.tensor(norm_adj(A)))
        self.g1 = nn.Linear(fnode, hid)
        self.g2 = nn.Linear(hid, hid)
        self.rnn = nn.GRU(2 * hid, hid, batch_first=True)
        self.out = nn.Linear(hid, K)
        self.drop = nn.Dropout(drop)

    def forward(self, x, nodes=None, **_):            # nodes (B,T,nR,F)
        h = F.gelu(self.A @ self.g1(nodes))
        h = F.gelu(self.A @ self.g2(self.drop(h)))
        p = torch.cat([h.mean(2), h.amax(2)], -1)
        o, _ = self.rnn(p)
        return self.out(o)


class BusGraphTCN(nn.Module):
    """C1: bus-typed graph + dilated causal TCN with node-level localisation head.

    Node tokens -> per-node causal TCN (shared weights) interleaved with graph propagation over the
    region adjacency -> attention pooling over buses -> class logits; per-node attack logits.
    Parameter count is independent of grid size (transferable across systems).
    """

    def __init__(self, fnode, A, hid=48, dil=(1, 2, 4, 8), drop=0.1, use_graph=True, **_):
        super().__init__()
        self.register_buffer("A", torch.tensor(norm_adj(A)))
        self.use_graph = use_graph
        self.inp = nn.Linear(fnode, hid)
        self.tcn = nn.ModuleList([TCNBlock(hid, 3, d, drop) for d in dil])
        self.gmix = nn.ModuleList([nn.Linear(hid, hid) for _ in dil])
        self.att = nn.Linear(hid, 1)
        self.out = nn.Sequential(nn.Linear(2 * hid, hid), nn.GELU(), nn.Linear(hid, K))
        self.node_out = nn.Linear(hid, 1)

    def set_graph(self, A):
        self.A = torch.tensor(norm_adj(A), device=self.A.device)

    def forward(self, x, nodes=None, return_nodes=False, **_):
        B, T, nR, Fn = nodes.shape
        h = self.inp(nodes)                                        # (B,T,nR,h)
        for tcn, gm in zip(self.tcn, self.gmix):
            z = h.permute(0, 2, 3, 1).reshape(B * nR, -1, T)
            z = tcn(z).reshape(B, nR, -1, T).permute(0, 3, 1, 2)   # (B,T,nR,h)
            if self.use_graph:
                z = z + F.gelu(gm(self.A @ z))
            h = z
        a = torch.softmax(self.att(h), 2)                          # (B,T,nR,1)
        pooled = torch.cat([(a * h).sum(2), h.amax(2)], -1)
        logits = self.out(pooled)
        if return_nodes:
            return logits, self.node_out(h)[..., 0]
        return logits


MODELS = {"flat_tcn": FlatTCN, "cnn1d": CNN1D, "gru": lambda cin, **k: RNN(cin, cell="gru", **k),
          "lstm": lambda cin, **k: RNN(cin, cell="lstm", **k), "mlp": MLPStep, "transformer": CausalTransformer,
          "stdcnn": STDCNN, "mamba": Mamba}
GRAPH_MODELS = {"gcn_gru": GCNGRU, "bus_graph_tcn": BusGraphTCN}


def n_params(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)
