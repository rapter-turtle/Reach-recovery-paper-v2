"""Value V(s) and policy of a raw training checkpoint (finalfinal/model/w095_ema_20k.pt).

Rebuilt from ris/ship_ris_heron.py of the finalfinal code: the observation is the 8 state
normalised by the state box with psi replaced by (sin, cos), the critics are 512 x 3 MLPs
with LayerNorm taking [obs, a, w], the two actors are 512 x 3 trunks to 64 then a linear head
with tanh, and V = min(Q1, Q2)(obs, pi(obs), mu(obs)).
"""
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

MINS = torch.tensor([-16.0, -3.0, -math.pi / 2, -0.8, -0.7, -0.9, -52.365333, -3.0])
MAXS = torch.tensor([2.0, 8.0, math.pi / 2, max(1.8, 1.10 * 0.0594 * 9.0 / 0.1985), 0.7, 0.9, 52.365333, 9.0])
MEAN, HALF = 0.5 * (MINS + MAXS), 0.5 * (MAXS - MINS)


def _mlp(i, o, h=512, ln=False):
    if ln:
        return nn.Sequential(nn.Linear(i, h), nn.LayerNorm(h), nn.ReLU(), nn.Linear(h, h), nn.LayerNorm(h),
                             nn.ReLU(), nn.Linear(h, h), nn.LayerNorm(h), nn.ReLU(), nn.Linear(h, o))
    return nn.Sequential(nn.Linear(i, h), nn.ReLU(), nn.Linear(h, h), nn.ReLU(), nn.Linear(h, h), nn.ReLU(),
                         nn.Linear(h, o))


class _Q(nn.Module):
    def __init__(self):
        super().__init__(); self.net = _mlp(14, 1, ln=True)


class _Pol(nn.Module):
    def __init__(self, out):
        super().__init__(); self.body = _mlp(9, 64); self.mu = nn.Linear(64, out)
        self.logstd = nn.Parameter(torch.zeros(out))


class Model:
    def __init__(self, path):
        ck = torch.load(str(path), map_location='cpu', weights_only=False)
        self.q1, self.q2, self.pi, self.adv = _Q(), _Q(), _Pol(2), _Pol(3)
        self.q1.load_state_dict(ck['q1']); self.q2.load_state_dict(ck['q2'])
        self.pi.load_state_dict(ck['pih']); self.adv.load_state_dict(ck['mu'])
        for m in (self.q1, self.q2, self.pi, self.adv):
            m.eval()
        self.meta = {k: v for k, v in ck.items() if not isinstance(v, dict)}

    @staticmethod
    def obs(s):
        z = (s - MEAN) / HALF
        return torch.cat([z[:, :2], torch.sin(s[:, 2:3]), torch.cos(s[:, 2:3]), z[:, 3:]], -1)

    @torch.no_grad()
    def value(self, states):
        out = []
        for i in range(0, len(states), 200000):
            s = torch.as_tensor(np.asarray(states[i:i + 200000]), dtype=torch.float32)
            o = self.obs(s)
            a = torch.tanh(self.pi.mu(self.pi.body(o)))
            w = torch.tanh(self.adv.mu(self.adv.body(o)))
            x = torch.cat([o, a, w], -1)
            out.append(torch.minimum(self.q1.net(x), self.q2.net(x)).squeeze(-1).numpy())
        return np.concatenate(out)


if __name__ == '__main__':
    import pickle
    root = Path(__file__).resolve().parents[2] / 'finalfinal'
    m = Model(root / 'model' / 'w095_ema_20k.pt')
    print('checkpoint meta', m.meta)
    d = pickle.load(open(root / 'runs' / 'proposed_as0.60.pkl', 'rb'))
    diffs = []
    for r in d['results']:
        t = r['traj']
        S = np.stack([np.asarray(t['x']) - np.asarray(t['CD_x']), t['y'], t['psi'], t['u'], t['v'], t['r'],
                      t['delta'], t['F']], 1)
        V = m.value(S); L = np.asarray(t['V'], float)
        ok = ~np.isnan(L[1:])
        diffs.append(np.abs(L[1:][ok] - V[:-1][ok]))
    dd = np.concatenate(diffs)
    print('logged V[k] vs rebuilt V(state[k-1]) over all proposed trials: '
          f'median {np.median(dd):.2e}, max {dd.max():.2e}, n {len(dd)}')
