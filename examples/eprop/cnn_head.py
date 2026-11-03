"""A small convolutional actor-critic head in NumPy, trained online (one sample per move, no replay).

It reads an image of the network's activity (e.g. the field layers' spike counts during a move, sheet x channels)
and outputs action logits and a value. Two 3x3 convolutions with circular padding (the field sheets are toroidal),
ReLU, then two linear outputs from the shared trunk. One-step actor-critic: for the transition (x, a) -> r, x',

    delta = r + gamma * V(x') * (1 - done) - V(x),
    actor:  minimise -delta * log pi(a | x) - entropy * H(pi(. | x)),     critic: minimise 0.5 * delta^2,

with delta treated as a constant, Adam on all parameters. The head is task-specific machinery on top of the spiking
network: backpropagation runs within the head only; the network below learns from the head's broadcast delta.
"""
import numpy as np


def _patches(x):
    """(H, W, C) -> (H, W, 9 C): the 3x3 circular neighbourhood of every position (offset-major)."""
    return np.concatenate([np.roll(x, (-dy, -dx), axis=(0, 1)) for dy in (-1, 0, 1) for dx in (-1, 0, 1)], axis=-1)


def _patches_backward(dp, c):
    """Adjoint of _patches: (H, W, 9 C) -> (H, W, C)."""
    dx = np.zeros(dp.shape[:2] + (c,), dp.dtype)
    k = 0
    for dy in (-1, 0, 1):
        for dx_ in (-1, 0, 1):
            dx += np.roll(dp[..., k * c:(k + 1) * c], (dy, dx_), axis=(0, 1))
            k += 1
    return dx


class CNNHead:
    def __init__(self, shape, n_actions, channels=(16, 16), lr=1e-4, betas=(0.9, 0.999), entropy=0.01, seed=0):
        h, w, c = shape
        rng = np.random.default_rng(seed)
        c1, c2 = channels
        he = lambda fan_in, *s: (rng.standard_normal(s) * np.sqrt(2.0 / fan_in)).astype(np.float64)
        self.p = {"W1": he(9 * c, 9 * c, c1), "b1": np.zeros(c1), "W2": he(9 * c1, 9 * c1, c2), "b2": np.zeros(c2),
                  "Wp": rng.standard_normal((h * w * c2, n_actions)) * 0.01, "bp": np.zeros(n_actions),
                  "Wv": rng.standard_normal((h * w * c2, 1)) * 0.01, "bv": np.zeros(1)}
        self.lr, self.b1, self.b2, self.entropy = lr, betas[0], betas[1], entropy
        self.m = {k: np.zeros_like(v) for k, v in self.p.items()}
        self.v = {k: np.zeros_like(v) for k, v in self.p.items()}
        self.t = 0
        self.c = c

    # ---- forward / backward
    def forward(self, x):
        p = self.p
        P1 = _patches(x)
        a1 = P1 @ p["W1"] + p["b1"]
        h1 = np.maximum(a1, 0.0)
        P2 = _patches(h1)
        a2 = P2 @ p["W2"] + p["b2"]
        h2 = np.maximum(a2, 0.0)
        f = h2.ravel()
        logits = f @ p["Wp"] + p["bp"]
        value = float((f @ p["Wv"] + p["bv"])[0])
        return logits, value, (x, P1, a1, h1, P2, a2, h2, f)

    def backward(self, cache, dlogits, dvalue):
        x, P1, a1, h1, P2, a2, h2, f = cache
        p = self.p
        g = {"Wp": np.outer(f, dlogits), "bp": dlogits.copy(), "Wv": np.outer(f, [dvalue]), "bv": np.array([dvalue])}
        df = p["Wp"] @ dlogits + p["Wv"][:, 0] * dvalue
        da2 = df.reshape(h2.shape) * (a2 > 0)
        g["W2"] = P2.reshape(-1, P2.shape[-1]).T @ da2.reshape(-1, da2.shape[-1])
        g["b2"] = da2.sum(axis=(0, 1))
        dh1 = _patches_backward(da2 @ p["W2"].T, h1.shape[-1])
        da1 = dh1 * (a1 > 0)
        g["W1"] = P1.reshape(-1, P1.shape[-1]).T @ da1.reshape(-1, da1.shape[-1])
        g["b1"] = da1.sum(axis=(0, 1))
        return g

    def adam(self, g):
        self.t += 1
        for k in self.p:
            self.m[k] = self.b1 * self.m[k] + (1 - self.b1) * g[k]
            self.v[k] = self.b2 * self.v[k] + (1 - self.b2) * g[k] ** 2
            mh = self.m[k] / (1 - self.b1 ** self.t)
            vh = self.v[k] / (1 - self.b2 ** self.t)
            self.p[k] -= self.lr * mh / (np.sqrt(vh) + 1e-8)

    # ---- actor-critic
    @staticmethod
    def policy(logits):
        z = np.exp(logits - logits.max())
        return z / z.sum()

    def loss_gradients(self, cache, logits, action, delta):
        """d/d(logits, value) of -delta log pi(a) - entropy H(pi) + 0.5 delta^2 (delta = target - value, constant
        target): dL/dlogits = delta (pi - onehot) + entropy * pi * (log pi + H); dL/dvalue = -delta."""
        pi = self.policy(logits)
        onehot = np.zeros_like(pi); onehot[action] = 1.0
        logp = np.log(pi + 1e-12)
        H = -(pi * logp).sum()
        dlogits = delta * (pi - onehot) + self.entropy * pi * (logp + H)
        return dlogits, -delta

    def learn(self, cache, logits, action, delta, train_value=True):
        """train_value False: actor only (delta comes from an external critic; the value output is not trained)."""
        dlogits, dvalue = self.loss_gradients(cache, logits, action, delta)
        self.adam(self.backward(cache, dlogits, dvalue if train_value else 0.0))


def gradient_check(seed=0, eps=1e-6):
    """Largest relative error between backward() and central finite differences, for a scalar loss
    L = c . logits + d * value on a random input; used by the tests."""
    rng = np.random.default_rng(seed)
    head = CNNHead((5, 6, 2), 3, channels=(3, 4), seed=seed)
    for k in head.p:                       # non-trivial readout weights for the check
        head.p[k] = head.p[k] + 0.3 * rng.standard_normal(head.p[k].shape)
    x = rng.standard_normal((5, 6, 2))
    c, d = rng.standard_normal(3), float(rng.standard_normal())
    loss = lambda: float(c @ head.forward(x)[0] + d * head.forward(x)[1])
    g = head.backward(head.forward(x)[2], c, d)
    worst = 0.0
    for k, v in head.p.items():
        flat = v.ravel()
        for idx in rng.choice(flat.size, size=min(20, flat.size), replace=False):
            old = flat[idx]
            flat[idx] = old + eps; lp = loss()
            flat[idx] = old - eps; lm = loss()
            flat[idx] = old
            num = (lp - lm) / (2 * eps)
            ana = g[k].ravel()[idx]
            worst = max(worst, abs(num - ana) / max(1e-8, abs(num) + abs(ana)))
    return worst
