"""Timing of numpy forward passes for amortized predictors (no training)."""
import time
import numpy as np
rng = np.random.default_rng(0)
relu = lambda x: np.maximum(x, 0)
# (a) DeepSets over the draw's evidence (<=300 labels), relation-aware pooling, cached per evidence set + per-call head
W1, W2 = rng.normal(size=(16, 64)), rng.normal(size=(64, 64))
H1, H2 = rng.normal(size=(64*4 + 16, 64)), rng.normal(size=(64, 1))
X = rng.normal(size=(300, 16))
def encode(X):
    return relu(relu(X @ W1) @ W2)
def head(Z, masks, q):
    pools = [Z[m].mean(0) if m.any() else np.zeros(64) for m in masks]
    h = relu(np.concatenate(pools + [q]) @ H1)
    return float(1/(1+np.exp(-(h @ H2)[0])))
masks = [rng.random(300) < f for f in (1.0, 0.1, 0.03, 0.01)]  # all / same pair / same benchmark-other pair / same item
q = rng.normal(size=16)
t = time.perf_counter(); 
for _ in range(200): Z = encode(X)
t_enc = (time.perf_counter() - t) / 200
t = time.perf_counter()
for _ in range(2000): head(Z, masks, q)
t_head = (time.perf_counter() - t) / 2000
# (b) sufficient-statistic MLP (12 -> 32 -> 32 -> 1)
A, B, C = rng.normal(size=(12, 32)), rng.normal(size=(32, 32)), rng.normal(size=(32, 1))
s = rng.normal(size=12)
t = time.perf_counter()
for _ in range(20000): float(1/(1+np.exp(-(relu(relu(s @ A) @ B) @ C)[0])))
t_mlp = (time.perf_counter() - t) / 20000
print(f"DeepSets encode 300 labels: {1e3*t_enc:.3f} ms (once per evidence set); per-call head: {1e3*t_head:.3f} ms; "
      f"suff-stat MLP per call: {1e3*t_mlp:.4f} ms")
n_par_a = W1.size + W2.size + H1.size + H2.size
print("DeepSets params", n_par_a, "| MLP params", A.size + B.size + C.size)
