"""Subject identity and attribute features, usable identically offline and inside a submission (stdlib only)."""
from __future__ import annotations

import math
import re

PROVIDER_WORDS = ("openai", "anthropic", "google", "meta", "deepseek", "xai", "alibaba", "qwen", "mistral",
                  "bytedance", "moonshot", "zhipu", "amazon", "microsoft", "nvidia", "cohere", "minimax")
QUALIFIERS = re.compile(r"\((?:[^)]*)\)")
SMALL = ("mini", "flash", "lite", "nano", "small", "haiku", "distill", "air", "tiny", "8b", "7b", "1.5b", "3b", "14b")
LARGE = ("pro", "opus", "max", "ultra", "large", "405b", "671b", "480b", "235b", "plus")
EFFORT = {"none": 0.0, "minimal": 0.5, "low": 1.0, "medium": 2.0, "high": 3.0, "xhigh": 4.0, "max": 4.0}
PROVIDERS = ("OpenAI", "Anthropic", "Google", "DeepSeek", "Alibaba", "Meta", "xAI", "ByteDance", "Mistral",
             "Moonshot", "Zhipu")


def _extra_field(subject: dict, key: str) -> str:
    for part in str(subject.get("subject_features_extra", "") or "").split(";"):
        if part.startswith(key + "="):
            return part.split("=", 1)[1].strip()
    return ""


def canonical_name(text: str) -> str:
    """'Anthropic Claude 3.7 Sonnet' and 'Claude-3.7-Sonnet (Think)' -> 'claude37sonnet'."""
    t = QUALIFIERS.sub(" ", str(text).lower())
    for w in PROVIDER_WORDS:
        t = re.sub(rf"\b{w}\b", " ", t)
    t = re.sub(r"(\d)\.0\b", r"\1", t)  # 4.0 -> 4
    return re.sub(r"[^a-z0-9]", "", t)


def name_key(subject: dict) -> str:
    name = str(subject.get("normalized_name", "") or "").strip()
    if not name:
        name = _extra_field(subject, "source_model_name")
    return canonical_name(name) if name else ""


def _year(date: str) -> float | None:
    m = re.match(r"(\d{4})-(\d{2})", str(date or ""))
    return int(m.group(1)) + (int(m.group(2)) - 1) / 12 if m else None


def attributes(subject: dict) -> list[float]:
    """Fixed-length numeric features for the ability regression."""
    raw = " ".join(str(subject.get(k, "") or "") for k in ("normalized_name", "subject_features_extra")).lower()
    provider = str(subject.get("provider", "") or "")
    if not provider:
        cfg = _extra_field(subject, "model_config").split("/")[0].lower()
        provider = next((p for p in PROVIDERS if p.lower().startswith(cfg[:4])), "") if cfg else ""
    year = _year(subject.get("release_date", ""))
    effort = str(subject.get("reasoning_effort", "") or "").lower()
    think = 1.0 if re.search(r"think|reason|\(high\)|\bhigh\b", raw) else 0.0
    feats = [
        1.0,
        (year - 2025.0) if year is not None else 0.0,
        1.0 if year is None else 0.0,
        EFFORT.get(effort, 0.0) / 4.0,
        think,
        1.0 if any(re.search(rf"\b{w}\b", raw) for w in SMALL) else 0.0,
        1.0 if any(re.search(rf"\b{w}\b", raw) for w in LARGE) else 0.0,
    ]
    feats += [1.0 if provider == p else 0.0 for p in PROVIDERS]
    return feats


def ridge_fit(X: list[list[float]], y: list[float], w: list[float], alpha: float = 1.0):
    """Weighted ridge in pure Python (tiny problems); returns (coef, residual variance)."""
    n, d = len(X), len(X[0])
    A = [[alpha * (i == j and i > 0) for j in range(d)] for i in range(d)]
    b = [0.0] * d
    for xi, yi, wi in zip(X, y, w):
        for i in range(d):
            b[i] += wi * xi[i] * yi
            for j in range(d):
                A[i][j] += wi * xi[i] * xi[j]
    # Gauss-Jordan solve
    M = [row[:] + [b[i]] for i, row in enumerate(A)]
    for c in range(d):
        piv = max(range(c, d), key=lambda r: abs(M[r][c]))
        M[c], M[piv] = M[piv], M[c]
        if abs(M[c][c]) < 1e-12:
            continue
        f = M[c][c]
        M[c] = [v / f for v in M[c]]
        for r in range(d):
            if r != c and M[r][c]:
                g = M[r][c]
                M[r] = [a - g * bb for a, bb in zip(M[r], M[c])]
    coef = [M[i][d] for i in range(d)]
    resid = [yi - sum(c * x for c, x in zip(coef, xi)) for xi, yi in zip(X, y)]
    tot_w = sum(w) or 1.0
    var = sum(wi * r * r for wi, r in zip(w, resid)) / tot_w
    return coef, max(var, 1e-3)


def predict_attr(coef: list[float], subject: dict) -> float:
    return sum(c * x for c, x in zip(coef, attributes(subject)))


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))
