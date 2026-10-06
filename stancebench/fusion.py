"""Relevance-weighted logarithmic pooling and development-only calibration."""
from __future__ import annotations
import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp
from .data import positions, candidate_order


def evidence_profile(comments, labels):
    """Renormalize over stances, weight by 1-P(U), then center the log pool."""
    if not comments:
        return np.zeros(len(labels))
    values = np.asarray([[comment["lp"][label] for label in labels] for comment in comments], float)
    unknown = np.asarray([comment["U"] for comment in comments], float)
    if not np.isfinite(values).all() or not np.isfinite(unknown).all() or np.any(unknown > 1e-6):
        raise ValueError("Reader scores must be finite")
    values -= logsumexp(values, axis=1, keepdims=True)
    weight = np.clip(1 - np.exp(unknown), 0, 1)
    if weight.sum() <= 1e-9:
        return np.zeros(len(labels))
    pooled = (weight[:, None] * values).sum(axis=0) / weight.sum()
    return pooled - pooled.mean()


def score_vectors(context, score):
    labels = [label for label in candidate_order(context) if label != "OTHER"]
    expected = set(labels) | {"OTHER"}
    if set(score["direct"]) != expected:
        raise ValueError("Direct scores must contain all listed stances plus OTHER")
    direct = np.asarray([score["direct"][label] for label in labels + ["OTHER"]], float)
    if not np.isfinite(direct).all():
        raise ValueError("Direct scores must be finite")
    evidence = np.r_[evidence_profile(score["evidence"], labels), 0.0]
    return labels + ["OTHER"], direct, evidence


def fit_coefficients(contexts, targets, scores):
    """Fit only development targets whose gold is a listed stance."""
    unknown = set(scores) - set(contexts)
    if unknown:
        raise ValueError("Scores contain unknown task IDs")
    if any(contexts[task]["split"] != "dev" for task in scores):
        raise ValueError("Coefficient fitting accepts development scores only")
    wanted = sorted(task for task, context in contexts.items()
                    if context["split"] == "dev" and targets[task]["label_id"] in positions(context))
    test_users = {row.get("profile_id") for row in contexts.values() if row["split"] == "test"}
    if any(contexts[task].get("profile_id") is not None and
           contexts[task]["profile_id"] in test_users for task in wanted):
        raise ValueError("Development and test users overlap")
    if not wanted or set(wanted) - set(scores):
        raise ValueError("Scores must cover all development listed-stance targets")
    items = []
    for task in wanted:
        labels, direct, evidence = score_vectors(contexts[task], scores[task])
        items.append((direct[:-1], evidence[:-1], labels.index(targets[task]["label_id"])))

    def objective(theta):
        total, gradient = 0.0, np.zeros(2)
        for direct, evidence, gold in items:
            logits = theta[0] * direct + theta[1] * evidence
            logits -= logits.max()
            prob = np.exp(logits) / np.exp(logits).sum()
            total -= np.log(prob[gold] + 1e-300)
            gradient[0] -= direct[gold] - (prob * direct).sum()
            gradient[1] -= evidence[gold] - (prob * evidence).sum()
        return total / len(items) + 1e-4 * (theta @ theta), gradient / len(items) + 2e-4 * theta

    result = minimize(objective, np.array([1 / 25, 0.0]), jac=True, method="L-BFGS-B")
    if not result.success:
        raise RuntimeError("Coefficient fitting did not converge: " + str(result.message))
    return {"a": float(result.x[0]), "b": float(result.x[1]), "fit_n": len(items),
            "fit_split": "dev",
            "objective": "mean listed-stance NLL + 0.0001 * (a*a + b*b)",
            "scope": "All benchmark development listed-stance targets"}


def predict(context, score, coefficients):
    labels, direct, evidence = score_vectors(context, score)
    logits = float(coefficients["a"]) * direct + float(coefficients["b"]) * evidence
    if not np.isfinite(logits).all():
        raise ValueError("Fused scores must be finite")
    top = np.flatnonzero(np.abs(logits - logits.max()) < 1e-12)
    return labels[int(top[0])] if len(top) == 1 else None
