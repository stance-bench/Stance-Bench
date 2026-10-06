"""Gold-supported discussion-label metrics and participant bootstrap."""
from __future__ import annotations
from collections import Counter
import numpy as np
from .data import candidate_order, positions


def macro_f1(gold, predicted, discussions):
    classes = set(zip(discussions, gold))
    tp, fp, fn = Counter(), Counter(), Counter()
    for discussion, truth, guess in zip(discussions, gold, predicted):
        if truth == guess:
            tp[discussion, truth] += 1
        else:
            fn[discussion, truth] += 1
            if guess:
                fp[discussion, guess] += 1
    return float(np.mean([2 * tp[key] / (2 * tp[key] + fp[key] + fn[key]) for key in classes])) if classes else 0.0


def evaluate(contexts, targets, predictions, split="test", allow_missing=False):
    if set(predictions) - set(contexts):
        raise ValueError("Predictions contain unknown task IDs")
    if any(contexts[task]["split"] != split for task in predictions):
        raise ValueError("Predictions contain tasks outside the evaluation split")
    required = {task for task, context in contexts.items() if context["split"] == split
                and targets[task]["label_id"] != "NO_STANCE"}
    missing = required - set(predictions)
    if missing and not allow_missing:
        raise ValueError(f"Missing predictions for {len(missing)} evaluation tasks; enable allow_missing to count them as misses")
    result = {}
    for name, listed_only in (("listed_stances", True), ("including_OTHER", False)):
        tasks = [task for task, context in contexts.items() if context["split"] == split
                 and targets[task]["label_id"] != "NO_STANCE"
                 and (not listed_only or targets[task]["label_id"] in positions(context))]
        gold, guessed, discussion = [], [], []
        for task in tasks:
            context = contexts[task]
            label = predictions.get(task, {}).get("label_id")
            gold.append(targets[task]["label_id"])
            guessed.append(label if label in candidate_order(context) else None)
            discussion.append(context["discussion_id"])
        result[name] = {"n": len(tasks), "classes": len(set(zip(discussion, gold))),
                        "accuracy": float(np.mean([a == b for a, b in zip(gold, guessed)])) if tasks else 0.0,
                        "macro_f1": macro_f1(gold, guessed, discussion),
                        "invalid_or_missing": sum(label is None for label in guessed)}
    return result


def bootstrap_intervals(contexts, targets, predictions, split="test", n_boot=2000, seed=17):
    """Resample participants; recompute gold-supported classes within each draw."""
    if n_boot < 1:
        raise ValueError("The bootstrap count must be positive")
    tasks = [task for task, context in contexts.items() if context["split"] == split
             and targets[task]["label_id"] != "NO_STANCE"]
    if not tasks:
        raise ValueError("Cannot bootstrap an empty evaluation set")
    groups = np.asarray([contexts[task]["profile_id"] for task in tasks], dtype=object)
    gold = np.asarray([targets[task]["label_id"] for task in tasks], dtype=object)
    discussion = np.asarray([contexts[task]["discussion_id"] for task in tasks], dtype=object)
    predicted = np.asarray([predictions.get(task, {}).get("label_id") for task in tasks], dtype=object)
    for index, task in enumerate(tasks):
        if predicted[index] not in candidate_order(contexts[task]):
            predicted[index] = None
    listed = np.asarray([gold[index] in positions(contexts[task]) for index, task in enumerate(tasks)])
    users, inverse = np.unique(groups, return_inverse=True)
    by_user = [np.flatnonzero(inverse == index) for index in range(len(users))]
    rng = np.random.default_rng(seed)
    samples = {name: {metric: [] for metric in ("accuracy", "macro_f1")}
               for name in ("listed_stances", "including_OTHER")}
    for _ in range(n_boot):
        draw = np.concatenate([by_user[index] for index in rng.integers(0, len(users), len(users))])
        for name, indices in (("listed_stances", draw[listed[draw]]), ("including_OTHER", draw)):
            if not len(indices):
                continue
            samples[name]["accuracy"].append(float(np.mean(gold[indices] == predicted[indices])))
            samples[name]["macro_f1"].append(macro_f1(gold[indices], predicted[indices], discussion[indices]))
    result = {"n_boot": n_boot, "seed": seed, "unit": "participant", "interval": "percentile 95%"}
    for name, metrics in samples.items():
        result[name] = {metric: list(map(float, np.percentile(values, [2.5, 97.5]))) if values else None
                        for metric, values in metrics.items()}
        result[name]["valid_draws"] = len(metrics["accuracy"])
    return result
