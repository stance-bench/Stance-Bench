"""Dataset loading and JSONL input/output utilities."""
from __future__ import annotations
import gzip
import json
import hashlib
import os
import tempfile
from pathlib import Path


def read_jsonl(path):
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = "".join(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n" for row in rows).encode("utf-8")
    if str(path).endswith(".gz"):
        payload = gzip.compress(payload, mtime=0)
    atomic_write(path, payload)


def atomic_write(path, payload):
    """Replace a completed artifact atomically, without partial output on failure."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=".writing-", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def write_json(path, value):
    atomic_write(path, (json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8"))


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()


def unique_rows(rows):
    result = {}
    for row in rows:
        if row["task_id"] in result:
            raise ValueError("Duplicate task ID")
        result[row["task_id"]] = row
    return result


def load_dataset(root):
    root = Path(root)
    contexts = unique_rows(read_jsonl(root / "processed/task_contexts.jsonl.gz"))
    targets = unique_rows(read_jsonl(root / "processed/task_targets.jsonl.gz"))
    if contexts.keys() != targets.keys():
        raise ValueError("Context and target task IDs differ")
    return contexts, targets


def select_task_ids(contexts, targets, split):
    if split not in ("dev", "test"):
        raise ValueError("Task split must be dev or test")
    return [task for task, context in contexts.items() if context["split"] == split
            and targets[task]["label_id"] != "NO_STANCE"]


def positions(context):
    return [option["id"] for option in context["target_stimulus"]["candidate_labels"]
            if option["id"] not in ("OTHER", "NO_STANCE")]


def candidate_order(context):
    order = context["candidate_order"]
    expected = set(positions(context)) | {"OTHER"}
    if len(order) != len(expected) or set(order) != expected:
        raise ValueError("Candidate order must contain each listed stance and OTHER once")
    return order
