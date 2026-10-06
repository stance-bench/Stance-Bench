"""Frozen model inference with resumable scoring."""
from __future__ import annotations
import gc
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import numpy as np
from .data import candidate_order, positions, unique_rows, read_jsonl, write_jsonl, write_json, fingerprint, select_task_ids
from .prompts import SYSTEM_DIRECT, SYSTEM_READER, LETTERS, direct_prompt, reader_prompt


class FrozenModel:
    """A simple full-forward scorer; model weights are never updated."""

    def __init__(self, model_id, device, dtype, revision="main", reader=False):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
        kwargs = {"dtype": getattr(torch, dtype), "revision": revision}
        if device == "auto":
            kwargs["device_map"] = "auto"
        self.model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs).eval()
        if device != "auto":
            self.model.to(device)
        self.device = self.model.get_input_embeddings().weight.device
        self.resolved_revision = getattr(self.model.config, "_commit_hash", None)
        if reader and dtype == "bfloat16":
            head = self.model.get_output_embeddings()
            if getattr(self.model.config, "tie_word_embeddings", False):
                weight, bias = head.weight, getattr(head, "bias", None)
                head.forward = lambda value: torch.nn.functional.linear(
                    value.float(), weight.float(), None if bias is None else bias.float())
            else:
                head.float()
                head.register_forward_pre_hook(lambda module, arguments: (arguments[0].float(),) + arguments[1:])
        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False

    def chat(self, system, prompt):
        return self.tokenizer.apply_chat_template(
            [{"role": "system", "content": system}, {"role": "user", "content": prompt}],
            tokenize=False, add_generation_prompt=True, enable_thinking=False)

    def direct(self, context):
        torch, tokenizer = self.torch, self.tokenizer
        text = self.chat(SYSTEM_DIRECT, direct_prompt(context)) + '{"label_id": "'
        prefix = tokenizer(text, add_special_tokens=False)["input_ids"]
        result = {}
        with torch.inference_mode():
            for label in candidate_order(context):
                full = tokenizer(text + label + '", "', add_special_tokens=False)["input_ids"]
                if full[:len(prefix)] != prefix:
                    raise ValueError("Candidate tokenization changed the prompt prefix")
                count = len(full) - len(prefix)
                inputs = torch.tensor([full[:-1]], device=self.device)
                logits = self.model(input_ids=inputs, logits_to_keep=count).logits[0, -count:].float().log_softmax(-1)
                suffix = torch.tensor(full[len(prefix):], device=logits.device)
                result[label] = float(logits.gather(1, suffix[:, None]).sum())
        return result

    def evidence(self, context, items):
        if not items:
            return []
        labels, orders = positions(context), context["reader_orders"]
        if not orders or len(labels) > len(LETTERS):
            raise ValueError("Missing reader orders or too many positions")
        for order in orders:
            if len(order) != len(labels) or set(order) != set(labels):
                raise ValueError("Reader order must contain every listed stance once")
        torch, tokenizer = self.torch, self.tokenizer
        result = []
        with torch.inference_mode():
            for item in items:
                values = []
                for order in orders:
                    text = self.chat(SYSTEM_READER, reader_prompt(context, order, item["text"])) + '{"answer": "'
                    prefix = tokenizer(text, add_special_tokens=False)["input_ids"]
                    token_ids = []
                    for letter in LETTERS[:len(labels)] + "U":
                        full = tokenizer(text + letter + '"}', add_special_tokens=False)["input_ids"]
                        if full[:len(prefix)] != prefix or tokenizer.decode([full[len(prefix)]]) != letter:
                            raise ValueError("Reader letters must each be a single token")
                        token_ids.append(full[len(prefix)])
                    logits = self.model(input_ids=torch.tensor([prefix], device=self.device), logits_to_keep=1).logits[0, -1].float()
                    lp = logits[token_ids].log_softmax(-1).cpu().numpy()
                    values.append([float(lp[order.index(label)]) for label in labels] + [float(lp[-1])])
                mean = np.mean(values, axis=0)
                result.append({"lp": dict(zip(labels, map(float, mean[:-1]))), "U": float(mean[-1])})
        return result


def parse_shard(value):
    try:
        index, count = map(int, value.split("/"))
    except (ValueError, AttributeError):
        raise ValueError("Shard must be INDEX/TOTAL, for example 0/4") from None
    if count < 1 or not 0 <= index < count:
        raise ValueError("Shard index must be in [0, TOTAL)")
    return index, count


def implementation_metadata():
    versions = {}
    for package in ("numpy", "torch", "transformers", "accelerate"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    source = b"".join((Path(__file__).parent / name).read_bytes()
                      for name in ("inference.py", "prompts.py", "data.py"))
    return {"implementation_sha256": hashlib.sha256(source).hexdigest(), "dependencies": versions}


def validate_branch(context, branch, value):
    if branch == "direct":
        if set(value) != set(candidate_order(context)) or not np.isfinite(list(value.values())).all():
            raise ValueError("Malformed direct scores")
    elif branch == "evidence":
        if not isinstance(value, list):
            raise ValueError("Evidence scores must be a list; [] denotes an explicitly empty evidence set")
        for comment in value:
            if set(comment["lp"]) != set(positions(context)):
                raise ValueError("Reader labels do not match the task")
            values = list(comment["lp"].values()) + [comment["U"]]
            if not np.isfinite(values).all() or comment["U"] > 1e-6:
                raise ValueError("Malformed reader log-probabilities")
    else:
        raise ValueError("Unknown score branch")


def recover_checkpoint(path, contexts, selected):
    """Keep complete records after interruption; repair only a truncated final line."""
    rows = {}
    if not path.exists():
        return rows
    payload = path.read_bytes()
    boundary = 0
    lines = payload.splitlines(keepends=True)
    for index, line in enumerate(lines):
        try:
            event = json.loads(line)
        except (ValueError, UnicodeDecodeError):
            if index != len(lines) - 1 or line.endswith(b"\n"):
                raise ValueError("Malformed checkpoint record before end of file") from None
            with path.open("r+b") as handle:
                handle.truncate(boundary)
            break
        task, branch = event["task_id"], event["branch"]
        if task not in selected:
            raise ValueError("Checkpoint contains a task outside the selected shard")
        validate_branch(contexts[task], branch, event["value"])
        row = rows.setdefault(task, {"task_id": task})
        if branch in row:
            raise ValueError("Duplicate checkpoint branch for a task")
        row[branch] = event["value"]
        boundary += len(line)
        if index == len(lines) - 1 and not line.endswith(b"\n"):
            with path.open("ab") as handle:
                handle.write(b"\n")
    return rows


def run_scoring(contexts, targets, evidence, output, *, split="test",
                model_id="Qwen/Qwen3-8B", revision="main", device="cuda", direct_dtype="bfloat16",
                reader_dtype="float32", branch="both", shard="0/1", limit=0, resume=False):
    """Score one shard with per-task checkpoints; the output file is written only when complete."""
    index, count = parse_shard(shard)
    if limit < 0 or branch not in ("both", "direct", "reader"):
        raise ValueError("Invalid scoring limit or branch")
    task_ids = select_task_ids(contexts, targets, split)
    if limit:
        task_ids = task_ids[:limit]
    selected = [task for number, task in enumerate(task_ids) if number % count == index]
    if not selected:
        raise ValueError("No tasks selected for this shard")
    branches = ["direct", "evidence"] if branch == "both" else ["evidence" if branch == "reader" else "direct"]
    if "evidence" in branches and any(task not in evidence for task in task_ids):
        raise ValueError("Evidence must explicitly cover every selected task, including empty evidence sets")
    config = {"format_version": 2, "split": split, "model": model_id, "revision": revision,
              "device": device, "direct_dtype": direct_dtype, "reader_dtype": reader_dtype,
              "branch": branch, "shard": [index, count], "limit": limit,
              "task_ids_sha256": fingerprint(task_ids),
              "contexts_sha256": fingerprint([contexts[task] for task in task_ids]),
              "evidence_sha256": fingerprint([evidence[task] for task in task_ids]) if "evidence" in branches else None}
    config.update(implementation_metadata())
    output = Path(output)
    manifest_path = Path(str(output) + ".manifest.json")
    checkpoint_path = Path(str(output) + ".checkpoint.jsonl")
    signature = fingerprint(config)
    if any(path.exists() for path in (output, manifest_path, checkpoint_path)):
        if not resume or not manifest_path.exists():
            raise ValueError("Scoring artifacts already exist; use --resume with the same inputs or choose a new output")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest["signature"] != signature:
            raise ValueError("Resume inputs or configuration differ from the saved manifest")
    else:
        manifest = {"signature": signature, "config": config, "selected_tasks": len(selected),
                    "status": "running", "model_revisions": {}}
        write_json(manifest_path, manifest)
    rows = recover_checkpoint(checkpoint_path, contexts, set(selected))
    output.parent.mkdir(parents=True, exist_ok=True)
    for active in branches:
        todo = [task for task in selected if active not in rows.get(task, {})]
        if not todo:
            continue
        resolved = (manifest["model_revisions"].get(active)
                    or next(iter(manifest["model_revisions"].values()), None) or revision)
        model = FrozenModel(model_id, device, direct_dtype if active == "direct" else reader_dtype,
                            revision=resolved, reader=active == "evidence")
        if model.resolved_revision:
            manifest["model_revisions"][active] = model.resolved_revision
            write_json(manifest_path, manifest)
        with checkpoint_path.open("ab") as handle:
            for number, task in enumerate(todo, 1):
                value = (model.direct(contexts[task]) if active == "direct"
                         else model.evidence(contexts[task], evidence[task]["items"]))
                validate_branch(contexts[task], active, value)
                event = {"task_id": task, "branch": active, "value": value}
                handle.write((json.dumps(event, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8"))
                handle.flush()
                os.fsync(handle.fileno())
                rows.setdefault(task, {"task_id": task})[active] = value
                if number % 25 == 0 or number == len(todo):
                    print(f"{active}: {number}/{len(todo)} remaining tasks", flush=True)
        torch = model.torch
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    write_jsonl(output, [rows[task] for task in selected])
    manifest["status"] = "complete"
    manifest["output_sha256"] = fingerprint([rows[task] for task in selected])
    write_json(manifest_path, manifest)
    return manifest


def merge_score_files(paths, contexts, expected_ids=None):
    """Merge disjoint shards or complementary direct/reader scores, never duplicates."""
    rows, compatible, evidence_digest, revisions = {}, None, None, {}
    for path in paths:
        file_rows = read_jsonl(path)
        sidecar = Path(str(path) + ".manifest.json")
        if sidecar.exists():
            manifest = json.loads(sidecar.read_text(encoding="utf-8"))
            if manifest["status"] != "complete":
                raise ValueError("Cannot merge an incomplete score run")
            if manifest.get("output_sha256") and fingerprint(file_rows) != manifest["output_sha256"]:
                raise ValueError("Score file content differs from its manifest")
            if "config" in manifest:
                settings = {key: value for key, value in manifest["config"].items()
                            if key not in ("shard", "branch", "evidence_sha256", "device")}
                if compatible is not None and settings != compatible:
                    raise ValueError("Score manifests use incompatible models, data, or selection settings")
                compatible = settings
                digest = manifest["config"].get("evidence_sha256")
                if digest is not None and evidence_digest is not None and digest != evidence_digest:
                    raise ValueError("Reader score manifests use different evidence")
                evidence_digest = digest or evidence_digest
                for branch, revision in manifest.get("model_revisions", {}).items():
                    if branch in revisions and revisions[branch] != revision:
                        raise ValueError("Score manifests use different resolved model revisions")
                    revisions[branch] = revision
                if len(set(revisions.values())) > 1:
                    raise ValueError("Direct and reader branches use different resolved model revisions")
        for task, row in unique_rows(file_rows).items():
            if task not in contexts:
                raise ValueError("Score file contains an unknown task")
            merged = rows.setdefault(task, {"task_id": task})
            for branch in ("direct", "evidence"):
                if branch not in row:
                    continue
                if branch in merged:
                    raise ValueError("Duplicate task branch across score files")
                validate_branch(contexts[task], branch, row[branch])
                merged[branch] = row[branch]
    if expected_ids is not None and set(rows) != set(expected_ids):
        raise ValueError("Merged scores do not exactly cover the requested task selection")
    if any("direct" not in row or "evidence" not in row for row in rows.values()):
        raise ValueError("Merged scores require both branches for every task")
    return rows
