"""Command-line interface for model inference, fitting, prediction and evaluation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .data import (load_dataset, read_jsonl, unique_rows, write_json,
                   write_jsonl, select_task_ids, fingerprint)
from .fusion import fit_coefficients, predict
from .metrics import evaluate, bootstrap_intervals
from .inference import run_scoring, merge_score_files


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data", help="Dataset directory")
    commands = parser.add_subparsers(dest="command", required=True)
    score = commands.add_parser("score", help="Score frozen direct and evidence branches")
    score.add_argument("--split", choices=("dev", "test"), default="test")
    score.add_argument("--model", default="Qwen/Qwen3-8B")
    score.add_argument("--revision", default="main", help="Model revision; resolved commits are saved for resume")
    score.add_argument("--device", default="cuda", help="cuda, cuda:INDEX, cpu, or auto")
    score.add_argument("--direct-dtype", choices=("float32", "bfloat16"), default="bfloat16")
    score.add_argument("--reader-dtype", choices=("float32", "bfloat16"), default="float32")
    score.add_argument("--branch", choices=("both", "direct", "reader"), default="both")
    score.add_argument("--limit", type=int, default=0, help="Limit tasks before sharding; 0 means all")
    score.add_argument("--shard", default="0/1", help="Zero-based INDEX/TOTAL")
    score.add_argument("--resume", action="store_true")
    score.add_argument("--output", required=True)
    fit = commands.add_parser("fit", help="Fit fusion coefficients on development listed-stance targets")
    fit.add_argument("--scores", required=True)
    fit.add_argument("--output", required=True)
    prediction = commands.add_parser("predict", help="Fuse direct and reader scores")
    prediction.add_argument("--scores", required=True)
    prediction.add_argument("--coefficients", required=True)
    prediction.add_argument("--output", required=True)
    evaluation = commands.add_parser("evaluate", help="Evaluate both cohorts from one prediction set")
    evaluation.add_argument("--predictions", required=True)
    evaluation.add_argument("--split", choices=("dev", "test"), default="test")
    evaluation.add_argument("--allow-missing", action="store_true", help="Count missing predictions as misses")
    evaluation.add_argument("--bootstrap", type=int, default=2000,
                            help="Participant resamples for 95%% metric intervals; 0 disables intervals")
    evaluation.add_argument("--seed", type=int, default=17)
    evaluation.add_argument("--output", help="Also save metrics as JSON")
    merge = commands.add_parser("merge-scores", help="Merge score shards or complementary branch outputs")
    merge.add_argument("--inputs", nargs="+", required=True)
    merge.add_argument("--split", choices=("dev", "test"), default="test")
    merge.add_argument("--allow-partial", action="store_true", help="Permit incomplete task coverage (both branches still required)")
    merge.add_argument("--output", required=True)
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    root = Path(args.data)
    contexts, targets = load_dataset(root)
    if args.command == "score":
        evidence = {} if args.branch == "direct" else unique_rows(
            read_jsonl(root / f"evidence_{args.split}.jsonl.gz"))
        manifest = run_scoring(contexts, targets, evidence, args.output, split=args.split,
                               model_id=args.model, revision=args.revision, device=args.device,
                               direct_dtype=args.direct_dtype, reader_dtype=args.reader_dtype,
                               branch=args.branch, shard=args.shard, limit=args.limit, resume=args.resume)
        print(json.dumps({"status": manifest["status"], "tasks": manifest["selected_tasks"]}, indent=2))
    elif args.command == "fit":
        fitted = fit_coefficients(contexts, targets, unique_rows(read_jsonl(args.scores)))
        write_json(args.output, fitted)
        print(json.dumps(fitted, indent=2))
    elif args.command == "predict":
        coefficients = json.loads(Path(args.coefficients).read_text(encoding="utf-8"))
        scores = unique_rows(read_jsonl(args.scores))
        if set(scores) - set(contexts):
            raise ValueError("Scores contain unknown task IDs")
        write_jsonl(args.output, [{"task_id": task, "label_id": predict(contexts[task], row, coefficients)}
                                 for task, row in scores.items()])
    elif args.command == "evaluate":
        if args.bootstrap < 0:
            parser.error("--bootstrap must be nonnegative")
        predictions = unique_rows(read_jsonl(args.predictions))
        result = evaluate(contexts, targets, predictions, args.split, allow_missing=args.allow_missing)
        if args.bootstrap:
            result["bootstrap"] = bootstrap_intervals(contexts, targets, predictions, args.split,
                                                       n_boot=args.bootstrap, seed=args.seed)
        if args.output:
            write_json(args.output, result)
        print(json.dumps(result, indent=2))
    else:
        expected = select_task_ids(contexts, targets, args.split)
        rows = merge_score_files(args.inputs, contexts, None if args.allow_partial else expected)
        if set(rows) - set(expected):
            raise ValueError("Merged scores contain tasks outside the requested split/selection")
        write_jsonl(args.output, [rows[task] for task in expected if task in rows])
        write_json(str(args.output) + ".manifest.json",
                   {"format_version": 2, "status": "complete", "operation": "merge-scores",
                    "tasks": len(rows), "input_count": len(args.inputs),
                    "task_ids_sha256": fingerprint([task for task in expected if task in rows])})


if __name__ == "__main__":
    main()
