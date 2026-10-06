"""StanceBench frozen evidence-reading method."""
from .data import read_jsonl, write_jsonl, unique_rows, load_dataset, positions, candidate_order
from .prompts import direct_prompt, reader_prompt
from .fusion import evidence_profile, score_vectors, fit_coefficients, predict
from .metrics import macro_f1, evaluate, bootstrap_intervals

__version__ = "0.2.0"
