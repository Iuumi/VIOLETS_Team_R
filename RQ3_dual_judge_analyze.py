"""
RQ3_dual_judge_analyze.py
==========================
Fits the same RQ3 mixed-effects models (overall + by-category VIOLETS vs.
Baseline effect) once per embedding model on a *_dual_judge.jsonl file
(primary text-embedding-3-small's `similarity_score` column, secondary
qwen3-embedding-8b's `similarity_score_2nd` column — see secondary_judge.py
/ rescore_secondary_rq3.py), then plots both embeddings' estimates as
separate, color-coded series on the same coefficient plot. RQ3 has no LLM
judge to cross-check (similarity_score is a deterministic cosine
similarity, not a judgment call), so this is the RQ3-equivalent robustness
check: re-embedding with an independent model rather than a second judge.

Usage:
  python RQ3_dual_judge_analyze.py --input output/rq3/eval_dataset_20260710_dual_judge.jsonl --output_dir output/rq3/analysis_mixed_20260710
"""

import argparse
from pathlib import Path

import pandas as pd

from RQ3_analyze import (
    load_jsonl,
    preprocess,
    build_table1_overall,
    build_table2_category,
    build_multi_embedding_coefficient_figure,
)

EMBEDDINGS = [
    {"label": "text-embedding-3-small (primary)", "color": "#7B2FBE", "outcome": "similarity_score"},
    {"label": "qwen3-embedding-8b (secondary)", "color": "#E8871E", "outcome": "similarity_score_2nd"},
]


def run(input_path: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_jsonl(input_path)
    df = preprocess(df)
    df["similarity_score_2nd"] = pd.to_numeric(df["similarity_score_2nd"], errors="coerce")

    embeddings = []
    for e in EMBEDDINGS:
        sub = df.dropna(subset=[e["outcome"]])
        table1, _ = build_table1_overall(sub, outcome=e["outcome"])
        table2, _ = build_table2_category(sub, outcome=e["outcome"])
        embeddings.append({**e, "table1": table1, "table2": table2})
        table1.to_csv(output_dir / f"table1_overall_{e['outcome']}.csv", index=False)
        table2.to_csv(output_dir / f"table2_category_{e['outcome']}.csv", index=False)

    build_multi_embedding_coefficient_figure(
        df=df,
        embeddings=embeddings,
        output_path=output_dir / "rq3_dual_embedding_figure.png",
    )
    print(f"Dual-embedding figure + tables written to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()
    run(Path(args.input), Path(args.output_dir))
