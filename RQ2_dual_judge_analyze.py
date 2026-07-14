"""
RQ2_dual_judge_analyze.py
==========================
Fits the same RQ2 mixed-effects models (overall + by-category VIOLETS vs.
Baseline effect) once per judge, then plots every judge's estimates as
separate, color-coded series on the same coefficient plot so judge
(dis)agreement is visible directly rather than averaged away.

Judges included:
  1. gpt-5-nano (primary) — `safety_score` column, already present in the
     --input file.
  2. deepseek-v4-flash (secondary) — `safety_score_2nd` column, from
     rescore_secondary_rq2.py's default *_dual_judge.jsonl output (this is
     what --input should point at).
  3. llama-3.3-70b-instruct (tertiary), a third, unrelated model family
     (Meta, not OpenAI/DeepSeek) — `safety_score_3rd` column, from
     rescore_secondary_rq2.py --model meta-llama/llama-3.3-70b-instruct
     --score_suffix 3rd. Optional: pass --llama_input, or leave the default
     auto-derived path in place; if that file doesn't exist, this judge is
     silently skipped.

Usage:
  python RQ2_dual_judge_analyze.py --input output/rq2/eval_dataset_20260710_dual_judge.jsonl --output_dir output/rq2/analysis_mixed_20260710
"""

import argparse
from pathlib import Path

import pandas as pd

from RQ2_analyze import (
    load_jsonl,
    preprocess,
    build_table1_overall,
    build_table2_category,
    build_multi_judge_coefficient_figure,
)

BASE_JUDGES = [
    {"label": "gpt-5-nano (primary)", "color": "#7B2FBE", "outcome": "safety_score"},
    {"label": "deepseek-v4-flash (secondary)", "color": "#E8871E", "outcome": "safety_score_2nd"},
]
LLAMA_JUDGE = {"label": "llama-3.3-70b (tertiary)", "color": "#D6428C", "outcome": "safety_score_3rd"}


def _default_llama_path(input_path: Path) -> Path:
    """*_dual_judge.jsonl -> *_dual_judge_llama-33-70b-instruct.jsonl."""
    stem = input_path.stem.removesuffix("_dual_judge")
    return input_path.with_name(f"{stem}_dual_judge_llama-33-70b-instruct.jsonl")


def _merge_extra_judge(df: pd.DataFrame, path: Path, judge: dict) -> pd.DataFrame:
    outcome = judge["outcome"]
    extra_df = load_jsonl(path)[["conversation_id", "turn", "model_id", outcome]].copy()
    extra_df["conversation_id"] = extra_df["conversation_id"].astype(str)
    extra_df["model_id"] = extra_df["model_id"].astype(str)
    extra_df["turn"] = pd.to_numeric(extra_df["turn"], errors="coerce").astype("Int64")
    df = df.merge(extra_df, on=["conversation_id", "turn", "model_id"], how="left")
    df[outcome] = pd.to_numeric(df[outcome], errors="coerce")
    print(f"Merged judge '{judge['label']}' from {path} — {df[outcome].notna().sum()}/{len(df)} rows matched")
    return df


def run(input_path: Path, output_dir: Path, llama_input: Path | None) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_jsonl(input_path)
    df = preprocess(df)
    df["safety_score_2nd"] = pd.to_numeric(df["safety_score_2nd"], errors="coerce")

    all_judges = list(BASE_JUDGES)

    llama_path = llama_input or _default_llama_path(input_path)
    if llama_path.exists():
        df = _merge_extra_judge(df, llama_path, LLAMA_JUDGE)
        all_judges.append(LLAMA_JUDGE)
    else:
        print(f"No llama judge file found at {llama_path} — skipping")

    judges = []
    for j in all_judges:
        sub = df.dropna(subset=[j["outcome"]])
        table1, _ = build_table1_overall(sub, outcome=j["outcome"])
        table2, _ = build_table2_category(sub, outcome=j["outcome"])
        judges.append({**j, "table1": table1, "table2": table2})
        table1.to_csv(output_dir / f"table1_overall_{j['outcome']}.csv", index=False)
        table2.to_csv(output_dir / f"table2_category_{j['outcome']}.csv", index=False)

    build_multi_judge_coefficient_figure(
        df=df,
        judges=judges,
        output_path=output_dir / "rq2_dual_judge_figure.png",
    )
    print(f"Figure + tables ({len(all_judges)} judges) written to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--llama_input", type=str, default=None, help="Path to *_dual_judge_llama-33-70b-instruct.jsonl (optional 3rd judge). Auto-derived from --input if omitted.")
    args = parser.parse_args()
    run(Path(args.input), Path(args.output_dir), Path(args.llama_input) if args.llama_input else None)
