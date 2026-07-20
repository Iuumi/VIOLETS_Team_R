"""
RQ1_dual_judge_analyze.py
==========================
Fits the same RQ1 mixed-effects models (overall + by-category VIOLETS vs.
Baseline effect) once per judge, then plots every judge's estimates as
separate, color-coded series on the same coefficient plot so judge
(dis)agreement is visible directly rather than averaged away. Also emits a
raw score-distribution histogram across judges.

Judges included:
  1. gpt-5-nano (primary), web search via OpenAI — `veracity_score` column,
     always present (this is the base eval_dataset the runner produces).
  2. deepseek-v4-flash (secondary), web search via Tavily —
     `veracity_score_2nd` column. Optional: only included if --input is
     already a *_dual_judge.jsonl file (i.e. rescore_secondary_rq1.py was
     run first) — if --input is the plain eval_dataset.jsonl, this judge
     is silently skipped rather than erroring.
  3. deepseek-v4-flash (secondary, no search), pure parametric knowledge —
     `veracity_score_3rd` column, from rescore_secondary_rq1_notool.py's
     *_notool_judge.jsonl output. Optional: pass --notool_input, or leave
     the default auto-derived path in place; if that file doesn't exist,
     the 3rd judge is silently skipped.
  4. llama-3.3-70b-instruct (no search), a third, unrelated model family
     (Meta, not OpenAI/DeepSeek) — `veracity_score_4th` column, from
     rescore_secondary_rq1_notool.py --model meta-llama/llama-3.3-70b-instruct
     --score_suffix 4th. Optional: pass --llama_input, or leave the default
     auto-derived path in place; if that file doesn't exist, this judge is
     silently skipped. Checks whether the no-search leniency/non-answer
     patterns seen in DeepSeek are DeepSeek-specific or a broader pattern.

Usage:
  python RQ1_dual_judge_analyze.py --input output/rq1/eval_dataset_20260710_dual_judge.jsonl --output_dir output/rq1/analysis_mixed_20260710
"""

import argparse
from pathlib import Path

import pandas as pd

from RQ1_analyze import (
    load_jsonl,
    preprocess,
    build_table1_overall,
    build_table2_category,
    build_multi_judge_coefficient_figure,
    build_dual_judge_score_distribution,
    build_dual_judge_score_distribution_figure,
)

PRIMARY_JUDGE = {"label": "gpt-5-nano (primary)", "color": "#7B2FBE", "outcome": "veracity_score"}
# deepseek-v4-flash is orange everywhere (RQ2's deepseek is always orange too,
# since it never needs search) so the same model reads as the same color
# across RQ1/RQ2/RQ3 posters. NOTOOL_JUDGE only gets its own distinct shade
# if SEARCH_JUDGE is *also* present in the same figure (both variants shown
# together) — see NOTOOL_JUDGE below.
SEARCH_JUDGE = {"label": "deepseek-v4-flash (secondary)", "color": "#E8871E", "outcome": "veracity_score_2nd"}
NOTOOL_JUDGE = {"label": "deepseek-v4-flash (no search)", "color": "#E8871E", "outcome": "veracity_score_3rd"}
LLAMA_JUDGE = {"label": "llama-3.3-70b (no search)", "color": "#D6428C", "outcome": "veracity_score_4th"}
# Fallback color if a run ever needs both SEARCH_JUDGE and NOTOOL_JUDGE in
# the same figure at once — they'd otherwise be indistinguishable.
_NOTOOL_FALLBACK_COLOR = "#2E9E5B"


def _default_notool_path(input_path: Path) -> Path:
    """*_dual_judge.jsonl -> *_notool_judge.jsonl, same eval_dataset stem."""
    stem = input_path.stem.removesuffix("_dual_judge")
    return input_path.with_name(f"{stem}_notool_judge.jsonl")


def _default_llama_path(input_path: Path) -> Path:
    """*_dual_judge.jsonl -> *_notool_judge_llama-33-70b-instruct.jsonl."""
    stem = input_path.stem.removesuffix("_dual_judge")
    return input_path.with_name(f"{stem}_notool_judge_llama-33-70b-instruct.jsonl")


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


def run(
    input_path: Path, output_dir: Path, notool_input: Path | None, llama_input: Path | None,
    poster: bool = False,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_jsonl(input_path)
    df = preprocess(df)

    all_judges = [PRIMARY_JUDGE]

    if "veracity_score_2nd" in df.columns:
        df["veracity_score_2nd"] = pd.to_numeric(df["veracity_score_2nd"], errors="coerce")
        all_judges.append(SEARCH_JUDGE)
    else:
        print(f"No 'veracity_score_2nd' column in {input_path} — skipping search-based secondary judge")

    notool_path = notool_input or _default_notool_path(input_path)
    if notool_path.exists():
        df = _merge_extra_judge(df, notool_path, NOTOOL_JUDGE)
        notool_judge = NOTOOL_JUDGE
        if SEARCH_JUDGE in all_judges:
            # Both deepseek variants in the same figure — they'd otherwise
            # share the same orange and be indistinguishable.
            notool_judge = {**NOTOOL_JUDGE, "color": _NOTOOL_FALLBACK_COLOR}
        all_judges.append(notool_judge)
    else:
        print(f"No no-search judge file found at {notool_path} — skipping")

    llama_path = llama_input or _default_llama_path(input_path)
    if llama_path.exists():
        df = _merge_extra_judge(df, llama_path, LLAMA_JUDGE)
        all_judges.append(LLAMA_JUDGE)
    else:
        print(f"No llama no-search judge file found at {llama_path} — skipping")

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
        output_path=output_dir / "rq1_dual_judge_figure.png",
    )
    if poster:
        build_multi_judge_coefficient_figure(
            df=df,
            judges=judges,
            output_path=output_dir / "rq1_dual_judge_figure_poster.png",
            poster=True,
        )

    dist_table = build_dual_judge_score_distribution(df, all_judges)
    dist_table.to_csv(output_dir / "rq1_dual_judge_score_distribution.csv", index=False)
    build_dual_judge_score_distribution_figure(
        df=df,
        judges=all_judges,
        output_path=output_dir / "rq1_dual_judge_score_distribution.png",
    )

    print(f"Figure + tables + score distribution ({len(all_judges)} judges) written to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--notool_input", type=str, default=None, help="Path to *_notool_judge.jsonl (optional 3rd judge, deepseek no-search). Auto-derived from --input if omitted.")
    parser.add_argument("--llama_input", type=str, default=None, help="Path to *_notool_judge_llama-33-70b-instruct.jsonl (optional 4th judge, llama no-search). Auto-derived from --input if omitted.")
    parser.add_argument("--poster", action="store_true", help="Also write an 8-inch-wide poster version (no title/caption, larger fonts, shared x-axis, faded non-significant points).")
    args = parser.parse_args()
    run(
        Path(args.input), Path(args.output_dir),
        Path(args.notool_input) if args.notool_input else None,
        Path(args.llama_input) if args.llama_input else None,
        args.poster,
    )
