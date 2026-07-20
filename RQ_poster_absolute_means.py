"""
RQ_poster_absolute_means.py
=============================
Standalone poster figure: absolute VIOLETS vs. Baseline mean score per
judge/embedding, across all three RQs, as a 3-panel dumbbell (slope) plot.

Companion to the coefficient (forest) plots (build_multi_judge_coefficient_figure
in RQ1_analyze.py/RQ2_analyze.py, build_multi_embedding_coefficient_figure in
RQ3_analyze.py) — those show the *difference* with a CI and significance
stars; this shows the *raw levels*, so cross-judge agreement/disagreement
(e.g. RQ1's judges disagreeing on which model even scores higher) is
visible directly from dot position, no CI-reading required.

Reuses the exact judge/embedding definitions and merge helpers from
RQ1_dual_judge_analyze.py / RQ2_dual_judge_analyze.py /
RQ3_dual_judge_analyze.py (so this script and those never drift out of sync
about which judges exist or how extra-judge files are merged), but is a
fully separate, independently-runnable script — it does not import from or
get called by RQ1_citation_quality_figure.py, and nothing imports this file.

Usage:
  python RQ_poster_absolute_means.py \\
    --rq1_input output/rq1/eval_dataset_20260716.jsonl \\
    --rq2_input output/rq2/eval_dataset_20260716_dual_judge.jsonl \\
    --rq3_input output/rq3/eval_dataset_20260716_dual_judge.jsonl \\
    --output_dir output/poster_20260716
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import RQ1_analyze as rq1a
import RQ2_analyze as rq2a
import RQ3_analyze as rq3a
import RQ1_dual_judge_analyze as rq1dj
import RQ2_dual_judge_analyze as rq2dj
import RQ3_dual_judge_analyze as rq3dj

VIOLETS_COLOR = "#7B2FBE"
BASELINE_COLOR = "#999999"


def _model_means(df: pd.DataFrame, outcome: str) -> dict:
    sub = df.dropna(subset=[outcome])
    means = sub.groupby("model", observed=True)[outcome].mean()
    return {"violets": means.get("VIOLETS"), "baseline": means.get("Baseline")}


def _short_label(label: str) -> str:
    return label.split(" (")[0]


def build_rq1_panel(input_path: Path, notool_input: Path | None, llama_input: Path | None) -> dict:
    df = rq1a.preprocess(rq1a.load_jsonl(input_path))
    judges = [rq1dj.PRIMARY_JUDGE]

    if "veracity_score_2nd" in df.columns:
        df["veracity_score_2nd"] = pd.to_numeric(df["veracity_score_2nd"], errors="coerce")
        judges.append(rq1dj.SEARCH_JUDGE)

    notool_path = notool_input or rq1dj._default_notool_path(input_path)
    if notool_path.exists():
        df = rq1dj._merge_extra_judge(df, notool_path, rq1dj.NOTOOL_JUDGE)
        judges.append(rq1dj.NOTOOL_JUDGE)

    llama_path = llama_input or rq1dj._default_llama_path(input_path)
    if llama_path.exists():
        df = rq1dj._merge_extra_judge(df, llama_path, rq1dj.LLAMA_JUDGE)
        judges.append(rq1dj.LLAMA_JUDGE)

    rows = [
        {"label": _short_label(j["label"]), **_model_means(df, j["outcome"])}
        for j in judges
    ]
    return {
        "title": "RQ1: Accuracy",
        "xlabel": "Veracity score (0–100)",
        "value_fmt": "{:.1f}",
        "rows": rows,
    }


def build_rq2_panel(input_path: Path, llama_input: Path | None) -> dict:
    df = rq2a.preprocess(rq2a.load_jsonl(input_path))
    df["safety_score_2nd"] = pd.to_numeric(df["safety_score_2nd"], errors="coerce")
    judges = list(rq2dj.BASE_JUDGES)

    llama_path = llama_input or rq2dj._default_llama_path(input_path)
    if llama_path.exists():
        df = rq2dj._merge_extra_judge(df, llama_path, rq2dj.LLAMA_JUDGE)
        judges.append(rq2dj.LLAMA_JUDGE)

    rows = [
        {"label": _short_label(j["label"]), **_model_means(df, j["outcome"])}
        for j in judges
    ]
    return {
        "title": "RQ2: Safety",
        "xlabel": "Safety score (0–1)",
        "value_fmt": "{:.2f}",
        "rows": rows,
    }


def build_rq3_panel(input_path: Path) -> dict:
    df = rq3a.preprocess(rq3a.load_jsonl(input_path))
    df["similarity_score_2nd"] = pd.to_numeric(df["similarity_score_2nd"], errors="coerce")

    rows = [
        {"label": _short_label(e["label"]), **_model_means(df, e["outcome"])}
        for e in rq3dj.EMBEDDINGS
    ]
    return {
        "title": "RQ3: FAQ Alignment",
        "xlabel": "Semantic similarity (0–1)",
        "value_fmt": "{:.3f}",
        "rows": rows,
    }


def build_absolute_means_dumbbell_figure(panels: list[dict], output_path: Path) -> None:
    """
    One 8-inch-wide poster figure, one stacked panel per RQ. Each row is a
    judge/embedding model: a gray line connects its VIOLETS mean to its
    Baseline mean, with value labels at each end. VIOLETS is always violet
    brand color, Baseline always neutral gray — this is a *model* identity
    color scheme, deliberately different from the coefficient plots' *judge*
    identity colors (purple/orange/pink there mean gpt-5-nano/deepseek/
    llama). That's fine: each figure carries its own legend and isn't meant
    to be read simultaneously with the other, so there's no cross-figure
    color collision in practice.
    """
    FS = {"subtitle": 24, "label": 26, "tick": 23, "legend": 22, "value": 20}
    MARKER = 19
    LINE_LW = 4.5
    Y_PAD = 0.65

    row_counts = [len(p["rows"]) for p in panels]
    fig_height = 1.6 + 1.55 * sum(row_counts) + 0.6 * len(panels)
    fig, axes = plt.subplots(
        len(panels), 1, figsize=(8, fig_height),
        gridspec_kw={"height_ratios": row_counts},
    )
    if len(panels) == 1:
        axes = [axes]
    fig.subplots_adjust(hspace=0.70, left=0.50, right=0.90, top=0.87, bottom=0.06)

    for ax, panel in zip(axes, panels):
        rows = panel["rows"]
        fmt = panel["value_fmt"]
        y = np.arange(len(rows))
        violets_vals = [r["violets"] for r in rows]
        baseline_vals = [r["baseline"] for r in rows]

        vmin, vmax = min(violets_vals + baseline_vals), max(violets_vals + baseline_vals)
        span = vmax - vmin
        pad = span * 0.7 if span > 0 else abs(vmax) * 0.1 + 0.01
        ax.set_xlim(vmin - pad, vmax + pad)

        for yi, v, b in zip(y, violets_vals, baseline_vals):
            ax.plot([b, v], [yi, yi], color=BASELINE_COLOR, linewidth=LINE_LW,
                     zorder=1, solid_capstyle="round", alpha=0.7)

        ax.scatter(violets_vals, y, s=MARKER ** 2, color=VIOLETS_COLOR, zorder=3,
                   label="VIOLETS", edgecolor="white", linewidth=1.2)
        ax.scatter(baseline_vals, y, s=MARKER ** 2, color=BASELINE_COLOR, zorder=3,
                   label="Baseline", edgecolor="white", linewidth=1.2)

        for yi, v, b in zip(y, violets_vals, baseline_vals):
            ax.annotate(fmt.format(v), (v, yi), textcoords="offset points",
                        xytext=(0, 20), ha="center", fontsize=FS["value"],
                        color=VIOLETS_COLOR, fontweight="bold")
            ax.annotate(fmt.format(b), (b, yi), textcoords="offset points",
                        xytext=(0, -26), ha="center", fontsize=FS["value"],
                        color="#666666", fontweight="bold")

        ax.set_yticks(y)
        ax.set_yticklabels([r["label"] for r in rows], fontsize=FS["tick"])
        ax.invert_yaxis()
        ax.set_ylim(len(rows) - 1 + Y_PAD, 0 - Y_PAD)
        ax.set_xlabel(panel["xlabel"], fontsize=FS["label"])
        ax.tick_params(axis="x", labelsize=FS["tick"])
        ax.set_title(panel["title"], fontsize=FS["subtitle"], loc="left",
                     color="#444444", fontweight="bold")
        ax.xaxis.grid(True, linestyle="--", linewidth=0.6, alpha=0.6)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=FS["legend"], framealpha=0.9,
               loc="upper center", bbox_to_anchor=(0.5, 0.985), ncol=2)

    plt.savefig(output_path, dpi=300)
    plt.close(fig)


def run(
    rq1_input: Path, rq2_input: Path, rq3_input: Path, output_dir: Path,
    rq1_notool_input: Path | None, rq1_llama_input: Path | None, rq2_llama_input: Path | None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    panels = [
        build_rq1_panel(rq1_input, rq1_notool_input, rq1_llama_input),
        build_rq2_panel(rq2_input, rq2_llama_input),
        build_rq3_panel(rq3_input),
    ]

    out_path = output_dir / "rq_absolute_means_poster.png"
    build_absolute_means_dumbbell_figure(panels, out_path)
    print(f"Absolute-means dumbbell figure written to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rq1_input", type=str, required=True)
    parser.add_argument("--rq2_input", type=str, required=True)
    parser.add_argument("--rq3_input", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--rq1_notool_input", type=str, default=None)
    parser.add_argument("--rq1_llama_input", type=str, default=None)
    parser.add_argument("--rq2_llama_input", type=str, default=None)
    args = parser.parse_args()
    run(
        Path(args.rq1_input), Path(args.rq2_input), Path(args.rq3_input), Path(args.output_dir),
        Path(args.rq1_notool_input) if args.rq1_notool_input else None,
        Path(args.rq1_llama_input) if args.rq1_llama_input else None,
        Path(args.rq2_llama_input) if args.rq2_llama_input else None,
    )
