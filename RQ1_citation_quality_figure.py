"""
RQ1_citation_quality_figure.py
================================
Standalone poster figure: RQ1.2's citation-quality chain as a 3-stage
horizontal staged bar — cited -> accessible -> accurate.

Stages 2 and 3 are conditional on the previous stage (accessible = % of
*citations* that resolve; accurate = % of *accessible* pages that back the
claim), not raw percentages of all VIOLETS turns — this script computes the
conditional accurate-of-accessible number explicitly, since
RQ1_analyze.build_url_citation_summary()'s pct_accurate field is
unconditional (% of all cited URLs, not just the reachable ones) and would
understate stage 3 if used directly.

Only depends on RQ1_analyze.py (load_jsonl, preprocess,
build_url_citation_summary) — fully independent of
RQ_poster_absolute_means.py; neither script imports the other.

Usage:
  python RQ1_citation_quality_figure.py --input output/rq1/eval_dataset_20260716.jsonl --output_dir output/poster_20260716
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt

from RQ1_analyze import load_jsonl, preprocess, build_url_citation_summary

BAR_COLOR = "#7B2FBE"
TRACK_COLOR = "#E6DAF5"


def build_citation_quality_figure(summary: dict, output_path: Path) -> None:
    n_urls_total = summary["n_urls_total"]
    n_urls_accessible = summary["n_urls_accessible"]
    n_urls_accurate = summary["n_urls_accurate"]

    pct_accurate_of_accessible = (
        100 * n_urls_accurate / n_urls_accessible if n_urls_accessible else 0.0
    )

    stages = [
        ("Cited", summary["pct_cited"], f"{int(summary['n_turns_cited'])}/{int(summary['n_turns'])} turns"),
        ("Accessible", summary["pct_accessible"], f"{int(n_urls_accessible)}/{int(n_urls_total)} citations"),
        ("Accurate", pct_accurate_of_accessible, f"{int(n_urls_accurate)}/{int(n_urls_accessible)} accessible pages"),
    ]

    FS = {"label": 19, "value": 20, "sub": 13, "connector": 13}
    fig, ax = plt.subplots(figsize=(8, 3.6))
    fig.subplots_adjust(left=0.24, right=0.96, top=0.90, bottom=0.10)

    y = list(range(len(stages)))
    bar_h = 0.55
    for yi, (name, pct, sub) in zip(y, stages):
        ax.barh(yi, 100, height=bar_h, color=TRACK_COLOR, zorder=1)
        ax.barh(yi, pct, height=bar_h, color=BAR_COLOR, zorder=2)
        ax.text(pct + 2, yi, f"{pct:.1f}%", va="center", ha="left",
                 fontsize=FS["value"], color=BAR_COLOR, fontweight="bold")
        ax.text(1.5, yi, sub, va="center", ha="left",
                 fontsize=FS["sub"], color="white", fontweight="bold", zorder=3)

    for yi in y[:-1]:
        ax.annotate(
            "", xy=(8, yi + 1 - bar_h / 2 - 0.03), xytext=(8, yi + bar_h / 2 + 0.03),
            arrowprops=dict(arrowstyle="-|>", color="#666666", lw=1.6),
        )

    ax.annotate("of citations", (14, 0.5), fontsize=FS["connector"], color="#666666",
                ha="left", va="center", style="italic")
    ax.annotate("of accessible pages", (14, 1.5), fontsize=FS["connector"], color="#666666",
                ha="left", va="center", style="italic")

    ax.set_yticks(y)
    ax.set_yticklabels([s[0] for s in stages], fontsize=FS["label"])
    ax.invert_yaxis()
    ax.set_ylim(len(stages) - 1 + 0.7, 0 - 0.7)
    ax.set_xlim(0, 118)
    ax.set_xticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    plt.savefig(output_path, dpi=300)
    plt.close(fig)


def run(input_path: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    df = preprocess(load_jsonl(input_path))
    if "url_citation_rate_score" not in df.columns:
        raise ValueError(
            f"{input_path} has no url_citation_rate_score column — "
            "this JSONL predates the RQ1.2 grounding-pipeline fix."
        )

    summary = build_url_citation_summary(df).iloc[0].to_dict()
    out_path = output_dir / "rq1_citation_quality_poster.png"
    build_citation_quality_figure(summary, out_path)
    print(f"Citation-quality figure written to {out_path}")
    print(
        f"  cited={summary['pct_cited']:.1f}%  accessible={summary['pct_accessible']:.1f}%  "
        f"accurate_of_accessible={100 * summary['n_urls_accurate'] / summary['n_urls_accessible']:.1f}%"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()
    run(Path(args.input), Path(args.output_dir))
