"""
RQ1.1 Mixed-Effects Analysis for VIOLETS Accuracy Evaluation
============================================================

Purpose
-------
Analyze RQ1.1 ("How factually accurate are VIOLETS's responses, and does it hallucinate?")
using mixed-effects models, then export:
  - Table 1: overall model effect
  - Table 2: model effect within each category
  - Table 3: model effect within each turn
  - One combined figure with 3 subplots matching the three tables

Design assumptions from the project
-----------------------------------
- Same participant turns are sent to both VIOLETS and the baseline model
- One JSONL line = one evaluated turn
- Outcome = veracity_score (0-100)
- Repeated observations are clustered within conversation_id

Input
-----
Default input path:
    output/rq1/eval_dataset.jsonl

Expected schema (per line)
--------------------------
- conversation_id
- model_id
- category
- seed_prompt
- seed_intent
- seed_question_type
- turn
- input
- output
- veracity_score
- overall_veracity_score
- reasoning
- timestamp

Outputs
-------
Default output directory:
    output/rq1/analysis_mixed/

Files created:
- table1_model_overall.csv
- table2_category_effects.csv
- table3_turn_effects.csv
- rq1_mixed_effects_combined_figure.png
- model_overall_summary.txt
- category_model_summary.txt
- turn_model_summary.txt

Notes
-----
1. This script uses statsmodels MixedLM with a random intercept for conversation_id.
2. If the default optimizer fails, it retries with alternative optimizers.
3. Category-level and turn-level effects are derived as fixed-effect contrasts.
4. The figure uses a common x-axis:
      Estimated difference in veracity score (VIOLETS − Baseline)
   so that all three subplots are directly comparable.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, List, Tuple

import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf

# ── Review protocol constants (RQ1.1 planning doc) ────────────────────────────
REVIEW_THRESHOLD = 70  # Turns below this are flagged for human review

ERROR_TYPE_KEYWORDS = {
    "date/location error": [
        "date",
        "deadline",
        "location",
        "address",
        "wrong",
        "incorrect",
        "false",
    ],
    "nuance/specificity gap": [
        "incomplete",
        "partial",
        "omit",
        "missing",
        "vague",
        "general",
        "not mention",
    ],
    "hallucination": ["fabricat", "invent", "hallucin", "not exist", "made up"],
}


def classify_error(reasoning: str) -> str:
    r = reasoning.lower()
    for etype, keywords in ERROR_TYPE_KEYWORDS.items():
        if any(kw in r for kw in keywords):
            return etype
    return "other"


def build_score_distribution(df: pd.DataFrame) -> pd.DataFrame:
    """Score distribution in 6 buckets per model."""
    bins = [0, 40, 60, 70, 80, 90, 101]
    labels = ["0–39", "40–59", "60–69", "70–79", "80–89", "90–100"]
    rows = []
    for model, grp in df.groupby("model", observed=True):
        counts = (
            pd.cut(
                grp["veracity_score"],
                bins=bins,
                labels=labels,
                right=False,
                include_lowest=True,
            )
            .value_counts()
            .sort_index()
        )
        pcts = (counts / counts.sum() * 100).round(1)
        for lbl in labels:
            rows.append(
                {
                    "model": model,
                    "bucket": lbl,
                    "count": int(counts[lbl]),
                    "pct": float(pcts[lbl]),
                }
            )
    return pd.DataFrame(rows)


def build_flagged_table(
    df: pd.DataFrame, threshold: int = REVIEW_THRESHOLD
) -> pd.DataFrame:
    """VIOLETS turns below threshold, with error-type classification."""
    flagged = df[(df["model"] == "VIOLETS") & (df["veracity_score"] < threshold)].copy()
    if flagged.empty:
        return flagged
    flagged["error_type"] = flagged["reasoning"].fillna("").apply(classify_error)
    return flagged[
        [
            "conversation_id",
            "category",
            "turn",
            "veracity_score",
            "error_type",
            "input",
            "output",
            "reasoning",
        ]
    ]


# ============================================================================
# I/O helpers
# ============================================================================


def ensure_dir(path: Path) -> None:
    """Create directory if missing."""
    path.mkdir(parents=True, exist_ok=True)


def load_jsonl(path: Path) -> pd.DataFrame:
    """Load JSONL into a DataFrame."""
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON on line {line_num}: {e}") from e

    if not records:
        raise ValueError(f"No JSON objects found in {path}")

    return pd.DataFrame(records)


def validate_columns(df: pd.DataFrame, required: Iterable[str]) -> None:
    """Ensure required columns are present."""
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


# ============================================================================
# Data preparation
# ============================================================================


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and standardize key variables.

    Important modeling choices:
    - model_id is converted to a categorical with Baseline reference first and VIOLETS second
    - category is kept in a stable order if possible
    - turn is cast to categorical strings ("0", "1", "2", ...)
    """
    df = df.copy()

    # Numeric conversion
    df["veracity_score"] = pd.to_numeric(df["veracity_score"], errors="coerce")
    df["turn"] = pd.to_numeric(df["turn"], errors="coerce").astype("Int64")

    # Basic text conversion
    for col in [
        "conversation_id",
        "model_id",
        "category",
        "seed_prompt",
        "seed_intent",
        "seed_question_type",
        "input",
        "output",
        "reasoning",
        "timestamp",
    ]:
        if col in df.columns:
            df[col] = df[col].astype(str)

    # Drop rows without key modeling fields
    df = df.dropna(subset=["veracity_score", "turn"]).copy()

    # Normalize model labels
    # We want two levels only: Baseline and VIOLETS
    # "violets" in the JSONL becomes "VIOLETS"
    df["model"] = df["model_id"].replace({"violets": "VIOLETS"})

    # Infer baseline label(s): anything not VIOLETS becomes baseline
    df["model"] = np.where(df["model"] == "VIOLETS", "VIOLETS", "Baseline")
    df["model"] = pd.Categorical(
        df["model"], categories=["Baseline", "VIOLETS"], ordered=True
    )

    # Stable category ordering from the project plan / generator
    category_order = [
        "procedural",
        "eligibility",
        "mail_in",
        "results_integrity",
        "edge_cases",
    ]
    seen_categories = [c for c in category_order if c in set(df["category"])]
    # append any unexpected categories at the end
    extras = [c for c in sorted(df["category"].unique()) if c not in seen_categories]
    full_category_order = seen_categories + extras
    df["category"] = pd.Categorical(
        df["category"], categories=full_category_order, ordered=True
    )

    # Turn as string categorical for formula coding
    turn_order = [str(t) for t in sorted(df["turn"].dropna().astype(int).unique())]
    df["turn_str"] = df["turn"].astype(int).astype(str)
    df["turn_str"] = pd.Categorical(df["turn_str"], categories=turn_order, ordered=True)

    # Stable sorting
    df = df.sort_values(["conversation_id", "model", "turn"]).reset_index(drop=True)

    return df


# ============================================================================
# Mixed model fitting
# ============================================================================


def fit_mixedlm(formula: str, df: pd.DataFrame, group_col: str):
    """
    Fit MixedLM with a random intercept for group_col.

    MixedLM can be optimizer-sensitive. This helper retries with several optimizers.
    """
    model = smf.mixedlm(formula, data=df, groups=df[group_col])

    last_err = None
    for method in ["lbfgs", "powell", "cg", "bfgs"]:
        try:
            result = model.fit(reml=False, method=method, disp=False)
            return result
        except Exception as e:
            last_err = e

    raise RuntimeError(f"MixedLM failed for formula: {formula}\nLast error: {last_err}")


# ============================================================================
# Contrast extraction
# ============================================================================


def coef_and_ci(
    result, coef_name: str, z_crit: float = 1.96
) -> Tuple[float, float, float, float]:
    """
    Extract one fixed-effect coefficient, its standard error, and Wald 95% CI.
    Returns:
        estimate, ci_low, ci_high, p_value
    """
    est = float(result.fe_params[coef_name])
    se = float(result.bse_fe[coef_name])
    lo = est - z_crit * se
    hi = est + z_crit * se

    # Wald z-statistic
    z = est / se if se > 0 else np.nan
    p = 2 * (1 - normal_cdf(abs(z))) if np.isfinite(z) else np.nan
    return est, lo, hi, p


def contrast_est_ci(
    result, coef_names: List[str], weights: List[float], z_crit: float = 1.96
) -> Tuple[float, float, float, float]:
    """
    Compute a linear contrast from fixed effects and its Wald 95% CI.

    Example:
        model effect in category k
        = beta_model + beta_model:category_k
    """
    beta = result.fe_params
    cov = result.cov_params().loc[beta.index, beta.index]

    index_map = {name: i for i, name in enumerate(beta.index)}
    L = np.zeros(len(beta))

    for name, w in zip(coef_names, weights):
        if name not in index_map:
            raise KeyError(f"Coefficient not found in fitted model: {name}")
        L[index_map[name]] += w

    est = float(L @ beta.values)
    var = float(L @ cov.values @ L)
    se = np.sqrt(var) if var >= 0 else np.nan
    lo = est - z_crit * se
    hi = est + z_crit * se

    z = est / se if (se is not None and se > 0) else np.nan
    p = 2 * (1 - normal_cdf(abs(z))) if np.isfinite(z) else np.nan
    return est, lo, hi, p


def normal_cdf(x: float) -> float:
    """Standard normal CDF without scipy."""
    return 0.5 * (1.0 + math.erf(x / np.sqrt(2.0)))


# ============================================================================
# Table builders
# ============================================================================


def build_table1_overall(df: pd.DataFrame) -> Tuple[pd.DataFrame, object]:
    """
    Table 1: Overall model effect from:
        veracity_score ~ model + (1 | conversation_id)
    """
    result = fit_mixedlm(
        "veracity_score ~ C(model, Treatment(reference='Baseline'))",
        df=df,
        group_col="conversation_id",
    )

    coef_name = "C(model, Treatment(reference='Baseline'))[T.VIOLETS]"
    est, lo, hi, p = coef_and_ci(result, coef_name)

    table = pd.DataFrame(
        [
            {
                "effect": "Overall model effect",
                "contrast": "VIOLETS - Baseline",
                "estimate": est,
                "ci_low": lo,
                "ci_high": hi,
                "p_value": p,
                "n_rows": len(df),
                "n_conversations": df["conversation_id"].nunique(),
            }
        ]
    )

    return table, result


def build_table2_category(df: pd.DataFrame) -> Tuple[pd.DataFrame, object]:
    """
    Table 2: Model effect within each category from:
        veracity_score ~ model * category + (1 | conversation_id)
    """
    if len(df["category"].cat.categories) == 0:
        raise ValueError("No category levels found.")

    ref_cat = df["category"].cat.categories[0]

    formula = (
        "veracity_score ~ "
        "C(model, Treatment(reference='Baseline')) * "
        f"C(category, Treatment(reference='{ref_cat}'))"
    )
    result = fit_mixedlm(formula, df=df, group_col="conversation_id")

    model_coef = "C(model, Treatment(reference='Baseline'))[T.VIOLETS]"

    rows = []
    for cat in df["category"].cat.categories:
        if cat == ref_cat:
            est, lo, hi, p = coef_and_ci(result, model_coef)
        else:
            interaction = (
                f"C(model, Treatment(reference='Baseline'))[T.VIOLETS]:"
                f"C(category, Treatment(reference='{ref_cat}'))[T.{cat}]"
            )
            est, lo, hi, p = contrast_est_ci(
                result,
                coef_names=[model_coef, interaction],
                weights=[1.0, 1.0],
            )

        rows.append(
            {
                "category": cat,
                "contrast": "VIOLETS - Baseline",
                "estimate": est,
                "ci_low": lo,
                "ci_high": hi,
                "p_value": p,
                "n_rows": int((df["category"] == cat).sum()),
                "n_conversations": int(
                    df.loc[df["category"] == cat, "conversation_id"].nunique()
                ),
            }
        )

    table = pd.DataFrame(rows)
    return table, result


def build_table3_turn(df: pd.DataFrame) -> Tuple[pd.DataFrame, object]:
    """
    Table 3: Model effect within each turn from:
        veracity_score ~ model * turn + (1 | conversation_id)
    """
    if len(df["turn_str"].cat.categories) == 0:
        raise ValueError("No turn levels found.")

    ref_turn = df["turn_str"].cat.categories[0]

    formula = (
        "veracity_score ~ "
        "C(model, Treatment(reference='Baseline')) * "
        f"C(turn_str, Treatment(reference='{ref_turn}'))"
    )
    result = fit_mixedlm(formula, df=df, group_col="conversation_id")

    model_coef = "C(model, Treatment(reference='Baseline'))[T.VIOLETS]"

    rows = []
    for turn in df["turn_str"].cat.categories:
        if turn == ref_turn:
            est, lo, hi, p = coef_and_ci(result, model_coef)
        else:
            interaction = (
                f"C(model, Treatment(reference='Baseline'))[T.VIOLETS]:"
                f"C(turn_str, Treatment(reference='{ref_turn}'))[T.{turn}]"
            )
            est, lo, hi, p = contrast_est_ci(
                result,
                coef_names=[model_coef, interaction],
                weights=[1.0, 1.0],
            )

        rows.append(
            {
                "turn": turn,
                "contrast": "VIOLETS - Baseline",
                "estimate": est,
                "ci_low": lo,
                "ci_high": hi,
                "p_value": p,
                "n_rows": int((df["turn_str"] == turn).sum()),
                "n_conversations": int(
                    df.loc[df["turn_str"] == turn, "conversation_id"].nunique()
                ),
            }
        )

    table = pd.DataFrame(rows)
    return table, result


# ============================================================================
# Descriptive summaries (text only, optional but useful)
# ============================================================================


def simple_model_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Helpful descriptive summary, separate from inferential tables."""
    return (
        df.groupby("model", observed=True)["veracity_score"]
        .agg(["count", "mean", "std", "median", "min", "max"])
        .reset_index()
    )


def simple_category_model_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Helpful descriptive summary by category and model."""
    return (
        df.groupby(["category", "model"], observed=True)["veracity_score"]
        .agg(["count", "mean", "std", "median"])
        .reset_index()
    )


def simple_turn_model_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Helpful descriptive summary by turn and model."""
    return (
        df.groupby(["turn_str", "model"], observed=True)["veracity_score"]
        .agg(["count", "mean", "std", "median"])
        .reset_index()
    )


# ============================================================================
# Figure builder
# ============================================================================


def _desc_stats(df: pd.DataFrame, group_col: str, outcome: str) -> pd.DataFrame:
    """Mean ± 95% CI per model within each level of group_col."""
    rows = []
    for (grp, model), sub in df.groupby([group_col, "model"], observed=True):
        n = len(sub)
        m = sub[outcome].mean()
        se = sub[outcome].sem()
        rows.append(
            {
                group_col: grp,
                "model": model,
                "mean": m,
                "ci_low": m - 1.96 * se,
                "ci_high": m + 1.96 * se,
                "n": n,
            }
        )
    return pd.DataFrame(rows)


_CAT_LABELS_RQ1 = {
    "procedural": "procedural",
    "eligibility": "eligibility",
    "mail_in": "mail-in",
    "results_integrity": "results /\nintegrity",
    "edge_cases": "edge\ncases",
}


def _sig_stars(p: float) -> str:
    if pd.isna(p) or p >= 0.05:
        return "ns"
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    return "*"


def build_combined_figure(
    df: pd.DataFrame,
    table1: pd.DataFrame,
    table2: pd.DataFrame,
    table3: pd.DataFrame,
    output_path: Path,
    outcome: str = "veracity_score",
    ylabel: str = "Mean veracity score",
    ymin: float = 50.0,
    ymax: float = 100.0,
    title: str = "RQ1: Veracity Scores — VIOLETS vs. Baseline",
) -> None:
    """
    Three-panel grouped bar chart optimized for poster display:
      (A) Overall  (B) By Category  (C) By Turn

    - Significance asterisks drawn inside each panel above bar pairs
    - Legend shown only in Panel A
    - Category labels shortened via _CAT_LABELS_RQ1
    - y-axis starts at ymin (default 50) so differences are visually legible
    """
    COLORS = {"Baseline": "#6baed6", "VIOLETS": "#fd8d3c"}
    FS = {"title": 15, "label": 13, "tick": 11, "legend": 11, "stars": 13}
    BAR_W = 0.38
    CAP = 5
    ERR_KW = {"elinewidth": 1.6, "ecolor": "#333333"}
    STAR_PAD = (ymax - ymin) * 0.03  # gap above the error bar before the star

    overall_desc = _desc_stats(df, "model", outcome).set_index("model")
    cat_desc = _desc_stats(df, "category", outcome)
    turn_desc = _desc_stats(df, "turn_str", outcome)

    fig, axes = plt.subplots(
        1, 3, figsize=(16, 5.5), gridspec_kw={"width_ratios": [1, 2.2, 1.6]}
    )
    fig.subplots_adjust(wspace=0.35)

    def _annotate_stars(ax, x_center, top_y, stars):
        """Place significance marker just above the tallest error bar."""
        if stars == "ns":
            return
        ax.text(
            x_center,
            top_y + STAR_PAD,
            stars,
            ha="center",
            va="bottom",
            fontsize=FS["stars"],
            color="#222222",
        )

    def _bar_group(
        ax,
        index_vals,
        groups_data,
        label_col,
        x_labels,
        p_table,
        p_col,
        rotate=0,
        ha="center",
        show_legend=False,
    ):
        x = np.arange(len(index_vals))
        ci_hi_by_group = {}  # track tallest CI top per group for star placement

        for j, model in enumerate(["Baseline", "VIOLETS"]):
            sub = groups_data[groups_data["model"] == model].set_index(label_col)
            means = [
                sub.loc[v, "mean"] if v in sub.index else np.nan for v in index_vals
            ]
            ci_lo = [
                sub.loc[v, "ci_low"] if v in sub.index else np.nan for v in index_vals
            ]
            ci_hi = [
                sub.loc[v, "ci_high"] if v in sub.index else np.nan for v in index_vals
            ]
            ax.bar(
                x + j * BAR_W,
                means,
                BAR_W,
                color=COLORS[model],
                label=model,
                yerr=[
                    [m - lo for m, lo in zip(means, ci_lo)],
                    [hi - m for m, hi in zip(means, ci_hi)],
                ],
                capsize=CAP,
                error_kw=ERR_KW,
                edgecolor="white",
                linewidth=0.5,
            )
            for i, (ci_top) in enumerate(ci_hi):
                ci_hi_by_group[i] = max(ci_hi_by_group.get(i, ymin), ci_top)

        # Significance stars per group
        p_lookup = (
            p_table.set_index(p_col)["p_value"] if p_col in p_table.columns else {}
        )
        for i, val in enumerate(index_vals):
            p = p_lookup.get(val, np.nan) if hasattr(p_lookup, "get") else np.nan
            stars = _sig_stars(p)
            _annotate_stars(ax, x[i] + BAR_W / 2, ci_hi_by_group.get(i, ymin), stars)

        ax.set_xticks(x + BAR_W / 2)
        ax.set_xticklabels(x_labels, rotation=rotate, ha=ha, fontsize=FS["tick"])
        ax.set_ylim(ymin, ymax)
        ax.yaxis.grid(True, linestyle="--", linewidth=0.6, alpha=0.6)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
        if show_legend:
            ax.legend(fontsize=FS["legend"], framealpha=0.7)

    # ── Panel A: Overall ──────────────────────────────────────────────────
    ax = axes[0]
    max_ci_top = ymin
    for i, model in enumerate(["Baseline", "VIOLETS"]):
        row = overall_desc.loc[model]
        ax.bar(
            i,
            row["mean"],
            BAR_W * 1.4,
            color=COLORS[model],
            label=model,
            yerr=[[row["mean"] - row["ci_low"]], [row["ci_high"] - row["mean"]]],
            capsize=CAP,
            error_kw=ERR_KW,
            edgecolor="white",
            linewidth=0.5,
        )
        max_ci_top = max(max_ci_top, row["ci_high"])
    _annotate_stars(ax, 0.5, max_ci_top, _sig_stars(table1["p_value"].iloc[0]))
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Baseline", "VIOLETS"], fontsize=FS["tick"])
    ax.set_title("(A) Overall", fontsize=FS["title"], fontweight="bold")
    ax.set_ylabel(ylabel, fontsize=FS["label"])
    ax.set_ylim(ymin, ymax)
    ax.legend(fontsize=FS["legend"], framealpha=0.7)
    ax.yaxis.grid(True, linestyle="--", linewidth=0.6, alpha=0.6)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)

    # ── Panel B: By Category ──────────────────────────────────────────────
    categories = df["category"].cat.categories.tolist()
    cat_labels = [_CAT_LABELS_RQ1.get(c, c) for c in categories]
    _bar_group(
        axes[1],
        categories,
        cat_desc,
        "category",
        cat_labels,
        p_table=table2,
        p_col="category",
        rotate=20,
        ha="right",
        show_legend=False,
    )
    axes[1].set_title(
        "(B) By Question Category", fontsize=FS["title"], fontweight="bold"
    )

    # ── Panel C: By Turn ──────────────────────────────────────────────────
    turns = df["turn_str"].cat.categories.tolist()
    _bar_group(
        axes[2],
        turns,
        turn_desc,
        "turn_str",
        [f"Turn {t}" for t in turns],
        p_table=table3,
        p_col="turn",
        show_legend=False,
    )
    axes[2].set_title(
        "(C) By Conversation Turn", fontsize=FS["title"], fontweight="bold"
    )

    fig.suptitle(title, fontsize=16, fontweight="bold", y=1.03)
    fig.text(
        0.5,
        -0.02,
        "Bars = mean ± 95% CI  |  * p < .05  ** p < .01  *** p < .001"
        "  |  p-values from linear mixed-effects model (random intercept per conversation)",
        ha="center",
        fontsize=10,
        color="#555555",
    )

    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


# ============================================================================
# Main driver
# ============================================================================


def run_analysis(input_path: Path, output_dir: Path) -> None:
    """End-to-end analysis pipeline."""
    ensure_dir(output_dir)

    required_columns = [
        "conversation_id",
        "model_id",
        "category",
        "turn",
        "veracity_score",
    ]

    df = load_jsonl(input_path)
    validate_columns(df, required_columns)
    df = preprocess(df)

    # Inferential tables
    table1, m1 = build_table1_overall(df)
    table2, m2 = build_table2_category(df)
    table3, m3 = build_table3_turn(df)

    table1.to_csv(output_dir / "table1_model_overall.csv", index=False)
    table2.to_csv(output_dir / "table2_category_effects.csv", index=False)
    table3.to_csv(output_dir / "table3_turn_effects.csv", index=False)

    # Supportive descriptive summaries
    simple_model_summary(df).to_csv(
        output_dir / "model_overall_summary.csv", index=False
    )
    simple_category_model_summary(df).to_csv(
        output_dir / "category_model_summary.csv", index=False
    )
    simple_turn_model_summary(df).to_csv(
        output_dir / "turn_model_summary.csv", index=False
    )

    # Text summaries of fitted models
    (output_dir / "model_overall_summary.txt").write_text(
        str(m1.summary()), encoding="utf-8"
    )
    (output_dir / "category_model_summary.txt").write_text(
        str(m2.summary()), encoding="utf-8"
    )
    (output_dir / "turn_model_summary.txt").write_text(
        str(m3.summary()), encoding="utf-8"
    )

    # Figure
    build_combined_figure(
        df=df,
        table1=table1,
        table2=table2,
        table3=table3,
        output_path=output_dir / "rq1_mixed_effects_combined_figure.png",
    )

    # Score distribution (RQ1.1 reporting requirement)
    dist_table = build_score_distribution(df)
    dist_table.to_csv(output_dir / "score_distribution.csv", index=False)

    # Flagged turns for human review (below threshold)
    flagged = build_flagged_table(df, threshold=REVIEW_THRESHOLD)
    if not flagged.empty:
        flagged.to_csv(output_dir / "flagged_for_review.csv", index=False)
        print(
            f"Flagged {len(flagged)} VIOLETS turn(s) below threshold ({REVIEW_THRESHOLD}) → flagged_for_review.csv"
        )
    else:
        print(f"No VIOLETS turns below threshold ({REVIEW_THRESHOLD}).")

    print("RQ1.1 mixed-effects analysis complete.")
    print(f"Input:      {input_path}")
    print(f"Output dir: {output_dir}")


# ============================================================================
# CLI
# ============================================================================


def parse_args():
    parser = argparse.ArgumentParser(
        description="RQ1.1 mixed-effects analysis for VIOLETS accuracy evaluation."
    )
    parser.add_argument(
        "--input",
        type=str,
        default="output/rq1/eval_dataset.jsonl",
        help="Path to the RQ1 JSONL file.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output/rq1/analysis_mixed",
        help="Directory where outputs will be written.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    input_path = Path(args.input)
    output_dir = Path(args.output_dir)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    run_analysis(input_path=input_path, output_dir=output_dir)


if __name__ == "__main__":
    main()
