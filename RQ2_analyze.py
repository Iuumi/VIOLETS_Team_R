"""
RQ2 Mixed-Effects Analysis for VIOLETS Safety Evaluation
========================================================

Purpose
-------
Analyze RQ2 ("Does VIOLETS correctly handle adversarial, out-of-scope,
sensitive, and misinformation-related queries safely?") using the primary
continuous outcome `safety_score`.

This script is intentionally aligned with the decisions made for RQ2:
  - Primary outcome: safety_score (0.0 to 1.0)
  - Primary inferential analysis: linear mixed-effects models
  - Primary figure: ONE combined 3-panel figure
  - No separate flagged-rate figure
  - flagged / conversation_flagged are retained only as supplementary summaries

Expected input
--------------
Default:
    output/rq2/eval_dataset.jsonl

Expected columns (one row per turn, per model)
----------------------------------------------
- conversation_id
- model_id
- category
- seed_prompt
- seed_intent
- seed_technique
- turn
- input
- output
- label
- safety_score
- overall_safety_score
- violation
- reasoning
- escalation_note
- flagged
- conversation_flagged
- timestamp

Primary outputs
---------------
- table1_model_overall.csv
- table2_category_effects.csv
- table3_turn_effects.csv
- rq2_mixed_effects_combined_figure.png

Supplementary outputs
---------------------
- model_overall_summary.csv
- category_model_summary.csv
- turn_model_summary.csv
- flagged_summary.csv
- conversation_flagged_summary.csv
- passfail_by_category.csv       (PASS/FAIL/WARN rates — primary rubric)
- passfail_by_turn.csv           (PASS/FAIL/WARN rates by turn)
- violation_breakdown.csv        (VIOLETS FAIL turns with violation strings)
- model_overall_mixedlm_summary.txt
- category_mixedlm_summary.txt
- turn_mixedlm_summary.txt

Statistical approach
--------------------
We fit three linear mixed-effects models, each with a random intercept for
conversation_id:

1. Overall model:
    safety_score ~ model + (1 | conversation_id)

2. Category interaction model:
    safety_score ~ model * category + (1 | conversation_id)

3. Turn interaction model:
    safety_score ~ model * turn + (1 | conversation_id)

Interpretation
--------------
All inferential estimates are reported as:
    VIOLETS - Baseline

Therefore:
- Positive values mean VIOLETS is SAFER than the baseline
- Negative values mean VIOLETS is LESS SAFE than the baseline

Important note on category/turn effects
---------------------------------------
For category-specific and turn-specific effects, the quantity of interest is not
always a single fixed-effect coefficient. Instead, it is a fixed-effect contrast:
the sum of the main model effect and the relevant interaction term.

In other words:
- Overall effect = beta_model
- Effect in category k = beta_model + beta_model:category_k
- Effect in turn t = beta_model + beta_model:turn_t
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf


# ============================================================================
# Basic utilities
# ============================================================================


def ensure_dir(path: Path) -> None:
    """Create a directory (including parents) if it does not already exist."""
    path.mkdir(parents=True, exist_ok=True)


def normal_cdf(x: float) -> float:
    """
    Standard normal CDF, implemented without scipy.

    This is used to convert Wald z-statistics into two-sided p-values for
    coefficient estimates and linear contrasts.
    """
    return 0.5 * (1.0 + np.math.erf(x / np.sqrt(2.0)))


def format_p_value(p: float) -> str:
    """
    Pretty formatter for p-values for use in figure annotations.

    Examples:
      0.0004 -> 'p < .001'
      0.0132 -> 'p = .013'
    """
    if pd.isna(p):
        return "p = NA"
    if p < 0.001:
        return "p < .001"
    return f"p = {p:.3f}".replace("0.", ".")


# ============================================================================
# JSONL loading and validation
# ============================================================================


def load_jsonl(path: Path) -> pd.DataFrame:
    """
    Load a JSONL file into a pandas DataFrame.

    Each non-empty line must be a valid JSON object.
    """
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON at line {line_num}: {e}") from e

    if not rows:
        raise ValueError(f"No JSON objects found in {path}")

    return pd.DataFrame(rows)


def validate_columns(df: pd.DataFrame, required: Iterable[str]) -> None:
    """Raise an error if any required columns are missing."""
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


# ============================================================================
# Data preparation
# ============================================================================


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and standardize the RQ2 dataset for modeling.

    Key steps:
    - Cast safety_score to numeric
    - Cast turn to integer, then a categorical string column for formula use
    - Collapse model_id into two analytic levels:
        Baseline, VIOLETS
    - Stabilize category order using the RQ2 design categories
    - Preserve flagged-related fields for supplementary summaries
    """
    df = df.copy()

    # ----------------------------------------------------------------------
    # Numeric fields
    # ----------------------------------------------------------------------
    df["safety_score"] = pd.to_numeric(df["safety_score"], errors="coerce")
    df["turn"] = pd.to_numeric(df["turn"], errors="coerce").astype("Int64")

    # Binary-like fields may be present as bool or string; normalize gently
    if "flagged" in df.columns:
        df["flagged"] = df["flagged"].astype(bool)
    if "conversation_flagged" in df.columns:
        df["conversation_flagged"] = df["conversation_flagged"].astype(bool)

    # ----------------------------------------------------------------------
    # Basic text fields
    # ----------------------------------------------------------------------
    for col in [
        "conversation_id",
        "model_id",
        "category",
        "seed_prompt",
        "seed_intent",
        "seed_technique",
        "input",
        "output",
        "label",
        "violation",
        "reasoning",
        "escalation_note",
        "timestamp",
    ]:
        if col in df.columns:
            df[col] = df[col].astype(str)

    # ----------------------------------------------------------------------
    # Remove rows unusable for primary modeling
    # ----------------------------------------------------------------------
    df = df.dropna(subset=["safety_score", "turn"]).copy()

    # ----------------------------------------------------------------------
    # Normalize model labels for analysis
    #
    # The raw JSONL contains:
    #   - "violets"
    #   - baseline model name such as "gpt-4o-mini"
    #
    # For inferential modeling, we only need two analytic groups:
    #   - Baseline
    #   - VIOLETS
    # ----------------------------------------------------------------------
    df["model"] = df["model_id"].replace({"violets": "VIOLETS"})
    df["model"] = np.where(df["model"] == "VIOLETS", "VIOLETS", "Baseline")
    df["model"] = pd.Categorical(
        df["model"], categories=["Baseline", "VIOLETS"], ordered=True
    )

    # ----------------------------------------------------------------------
    # Stable category order based on RQ2 design categories
    # ----------------------------------------------------------------------
    planned_categories = [
        "harmful_content",
        "off_topic_drift",
        "misinformation",
        "sensitive_personal",
        "political_electoral",
    ]
    present = [c for c in planned_categories if c in set(df["category"])]
    extras = [c for c in sorted(df["category"].unique()) if c not in present]
    category_order = present + extras
    df["category"] = pd.Categorical(
        df["category"], categories=category_order, ordered=True
    )

    # ----------------------------------------------------------------------
    # Turn as string categorical for formula coding
    #
    # We model turns as categorical, not numeric trend, because the effect of
    # later turns need not be linear.
    # ----------------------------------------------------------------------
    turn_order = [str(t) for t in sorted(df["turn"].dropna().astype(int).unique())]
    df["turn_str"] = df["turn"].astype(int).astype(str)
    df["turn_str"] = pd.Categorical(df["turn_str"], categories=turn_order, ordered=True)

    # ----------------------------------------------------------------------
    # Stable sorting for readability and reproducibility
    # ----------------------------------------------------------------------
    df = df.sort_values(["conversation_id", "model", "turn"]).reset_index(drop=True)

    return df


# ============================================================================
# Mixed model fitting
# ============================================================================


def fit_mixedlm(formula: str, df: pd.DataFrame, group_col: str):
    """
    Fit a linear mixed-effects model with a random intercept for group_col.

    MixedLM can occasionally be optimizer-sensitive. To make the script more
    robust on real evaluation data, we retry using multiple optimizers.
    """
    model = smf.mixedlm(formula, data=df, groups=df[group_col])

    last_err = None
    for method in ["lbfgs", "powell", "cg", "bfgs"]:
        try:
            result = model.fit(reml=False, method=method, disp=False)
            return result
        except Exception as e:
            last_err = e

    raise RuntimeError(
        f"MixedLM failed for formula:\n{formula}\nLast error: {last_err}"
    )


# ============================================================================
# Fixed-effect coefficient and contrast extraction
# ============================================================================


def coef_and_ci(
    result, coef_name: str, z_crit: float = 1.96
) -> Tuple[float, float, float, float]:
    """
    Extract a single fixed-effect coefficient and compute its 95% Wald CI.

    Returns:
        estimate, ci_low, ci_high, p_value

    Why manual extraction?
    ----------------------
    statsmodels does provide confidence intervals for coefficients. However, we
    compute these explicitly here for two reasons:
      1. we want consistent handling for both coefficients and linear contrasts
      2. category/turn-specific effects are often not single coefficients but
         linear combinations of coefficients
    """
    est = float(result.fe_params[coef_name])
    se = float(result.bse_fe[coef_name])

    lo = est - z_crit * se
    hi = est + z_crit * se

    z = est / se if se > 0 else np.nan
    p = 2 * (1 - normal_cdf(abs(z))) if np.isfinite(z) else np.nan

    return est, lo, hi, p


def contrast_est_ci(
    result, coef_names: List[str], weights: List[float], z_crit: float = 1.96
) -> Tuple[float, float, float, float]:
    """
    Compute a linear contrast of fixed effects and its 95% Wald CI.

    General form:
        estimate = L * beta
        variance = L * Cov(beta) * L'
        se       = sqrt(variance)

    This is how we compute:
      - category-specific model effects
      - turn-specific model effects

    Example
    -------
    If the reference category is 'harmful_content', then the effect of VIOLETS
    in 'misinformation' is:

        beta_model + beta_model:category[T.misinformation]
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


# ============================================================================
# Table builders
# ============================================================================


def build_table1_overall(df: pd.DataFrame) -> Tuple[pd.DataFrame, object]:
    """
    Table 1: Overall model effect.

    Model:
        safety_score ~ model + (1 | conversation_id)

    Quantity of interest:
        effect of VIOLETS relative to Baseline
    """
    result = fit_mixedlm(
        "safety_score ~ C(model, Treatment(reference='Baseline'))",
        df=df,
        group_col="conversation_id",
    )

    coef_name = "C(model, Treatment(reference='Baseline'))[T.VIOLETS]"
    est, lo, hi, p = coef_and_ci(result, coef_name)

    # Standardized effect size (simple reporting convenience)
    # This is not the only possible effect size, but it is a readable
    # scale-free companion for the unstandardized coefficient.
    outcome_sd = float(df["safety_score"].std(ddof=1))
    std_effect = est / outcome_sd if outcome_sd > 0 else np.nan

    table = pd.DataFrame(
        [
            {
                "effect": "Overall model effect",
                "contrast": "VIOLETS - Baseline",
                "estimate": est,
                "ci_low": lo,
                "ci_high": hi,
                "p_value": p,
                "std_effect": std_effect,
                "n_rows": len(df),
                "n_conversations": df["conversation_id"].nunique(),
            }
        ]
    )

    return table, result


def build_table2_category(df: pd.DataFrame) -> Tuple[pd.DataFrame, object]:
    """
    Table 2: Category-specific model effects.

    Model:
        safety_score ~ model * category + (1 | conversation_id)

    Reference category:
        the first level of df["category"]

    Category-specific effect:
        beta_model + beta_model:category_k
    """
    if len(df["category"].cat.categories) == 0:
        raise ValueError("No category levels available.")

    ref_cat = df["category"].cat.categories[0]

    formula = (
        "safety_score ~ "
        "C(model, Treatment(reference='Baseline')) * "
        f"C(category, Treatment(reference='{ref_cat}'))"
    )
    result = fit_mixedlm(formula, df=df, group_col="conversation_id")

    model_coef = "C(model, Treatment(reference='Baseline'))[T.VIOLETS]"
    outcome_sd = float(df["safety_score"].std(ddof=1))

    rows = []
    for cat in df["category"].cat.categories:
        if cat == ref_cat:
            est, lo, hi, p = coef_and_ci(result, model_coef)
        else:
            interaction_name = (
                f"C(model, Treatment(reference='Baseline'))[T.VIOLETS]:"
                f"C(category, Treatment(reference='{ref_cat}'))[T.{cat}]"
            )
            est, lo, hi, p = contrast_est_ci(
                result,
                coef_names=[model_coef, interaction_name],
                weights=[1.0, 1.0],
            )

        std_effect = est / outcome_sd if outcome_sd > 0 else np.nan

        rows.append(
            {
                "category": cat,
                "contrast": "VIOLETS - Baseline",
                "estimate": est,
                "ci_low": lo,
                "ci_high": hi,
                "p_value": p,
                "std_effect": std_effect,
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
    Table 3: Turn-specific model effects.

    Model:
        safety_score ~ model * turn + (1 | conversation_id)

    Reference turn:
        the first level of df["turn_str"]

    Turn-specific effect:
        beta_model + beta_model:turn_t
    """
    if len(df["turn_str"].cat.categories) == 0:
        raise ValueError("No turn levels available.")

    ref_turn = df["turn_str"].cat.categories[0]

    formula = (
        "safety_score ~ "
        "C(model, Treatment(reference='Baseline')) * "
        f"C(turn_str, Treatment(reference='{ref_turn}'))"
    )
    result = fit_mixedlm(formula, df=df, group_col="conversation_id")

    model_coef = "C(model, Treatment(reference='Baseline'))[T.VIOLETS]"
    outcome_sd = float(df["safety_score"].std(ddof=1))

    rows = []
    for turn in df["turn_str"].cat.categories:
        if turn == ref_turn:
            est, lo, hi, p = coef_and_ci(result, model_coef)
        else:
            interaction_name = (
                f"C(model, Treatment(reference='Baseline'))[T.VIOLETS]:"
                f"C(turn_str, Treatment(reference='{ref_turn}'))[T.{turn}]"
            )
            est, lo, hi, p = contrast_est_ci(
                result,
                coef_names=[model_coef, interaction_name],
                weights=[1.0, 1.0],
            )

        std_effect = est / outcome_sd if outcome_sd > 0 else np.nan

        rows.append(
            {
                "turn": turn,
                "contrast": "VIOLETS - Baseline",
                "estimate": est,
                "ci_low": lo,
                "ci_high": hi,
                "p_value": p,
                "std_effect": std_effect,
                "n_rows": int((df["turn_str"] == turn).sum()),
                "n_conversations": int(
                    df.loc[df["turn_str"] == turn, "conversation_id"].nunique()
                ),
            }
        )

    table = pd.DataFrame(rows)
    return table, result


# ============================================================================
# Supplementary summaries (descriptive only)
# ============================================================================


def simple_model_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Descriptive summary of safety_score by model.

    This is supplementary and should not be interpreted as the primary
    inferential result.
    """
    return (
        df.groupby("model", observed=True)["safety_score"]
        .agg(["count", "mean", "std", "median", "min", "max"])
        .reset_index()
    )


def simple_category_model_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Descriptive safety_score summary by category and model."""
    return (
        df.groupby(["category", "model"], observed=True)["safety_score"]
        .agg(["count", "mean", "std", "median"])
        .reset_index()
    )


def simple_turn_model_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Descriptive safety_score summary by turn and model."""
    return (
        df.groupby(["turn_str", "model"], observed=True)["safety_score"]
        .agg(["count", "mean", "std", "median"])
        .reset_index()
    )


def flagged_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Supplementary turn-level flagged rate summary.

    Important:
    ----------
    `flagged` is a threshold-derived operational metric, not the primary
    inferential outcome. We include it only as a descriptive supplement.
    """
    if "flagged" not in df.columns:
        return pd.DataFrame()

    out = (
        df.groupby("model", observed=True)["flagged"]
        .agg(["count", "mean"])
        .reset_index()
        .rename(columns={"mean": "flagged_rate"})
    )
    return out


def conversation_flagged_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Supplementary conversation-level flagged summary.

    We collapse to one row per conversation_id x model before averaging
    conversation_flagged.
    """
    if "conversation_flagged" not in df.columns:
        return pd.DataFrame()

    conv = df.groupby(["conversation_id", "model"], observed=True, as_index=False).agg(
        conversation_flagged=("conversation_flagged", "max")
    )

    out = (
        conv.groupby("model", observed=True)["conversation_flagged"]
        .agg(["count", "mean"])
        .reset_index()
        .rename(
            columns={"count": "n_conversations", "mean": "conversation_flagged_rate"}
        )
    )
    return out


def build_passfail_table(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    """
    PASS/FAIL/WARN counts and rates by model × group_col (category or turn_str).

    This is the primary pass/fail rubric output called for in the RQ2 plan.
    """
    if "label" not in df.columns:
        return pd.DataFrame()

    rows = []
    for (grp_val, model), sub in df.groupby([group_col, "model"], observed=True):
        n_total = len(sub)
        n_pass = int((sub["label"] == "PASS").sum())
        n_warn = int((sub["label"] == "WARN").sum())
        n_fail = int((sub["label"] == "FAIL").sum())
        rows.append(
            {
                group_col: grp_val,
                "model": model,
                "n_total": n_total,
                "n_pass": n_pass,
                "n_warn": n_warn,
                "n_fail": n_fail,
                "pass_rate": round(n_pass / n_total, 4) if n_total else None,
                "warn_rate": round(n_warn / n_total, 4) if n_total else None,
                "fail_rate": round(n_fail / n_total, 4) if n_total else None,
            }
        )
    return pd.DataFrame(rows)


def build_violation_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Breakdown of violation strings for FAIL-labelled VIOLETS turns.

    Helps characterize what kinds of failures occur (analogous to RQ1's
    error-type classification).
    """
    if "label" not in df.columns or "violation" not in df.columns:
        return pd.DataFrame()

    fails = df[(df["model"] == "VIOLETS") & (df["label"] == "FAIL")].copy()
    if fails.empty:
        return fails

    return fails[
        [
            "conversation_id",
            "category",
            "turn",
            "safety_score",
            "violation",
            "input",
            "output",
            "reasoning",
        ]
    ].reset_index(drop=True)


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


_CAT_LABELS_RQ2 = {
    "harmful_content": "harmful\ncontent",
    "off_topic_drift": "off-topic\ndrift",
    "misinformation": "misinfo",
    "sensitive_personal": "sensitive\npersonal",
    "political_electoral": "political /\nelectoral",
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
    outcome: str = "safety_score",
    ylabel: str = "Mean safety score",
    ymin: float = 0.5,
    ymax: float = 1.0,
    title: str = "RQ2: Safety Scores — VIOLETS vs. Baseline",
) -> None:
    """
    Three-panel grouped bar chart optimized for poster display:
      (A) Overall  (B) By Category  (C) By Turn

    - Significance asterisks drawn inside each panel above bar pairs
    - Legend shown only in Panel A
    - Category labels shortened via _CAT_LABELS_RQ2
    - y-axis starts at ymin (default 0.5) so differences are visually legible
    """
    COLORS = {"Baseline": "#6baed6", "VIOLETS": "#fd8d3c"}
    FS = {"title": 15, "label": 13, "tick": 11, "legend": 11, "stars": 13}
    BAR_W = 0.38
    CAP = 5
    ERR_KW = {"elinewidth": 1.6, "ecolor": "#333333"}
    STAR_PAD = (ymax - ymin) * 0.03

    overall_desc = _desc_stats(df, "model", outcome).set_index("model")
    cat_desc = _desc_stats(df, "category", outcome)
    turn_desc = _desc_stats(df, "turn_str", outcome)

    fig, axes = plt.subplots(
        1, 3, figsize=(16, 5.5), gridspec_kw={"width_ratios": [1, 2.2, 1.6]}
    )
    fig.subplots_adjust(wspace=0.35)

    def _annotate_stars(ax, x_center, top_y, stars):
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
        ci_hi_by_group = {}

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
            for i, ci_top in enumerate(ci_hi):
                ci_hi_by_group[i] = max(ci_hi_by_group.get(i, ymin), ci_top)

        p_lookup = (
            p_table.set_index(p_col)["p_value"] if p_col in p_table.columns else {}
        )
        for i, val in enumerate(index_vals):
            p = p_lookup.get(val, np.nan) if hasattr(p_lookup, "get") else np.nan
            _annotate_stars(
                ax, x[i] + BAR_W / 2, ci_hi_by_group.get(i, ymin), _sig_stars(p)
            )

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
    cat_labels = [_CAT_LABELS_RQ2.get(c, c) for c in categories]
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
    axes[1].set_title("(B) By Threat Category", fontsize=FS["title"], fontweight="bold")

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
# Main analysis routine
# ============================================================================


def run_analysis(input_path: Path, output_dir: Path) -> None:
    """
    Full RQ2 analysis pipeline:
      1. load and preprocess JSONL
      2. fit primary mixed models
      3. export inferential tables
      4. export supplementary summaries
      5. build one combined figure
    """
    ensure_dir(output_dir)

    required_columns = [
        "conversation_id",
        "model_id",
        "category",
        "turn",
        "safety_score",
    ]

    df = load_jsonl(input_path)
    validate_columns(df, required_columns)
    df = preprocess(df)

    # ----------------------------------------------------------------------
    # Primary inferential tables
    # ----------------------------------------------------------------------
    table1, m1 = build_table1_overall(df)
    table2, m2 = build_table2_category(df)
    table3, m3 = build_table3_turn(df)

    table1.to_csv(output_dir / "table1_model_overall.csv", index=False)
    table2.to_csv(output_dir / "table2_category_effects.csv", index=False)
    table3.to_csv(output_dir / "table3_turn_effects.csv", index=False)

    # ----------------------------------------------------------------------
    # Supplementary summaries
    # ----------------------------------------------------------------------
    simple_model_summary(df).to_csv(
        output_dir / "model_overall_summary.csv", index=False
    )
    simple_category_model_summary(df).to_csv(
        output_dir / "category_model_summary.csv", index=False
    )
    simple_turn_model_summary(df).to_csv(
        output_dir / "turn_model_summary.csv", index=False
    )

    fs = flagged_summary(df)
    if not fs.empty:
        fs.to_csv(output_dir / "flagged_summary.csv", index=False)

    cfs = conversation_flagged_summary(df)
    if not cfs.empty:
        cfs.to_csv(output_dir / "conversation_flagged_summary.csv", index=False)

    # ----------------------------------------------------------------------
    # Pass/fail rate tables (primary rubric per RQ2 plan)
    # ----------------------------------------------------------------------
    pf_cat = build_passfail_table(df, group_col="category")
    if not pf_cat.empty:
        pf_cat.to_csv(output_dir / "passfail_by_category.csv", index=False)

    pf_turn = build_passfail_table(df, group_col="turn_str")
    if not pf_turn.empty:
        pf_turn.to_csv(output_dir / "passfail_by_turn.csv", index=False)

    # Violation breakdown for VIOLETS FAILs
    viol = build_violation_table(df)
    if not viol.empty:
        viol.to_csv(output_dir / "violation_breakdown.csv", index=False)
        print(f"Recorded {len(viol)} VIOLETS FAIL turn(s) → violation_breakdown.csv")

    # ----------------------------------------------------------------------
    # Save statsmodels summaries for full inspection / appendix use
    # ----------------------------------------------------------------------
    (output_dir / "model_overall_mixedlm_summary.txt").write_text(
        str(m1.summary()), encoding="utf-8"
    )
    (output_dir / "category_mixedlm_summary.txt").write_text(
        str(m2.summary()), encoding="utf-8"
    )
    (output_dir / "turn_mixedlm_summary.txt").write_text(
        str(m3.summary()), encoding="utf-8"
    )

    # ----------------------------------------------------------------------
    # Primary figure
    # ----------------------------------------------------------------------
    build_combined_figure(
        df=df,
        table1=table1,
        table2=table2,
        table3=table3,
        output_path=output_dir / "rq2_mixed_effects_combined_figure.png",
    )

    print("RQ2 mixed-effects analysis complete.")
    print(f"Input:      {input_path}")
    print(f"Output dir: {output_dir}")


# ============================================================================
# CLI
# ============================================================================


def parse_args():
    parser = argparse.ArgumentParser(
        description="RQ2 mixed-effects analysis for VIOLETS safety evaluation."
    )
    parser.add_argument(
        "--input",
        type=str,
        default="output/rq2/eval_dataset.jsonl",
        help="Path to the RQ2 JSONL file.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output/rq2/analysis_mixed",
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
