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

import math

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
    return 0.5 * (1.0 + math.erf(x / np.sqrt(2.0)))


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


def build_table1_overall(df: pd.DataFrame, outcome: str = "safety_score") -> Tuple[pd.DataFrame, object]:
    """
    Table 1: Overall model effect.

    Model:
        {outcome} ~ model + (1 | conversation_id)

    Quantity of interest:
        effect of VIOLETS relative to Baseline

    `outcome` defaults to the primary judge's safety_score column, but can be
    pointed at a secondary judge's column (e.g. "safety_score_2nd") to fit
    the identical model on an independent judge's scores for comparison.
    """
    result = fit_mixedlm(
        f"{outcome} ~ C(model, Treatment(reference='Baseline'))",
        df=df,
        group_col="conversation_id",
    )

    coef_name = "C(model, Treatment(reference='Baseline'))[T.VIOLETS]"
    est, lo, hi, p = coef_and_ci(result, coef_name)

    # Standardized effect size (simple reporting convenience)
    # This is not the only possible effect size, but it is a readable
    # scale-free companion for the unstandardized coefficient.
    outcome_sd = float(df[outcome].std(ddof=1))
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


def build_table2_category(df: pd.DataFrame, outcome: str = "safety_score") -> Tuple[pd.DataFrame, object]:
    """
    Table 2: Category-specific model effects.

    Model:
        {outcome} ~ model * category + (1 | conversation_id)

    Reference category:
        the first level of df["category"]

    Category-specific effect:
        beta_model + beta_model:category_k

    `outcome` defaults to the primary judge's safety_score column, but can be
    pointed at a secondary judge's column (e.g. "safety_score_2nd") to fit
    the identical model on an independent judge's scores for comparison.
    """
    if len(df["category"].cat.categories) == 0:
        raise ValueError("No category levels available.")

    ref_cat = df["category"].cat.categories[0]

    formula = (
        f"{outcome} ~ "
        "C(model, Treatment(reference='Baseline')) * "
        f"C(category, Treatment(reference='{ref_cat}'))"
    )
    result = fit_mixedlm(formula, df=df, group_col="conversation_id")

    model_coef = "C(model, Treatment(reference='Baseline'))[T.VIOLETS]"
    outcome_sd = float(df[outcome].std(ddof=1))

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


def build_coefficient_figure(
    df: pd.DataFrame,
    table1: pd.DataFrame,
    table2: pd.DataFrame,
    output_path: Path,
    xlabel: str = "VIOLETS − Baseline (safety score)",
    title: str = "RQ2: Safety — VIOLETS vs. Baseline (estimated effect)",
) -> None:
    """
    Two-panel coefficient (forest) plot:
      (A) Overall effect   (B) Effect by Threat Category

    Each point is the VIOLETS-minus-Baseline estimate from the mixed-effects
    model (table1/table2), with a 95% CI whisker; the dashed vertical line
    at 0 marks "no difference". This plots exactly the quantity the
    significance test is about, so there is only one interval to read per
    row — unlike a grouped bar chart with one CI per model, where two
    overlapping per-group CIs can visually look non-significant even when
    the (correctly, paired/clustered) tested difference is significant.
    """
    COLOR = "#7B2FBE"
    FS = {"title": 20, "label": 16, "tick": 15, "stars": 15}
    CAP = 5
    ERR_KW = {"elinewidth": 2.0, "ecolor": COLOR, "capthick": 2.0}

    categories = df["category"].cat.categories.tolist()
    cat_labels = [_CAT_LABELS_RQ2.get(c, c).replace("\n", " ") for c in categories]
    cat2 = table2.set_index("category").loc[categories]

    fig, axes = plt.subplots(
        2, 1, figsize=(10, 7.5), gridspec_kw={"height_ratios": [1, len(categories)]}
    )
    fig.subplots_adjust(hspace=0.55, left=0.28)

    def _forest_panel(ax, labels, est, lo, hi, p_values):
        y = np.arange(len(labels))
        ax.axvline(0, color="#999999", linewidth=1.2, linestyle="--", zorder=1)
        ax.errorbar(
            est, y,
            xerr=[np.array(est) - np.array(lo), np.array(hi) - np.array(est)],
            fmt="o", color=COLOR, markersize=8, capsize=CAP, **ERR_KW, zorder=3,
        )
        for yi, hi_i, p in zip(y, hi, p_values):
            stars = _sig_stars(p)
            if stars != "ns":
                ax.text(hi_i, yi, f"  {stars}", va="center", ha="left",
                         fontsize=FS["stars"], color="#222222", fontweight="bold")
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=FS["tick"])
        ax.invert_yaxis()
        ax.set_xlabel(xlabel, fontsize=FS["label"])
        ax.xaxis.grid(True, linestyle="--", linewidth=0.6, alpha=0.6)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)

    # ── Panel A: Overall ──────────────────────────────────────────────────
    row = table1.iloc[0]
    _forest_panel(
        axes[0], ["Overall"], [row["estimate"]], [row["ci_low"]], [row["ci_high"]],
        [row["p_value"]],
    )
    axes[0].set_title("(A) Overall", fontsize=FS["title"], fontweight="bold", loc="left")

    # ── Panel B: By Category ──────────────────────────────────────────────
    _forest_panel(
        axes[1], cat_labels, cat2["estimate"].tolist(), cat2["ci_low"].tolist(),
        cat2["ci_high"].tolist(), cat2["p_value"].tolist(),
    )
    axes[1].set_title(
        "(B) By Threat Category", fontsize=FS["title"], fontweight="bold", loc="left"
    )

    fig.suptitle(title, fontsize=20, fontweight="bold", y=1.02)
    fig.text(
        0.5, -0.02,
        "Points = mixed-model estimate of VIOLETS − Baseline, whiskers = 95% CI  |  "
        "dashed line = no difference  |  * p < .05  ** p < .01  *** p < .001",
        ha="center", fontsize=12, color="#555555",
    )

    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def build_multi_judge_coefficient_figure(
    df: pd.DataFrame,
    judges: list[dict],
    output_path: Path,
    xlabel: str = "VIOLETS − Baseline (safety score)",
    title: str = "RQ2: Safety — VIOLETS vs. Baseline, by judge model",
) -> None:
    """
    Same two-panel coefficient (forest) plot as build_coefficient_figure, but
    overlays one point+CI series per judge model instead of collapsing them
    into a single average or majority vote. Disagreement between judges is
    then visible directly (differently-colored points at different
    positions, or on different sides of the zero line) rather than hidden
    inside an aggregate.

    `judges` is a list of dicts, each:
        {"label": str, "color": str, "table1": DataFrame, "table2": DataFrame}
    where table1/table2 are the outputs of build_table1_overall /
    build_table2_category run with that judge's outcome column (e.g.
    safety_score vs. safety_score_2nd).
    """
    FS = {"title": 20, "label": 16, "tick": 15, "legend": 13, "stars": 13}
    CAP = 5
    n_judges = len(judges)
    # Small vertical offsets so overlapping judges' points/whiskers don't
    # sit exactly on top of each other.
    offsets = np.linspace(-0.16, 0.16, n_judges) if n_judges > 1 else [0.0]

    categories = df["category"].cat.categories.tolist()
    cat_labels = [_CAT_LABELS_RQ2.get(c, c).replace("\n", " ") for c in categories]

    fig, axes = plt.subplots(
        2, 1, figsize=(10, 7.5), gridspec_kw={"height_ratios": [1, len(categories)]}
    )
    fig.subplots_adjust(hspace=0.55, left=0.28)

    def _forest_panel(ax, labels, judge_rows_list):
        y_base = np.arange(len(labels))
        ax.axvline(0, color="#999999", linewidth=1.2, linestyle="--", zorder=1)
        for judge, offset, rows in zip(judges, offsets, judge_rows_list):
            y = y_base + offset
            est = rows["estimate"].tolist()
            lo = rows["ci_low"].tolist()
            hi = rows["ci_high"].tolist()
            ax.errorbar(
                est, y,
                xerr=[np.array(est) - np.array(lo), np.array(hi) - np.array(est)],
                fmt="o", color=judge["color"], markersize=7, capsize=CAP,
                elinewidth=2.0, ecolor=judge["color"], capthick=2.0, zorder=3,
                label=judge["label"],
            )
            for yi, hi_i, p in zip(y, hi, rows["p_value"].tolist()):
                stars = _sig_stars(p)
                if stars != "ns":
                    ax.text(hi_i, yi, f"  {stars}", va="center", ha="left",
                             fontsize=FS["stars"], color=judge["color"], fontweight="bold")
        ax.set_yticks(y_base)
        ax.set_yticklabels(labels, fontsize=FS["tick"])
        ax.invert_yaxis()
        ax.set_xlabel(xlabel, fontsize=FS["label"])
        ax.xaxis.grid(True, linestyle="--", linewidth=0.6, alpha=0.6)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)

    # ── Panel A: Overall ──────────────────────────────────────────────────
    _forest_panel(axes[0], ["Overall"], [j["table1"] for j in judges])
    axes[0].set_title("(A) Overall", fontsize=FS["title"], fontweight="bold", loc="left")

    # ── Panel B: By Category ──────────────────────────────────────────────
    judge_cat_rows = [j["table2"].set_index("category").loc[categories] for j in judges]
    _forest_panel(axes[1], cat_labels, judge_cat_rows)
    axes[1].set_title(
        "(B) By Threat Category", fontsize=FS["title"], fontweight="bold", loc="left"
    )

    # Figure-level legend (not tied to either axes) so it can't overlap the
    # title text or the Panel A data points.
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=FS["legend"], framealpha=0.9,
               loc="upper center", bbox_to_anchor=(0.5, 1.06), ncol=n_judges)

    fig.suptitle(title, fontsize=20, fontweight="bold", y=1.14)
    fig.text(
        0.5, -0.02,
        "Points = mixed-model estimate of VIOLETS − Baseline per judge model, whiskers = 95% CI  |  "
        "dashed line = no difference  |  * p < .05  ** p < .01  *** p < .001",
        ha="center", fontsize=12, color="#555555",
    )

    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def build_passfail_figure(
    df: pd.DataFrame,
    passfail_by_category: pd.DataFrame,
    output_path: Path,
    title: str = "RQ2: PASS / WARN / FAIL Rate by Threat Category",
) -> None:
    """
    Supplementary stacked-bar panel: PASS/WARN/FAIL proportions per model
    within each threat category. safety_score is a continuous number but it
    is built from fixed categorical anchors (PASS/WARN/FAIL); a mean-based
    coefficient plot can hide a genuinely alarming rate (e.g. a 75% FAIL
    rate can still average out to a middling mean), so this panel shows the
    underlying label distribution directly.
    """
    COLORS = {"PASS": "#4CAF50", "WARN": "#FFB300", "FAIL": "#D32F2F"}
    HATCH = {"Baseline": "////", "VIOLETS": None}
    FS = {"title": 18, "label": 15, "tick": 14, "legend": 12}
    BAR_W = 0.38

    categories = df["category"].cat.categories.tolist()
    cat_labels = [_CAT_LABELS_RQ2.get(c, c).replace("\n", " ") for c in categories]
    pf = passfail_by_category.set_index(["category", "model"])

    fig, ax = plt.subplots(figsize=(11, 5.5))
    x = np.arange(len(categories))

    for j, model in enumerate(["Baseline", "VIOLETS"]):
        bottoms = np.zeros(len(categories))
        for label in ["PASS", "WARN", "FAIL"]:
            rates = np.array([
                pf.loc[(c, model), f"{label.lower()}_rate"]
                if (c, model) in pf.index else np.nan
                for c in categories
            ])
            ax.bar(
                x + j * BAR_W, rates, BAR_W, bottom=bottoms,
                color=COLORS[label], label=label if j == 0 else None,
                hatch=HATCH[model], edgecolor="white", linewidth=0.5,
            )
            bottoms += np.nan_to_num(rates)

    # Proxy legend entries for the Baseline/VIOLETS hatch pattern, shown
    # alongside the PASS/WARN/FAIL color legend.
    model_handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor="white", edgecolor="black",
                       hatch=HATCH["Baseline"], label="Baseline (hatched)"),
        plt.Rectangle((0, 0), 1, 1, facecolor="white", edgecolor="black",
                       label="VIOLETS (solid)"),
    ]

    ax.set_xticks(x + BAR_W / 2)
    ax.set_xticklabels(cat_labels, fontsize=FS["tick"])
    ax.set_ylabel("Share of turns", fontsize=FS["label"])
    ax.set_ylim(0, 1.0)
    ax.yaxis.grid(True, linestyle="--", linewidth=0.6, alpha=0.6)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    color_handles, color_labels = ax.get_legend_handles_labels()
    ax.legend(
        handles=color_handles + model_handles,
        fontsize=FS["legend"], framealpha=0.9, loc="upper center",
        bbox_to_anchor=(0.5, 1.16), ncol=5,
    )
    ax.set_title(title, fontsize=FS["title"], fontweight="bold", y=1.14)
    fig.subplots_adjust(bottom=0.15)

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
    # Primary figure: coefficient plot (VIOLETS − Baseline, with 95% CI)
    # ----------------------------------------------------------------------
    build_coefficient_figure(
        df=df,
        table1=table1,
        table2=table2,
        output_path=output_dir / "rq2_poster_figure.png",
    )

    # ----------------------------------------------------------------------
    # Supplementary figure: PASS/WARN/FAIL rate by category
    # ----------------------------------------------------------------------
    if not pf_cat.empty:
        build_passfail_figure(
            df=df,
            passfail_by_category=pf_cat,
            output_path=output_dir / "rq2_passfail_figure.png",
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
