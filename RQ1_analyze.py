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
- url_citation_summary.csv        (RQ1.2 — VIOLETS-only, no baseline citations to compare against)
- url_citation_by_category.csv    (RQ1.2, per category)
- url_citations_flagged_for_review.csv  (RQ1.2, cited URLs whose content didn't
  clearly support the claim — only written if the input JSONL has url_* columns,
  i.e. it was produced after the RQ1.2 grounding-pipeline fix)

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

from url_validity_judge import compute_url_aggregate_stats

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
# URL citation quality (RQ1.2 — Grounding and Citation Reliability)
# ============================================================================
#
# Baseline never produces RAG-style citations, so these are VIOLETS-only
# descriptive summaries — not a VIOLETS-vs-Baseline mixed-effects comparison
# like the tables above. They reuse compute_url_aggregate_stats() from
# url_validity_judge.py so the aggregation logic matches exactly what the
# runner already computes per-conversation, instead of reimplementing it.


def _violets_url_turns(df: pd.DataFrame) -> list[dict]:
    """Reshape VIOLETS rows into the {citation_rate_score, url_details} shape
    compute_url_aggregate_stats() expects (matching the raw judge output keys,
    not the url_-prefixed column names dataset_writer.py stores them under)."""
    violets = df[df["model"] == "VIOLETS"]
    return [
        {
            "citation_rate_score": row.get("url_citation_rate_score"),
            "url_details": row.get("url_details") or [],
        }
        for row in violets.to_dict("records")
    ]


def build_url_citation_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Overall pct_cited / pct_accessible / pct_accurate across all VIOLETS turns."""
    stats = compute_url_aggregate_stats(_violets_url_turns(df))
    return pd.DataFrame([stats])


def build_url_citation_by_category(df: pd.DataFrame) -> pd.DataFrame:
    """Same aggregate stats, broken out per question category."""
    rows = []
    violets = df[df["model"] == "VIOLETS"]
    for cat in df["category"].cat.categories:
        grp = violets[violets["category"] == cat]
        if grp.empty:
            continue
        stats = compute_url_aggregate_stats(_violets_url_turns(grp))
        rows.append({"category": cat, **stats})
    return pd.DataFrame(rows)


def build_url_flagged_table(
    df: pd.DataFrame,
    accuracy_threshold: int = 60,
    accessibility_threshold: int = 50,
) -> pd.DataFrame:
    """
    Staged review table for VIOLETS's citation quality, in three tiers so a
    reviewer can triage worst-first rather than one flat accuracy filter
    (which silently missed unreachable URLs that happened to still get an
    accuracy >= threshold, e.g. a 404'd PDF scored accuracy=60 because the
    judge couldn't verify it either way):

      1. no_citation  — turn cited no URL at all (citation_rate_score == 0)
      2. inaccessible — a cited URL couldn't be reached/loaded at all
                         (accessibility < accessibility_threshold), regardless
                         of what accuracy it was given
      3. inaccurate    — a cited URL was reachable but its content didn't
                         clearly support the claim (accuracy < accuracy_threshold)

    Stages are mutually exclusive and ordered worst-first: an inaccessible
    URL is never also listed under "inaccurate".
    """
    violets = df[df["model"] == "VIOLETS"]
    rows = []

    for _, r in violets.iterrows():
        if r.get("url_citation_rate_score") == 0:
            rows.append(
                {
                    "stage": "no_citation",
                    "conversation_id": r["conversation_id"],
                    "category": r["category"],
                    "turn": r["turn"],
                    "url": None,
                    "accessibility": None,
                    "accuracy": None,
                    "note": "No URL cited in this response.",
                }
            )

    for _, r in violets.iterrows():
        for d in (r.get("url_details") or []):
            access = d.get("accessibility")
            acc = d.get("accuracy")
            if access is not None and access < accessibility_threshold:
                stage = "inaccessible"
            elif acc is not None and acc < accuracy_threshold:
                stage = "inaccurate"
            else:
                continue
            rows.append(
                {
                    "stage": stage,
                    "conversation_id": r["conversation_id"],
                    "category": r["category"],
                    "turn": r["turn"],
                    "url": d.get("url"),
                    "accessibility": access,
                    "accuracy": acc,
                    "note": d.get("note", ""),
                }
            )

    result = pd.DataFrame(rows)
    if result.empty:
        return result
    stage_order = {"no_citation": 0, "inaccessible": 1, "inaccurate": 2}
    result["_stage_order"] = result["stage"].map(stage_order)
    result = result.sort_values(["_stage_order", "conversation_id", "turn"]).drop(columns="_stage_order")
    return result.reset_index(drop=True)


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


def build_table1_overall(df: pd.DataFrame, outcome: str = "veracity_score") -> Tuple[pd.DataFrame, object]:
    """
    Table 1: Overall model effect from:
        {outcome} ~ model + (1 | conversation_id)

    `outcome` defaults to the primary judge's veracity_score column, but can
    be pointed at a secondary judge's column (e.g. "veracity_score_2nd") to
    fit the identical model for comparison.
    """
    result = fit_mixedlm(
        f"{outcome} ~ C(model, Treatment(reference='Baseline'))",
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


def build_table2_category(df: pd.DataFrame, outcome: str = "veracity_score") -> Tuple[pd.DataFrame, object]:
    """
    Table 2: Model effect within each category from:
        {outcome} ~ model * category + (1 | conversation_id)

    `outcome` defaults to the primary judge's veracity_score column, but can
    be pointed at a secondary judge's column (e.g. "veracity_score_2nd") to
    fit the identical model for comparison.
    """
    if len(df["category"].cat.categories) == 0:
        raise ValueError("No category levels found.")

    ref_cat = df["category"].cat.categories[0]

    formula = (
        f"{outcome} ~ "
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


def build_coefficient_figure(
    df: pd.DataFrame,
    table1: pd.DataFrame,
    table2: pd.DataFrame,
    output_path: Path,
    xlabel: str = "VIOLETS − Baseline (veracity points)",
    title: str = "RQ1: Veracity — VIOLETS vs. Baseline (estimated effect)",
) -> None:
    """
    Two-panel coefficient (forest) plot:
      (A) Overall effect   (B) Effect by Question Category

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
    cat_labels = [_CAT_LABELS_RQ1.get(c, c).replace("\n", " ") for c in categories]
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
        "(B) By Question Category", fontsize=FS["title"], fontweight="bold", loc="left"
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
    xlabel: str = "VIOLETS − Baseline (veracity points)",
    title: str = "RQ1: Veracity — VIOLETS vs. Baseline, by judge model",
    poster: bool = False,
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
    veracity_score vs. veracity_score_2nd).

    `poster=True` renders at a fixed 8-inch width for a 3-panels-across
    poster row: no suptitle/caption (those go in the poster's own text
    boxes instead), larger content fonts, no redundant xlabel on panel A,
    a shared x-axis range across both panels, and a heavier zero-line.
    """
    n_judges = len(judges)
    if poster:
        FS = {"title": 22, "subtitle": 17, "label": 20, "tick": 18, "legend": 16, "stars": 19}
        CAP, MARKER, LW = 8, 12, 3.0
        figsize = (8, 7.2)
        left_margin = 0.32
        top_margin, bottom_margin = 0.85, 0.13
        hspace = 0.48
        zero_lw, zero_dashes = 2.2, (6, 5)
        offset_span = 0.12
    else:
        FS = {"title": 20, "label": 16, "tick": 15, "legend": 13, "stars": 13}
        CAP, MARKER, LW = 5, 7, 2.0
        figsize = (10, 7.5)
        left_margin = 0.28
        top_margin, bottom_margin = None, None
        hspace = 0.55
        zero_lw, zero_dashes = 1.2, (4, 3)
        offset_span = 0.16
    offsets = np.linspace(-offset_span, offset_span, n_judges) if n_judges > 1 else [0.0]

    categories = df["category"].cat.categories.tolist()
    cat_labels = [_CAT_LABELS_RQ1.get(c, c).replace("\n", " ") for c in categories]

    # Panel A holds n_judges offset points on a single row, same as any one
    # row of panel B — poster mode drops panel A's title (see below), and
    # without a bigger height_ratio that freed space would just become
    # blank margin instead of actually giving panel A's whiskers more room.
    panel_a_ratio = 2 if poster else 1
    fig, axes = plt.subplots(
        2, 1, figsize=figsize, gridspec_kw={"height_ratios": [panel_a_ratio, len(categories)]}
    )
    adjust_kwargs = {"hspace": hspace, "left": left_margin}
    if poster:
        adjust_kwargs.update(top=top_margin, bottom=bottom_margin)
    fig.subplots_adjust(**adjust_kwargs)

    # Shared x-axis range across both panels (poster only) so the same
    # veracity-point scale reads directly across (A) and (B) — computed
    # from every judge's CI bounds in both tables, with a small pad so no
    # whisker or star annotation touches the very edge.
    shared_xlim = None
    if poster:
        bounds = []
        for j in judges:
            bounds += j["table1"]["ci_low"].tolist() + j["table1"]["ci_high"].tolist()
            bounds += j["table2"]["ci_low"].tolist() + j["table2"]["ci_high"].tolist()
        xmin, xmax = min(bounds), max(bounds)
        pad = (xmax - xmin) * 0.12
        shared_xlim = (xmin - pad, xmax + pad)

    def _forest_panel(ax, labels, judge_rows_list, show_xlabel):
        y_base = np.arange(len(labels))
        ax.axvline(0, color="#666666", linewidth=zero_lw, dashes=zero_dashes, zorder=1)
        for judge, offset, rows in zip(judges, offsets, judge_rows_list):
            y = y_base + offset
            est = rows["estimate"].tolist()
            lo = rows["ci_low"].tolist()
            hi = rows["ci_high"].tolist()
            p_values = rows["p_value"].tolist()
            if poster:
                # Non-significant points/whiskers fade out so the
                # significant ones read at a glance without needing to
                # check every star annotation individually.
                for est_i, lo_i, hi_i, y_i, p in zip(est, lo, hi, y, p_values):
                    faded = _sig_stars(p) == "ns"
                    ax.errorbar(
                        [est_i], [y_i],
                        xerr=[[est_i - lo_i], [hi_i - est_i]],
                        fmt="o", color=judge["color"], markersize=MARKER, capsize=CAP,
                        elinewidth=LW, ecolor=judge["color"], capthick=LW, zorder=3,
                        alpha=0.35 if faded else 1.0,
                    )
                # Invisible full-series artist purely to give the legend one
                # correctly-labeled, full-opacity handle per judge.
                ax.errorbar([], [], fmt="o", color=judge["color"], markersize=MARKER,
                            elinewidth=LW, ecolor=judge["color"], capthick=LW,
                            label=judge["label"])
            else:
                ax.errorbar(
                    est, y,
                    xerr=[np.array(est) - np.array(lo), np.array(hi) - np.array(est)],
                    fmt="o", color=judge["color"], markersize=MARKER, capsize=CAP,
                    elinewidth=LW, ecolor=judge["color"], capthick=LW, zorder=3,
                    label=judge["label"],
                )
            for yi, hi_i, p in zip(y, hi, p_values):
                stars = _sig_stars(p)
                if stars != "ns":
                    ax.text(hi_i, yi, f"  {stars}", va="center", ha="left",
                             fontsize=FS["stars"], color=judge["color"], fontweight="bold")
        ax.set_yticks(y_base)
        ax.set_yticklabels(labels, fontsize=FS["tick"])
        ax.invert_yaxis()
        if show_xlabel:
            ax.set_xlabel(xlabel, fontsize=FS["label"])
        if poster:
            ax.tick_params(axis="x", labelsize=FS["tick"])
            ax.set_xlim(shared_xlim)
            # Explicit margin instead of relying on autoscale's default 5%:
            # panel A's y-range is only ~2*offset_span wide (a single row of
            # offset points), so a data-range-relative margin is too thin in
            # absolute terms to keep the outermost marker/whisker cap from
            # visually touching the panel edge — an absolute pad fixes that
            # regardless of how narrow the panel's own data range is.
            ax.set_ylim(y_base.max() + 0.4, y_base.min() - 0.4)
        ax.xaxis.grid(True, linestyle="--", linewidth=0.6, alpha=0.6)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)

    # ── Panel A: Overall ──────────────────────────────────────────────────
    _forest_panel(axes[0], ["Overall"], [j["table1"] for j in judges], show_xlabel=not poster)
    if not poster:
        axes[0].set_title("(A) Overall", fontsize=FS["title"], fontweight="bold", loc="left")

    # ── Panel B: By Category ──────────────────────────────────────────────
    judge_cat_rows = [j["table2"].set_index("category").loc[categories] for j in judges]
    _forest_panel(axes[1], cat_labels, judge_cat_rows, show_xlabel=True)
    if poster:
        # No "(A)"/"(B)" panel lettering — panel A's single "Overall" row is
        # self-explanatory, and panel B keeps only a small, unobtrusive
        # group label instead of a bold title, freeing vertical space.
        axes[1].set_title("By Question Category", fontsize=FS["subtitle"], loc="left", color="#444444")
    else:
        axes[1].set_title(
            "(B) By Question Category", fontsize=FS["title"], fontweight="bold", loc="left"
        )

    handles, labels = axes[0].get_legend_handles_labels()
    if poster:
        # Legend sits inside the reserved top_margin band (y < 1.0) so it's
        # captured by the fixed-figsize save below — the screen-mode
        # y=1.06 position only works because bbox_inches="tight" expands
        # the canvas to include it. Labels are shortened to just the model
        # name — at 8 inches wide, the full "(primary)"/"(no search)" role
        # suffixes don't fit on one row, and that role context belongs in
        # the poster's own text box now that the caption is gone anyway.
        short_labels = [lab.split(" (")[0] for lab in labels]
        fig.legend(handles, short_labels, fontsize=FS["legend"], framealpha=0.9,
                   loc="upper center", bbox_to_anchor=(0.5, 0.985), ncol=n_judges,
                   handletextpad=0.5, columnspacing=1.2)
        plt.savefig(output_path, dpi=300)
    else:
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


def build_dual_judge_score_distribution(
    df: pd.DataFrame, judges: list[dict]
) -> pd.DataFrame:
    """
    Bucketed (0-9, 10-19, ..., 90-100) score distribution per judge per
    model, long-format table: judge, model, bucket, count, pct.

    `judges` is a list of dicts, each {"label": str, "outcome": str}
    (color/table1/table2 are not needed here, unlike the coefficient-figure
    judges list, but the same dicts used there can be passed as-is).
    """
    bin_edges = list(range(0, 91, 10)) + [101]
    bucket_labels = [f"{b}-{b+9}" for b in range(0, 90, 10)] + ["90-100"]

    rows = []
    for j in judges:
        for model, grp in df.groupby("model", observed=True):
            counts = (
                pd.cut(
                    grp[j["outcome"]], bins=bin_edges, labels=bucket_labels,
                    right=False, include_lowest=True,
                )
                .value_counts()
                .reindex(bucket_labels, fill_value=0)
            )
            total = counts.sum()
            pcts = (counts / total * 100).round(1) if total else counts.astype(float)
            for bucket in bucket_labels:
                rows.append(
                    {
                        "judge": j["label"],
                        "model": model,
                        "bucket": bucket,
                        "count": int(counts[bucket]),
                        "pct": float(pcts[bucket]),
                    }
                )
    return pd.DataFrame(rows)


def build_dual_judge_score_distribution_figure(
    df: pd.DataFrame,
    judges: list[dict],
    output_path: Path,
    title: str = "RQ1: Veracity score distribution — by judge model",
) -> None:
    """
    Dodged (side-by-side) grouped bar chart of the bucketed score
    distribution, one panel per model (VIOLETS / Baseline), one color per
    judge. Overlaid semi-transparent histograms (the original design) become
    unreadable once there are 3+ judges — the alpha-blended colors merge
    into indistinguishable blobs. Dodging keeps every judge visually
    separable at any judge count, at the cost of x-axis space, hence the
    10-bucket (not 100-value) granularity here.

    Uses % of that judge's responses (not raw count) per bucket, so judges
    with a few missing/null scores (e.g. a failed API call) are still
    directly comparable to one with none missing.
    """
    dist_table = build_dual_judge_score_distribution(df, judges)
    bucket_labels = [f"{b}-{b+9}" for b in range(0, 90, 10)] + ["90-100"]
    models = df["model"].cat.categories.tolist()
    n_judges = len(judges)
    bar_w = 0.8 / n_judges

    fig, axes = plt.subplots(1, len(models), figsize=(8 * len(models), 5.5), sharey=True)
    if len(models) == 1:
        axes = [axes]

    x = np.arange(len(bucket_labels))
    for ax, model in zip(axes, models):
        n = int((df["model"] == model).sum())
        sub = dist_table[dist_table["model"] == model]
        for i, j in enumerate(judges):
            jrow = sub[sub["judge"] == j["label"]].set_index("bucket").reindex(bucket_labels).fillna(0)
            offset = (i - (n_judges - 1) / 2) * bar_w
            ax.bar(
                x + offset, jrow["pct"], width=bar_w * 0.92,
                color=j["color"], label=j["label"], edgecolor="white", linewidth=0.6,
            )
        ax.set_title(f"{model} (n={n})", fontsize=15, fontweight="bold")
        ax.set_xlabel("Veracity score", fontsize=13)
        ax.set_xticks(x)
        ax.set_xticklabels(bucket_labels, rotation=45, ha="right", fontsize=10)
        ax.yaxis.grid(True, linestyle="--", linewidth=0.6, alpha=0.6)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)

    axes[0].set_ylabel("% of responses", fontsize=13)
    axes[0].legend(fontsize=10, loc="upper left")
    fig.suptitle(title, fontsize=18, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=180, bbox_inches="tight")
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
    build_coefficient_figure(
        df=df,
        table1=table1,
        table2=table2,
        output_path=output_dir / "rq1_poster_figure.png",
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

    # URL citation quality (RQ1.2 — Grounding and Citation Reliability).
    # Only present if the input JSONL was produced after the RQ1.2 grounding
    # pipeline fix; older files won't have these columns.
    if "url_citation_rate_score" in df.columns:
        url_summary = build_url_citation_summary(df)
        url_summary.to_csv(output_dir / "url_citation_summary.csv", index=False)

        url_by_cat = build_url_citation_by_category(df)
        if not url_by_cat.empty:
            url_by_cat.to_csv(output_dir / "url_citation_by_category.csv", index=False)

        url_flagged = build_url_flagged_table(df)
        if not url_flagged.empty:
            url_flagged.to_csv(
                output_dir / "url_citations_flagged_for_review.csv", index=False
            )
            stage_counts = url_flagged["stage"].value_counts()
            print(
                f"Flagged {len(url_flagged)} citation issue(s) → url_citations_flagged_for_review.csv "
                f"(no_citation={stage_counts.get('no_citation', 0)}, "
                f"inaccessible={stage_counts.get('inaccessible', 0)}, "
                f"inaccurate={stage_counts.get('inaccurate', 0)})"
            )

        s = url_summary.iloc[0]
        print(
            f"URL citation summary: pct_cited={s['pct_cited']}%  "
            f"pct_accessible={s['pct_accessible']}%  pct_accurate={s['pct_accurate']}%  "
            f"({s['n_urls_total']} URLs across {s['n_turns']} VIOLETS turns)"
        )
    else:
        print(
            "No url_citation_rate_score column in input — skipping RQ1.2 "
            "citation-quality summary (this JSONL predates the grounding-pipeline fix)."
        )

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
