"""
RQ3 Mixed-Effects Analysis for VIOLETS FAQ Alignment Evaluation
================================================================

Purpose
-------
Analyze RQ3 ("How well do VIOLETS's responses align with official FAQ
guidance?") using mixed-effects models, mirroring the RQ1/RQ2 analysis
scripts, then export:
  - Table 1: overall model effect
  - Table 2: model effect within each FAQ category
  - Table 3: model effect within each query type (original vs. GLC-perturbed)
  - One combined figure with 2 subplots (Overall + By Category), matching
    the RQ1/RQ2 poster figure convention

Design assumptions from the project
------------------------------------
- The same FAQ question (original wording, and a GLC paraphrase) is sent
  to both VIOLETS and the baseline model
- One JSONL line = one (faq_id, query_type, model_id) scored response
- Outcome = similarity_score (0-1 cosine similarity to the official answer)
- Repeated observations are clustered within faq_id — each FAQ contributes
  up to 4 rows (original/perturbed x VIOLETS/baseline), so faq_id plays the
  same role conversation_id plays in RQ1/RQ2

Input
-----
Default input path:
    output/rq3/eval_dataset.jsonl

Expected schema (per line)
--------------------------
- faq_id
- category
- query_type      ("original" | "perturbed")
- model_id
- query
- official_answer
- model_response
- similarity_score
- timestamp

Outputs
-------
Default output directory:
    output/rq3/analysis_mixed/

Files created:
- table1_model_overall.csv
- table2_category_effects.csv
- table3_querytype_effects.csv
- rq3_poster_figure.png
- model_overall_summary.csv / .txt
- category_model_summary.csv
- querytype_model_summary.csv

Notes
-----
1. This script uses statsmodels MixedLM with a random intercept for faq_id
   (the natural pairing unit here, analogous to conversation_id in RQ1/RQ2).
2. If the default optimizer fails, it retries with alternative optimizers.
3. Category-level and query-type-level effects are derived as fixed-effect
   contrasts (main effect + interaction term), not raw interaction
   coefficients — see contrast_est_ci().
4. Sample-size caveat: only ~26 FAQs (so ~26 faq_id clusters) back this
   analysis, versus ~50 conversations for RQ1/RQ2 — category-level
   breakdowns in particular will have low power (a handful of FAQs per
   category), so a non-significant category effect may just mean
   underpowered, not "no difference."
5. query_type is the RQ3-specific axis that replaces RQ1/RQ2's turn
   dimension: it tests whether the GLC paraphrase degrades alignment more
   for one model than the other (robustness to query rewording), which is
   the stated purpose of the perturbation in the project design.
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

# ── FAQ category order (voter-journey sequence; unlisted categories appended) ─
CATEGORY_ORDER = [
    "voter_registration",
    "requesting_a_ballot",
    "state_of_application_ballot",
    "ballot_documents_envelope",
    "marking_reviewing_ballot",
    "returning_ballot",
    "ballot_drop_boxes",
    "in_person_voting",
    "general_election",
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
    - query_type is cast to categorical with "original" as the reference level
    """
    df = df.copy()

    # Numeric conversion
    df["similarity_score"] = pd.to_numeric(df["similarity_score"], errors="coerce")

    # Basic text conversion
    for col in [
        "faq_id",
        "model_id",
        "category",
        "query_type",
        "query",
        "official_answer",
        "model_response",
        "timestamp",
    ]:
        if col in df.columns:
            df[col] = df[col].astype(str)

    # Drop rows without key modeling fields
    df = df.dropna(subset=["similarity_score", "query_type"]).copy()

    # Normalize model labels
    # We want two levels only: Baseline and VIOLETS
    # "violets" in the JSONL becomes "VIOLETS"
    df["model"] = df["model_id"].replace({"violets": "VIOLETS"})

    # Infer baseline label(s): anything not VIOLETS becomes baseline
    df["model"] = np.where(df["model"] == "VIOLETS", "VIOLETS", "Baseline")
    df["model"] = pd.Categorical(
        df["model"], categories=["Baseline", "VIOLETS"], ordered=True
    )

    # Stable category ordering (voter-journey order from the project plan)
    seen_categories = [c for c in CATEGORY_ORDER if c in set(df["category"])]
    extras = [c for c in sorted(df["category"].unique()) if c not in seen_categories]
    full_category_order = seen_categories + extras
    df["category"] = pd.Categorical(
        df["category"], categories=full_category_order, ordered=True
    )

    # query_type as categorical, "original" as the reference level
    query_type_order = [
        qt for qt in ["original", "perturbed"] if qt in set(df["query_type"])
    ]
    extras_qt = [
        qt for qt in sorted(df["query_type"].unique()) if qt not in query_type_order
    ]
    df["query_type"] = pd.Categorical(
        df["query_type"], categories=query_type_order + extras_qt, ordered=True
    )

    # Stable sorting
    df = df.sort_values(["faq_id", "model", "query_type"]).reset_index(drop=True)

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


def build_table1_overall(df: pd.DataFrame, outcome: str = "similarity_score") -> Tuple[pd.DataFrame, object]:
    """
    Table 1: Overall model effect from:
        {outcome} ~ model + (1 | faq_id)

    `outcome` defaults to the primary embedding model's similarity_score
    column, but can be pointed at a secondary embedding's column (e.g.
    "similarity_score_2nd") to fit the identical model for comparison.
    """
    result = fit_mixedlm(
        f"{outcome} ~ C(model, Treatment(reference='Baseline'))",
        df=df,
        group_col="faq_id",
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
                "n_faqs": df["faq_id"].nunique(),
            }
        ]
    )

    return table, result


def build_table2_category(df: pd.DataFrame, outcome: str = "similarity_score") -> Tuple[pd.DataFrame, object]:
    """
    Table 2: Model effect within each category from:
        {outcome} ~ model * category + (1 | faq_id)

    `outcome` defaults to the primary embedding model's similarity_score
    column, but can be pointed at a secondary embedding's column (e.g.
    "similarity_score_2nd") to fit the identical model for comparison.
    """
    if len(df["category"].cat.categories) == 0:
        raise ValueError("No category levels found.")

    ref_cat = df["category"].cat.categories[0]

    formula = (
        f"{outcome} ~ "
        "C(model, Treatment(reference='Baseline')) * "
        f"C(category, Treatment(reference='{ref_cat}'))"
    )
    result = fit_mixedlm(formula, df=df, group_col="faq_id")

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
                "n_faqs": int(
                    df.loc[df["category"] == cat, "faq_id"].nunique()
                ),
            }
        )

    table = pd.DataFrame(rows)
    return table, result


def build_table3_querytype(df: pd.DataFrame) -> Tuple[pd.DataFrame, object]:
    """
    Table 3: Model effect within each query type (original vs. perturbed) from:
        similarity_score ~ model * query_type + (1 | faq_id)

    This is RQ3's robustness-to-rewording check: does the GLC paraphrase
    shrink (or widen) the VIOLETS-vs-baseline alignment gap relative to the
    original wording?
    """
    if len(df["query_type"].cat.categories) == 0:
        raise ValueError("No query_type levels found.")

    ref_qt = df["query_type"].cat.categories[0]

    formula = (
        "similarity_score ~ "
        "C(model, Treatment(reference='Baseline')) * "
        f"C(query_type, Treatment(reference='{ref_qt}'))"
    )
    result = fit_mixedlm(formula, df=df, group_col="faq_id")

    model_coef = "C(model, Treatment(reference='Baseline'))[T.VIOLETS]"

    rows = []
    for qt in df["query_type"].cat.categories:
        if qt == ref_qt:
            est, lo, hi, p = coef_and_ci(result, model_coef)
        else:
            interaction = (
                f"C(model, Treatment(reference='Baseline'))[T.VIOLETS]:"
                f"C(query_type, Treatment(reference='{ref_qt}'))[T.{qt}]"
            )
            est, lo, hi, p = contrast_est_ci(
                result,
                coef_names=[model_coef, interaction],
                weights=[1.0, 1.0],
            )

        rows.append(
            {
                "query_type": qt,
                "contrast": "VIOLETS - Baseline",
                "estimate": est,
                "ci_low": lo,
                "ci_high": hi,
                "p_value": p,
                "n_rows": int((df["query_type"] == qt).sum()),
                "n_faqs": int(
                    df.loc[df["query_type"] == qt, "faq_id"].nunique()
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
        df.groupby("model", observed=True)["similarity_score"]
        .agg(["count", "mean", "std", "median", "min", "max"])
        .reset_index()
    )


def simple_category_model_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Helpful descriptive summary by category and model."""
    return (
        df.groupby(["category", "model"], observed=True)["similarity_score"]
        .agg(["count", "mean", "std", "median"])
        .reset_index()
    )


def simple_querytype_model_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Helpful descriptive summary by query_type and model."""
    return (
        df.groupby(["query_type", "model"], observed=True)["similarity_score"]
        .agg(["count", "mean", "std", "median"])
        .reset_index()
    )


# ============================================================================
# Figure builder
# ============================================================================


_CAT_LABELS_RQ3 = {
    "voter_registration": "voter registration",
    "requesting_a_ballot": "requesting ballot",
    "state_of_application_ballot": "application status",
    "ballot_documents_envelope": "ballot docs/envelope",
    "marking_reviewing_ballot": "marking/reviewing",
    "returning_ballot": "returning ballot",
    "ballot_drop_boxes": "drop boxes",
    "in_person_voting": "in-person voting",
    "general_election": "general election",
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
    xlabel: str = "VIOLETS − Baseline (semantic similarity)",
    title: str = "RQ3: FAQ Alignment — VIOLETS vs. Baseline (estimated effect)",
) -> None:
    """
    Two-panel coefficient (forest) plot:
      (A) Overall effect   (B) Effect by FAQ Category

    Each point is the VIOLETS-minus-Baseline estimate from the mixed-effects
    model (table1/table2), with a 95% CI whisker; the dashed vertical line
    at 0 marks "no difference". This plots exactly the quantity the
    significance test is about, so there is only one interval to read per
    row — unlike a grouped bar chart with one CI per model, where two
    overlapping per-group CIs can visually look non-significant even when
    the (correctly, paired/clustered) tested difference is significant.
    """
    COLOR = "#7B2FBE"
    FS = {"title": 20, "label": 16, "tick": 14, "stars": 15}
    CAP = 5
    ERR_KW = {"elinewidth": 2.0, "ecolor": COLOR, "capthick": 2.0}

    categories = df["category"].cat.categories.tolist()
    cat_labels = [_CAT_LABELS_RQ3.get(c, c) for c in categories]
    cat2 = table2.set_index("category").loc[categories]

    # Panel B carries 9 FAQ categories (vs. 5 threat/question categories in
    # RQ1/RQ2), so it needs proportionally more height to avoid crowding.
    fig, axes = plt.subplots(
        2, 1, figsize=(10, 9.5), gridspec_kw={"height_ratios": [1, len(categories)]}
    )
    fig.subplots_adjust(hspace=0.45, left=0.3)

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
        "(B) By FAQ Category", fontsize=FS["title"], fontweight="bold", loc="left"
    )

    fig.suptitle(title, fontsize=20, fontweight="bold", y=1.01)
    fig.text(
        0.5, -0.02,
        "Points = mixed-model estimate of VIOLETS − Baseline, whiskers = 95% CI  |  "
        "dashed line = no difference  |  * p < .05  ** p < .01  *** p < .001",
        ha="center", fontsize=12, color="#555555",
    )

    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def build_multi_embedding_coefficient_figure(
    df: pd.DataFrame,
    embeddings: list[dict],
    output_path: Path,
    xlabel: str = "VIOLETS − Baseline (semantic similarity)",
    title: str = "RQ3: FAQ Alignment — VIOLETS vs. Baseline, by embedding model",
) -> None:
    """
    Same two-panel coefficient (forest) plot as build_coefficient_figure, but
    overlays one point+CI series per embedding model instead of relying on a
    single one. RQ3 has no LLM judge to cross-check (similarity_score is a
    deterministic cosine similarity, not a judgment call), so the analogous
    robustness check here is re-embedding with an independent model
    (openai/text-embedding-3-small vs. qwen/qwen3-embedding-8b) rather than a
    second judge — if both embeddings agree on which categories are
    significant, the finding isn't an artifact of one embedding space.

    `embeddings` is a list of dicts, each:
        {"label": str, "color": str, "table1": DataFrame, "table2": DataFrame}
    """
    FS = {"title": 20, "label": 16, "tick": 14, "legend": 13, "stars": 13}
    CAP = 5
    n_emb = len(embeddings)
    offsets = np.linspace(-0.16, 0.16, n_emb) if n_emb > 1 else [0.0]

    categories = df["category"].cat.categories.tolist()
    cat_labels = [_CAT_LABELS_RQ3.get(c, c) for c in categories]

    fig, axes = plt.subplots(
        2, 1, figsize=(10, 9.5), gridspec_kw={"height_ratios": [1, len(categories)]}
    )
    fig.subplots_adjust(hspace=0.45, left=0.3)

    def _forest_panel(ax, labels, emb_rows_list):
        y_base = np.arange(len(labels))
        ax.axvline(0, color="#999999", linewidth=1.2, linestyle="--", zorder=1)
        for emb, offset, rows in zip(embeddings, offsets, emb_rows_list):
            y = y_base + offset
            est = rows["estimate"].tolist()
            lo = rows["ci_low"].tolist()
            hi = rows["ci_high"].tolist()
            ax.errorbar(
                est, y,
                xerr=[np.array(est) - np.array(lo), np.array(hi) - np.array(est)],
                fmt="o", color=emb["color"], markersize=7, capsize=CAP,
                elinewidth=2.0, ecolor=emb["color"], capthick=2.0, zorder=3,
                label=emb["label"],
            )
            for yi, hi_i, p in zip(y, hi, rows["p_value"].tolist()):
                stars = _sig_stars(p)
                if stars != "ns":
                    ax.text(hi_i, yi, f"  {stars}", va="center", ha="left",
                             fontsize=FS["stars"], color=emb["color"], fontweight="bold")
        ax.set_yticks(y_base)
        ax.set_yticklabels(labels, fontsize=FS["tick"])
        ax.invert_yaxis()
        ax.set_xlabel(xlabel, fontsize=FS["label"])
        ax.xaxis.grid(True, linestyle="--", linewidth=0.6, alpha=0.6)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)

    # ── Panel A: Overall ──────────────────────────────────────────────────
    _forest_panel(axes[0], ["Overall"], [e["table1"] for e in embeddings])
    axes[0].set_title("(A) Overall", fontsize=FS["title"], fontweight="bold", loc="left")

    # ── Panel B: By Category ──────────────────────────────────────────────
    emb_cat_rows = [e["table2"].set_index("category").loc[categories] for e in embeddings]
    _forest_panel(axes[1], cat_labels, emb_cat_rows)
    axes[1].set_title(
        "(B) By FAQ Category", fontsize=FS["title"], fontweight="bold", loc="left"
    )

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=FS["legend"], framealpha=0.9,
               loc="upper center", bbox_to_anchor=(0.5, 1.05), ncol=n_emb)

    fig.suptitle(title, fontsize=20, fontweight="bold", y=1.1)
    fig.text(
        0.5, -0.02,
        "Points = mixed-model estimate of VIOLETS − Baseline per embedding model, whiskers = 95% CI  |  "
        "dashed line = no difference  |  * p < .05  ** p < .01  *** p < .001",
        ha="center", fontsize=12, color="#555555",
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
        "faq_id",
        "model_id",
        "category",
        "query_type",
        "similarity_score",
    ]

    df = load_jsonl(input_path)
    validate_columns(df, required_columns)
    df = preprocess(df)

    # Inferential tables
    table1, m1 = build_table1_overall(df)
    table2, m2 = build_table2_category(df)
    table3, m3 = build_table3_querytype(df)

    table1.to_csv(output_dir / "table1_model_overall.csv", index=False)
    table2.to_csv(output_dir / "table2_category_effects.csv", index=False)
    table3.to_csv(output_dir / "table3_querytype_effects.csv", index=False)

    # Supportive descriptive summaries
    simple_model_summary(df).to_csv(
        output_dir / "model_overall_summary.csv", index=False
    )
    simple_category_model_summary(df).to_csv(
        output_dir / "category_model_summary.csv", index=False
    )
    simple_querytype_model_summary(df).to_csv(
        output_dir / "querytype_model_summary.csv", index=False
    )

    # Text summaries of fitted models
    (output_dir / "model_overall_summary.txt").write_text(
        str(m1.summary()), encoding="utf-8"
    )
    (output_dir / "category_model_summary.txt").write_text(
        str(m2.summary()), encoding="utf-8"
    )
    (output_dir / "querytype_model_summary.txt").write_text(
        str(m3.summary()), encoding="utf-8"
    )

    # Figure
    build_coefficient_figure(
        df=df,
        table1=table1,
        table2=table2,
        output_path=output_dir / "rq3_poster_figure.png",
    )

    print("RQ3 mixed-effects analysis complete.")
    print(f"Input:      {input_path}")
    print(f"Output dir: {output_dir}")
    print(
        f"Note: only {df['faq_id'].nunique()} FAQ clusters back this analysis — "
        "category-level effects in particular may be underpowered."
    )


# ============================================================================
# CLI
# ============================================================================


def parse_args():
    parser = argparse.ArgumentParser(
        description="RQ3 mixed-effects analysis for VIOLETS FAQ alignment evaluation."
    )
    parser.add_argument(
        "--input",
        type=str,
        default="output/rq3/eval_dataset.jsonl",
        help="Path to the RQ3 JSONL file.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output/rq3/analysis_mixed",
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
