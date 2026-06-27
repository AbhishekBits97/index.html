"""
build_data_json.py
==================
Reads the scored parquet dataset and outputs data/website_data.json
for the Lumetha static website.

Run locally:   python build_data_json.py
Run by CI:     same command, triggered on push via GitHub Actions
"""

import json
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"

SCORED_PARQUET  = DATA_DIR / "explanations" / "listings_investment_scored.parquet"
FEATURE_PARQUET = DATA_DIR / "features" / "feature_matrix.parquet"
FORECAST_JSON   = DATA_DIR / "models" / "prophet_forecasts.json"
OUT_JSON        = DATA_DIR / "website_data.json"

# ── Load data ─────────────────────────────────────────────────────────────────
def load_df() -> pd.DataFrame:
    """Load best available source; fall back gracefully."""
    if SCORED_PARQUET.exists():
        df = pd.read_parquet(SCORED_PARQUET)
        print(f"✅  Loaded scored parquet: {len(df)} rows")
        return df
    if FEATURE_PARQUET.exists():
        df = pd.read_parquet(FEATURE_PARQUET)
        print(f"⚠️  Scored parquet missing — loaded feature matrix: {len(df)} rows")
        return df
    raise FileNotFoundError(
        "Neither listings_investment_scored.parquet nor feature_matrix.parquet found."
    )


def safe(val):
    """Convert numpy / nan types to JSON-safe Python scalars."""
    if val is None or (isinstance(val, float) and math.isnan(val)):
        return None
    if isinstance(val, (np.integer,)):
        return int(val)
    if isinstance(val, (np.floating,)):
        return float(val)
    return val


# ── Column name helpers ───────────────────────────────────────────────────────
COL_ALIASES = {
    # locality / area name
    "locality": ["locality", "Locality", "area", "Area", "micro_market", "location"],
    # price per sqft
    "price_per_sqft": ["price_per_sqft", "Price_per_sqft", "price_sqft",
                       "price_per_sqft_inr", "psf"],
    # investment / composite score
    "investment_score": ["investment_score", "Investment_Score", "composite_score",
                         "inv_score", "score"],
    # appreciation potential
    "appreciation_score": ["appreciation_score", "Appreciation_Score",
                           "appr_score", "appreciation_potential"],
    # risk-adjusted ROI
    "roi": ["risk_adj_roi", "Risk_Adj_ROI", "roi", "ROI", "risk_adjusted_roi",
            "net_roi", "adj_roi"],
    # investment grade
    "grade": ["investment_grade", "Investment_Grade", "grade", "Grade", "recommendation"],
    # livability
    "livability": ["livability_score", "Livability_Score", "livability_index",
                   "livability"],
    # infra impact
    "infra_impact": ["infra_impact_score", "Infra_Impact_Score", "infra_score",
                     "infrastructure_score"],
    # builder tier
    "builder_tier": ["builder_tier", "Builder_Tier", "tier", "Tier"],
    # BHK / property type
    "property_type": ["property_type", "Property_Type", "bhk", "BHK", "config"],
    # total price / listing price
    "total_price": ["total_price", "Total_Price", "listing_price", "price",
                    "Price", "price_inr"],
    # payback years
    "payback_years": ["payback_years", "Payback_Years", "payback", "payback_period"],
}


def find_col(df: pd.DataFrame, key: str):
    """Return the first matching column name or None."""
    for candidate in COL_ALIASES.get(key, [key]):
        if candidate in df.columns:
            return candidate
    return None


def col(df: pd.DataFrame, key: str) -> pd.Series | None:
    """Return series for a logical column, or None if absent."""
    c = find_col(df, key)
    return df[c] if c else None


# ── Build JSON ────────────────────────────────────────────────────────────────
def build(df: pd.DataFrame) -> dict:
    locality_col    = find_col(df, "locality")
    psf_col         = find_col(df, "price_per_sqft")
    score_col       = find_col(df, "investment_score")
    appr_col        = find_col(df, "appreciation_score")
    roi_col         = find_col(df, "roi")
    grade_col       = find_col(df, "grade")
    livability_col  = find_col(df, "livability")
    infra_col       = find_col(df, "infra_impact")
    tier_col        = find_col(df, "builder_tier")
    price_col       = find_col(df, "total_price")
    payback_col     = find_col(df, "payback_years")

    # ── KPIs ──────────────────────────────────────────────────────────────────
    kpis = {
        "total_listings"   : len(df),
        "avg_price_sqft"   : safe(round(df[psf_col].mean(), 0))    if psf_col    else None,
        "median_price_sqft": safe(round(df[psf_col].median(), 0))  if psf_col    else None,
        "avg_roi"          : safe(round(df[roi_col].mean(), 1))     if roi_col    else None,
        "avg_investment_score": safe(round(df[score_col].mean(), 1)) if score_col else None,
        "avg_livability"   : safe(round(df[livability_col].mean(), 1)) if livability_col else None,
        "avg_infra_impact" : safe(round(df[infra_col].mean(), 1))   if infra_col  else None,
        "avg_payback_years": safe(round(df[payback_col].mean(), 1)) if payback_col else None,
    }

    # ── Grade / recommendation breakdown ──────────────────────────────────────
    grade_counts = {}
    buy_count    = 0
    if grade_col:
        vc = df[grade_col].value_counts().to_dict()
        grade_counts = {str(k): int(v) for k, v in vc.items()}
        buy_keywords = {"buy", "grade b", "b", "a", "grade a", "a+", "b+"}
        buy_count = sum(v for k, v in grade_counts.items()
                        if str(k).strip().lower() in buy_keywords)

    kpis["buy_count"]        = buy_count
    kpis["buy_pct"]          = safe(round(buy_count / len(df) * 100, 1)) if len(df) else 0
    kpis["grade_breakdown"]  = grade_counts

    # ── Builder tier ──────────────────────────────────────────────────────────
    tier_counts = {}
    if tier_col:
        vc = df[tier_col].value_counts().to_dict()
        tier_counts = {str(k): int(v) for k, v in vc.items()}
    kpis["builder_tier_counts"] = tier_counts

    # ── Budget bands (total price) ────────────────────────────────────────────
    budget_bands: dict = {}
    if price_col:
        p = df[price_col].dropna()
        budget_bands = {
            "under_1cr"  : int((p < 1e7).sum()),
            "1cr_to_2cr" : int(((p >= 1e7) & (p < 2e7)).sum()),
            "2cr_to_5cr" : int(((p >= 2e7) & (p < 5e7)).sum()),
            "above_5cr"  : int((p >= 5e7).sum()),
        }
    kpis["budget_bands"] = budget_bands

    # ── Locality aggregates ───────────────────────────────────────────────────
    localities: list[dict] = []
    if locality_col:
        grp_cols = {k: v for k, v in {
            "price_per_sqft"   : psf_col,
            "investment_score" : score_col,
            "appreciation_score": appr_col,
            "roi"              : roi_col,
            "livability"       : livability_col,
            "infra_impact"     : infra_col,
        }.items() if v}

        agg_dict = {v: "mean" for v in grp_cols.values()}
        if grade_col:
            # most common grade per locality
            agg_dict[grade_col] = lambda x: x.mode()[0] if len(x) else None

        grp = df.groupby(locality_col).agg({**agg_dict, locality_col: "count"})
        grp.columns = [
            *[k for k in grp_cols],
            *([("grade" if grade_col else "")] if grade_col else []),
            "listing_count"
        ]
        grp = grp.reset_index()
        grp = grp[grp["listing_count"] >= 2]  # at least 2 listings

        if "investment_score" in grp.columns:
            grp = grp.sort_values("investment_score", ascending=False)

        for _, row in grp.iterrows():
            rec: dict = {"locality": str(row[locality_col]), "listing_count": int(row["listing_count"])}
            for k in grp_cols:
                if k in row and pd.notna(row[k]):
                    rec[k] = safe(round(float(row[k]), 1))
            if grade_col and "grade" in row:
                rec["grade"] = str(row["grade"]) if pd.notna(row["grade"]) else None
            localities.append(rec)

    # ── Ticker entries (top 15 by price/sqft) ────────────────────────────────
    ticker: list[dict] = []
    if locality_col and psf_col:
        top_psf = (
            df.groupby(locality_col)[psf_col]
            .mean()
            .sort_values(ascending=False)
            .head(15)
        )
        for loc, psf_val in top_psf.items():
            entry = {"locality": str(loc), "price_per_sqft": safe(round(psf_val, 0))}
            if score_col:
                s = df[df[locality_col] == loc][score_col].mean()
                entry["investment_score"] = safe(round(s, 1))
            ticker.append(entry)

    # ── Sector cards (top 6 by investment score) ──────────────────────────────
    sector_cards = localities[:6] if localities else []

    # ── Forecast data (if available) ──────────────────────────────────────────
    forecasts: dict = {}
    if FORECAST_JSON.exists():
        try:
            with open(FORECAST_JSON) as f:
                forecasts = json.load(f)
            print(f"✅  Loaded forecast JSON")
        except Exception as e:
            print(f"⚠️  Could not load forecast JSON: {e}")

    return {
        "meta": {
            "generated_at": pd.Timestamp.utcnow().isoformat() + "Z",
            "source"       : "listings_investment_scored.parquet",
            "total_rows"   : len(df),
        },
        "kpis"         : kpis,
        "localities"   : localities,
        "ticker"       : ticker,
        "sector_cards" : sector_cards,
        "forecasts"    : forecasts,
    }


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    df   = load_df()
    data = build(df)
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"✅  Written → {OUT_JSON}  ({OUT_JSON.stat().st_size // 1024} KB)")
