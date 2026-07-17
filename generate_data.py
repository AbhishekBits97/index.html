#!/usr/bin/env python3
"""
generate_data.py
----------------
Reads listings_investment_scored.csv and writes data.json.
Run locally:  python generate_data.py
Run by CI:    called automatically by .github/workflows/update-data.yml

Place this file at the ROOT of your GitHub repo.
"""

import csv, json, statistics
from collections import defaultdict
from datetime import date
from pathlib import Path

CSV_PATH  = Path(__file__).parent / "listings_investment_scored.csv"
JSON_PATH = Path(__file__).parent / "data.json"

# ── Load ──────────────────────────────────────────────────────────────
rows = []
with open(CSV_PATH, newline="") as f:
    for row in csv.DictReader(f):
        rows.append(row)

def safe_float(val):
    try:
        return float(val)
    except (ValueError, TypeError):
        return None

# ── Aggregate by locality ─────────────────────────────────────────────
lp = defaultdict(list)   # price/sqft
ls = defaultdict(list)   # investment_score
la = defaultdict(list)   # appreciation_potential_score
lr = defaultdict(list)   # roi_risk_adjusted

for r in rows:
    loc = r["locality"]
    for store, field in [(lp,"price_per_sqft"),(ls,"investment_score"),
                         (la,"appreciation_potential_score"),(lr,"roi_risk_adjusted")]:
        v = safe_float(r.get(field))
        if v is not None:
            store[loc].append(v)

# ── KPIs ──────────────────────────────────────────────────────────────
all_prices  = [safe_float(r["price_per_sqft"]) for r in rows if safe_float(r.get("price_per_sqft"))]
all_scores  = [safe_float(r["investment_score"]) for r in rows if safe_float(r.get("investment_score"))]
all_roi     = [safe_float(r["roi_risk_adjusted"]) for r in rows if safe_float(r.get("roi_risk_adjusted"))]
all_appr    = [safe_float(r["appreciation_potential_score"]) for r in rows if safe_float(r.get("appreciation_potential_score"))]
all_payback = [safe_float(r["payback_years"]) for r in rows if safe_float(r.get("payback_years"))]
buy_count   = sum(1 for r in rows if r.get("recommendation") == "Buy")
total       = len(rows)

kpis = {
    "avg_price_sqft":          round(statistics.mean(all_prices), 0),
    "median_price_sqft":       round(statistics.median(all_prices), 0),
    "avg_investment_score":    round(statistics.mean(all_scores), 1),
    "avg_roi_risk_adjusted":   round(statistics.mean(all_roi), 1),
    "avg_appreciation_potential": round(statistics.mean(all_appr), 1),
    "avg_payback_years":       round(statistics.mean(all_payback), 1),
    "buy_count":               buy_count,
    "hold_count":              total - buy_count,
    "buy_pct":                 round(buy_count / total * 100, 1),
    "total_listings":          total,
}

# ── Price bands ───────────────────────────────────────────────────────
bands = {"under_1cr": 0, "1_2cr": 0, "2_5cr": 0, "above_5cr": 0}
for r in rows:
    p = safe_float(r.get("price_cr"))
    if p is None: continue
    if   p < 1: bands["under_1cr"] += 1
    elif p < 2: bands["1_2cr"]     += 1
    elif p < 5: bands["2_5cr"]     += 1
    else:       bands["above_5cr"] += 1

# ── BHK / property type / builder tier ───────────────────────────────
bhk = defaultdict(int)
ptypes = defaultdict(int)
btiers = defaultdict(int)
for r in rows:
    v = safe_float(r.get("bhk_count"))
    if v: bhk[int(v)] += 1
    ptypes[r.get("property_type_clean", "other")] += 1
    btiers[r.get("builder_tier", "unknown")]       += 1

# ── Ticker — top 12 localities by avg price/sqft (min 3 listings) ────
ticker = sorted(
    [{"locality": k, "avg_price_sqft": round(statistics.mean(v), 0)}
     for k, v in lp.items() if len(v) >= 3],
    key=lambda x: -x["avg_price_sqft"]
)[:12]

# ── Sector rankings — top 10 by investment score (min 3 listings) ────
sector_rankings = sorted(
    [{"locality": k,
      "investment_score": round(statistics.mean(ls[k]), 1),
      "avg_price_sqft":   round(statistics.mean(lp[k]), 0) if lp[k] else 0,
      "appreciation":     round(statistics.mean(la[k]), 1) if la[k] else 0,
      "roi":              round(statistics.mean(lr[k]), 1) if lr[k] else 0,
      "listing_count":    len(ls[k])}
     for k in ls if len(ls[k]) >= 3],
    key=lambda x: -x["investment_score"]
)[:10]

# ── Price chart — top 12 localities by price ─────────────────────────
price_chart = sorted(
    [{"locality": k, "avg_price_sqft": round(statistics.mean(v), 0)}
     for k, v in lp.items() if len(v) >= 3],
    key=lambda x: -x["avg_price_sqft"]
)[:12]

# ── Assemble & write ──────────────────────────────────────────────────
output = {
    "meta": {
        "total_listings": total,
        "generated_at":   str(date.today()),
        "source":         CSV_PATH.name,
    },
    "kpis":             kpis,
    "price_bands":      bands,
    "bhk_distribution": dict(sorted(bhk.items())),
    "property_types":   dict(ptypes),
    "builder_tiers":    dict(btiers),
    "ticker":           ticker,
    "sector_rankings":  sector_rankings,
    "price_chart":      price_chart,
}

with open(JSON_PATH, "w") as f:
    json.dump(output, f, indent=2)

print(f"✓ data.json written — {total} listings, generated {date.today()}")
