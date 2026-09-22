#!/usr/bin/env python3
"""Analyze product profitability: rankings, segments, divisions, Pareto, cost diagnostics, dependency risk.

Usage:
    python3 analysis.py                    # reads product_metrics.csv + cleaned_data.csv, writes ./analysis/
Outputs (in analysis/): product_ranking.csv, cost_sales_scatter.csv, pareto_analysis.csv,
    pareto_curve.csv, dependency_risk.csv, segment_summary.csv, division_summary.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

REGION_RISK_PCT = 30.0   # one region above this share of sales/profit = dependency risk
STATE_RISK_PCT = 10.0    # one state above this share = dependency risk
COST_HEAVY_QUANTILE = 0.75  # cost/sales ratio in the top quartile = cost-heavy
DIVISION_IMBALANCE_PCT = 10.0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Analyze product profitability metrics.")
    p.add_argument("input", nargs="?", default="product_metrics.csv")
    p.add_argument("--cleaned", default="cleaned_data.csv")
    p.add_argument("-o", "--output", default="analysis")
    return p.parse_args()


def find_file(path_text: str) -> Path:
    path = Path(path_text).expanduser()
    if path.is_file():
        return path
    alt = Path(__file__).resolve().parent / path_text
    if alt.is_file():
        return alt
    raise FileNotFoundError(f"File not found: {path_text}")


def label_products(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    mp, mm, ms = df["gross_profit"].median(), df["Gross Margin %"].median(), df["sales"].median()

    def label(r):
        high_profit, high_margin, high_sales = r["gross_profit"] >= mp, r["Gross Margin %"] >= mm, r["sales"] >= ms
        if high_profit and high_margin:
            return "High-profit/High-margin"
        if high_sales and not high_margin:
            return "High-sales/Low-margin"
        if not high_sales and not high_profit:
            return "Low-sales/Low-profit"
        return "Other"

    df["segment"] = df.apply(label, axis=1)
    return df


def cost_flags(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["cost_sales_ratio"] = df["cost"] / df["sales"].replace(0, np.nan)
    df["cost_heavy"] = df["cost_sales_ratio"] >= df["cost_sales_ratio"].quantile(COST_HEAVY_QUANTILE)
    df["low_margin"] = df["Gross Margin %"] < df["Gross Margin %"].median()
    df["cost_heavy_low_margin"] = df["cost_heavy"] & df["low_margin"]
    return df


def add_actions(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    def decide(r):
        if r["segment"] == "Low-sales/Low-profit":
            return "Discontinuation Review"
        if r["cost_heavy_low_margin"]:
            return "Cost Renegotiation"
        if r["segment"] == "High-sales/Low-margin":
            return "Repricing"
        return "Maintain"

    df["action_needed"] = df.apply(decide, axis=1)
    return df


def division_summary(df: pd.DataFrame) -> pd.DataFrame:
    if "Division" not in df.columns:
        return pd.DataFrame()
    ts, tp = df["sales"].sum(), df["gross_profit"].sum()
    agg = (
        df.groupby("Division")
        .agg(products=("Product Name", "count"), sales=("sales", "sum"), gross_profit=("gross_profit", "sum"),
             avg_margin=("Gross Margin %", "mean"), avg_profit_per_unit=("Profit per Unit", "mean"))
        .reset_index()
    )
    agg["revenue_share"] = agg["sales"] / ts * 100
    agg["profit_share"] = agg["gross_profit"] / tp * 100
    agg["imbalance"] = (agg["revenue_share"] - agg["profit_share"]).abs()
    agg["imbalance_flag"] = agg["imbalance"] > DIVISION_IMBALANCE_PCT
    return agg


def pareto(df: pd.DataFrame, threshold: float = 80.0):
    """Returns (summary, curve). products_needed = fewest top products reaching `threshold`% of the total."""
    summary_rows, curves = [], []
    for name, col in [("Revenue", "sales"), ("Profit", "gross_profit")]:
        s = df[["Product Name", col]].sort_values(col, ascending=False).reset_index(drop=True)
        total = s[col].sum()
        s["cum_pct"] = s[col].cumsum() / total * 100 if total else 0.0
        s["rank"] = np.arange(1, len(s) + 1)
        s["metric"] = name
        needed = int(np.argmax((s["cum_pct"] >= threshold).to_numpy())) + 1
        summary_rows.append({
            "metric": name,
            "products_needed": needed,
            "pct_of_total_products": needed / len(s) * 100,
            "cumulative_value_at_threshold": float(s[col].iloc[:needed].sum()),
        })
        curves.append(s.rename(columns={col: "value"})[["metric", "rank", "Product Name", "value", "cum_pct"]])
    return pd.DataFrame(summary_rows), pd.concat(curves, ignore_index=True)


def dependency_risk(cleaned: pd.DataFrame) -> pd.DataFrame:
    cols = ["geography_type", "geography", "sales", "gross_profit", "sales_share", "profit_share", "risk_flag"]
    needed = ["Region", "State/Province", "Sales", "Gross Profit"]
    if any(c not in cleaned.columns for c in needed):
        return pd.DataFrame(columns=cols)
    cleaned = cleaned.copy()
    cleaned["Sales"] = pd.to_numeric(cleaned["Sales"], errors="coerce")
    cleaned["Gross Profit"] = pd.to_numeric(cleaned["Gross Profit"], errors="coerce")
    cleaned = cleaned.dropna(subset=["Sales", "Gross Profit"])

    frames = []
    for gtype, col, limit in [("Region", "Region", REGION_RISK_PCT), ("State/Province", "State/Province", STATE_RISK_PCT)]:
        agg = cleaned.groupby(col).agg(sales=("Sales", "sum"), gross_profit=("Gross Profit", "sum")).reset_index()
        agg["sales_share"] = agg["sales"] / agg["sales"].sum() * 100
        agg["profit_share"] = agg["gross_profit"] / agg["gross_profit"].sum() * 100
        agg["risk_flag"] = (agg["sales_share"] > limit) | (agg["profit_share"] > limit)
        agg["geography_type"] = gtype
        frames.append(agg.rename(columns={col: "geography"})[cols])
    return pd.concat(frames, ignore_index=True).sort_values(["geography_type", "sales_share"], ascending=[True, False])


def top_bottom(df: pd.DataFrame, col: str, label: str) -> None:
    print(f"\nTop 5 by {label}:")
    for _, r in df.nlargest(5, col).iterrows():
        print(f"  {r['Product Name']}: {r[col]:,.2f}")
    print(f"Bottom 5 by {label}:")
    for _, r in df.nsmallest(5, col).iterrows():
        print(f"  {r['Product Name']}: {r[col]:,.2f}")


def main() -> None:
    args = parse_args()
    input_path = find_file(args.input)
    out = Path(args.output).expanduser()
    out.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_path)
    df["rank_gross_profit"] = df["gross_profit"].rank(ascending=False, method="min")
    df["rank_margin"] = df["Gross Margin %"].rank(ascending=False, method="min")
    df = add_actions(cost_flags(label_products(df)))

    ranked = df.sort_values(["rank_gross_profit", "rank_margin"]).reset_index(drop=True)
    ranked.to_csv(out / "product_ranking.csv", index=False)
    ranked[["Product Name", "sales", "cost", "cost_sales_ratio", "Gross Margin %", "cost_heavy",
            "cost_heavy_low_margin", "segment", "action_needed"]].to_csv(out / "cost_sales_scatter.csv", index=False)

    par, curve = pareto(df)
    par.to_csv(out / "pareto_analysis.csv", index=False)
    curve.to_csv(out / "pareto_curve.csv", index=False)

    try:
        risk = dependency_risk(pd.read_csv(find_file(args.cleaned)))
    except FileNotFoundError:
        risk = dependency_risk(pd.DataFrame())
    risk.to_csv(out / "dependency_risk.csv", index=False)

    seg = df.groupby("segment").agg(products=("Product Name", "count"), sales=("sales", "sum"),
                                    gross_profit=("gross_profit", "sum")).reset_index()
    seg["revenue_share"] = seg["sales"] / seg["sales"].sum() * 100
    seg["profit_share"] = seg["gross_profit"] / seg["gross_profit"].sum() * 100
    seg.to_csv(out / "segment_summary.csv", index=False)

    div = division_summary(df)
    div.to_csv(out / "division_summary.csv", index=False)

    # ---- console summary ----
    print(f"Input: {input_path}\nOutputs written to: {out}\nProducts analyzed: {len(df):,}")
    print("\nPareto (80% threshold):")
    for _, r in par.iterrows():
        print(f"  {r['metric']}: {int(r['products_needed'])} of {len(df)} products ({r['pct_of_total_products']:.0f}%)")
    print("\nSegments:")
    for _, r in seg.iterrows():
        print(f"  {r['segment']}: {int(r['products'])} products, {r['revenue_share']:.1f}% revenue, {r['profit_share']:.1f}% profit")
    if not div.empty:
        print("\nDivisions:")
        for _, r in div.iterrows():
            flag = "  <- IMBALANCE" if r["imbalance_flag"] else ""
            print(f"  {r['Division']}: avg margin {r['avg_margin']:.1f}%, revenue {r['revenue_share']:.1f}%, profit {r['profit_share']:.1f}%{flag}")
    flagged = risk[risk["risk_flag"]] if not risk.empty else risk
    print(f"\nDependency risks flagged: {len(flagged)}")
    for _, r in flagged.iterrows():
        print(f"  {r['geography_type']} / {r['geography']}: sales {r['sales_share']:.1f}%, profit {r['profit_share']:.1f}%")
    top_bottom(df, "gross_profit", "gross profit")
    top_bottom(df, "Gross Margin %", "gross margin %")
    print("\nActions needed:")
    for action, n in df["action_needed"].value_counts().items():
        print(f"  {action}: {n}")


if __name__ == "__main__":
    main()