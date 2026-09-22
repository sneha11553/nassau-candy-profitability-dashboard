#!/usr/bin/env python3
"""Compute per-product profitability metrics from cleaned_data.csv -> product_metrics.csv.

Usage:
    python3 metrics.py                       # reads cleaned_data.csv
    python3 metrics.py cleaned_data.csv -o product_metrics.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = ["Order Date", "Product Name", "Sales", "Units", "Gross Profit", "Cost"]
# carried over per product if present (names match cleaning.py output)
FIRST_COLUMNS = ["Division", "Factory", "Factory_Latitude", "Factory_Longitude"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compute per-product profitability metrics.")
    p.add_argument("input", nargs="?", default="cleaned_data.csv")
    p.add_argument("-o", "--output", default="product_metrics.csv")
    return p.parse_args()


def find_file(path_text: str) -> Path:
    path = Path(path_text).expanduser()
    if path.is_file():
        return path
    alt = Path(__file__).resolve().parent / path_text
    if alt.is_file():
        return alt
    raise FileNotFoundError(f"Input file not found: {path_text}")


def compute_metrics(input_path: Path) -> pd.DataFrame:
    data = pd.read_csv(input_path)
    missing = [c for c in REQUIRED_COLUMNS if c not in data.columns]
    if missing:
        raise ValueError(f"Input CSV is missing required columns: {missing}")

    data["Order Date"] = pd.to_datetime(data["Order Date"], errors="coerce")
    for c in ["Sales", "Units", "Gross Profit", "Cost"]:
        data[c] = pd.to_numeric(data[c], errors="coerce")
    data["Product Name"] = data["Product Name"].astype("string").str.strip()
    data = data[data["Order Date"].notna() & data["Product Name"].notna() & (data["Product Name"] != "")].copy()
    if data.empty:
        raise ValueError("No valid rows found in the input file.")

    total_sales = data["Sales"].sum()
    total_profit = data["Gross Profit"].sum()

    prod = (
        data.groupby("Product Name")
        .agg(sales=("Sales", "sum"), units=("Units", "sum"),
             gross_profit=("Gross Profit", "sum"), cost=("Cost", "sum"))
        .reset_index()
    )
    prod["Gross Margin %"] = prod["gross_profit"] / prod["sales"].replace(0, np.nan) * 100
    prod["Profit per Unit"] = prod["gross_profit"] / prod["units"].replace(0, np.nan)
    prod["Revenue Contribution %"] = prod["sales"] / total_sales * 100 if total_sales else np.nan
    prod["Profit Contribution %"] = prod["gross_profit"] / total_profit * 100 if total_profit else np.nan

    # Margin volatility = std dev of the monthly gross margin % per product
    data["Month"] = data["Order Date"].dt.to_period("M")
    monthly = data.groupby(["Product Name", "Month"]).agg(s=("Sales", "sum"), g=("Gross Profit", "sum")).reset_index()
    monthly["margin_pct"] = monthly["g"] / monthly["s"].replace(0, np.nan) * 100
    vol = monthly.groupby("Product Name")["margin_pct"].std(ddof=1).fillna(0.0).rename("Margin Volatility").reset_index()
    prod = prod.merge(vol, on="Product Name", how="left")

    # carry over division / factory info
    extra_cols = [c for c in FIRST_COLUMNS if c in data.columns]
    if extra_cols:
        extra = data.groupby("Product Name")[extra_cols].first().reset_index()
        prod = prod.merge(extra, on="Product Name", how="left")
    if "unmatched_product" in data.columns:
        flag = data.groupby("Product Name")["unmatched_product"].any().rename("unmatched_product").reset_index()
        prod = prod.merge(flag, on="Product Name", how="left")

    order = ["Product Name", "Division", "Factory", "Factory_Latitude", "Factory_Longitude",
             "unmatched_product", "sales", "units", "gross_profit", "cost", "Gross Margin %",
             "Profit per Unit", "Revenue Contribution %", "Profit Contribution %", "Margin Volatility"]
    return prod[[c for c in order if c in prod.columns]]


def main() -> None:
    args = parse_args()
    input_path = find_file(args.input)
    out = Path(args.output).expanduser()
    metrics = compute_metrics(input_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(out, index=False)
    print(f"Input: {input_path}")
    print(f"Output: {out}")
    print(f"Products written: {len(metrics):,}")


if __name__ == "__main__":
    main()