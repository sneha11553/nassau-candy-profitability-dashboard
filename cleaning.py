"""
cleaning.py - Nassau Candy Distributor data cleaning + factory mapping.

Usage:
    python3 cleaning.py                                  # uses "Nassau Candy Distributor.csv"
    python3 cleaning.py "Nassau Candy Distributor.csv"   # or pass the file name

Outputs:
    cleaned_data.csv        cleaned rows + Factory, Factory_Latitude, Factory_Longitude, unmatched_product
    unmatched_products.csv  distinct product names with no factory match (with row counts)
"""
import re
import sys

import numpy as np
import pandas as pd

DEFAULT_INPUT = "Nassau Candy Distributor.csv"
OUTPUT = "cleaned_data.csv"
UNMATCHED_OUTPUT = "unmatched_products.csv"
DATE_FMT = "%d-%m-%Y"  # dates in the file are day-month-year

REQUIRED = [
    "Row ID", "Order ID", "Order Date", "Ship Date", "Ship Mode", "Customer ID",
    "Country/Region", "City", "State/Province", "Postal Code", "Division", "Region",
    "Product ID", "Product Name", "Sales", "Units", "Gross Profit", "Cost",
]

PRODUCT_FACTORY = {
    "Wonka Bar - Nutty Crunch Surprise": "Lot's O' Nuts",
    "Wonka Bar - Fudge Mallows": "Lot's O' Nuts",
    "Wonka Bar - Scrumdiddlyumptious": "Lot's O' Nuts",
    "Wonka Bar - Milk Chocolate": "Lot's O' Nuts",
    "Wonka Bar - Triple Dazzle Caramel": "Wicked Choccy's",
    "Laffy Taffy": "Sugar Shack",
    "SweeTARTS": "Sugar Shack",
    "Nerds": "Sugar Shack",
    "Fun Dip": "Sugar Shack",
    "Fizzy Lifting Drinks": "Sugar Shack",
    "Everlasting Gobstopper": "Secret Factory",
    "Hair Toffee": "The Other Factory",
    "Lickable Wallpaper": "Secret Factory",
    "Wonka Gum": "Secret Factory",
    "Kazookles": "The Other Factory",
}

FACTORY_COORDS = {
    "Lot's O' Nuts": (32.881893, -111.768036),
    "Wicked Choccy's": (32.076176, -81.088371),
    "Sugar Shack": (48.11914, -96.18115),
    "Secret Factory": (41.446333, -90.565487),
    "The Other Factory": (35.1175, -89.971107),
}


def norm_name(value):
    """Trim, collapse spaces, standardize ' - ' spacing and apostrophes."""
    s = str(value).replace("\u2019", "'").replace("\u2013", "-").replace("\u2014", "-")
    s = re.sub(r"\s+", " ", s).strip()
    return re.sub(r"\s*-\s*", " - ", s)


def name_key(value):
    return norm_name(value).lower()


PRODUCT_KEY_TO_NAME = {name_key(k): k for k in PRODUCT_FACTORY}


def to_number(series):
    cleaned = series.astype(str).str.replace(r"[$,\s]", "", regex=True)
    return pd.to_numeric(cleaned, errors="coerce")


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_INPUT
    df = pd.read_csv(path, dtype=str, keep_default_na=True)
    df.columns = [c.strip() for c in df.columns]

    missing_cols = [c for c in REQUIRED if c not in df.columns]
    if missing_cols:
        sys.exit(f"Missing columns in file: {missing_cols}\nFound: {list(df.columns)}")

    rows_read = len(df)

    # ---- trim text columns and standardize labels ----
    text_cols = [c for c in df.columns if c not in ("Sales", "Units", "Gross Profit", "Cost")]
    for c in text_cols:
        df[c] = df[c].astype(str).str.strip().replace({"nan": np.nan, "": np.nan})
    df["Division"] = df["Division"].str.title()
    df["Region"] = df["Region"].str.title()
    df["Product Name"] = df["Product Name"].map(lambda v: norm_name(v) if pd.notna(v) else v)
    # snap known products to their canonical spelling
    df["Product Name"] = df["Product Name"].map(
        lambda v: PRODUCT_KEY_TO_NAME.get(name_key(v), v) if pd.notna(v) else v
    )

    # ---- numeric + date parsing ----
    for c in ["Sales", "Units", "Gross Profit", "Cost"]:
        df[c] = to_number(df[c])
    df["Order Date"] = pd.to_datetime(df["Order Date"], format=DATE_FMT, errors="coerce")
    df["Ship Date"] = pd.to_datetime(df["Ship Date"], format=DATE_FMT, errors="coerce")

    # ---- fill what can be derived ----
    fill_cost = df["Cost"].isna() & df["Sales"].notna() & df["Gross Profit"].notna()
    df.loc[fill_cost, "Cost"] = df["Sales"] - df["Gross Profit"]
    fill_gp = df["Gross Profit"].isna() & df["Sales"].notna() & df["Cost"].notna()
    df.loc[fill_gp, "Gross Profit"] = df["Sales"] - df["Cost"]

    # ---- validation: reasons a row is removed ----
    reasons = {
        "duplicate Row ID": df.duplicated(subset="Row ID", keep="first"),
        "missing/invalid Sales": df["Sales"].isna(),
        "zero or negative Sales": df["Sales"].fillna(1) <= 0,
        "missing/invalid Cost": df["Cost"].isna(),
        "negative Cost": df["Cost"].fillna(0) < 0,
        "missing/invalid Order Date": df["Order Date"].isna(),
        "Ship Date before Order Date": (
            df["Ship Date"].notna() & df["Order Date"].notna() & (df["Ship Date"] < df["Order Date"])
        ),
        "missing Product Name": df["Product Name"].isna(),
    }
    remove = pd.Series(False, index=df.index)
    for mask in reasons.values():
        remove |= mask
    df = df[~remove].copy()

    # ---- handle missing / non-positive Units (impute, don't drop) ----
    bad_units = df["Units"].isna() | (df["Units"] <= 0)
    units_imputed = int(bad_units.sum())
    df.loc[bad_units, "Units"] = np.nan
    prod_median = df.groupby("Product Name")["Units"].transform("median")
    df["Units"] = df["Units"].fillna(prod_median).fillna(df["Units"].median()).round()

    # ---- consistency check: Gross Profit vs Sales - Cost (report only) ----
    gp_mismatch = int(((df["Gross Profit"] - (df["Sales"] - df["Cost"])).abs() > 0.01).sum())

    # ---- factory mapping ----
    df["Factory"] = df["Product Name"].map(PRODUCT_FACTORY)
    df["Factory_Latitude"] = df["Factory"].map(lambda f: FACTORY_COORDS[f][0] if pd.notna(f) else np.nan)
    df["Factory_Longitude"] = df["Factory"].map(lambda f: FACTORY_COORDS[f][1] if pd.notna(f) else np.nan)
    df["unmatched_product"] = df["Factory"].isna()

    unmatched = (
        df[df["unmatched_product"]]
        .groupby("Product Name", dropna=False).size()
        .reset_index(name="rows")
        .sort_values("rows", ascending=False)
    )
    unmatched.to_csv(UNMATCHED_OUTPUT, index=False)

    # ---- save ----
    df["Order Date"] = df["Order Date"].dt.strftime("%Y-%m-%d")
    df["Ship Date"] = df["Ship Date"].dt.strftime("%Y-%m-%d")
    df.to_csv(OUTPUT, index=False)

    # ---- summary ----
    print(f"Input: {path}")
    print(f"Rows read: {rows_read:,}")
    print(f"Rows written: {len(df):,}")
    print(f"Rows removed: {rows_read - len(df):,}")
    print("Rows removed by reason (can overlap):")
    for name, mask in reasons.items():
        print(f"  - {name}: {int(mask.sum()):,}")
    print(f"Units imputed: {units_imputed:,}")
    print(f"Gross Profit != Sales - Cost (kept, flagged only): {gp_mismatch:,}")
    print(f"Unmatched products (kept, Factory blank): {int(df['unmatched_product'].sum()):,} rows "
          f"across {len(unmatched)} product(s) -> {UNMATCHED_OUTPUT}")
    print(f"Saved: {OUTPUT}")


if __name__ == "__main__":
    main()