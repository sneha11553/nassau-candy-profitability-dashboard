"""
app.py - Nassau Candy Distributor: Product Line Profitability & Margin Performance dashboard.

Run:
    python3 -m streamlit run app.py

Reads cleaned_data.csv (created by cleaning.py) from the same folder and recomputes every
metric live from the sidebar filters.
"""
from __future__ import annotations

import inspect
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st

DATA_FILE = Path(__file__).resolve().parent / "cleaned_data.csv"
REGION_RISK_PCT = 30.0
STATE_RISK_PCT = 10.0
COST_HEAVY_QUANTILE = 0.75
DIVISION_IMBALANCE_PCT = 10.0

# ----------------------------------------------------------------------------- theme


# Professional palette: 1 primary accent + supporting neutrals
NAVY, SLATE, TEAL, AMBER, RED, GREEN, GREY = "#1E3A5F", "#4A5568", "#0D7A6E", "#B7791F", "#C53030", "#2F855A", "#8D96A8"
CHOC = "#7A4E1A"
GRID = "rgba(127,127,127,.18)"
CARD_BG, BORDER = "#FFFFFF", "rgba(127,127,127,.18)"
SUBTLE_BG = "rgba(127,127,127,.06)"
NOW_BAR = SALES_BAR = NAVY
SEQUENCE = [NAVY, TEAL, AMBER, RED, GREEN, SLATE, "#6B46C1"]
DIV_COLORS = {"Chocolate": CHOC, "Sugar": "#B7791F", "Other": SLATE}
ACTION_COLORS = {
    "Maintain": GREEN, "Repricing": AMBER, "Cost Renegotiation": NAVY, "Discontinuation Review": RED,
}

st.set_page_config(page_title="Nassau Candy - Profitability", page_icon="🍬", layout="wide")

# ----------------------------------------------------------------------------- theme
LIGHT_THEME = {
    "page": "#ffffff", "card": "#f6f8fa", "elevated": "#ffffff",
    "interactive": "#f6f8fa", "border": "#d0d7de", "border_subtle": "#e3e8ee",
    "text": "#1a2330", "text_secondary": "#5b6573", "text_tertiary": "#8a94a0",
    "grid": "rgba(127,127,127,0.18)", "shadow": "0 1px 2px rgba(0,0,0,.06), 0 1px 3px rgba(0,0,0,.05)",
}
DARK_THEME = {
    "page": "#0d1117", "card": "#161b22", "elevated": "#1c2128",
    "interactive": "#262c34", "border": "#30363d", "border_subtle": "#30363d",
    "text": "#e6edf3", "text_secondary": "#c9d1d9", "text_tertiary": "#8b949e",
    "grid": "rgba(255,255,255,0.14)", "shadow": "none",
}

def current_theme() -> tuple[str, dict[str, str]]:
    try:
        theme = st.context.theme.type
    except Exception:
        theme = "light"
    theme = theme if theme in ("light", "dark") else "light"
    return theme, DARK_THEME if theme == "dark" else LIGHT_THEME

THEME, TC = current_theme()

st.markdown(f"""
<style>
:root {{
  --surface-page: {TC['page']}; --surface-card: {TC['card']}; --surface-elevated: {TC['elevated']};
  --surface-interactive: {TC['interactive']}; --border-default: {TC['border']}; --border-subtle: {TC['border_subtle']};
  --text-primary: {TC['text']}; --text-secondary: {TC['text_secondary']}; --text-tertiary: {TC['text_tertiary']};
  --shadow-card: {TC['shadow']}; --plot-grid: {TC['grid']};
}}

html, body, [data-testid="stAppViewContainer"], [data-testid="stAppViewContainer"] > section {{
  background: var(--surface-page) !important; color: var(--text-primary) !important;
}}
footer {{visibility:hidden;}}
.block-container {{padding-top:1.2rem; max-width:1350px;}}
.app-header {{padding:.5rem 0 .25rem; border-bottom:1px solid var(--border-default); margin-bottom:1rem; background:var(--surface-page);}}
.app-header h1 {{font-size:1.35rem; font-weight:700; color:var(--text-primary) !important; margin:0; letter-spacing:-.02em;}}
.app-header p {{font-size:.82rem; color:var(--text-secondary) !important; margin:.15rem 0 0;}}
.kpi {{background:var(--surface-elevated); border:1px solid var(--border-default); border-top:3px solid var(--c); padding:.6rem .8rem; border-radius:8px; box-shadow:var(--shadow-card);}}
.kpi .l {{font-size:.68rem; text-transform:uppercase; letter-spacing:.06em; color:var(--text-secondary) !important;}}
.kpi .v {{font-size:1.2rem; font-weight:600; color:var(--text-primary) !important; line-height:1.2;}}
.kpi .s {{font-size:.7rem; color:var(--text-tertiary) !important;}}
.insight {{background:var(--surface-elevated); border-left:3px solid var(--c); padding:.55rem .75rem; font-size:.8rem; color:var(--text-primary) !important; border-radius:8px; border:1px solid var(--border-subtle);}}
.insight b {{color:var(--c) !important;}}
.section {{font-size:.95rem; font-weight:600; color:var(--text-primary) !important; margin:1.2rem 0 .4rem;}}
.stTabs [data-baseweb="tab-list"] {{gap:0; border-bottom:2px solid var(--border-default) !important; background:var(--surface-page) !important;}}
.stTabs [data-baseweb="tab"] {{background:var(--surface-page) !important; border-radius:0; padding:8px 14px; border-bottom:2px solid transparent; margin-bottom:-2px; color:var(--text-secondary) !important;}}
.stTabs [data-baseweb="tab"] p, .stTabs [data-baseweb="tab"] span {{color:var(--text-secondary) !important;}}
.stTabs [data-baseweb="tab"]:hover {{background:var(--surface-interactive) !important; color:var(--text-primary) !important;}}
.stTabs [aria-selected="true"] {{border-bottom-color:{NAVY} !important; background:var(--surface-page) !important;}}
.stTabs [aria-selected="true"] p, .stTabs [aria-selected="true"] span {{color:var(--text-primary) !important; font-weight:600;}}
[data-testid="stSidebar"] {{background:var(--surface-page) !important; border-right:1px solid var(--border-default) !important;}}
[data-testid="stSidebar"] .stMarkdown h2 {{font-size:1rem; color:var(--text-primary) !important;}}
[data-testid="stSidebar"] [data-testid="stCaptionContainer"] p {{color:var(--text-secondary) !important;}}
input, textarea, select, [data-baseweb="select"] > div, [data-baseweb="input"] > div {{background:var(--surface-elevated) !important; color:var(--text-primary) !important; border-color:var(--border-default) !important;}}
input::placeholder, textarea::placeholder {{color:var(--text-tertiary) !important;}}
[data-baseweb="select"] *, [data-baseweb="input"] * {{color:var(--text-primary) !important;}}
[data-testid="stDownloadButton"] button {{background:var(--surface-elevated) !important; color:var(--text-primary) !important; border-color:var(--border-default) !important;}}
button {{color:var(--text-primary);}}
@media(max-width:768px) {{.block-container {{padding-left:.5rem; padding-right:.5rem;}} .kpi {{padding:.5rem .6rem;}} .kpi .v {{font-size:1rem;}}}}
</style>
""", unsafe_allow_html=True)

_PLOTLY_HAS_WIDTH = "width" in inspect.signature(st.plotly_chart).parameters
_DF_HAS_WIDTH = "width" in inspect.signature(st.dataframe).parameters

def table_show(data, **kwargs) -> None:
    if _DF_HAS_WIDTH:
        st.dataframe(data, width="stretch", **kwargs)
    else:
        st.dataframe(data, use_container_width=True, **kwargs)

def kpi(label: str, value: str, sub: str, color: str) -> str:
    return f'<div class="kpi" style="--c:{color}"><div class="l">{label}</div><div class="v">{value}</div><div class="s">{sub}</div></div>'

def insight(html: str, color: str) -> str:
    return f'<div class="insight" style="--c:{color}">{html}</div>'

def section(title: str) -> None:
    st.markdown(f'<div class="section">{title}</div>', unsafe_allow_html=True)

def show(fig, height: int | None = None) -> None:
    is_dark = THEME == "dark"
    text_color = TC["text"]
    fig.update_layout(
        template="plotly_dark" if is_dark else "plotly_white",
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, Helvetica, Arial, sans-serif", size=13, color=text_color),
        margin=dict(l=10, r=10, t=30, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0, title_text="", font=dict(color=text_color)),
        hoverlabel=dict(bgcolor=TC["elevated"], bordercolor=TC["border"], font=dict(color=text_color)),
    )
    fig.update_xaxes(gridcolor=TC["grid"], zeroline=False, tickfont=dict(color=TC["text_secondary"]), title_font=dict(color=text_color), linecolor=TC["border"])
    fig.update_yaxes(gridcolor=TC["grid"], zeroline=False, tickfont=dict(color=TC["text_secondary"]), title_font=dict(color=text_color), linecolor=TC["border"])
    if height:
        fig.update_layout(height=height)
    if _PLOTLY_HAS_WIDTH:
        st.plotly_chart(fig, width="stretch")
    else:
        st.plotly_chart(fig, use_container_width=True)


# ----------------------------------------------------------------------------- data
@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["Order Date"] = pd.to_datetime(df["Order Date"], errors="coerce")
    for c in ["Sales", "Units", "Gross Profit", "Cost"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["Order Date", "Sales", "Gross Profit", "Cost"]).copy()
    df["Gross Margin %"] = df["Gross Profit"] / df["Sales"].replace(0, np.nan) * 100
    return df


def build_product_table(df: pd.DataFrame, margin_threshold: float) -> pd.DataFrame:
    total_sales, total_profit = df["Sales"].sum(), df["Gross Profit"].sum()
    p = (
        df.groupby(["Product Name", "Division"])
        .agg(sales=("Sales", "sum"), units=("Units", "sum"), gross_profit=("Gross Profit", "sum"), cost=("Cost", "sum"))
        .reset_index()
    )
    if "Factory" in df.columns:
        p = p.merge(df.groupby("Product Name")["Factory"].first().reset_index(), on="Product Name", how="left")

    p["margin_pct"] = p["gross_profit"] / p["sales"].replace(0, np.nan) * 100
    p["profit_per_unit"] = p["gross_profit"] / p["units"].replace(0, np.nan)
    p["revenue_contrib_pct"] = p["sales"] / total_sales * 100 if total_sales else np.nan
    p["profit_contrib_pct"] = p["gross_profit"] / total_profit * 100 if total_profit else np.nan

    monthly = (
        df.assign(Month=df["Order Date"].dt.to_period("M"))
        .groupby(["Product Name", "Month"]).agg(s=("Sales", "sum"), g=("Gross Profit", "sum")).reset_index()
    )
    monthly["m"] = monthly["g"] / monthly["s"].replace(0, np.nan) * 100
    vol = monthly.groupby("Product Name")["m"].std(ddof=1).fillna(0.0).rename("margin_volatility").reset_index()
    p = p.merge(vol, on="Product Name", how="left")

    # segments, cost flags, actions (same rules as analysis.py)
    mp, mm, ms = p["gross_profit"].median(), p["margin_pct"].median(), p["sales"].median()

    def segment(r):
        hp, hm, hs = r["gross_profit"] >= mp, r["margin_pct"] >= mm, r["sales"] >= ms
        if hp and hm:
            return "High-profit/High-margin"
        if hs and not hm:
            return "High-sales/Low-margin"
        if not hs and not hp:
            return "Low-sales/Low-profit"
        return "Other"

    p["segment"] = p.apply(segment, axis=1)
    p["cost_sales_ratio"] = p["cost"] / p["sales"].replace(0, np.nan)
    p["cost_heavy"] = p["cost_sales_ratio"] >= p["cost_sales_ratio"].quantile(COST_HEAVY_QUANTILE)
    p["cost_heavy_low_margin"] = p["cost_heavy"] & (p["margin_pct"] < mm)
    p["below_threshold"] = p["margin_pct"] < margin_threshold

    def action(r):
        if r["segment"] == "Low-sales/Low-profit":
            return "Discontinuation Review"
        if r["cost_heavy_low_margin"]:
            return "Cost Renegotiation"
        if r["segment"] == "High-sales/Low-margin":
            return "Repricing"
        return "Maintain"

    p["action_needed"] = p.apply(action, axis=1)
    return p


def pareto_count(p: pd.DataFrame, col: str, threshold: float = 80.0) -> int:
    s = p[col].sort_values(ascending=False)
    if s.sum() == 0:
        return 0
    cum = s.cumsum() / s.sum() * 100
    return int(np.argmax((cum >= threshold).to_numpy())) + 1


def money(x: float) -> str:
    if abs(x) >= 1_000_000:
        return f"${x / 1_000_000:,.2f}M"
    if abs(x) >= 10_000:
        return f"${x / 1_000:,.1f}K"
    return f"${x:,.0f}"


# ----------------------------------------------------------------------------- load + sidebar
if not DATA_FILE.is_file():
    st.error(f"cleaned_data.csv not found next to app.py ({DATA_FILE.parent}). Run cleaning.py first.")
    st.stop()

data = load_data(str(DATA_FILE))

st.sidebar.markdown("## Nassau Candy")
st.sidebar.caption(f"{len(data):,} order lines - {data['Product Name'].nunique()} products")
st.sidebar.caption("Theme: ⋮ menu (top right) → Settings → Choose app theme")
st.sidebar.markdown("### Filters")
min_d, max_d = data["Order Date"].min().date(), data["Order Date"].max().date()
date_range = st.sidebar.date_input("Order date range", value=(min_d, max_d), min_value=min_d, max_value=max_d)
divisions = sorted(data["Division"].dropna().unique())
sel_div = st.sidebar.multiselect("Division", divisions, default=divisions)
margin_threshold = st.sidebar.slider("Margin threshold (%)", 0, 100, 40, help="Products below this gross margin are flagged as margin risk.")
search = st.sidebar.text_input("Product search", placeholder="e.g. wonka")

picked = list(date_range) if isinstance(date_range, (tuple, list)) else [date_range]
start = picked[0] if picked else min_d
end = picked[-1] if picked else max_d

view = data[
    (data["Order Date"].dt.date >= start)
    & (data["Order Date"].dt.date <= end)
    & (data["Division"].isin(sel_div))
]
if search.strip():
    view = view[view["Product Name"].str.contains(search.strip(), case=False, na=False, regex=False)]

st.markdown(
    '<div class="app-header"><h1>🍬 Nassau Candy Distributor</h1>'
    "<p>Product Line Profitability &amp; Margin Performance Analysis</p></div>",
    unsafe_allow_html=True,
)

if view.empty:
    st.warning("No data for the selected filters.")
    st.stop()

products = build_product_table(view, margin_threshold)

total_sales, total_profit, total_units = view["Sales"].sum(), view["Gross Profit"].sum(), view["Units"].sum()
overall_margin = total_profit / total_sales * 100 if total_sales else 0.0
n_below = int(products["below_threshold"].sum())

k = st.columns(5)
k[0].markdown(kpi("Sales", money(total_sales), f"{len(view):,} order lines", NAVY), unsafe_allow_html=True)
k[1].markdown(kpi("Gross Profit", money(total_profit), f"{total_profit / total_sales * 100:.1f}% of sales" if total_sales else "", TEAL), unsafe_allow_html=True)
k[2].markdown(kpi("Gross Margin", f"{overall_margin:.1f}%", f"threshold {margin_threshold}%", AMBER), unsafe_allow_html=True)
k[3].markdown(kpi("Profit per Unit", f"${total_profit / total_units:,.2f}" if total_units else "n/a", f"{total_units:,.0f} units", GREEN), unsafe_allow_html=True)
k[4].markdown(kpi("Below Threshold", f"{n_below} / {len(products)}", "products flagged", RED if n_below else GREEN), unsafe_allow_html=True)

# key insights
n_rev, n_prof = pareto_count(products, "sales"), pareto_count(products, "gross_profit")
top = products.loc[products["gross_profit"].idxmax()]
weak = products.loc[products["margin_pct"].idxmin()]
n_review = int((products["action_needed"] != "Maintain").sum())
st.write("")
i = st.columns(4)
i[0].markdown(insight(f"<b>Top earner</b><br>{top['Product Name']} - {money(top['gross_profit'])} profit ({top['profit_contrib_pct']:.1f}% of total)", NAVY), unsafe_allow_html=True)
i[1].markdown(insight(f"<b>Weakest margin</b><br>{weak['Product Name']} at {weak['margin_pct']:.1f}% gross margin", RED), unsafe_allow_html=True)
i[2].markdown(insight(f"<b>Concentration</b><br>{n_prof} of {len(products)} products deliver 80% of profit", TEAL), unsafe_allow_html=True)
i[3].markdown(insight(f"<b>Needs attention</b><br>{n_review} products flagged for repricing, cost review or discontinuation", AMBER), unsafe_allow_html=True)
st.write("")

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
    ["Product Profitability", "Division Performance", "Cost vs Margin", "Profit Concentration",
     "Trends", "What-If Simulator"]
)

# ----------------------------------------------------------------------------- tab 1
with tab1:
    section("Product margin leaderboard")
    lb = products.sort_values("margin_pct", ascending=True)
    fig = px.bar(
        lb, x="margin_pct", y="Product Name", orientation="h", color="below_threshold", text_auto=True,
        color_discrete_map={True: RED, False: NAVY},
        labels={"margin_pct": "Gross margin %", "below_threshold": "Below threshold"},
        hover_data={"gross_profit": ":,.0f", "sales": ":,.0f"},
    )
    fig.add_vline(x=margin_threshold, line_dash="dash", line_color=GREY)
    fig.update_traces(marker_cornerradius=6, textposition="outside", texttemplate="%{x:.1f}")
    fig.update_layout(yaxis_title=None)
    show(fig, height=max(360, 30 * len(lb) + 120))

    section("Profit contribution by product")
    pc = products.sort_values("profit_contrib_pct", ascending=False)
    fig = px.bar(
        pc, x="Product Name", y="profit_contrib_pct", color="Division", color_discrete_map=DIV_COLORS,
        labels={"profit_contrib_pct": "Profit contribution %"},
    )
    fig.update_traces(marker_cornerradius=6)
    fig.update_layout(xaxis_tickangle=-40, xaxis_title=None)
    show(fig, height=430)

    section("Product metrics")
    table = products.rename(columns={
        "sales": "Sales", "units": "Units", "gross_profit": "Gross Profit", "cost": "Cost",
        "margin_pct": "Gross Margin %", "profit_per_unit": "Profit per Unit",
        "revenue_contrib_pct": "Revenue Contribution %", "profit_contrib_pct": "Profit Contribution %",
        "margin_volatility": "Margin Volatility", "segment": "Segment", "action_needed": "Action Needed",
    })
    show_cols = [c for c in ["Product Name", "Division", "Factory", "Sales", "Units", "Gross Profit", "Cost",
                             "Gross Margin %", "Profit per Unit", "Revenue Contribution %",
                             "Profit Contribution %", "Margin Volatility", "Segment", "Action Needed"] if c in table.columns]
    table = table[show_cols].sort_values("Gross Profit", ascending=False).round(2)
    config = {
        "Sales": st.column_config.NumberColumn(format="$%.0f"),
        "Gross Profit": st.column_config.NumberColumn(format="$%.0f"),
        "Cost": st.column_config.NumberColumn(format="$%.0f"),
        "Profit per Unit": st.column_config.NumberColumn(format="$%.2f"),
        "Gross Margin %": st.column_config.ProgressColumn("Gross Margin %", min_value=0, max_value=100, format="%.1f"),
        "Profit Contribution %": st.column_config.ProgressColumn("Profit Contribution %", min_value=0, max_value=max(1.0, float(table["Profit Contribution %"].max())), format="%.1f"),
    }
    table_show(table, hide_index=True, column_config={k_: v for k_, v in config.items() if k_ in table.columns})
    st.download_button("⬇ Download product metrics (CSV)", table.to_csv(index=False), "product_metrics_filtered.csv", "text/csv")

# ----------------------------------------------------------------------------- tab 2
with tab2:
    ds = (
        products.groupby("Division")
        .agg(products=("Product Name", "count"), sales=("sales", "sum"), gross_profit=("gross_profit", "sum"))
        .reset_index()
    )
    ds["avg_margin_pct"] = ds["gross_profit"] / ds["sales"].replace(0, np.nan) * 100
    ds["Revenue share %"] = ds["sales"] / ds["sales"].sum() * 100
    ds["Profit share %"] = ds["gross_profit"] / ds["gross_profit"].sum() * 100
    ds["imbalance"] = (ds["Revenue share %"] - ds["Profit share %"]).abs()
    ds["imbalance_flag"] = ds["imbalance"] > DIVISION_IMBALANCE_PCT

    c1, c2 = st.columns(2)
    with c1:
        section("Revenue vs profit share")
        long = ds.melt(id_vars="Division", value_vars=["Revenue share %", "Profit share %"], var_name="Measure", value_name="Share %")
        fig = px.bar(long, x="Division", y="Share %", color="Measure", barmode="group", text_auto=True,
                     color_discrete_map={"Revenue share %": NAVY, "Profit share %": TEAL})
        fig.update_traces(marker_cornerradius=6, texttemplate="%{y:.1f}")
        show(fig, height=400)
    with c2:
        section("Margin distribution (order lines)")
        fig = px.box(view, x="Division", y="Gross Margin %", color="Division", color_discrete_map=DIV_COLORS, points=False)
        fig.update_layout(showlegend=False)
        show(fig, height=400)

    section("Division summary")
    table_show(
        ds.rename(columns={"sales": "Sales", "gross_profit": "Gross Profit", "avg_margin_pct": "Margin %",
                           "imbalance": "Imbalance (pts)", "imbalance_flag": "Imbalance flag"}).round(2),
        hide_index=True,
        column_config={"Sales": st.column_config.NumberColumn(format="$%.0f"),
                       "Gross Profit": st.column_config.NumberColumn(format="$%.0f")},
    )

# ----------------------------------------------------------------------------- tab 3
with tab3:
    section("Cost vs sales")
    fig = px.scatter(
        products, x="sales", y="cost", size="units", color="action_needed", color_discrete_map=ACTION_COLORS,
        symbol="below_threshold", hover_name="Product Name",
        hover_data={"margin_pct": ":.1f", "cost_sales_ratio": ":.2f", "units": ":,.0f"},
        labels={"sales": "Sales ($)", "cost": "Cost ($)", "action_needed": "Action", "below_threshold": "Below margin threshold"},
    )
    fig.update_traces(marker=dict(line=dict(width=1, color=GREY), opacity=0.85))
    show(fig, height=520)

    section("Margin risk flags")
    flagged = products[products["below_threshold"] | products["cost_heavy_low_margin"] | (products["action_needed"] != "Maintain")]
    if flagged.empty:
        st.success("No products flagged for the current filters.")
    else:
        out = flagged[["Product Name", "Division", "sales", "cost", "margin_pct", "cost_sales_ratio",
                       "below_threshold", "cost_heavy_low_margin", "segment", "action_needed"]].rename(columns={
            "sales": "Sales", "cost": "Cost", "margin_pct": "Margin %", "cost_sales_ratio": "Cost/Sales",
            "below_threshold": "Below threshold", "cost_heavy_low_margin": "Cost-heavy & low margin",
            "segment": "Segment", "action_needed": "Action Needed"})
        table_show(out.sort_values("Margin %").round(2), hide_index=True,
                     column_config={"Sales": st.column_config.NumberColumn(format="$%.0f"),
                                    "Cost": st.column_config.NumberColumn(format="$%.0f")})

# ----------------------------------------------------------------------------- tab 4
with tab4:
    section("Pareto analysis")
    metric = st.radio("Metric", ["Profit", "Revenue"], horizontal=True, label_visibility="collapsed")
    col = "gross_profit" if metric == "Profit" else "sales"
    s = products.sort_values(col, ascending=False).reset_index(drop=True)
    s["cum_pct"] = s[col].cumsum() / s[col].sum() * 100 if s[col].sum() else 0.0

    c1, c2 = st.columns(2)
    c1.markdown(kpi("Products for 80% of revenue", f"{n_rev} of {len(products)}", f"{n_rev / len(products) * 100:.0f}% of the range", NAVY), unsafe_allow_html=True)
    c2.markdown(kpi("Products for 80% of profit", f"{n_prof} of {len(products)}", f"{n_prof / len(products) * 100:.0f}% of the range", TEAL), unsafe_allow_html=True)

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_bar(x=s["Product Name"], y=s[col], name=metric, marker_color=NAVY, secondary_y=False)
    fig.add_scatter(x=s["Product Name"], y=s["cum_pct"], name="Cumulative %", mode="lines+markers",
                    line=dict(color=AMBER, width=3), secondary_y=True)
    fig.add_hline(y=80, line_dash="dash", line_color=GREY, secondary_y=True)
    fig.update_yaxes(title_text=f"{metric} ($)", secondary_y=False)
    fig.update_yaxes(title_text="Cumulative %", range=[0, 105], secondary_y=True)
    fig.update_layout(xaxis_tickangle=-40)
    show(fig, height=460)

    section("Geographic dependency")
    geo = st.radio("Geography", ["Region", "State/Province"], horizontal=True, label_visibility="collapsed")
    limit = REGION_RISK_PCT if geo == "Region" else STATE_RISK_PCT
    g = view.groupby(geo).agg(sales=("Sales", "sum"), gross_profit=("Gross Profit", "sum")).reset_index()
    g["Sales share %"] = g["sales"] / g["sales"].sum() * 100
    g["Profit share %"] = g["gross_profit"] / g["gross_profit"].sum() * 100
    g["Risk"] = np.where((g["Sales share %"] > limit) | (g["Profit share %"] > limit), "Over-dependent", "OK")
    g = g.sort_values("Profit share %", ascending=False)
    fig = px.bar(g.head(15), x=geo, y="Profit share %", color="Risk", text_auto=True,
                 color_discrete_map={"Over-dependent": RED, "OK": NAVY})
    fig.add_hline(y=limit, line_dash="dash", line_color=GREY)
    fig.update_traces(marker_cornerradius=6, texttemplate="%{y:.1f}")
    show(fig, height=400)
    st.caption(f"Flag rule: a single {geo.lower()} above {limit:.0f}% of sales or profit is treated as over-dependency.")
    with st.expander("Show full table"):
        table_show(g.rename(columns={"sales": "Sales", "gross_profit": "Gross Profit"}).round(2),
                     hide_index=True,
                     column_config={"Sales": st.column_config.NumberColumn(format="$%.0f"),
                                    "Gross Profit": st.column_config.NumberColumn(format="$%.0f")})

    if {"Factory", "Factory_Latitude", "Factory_Longitude"} <= set(view.columns):
        fac = (
            view.dropna(subset=["Factory", "Factory_Latitude", "Factory_Longitude"])
            .groupby(["Factory", "Factory_Latitude", "Factory_Longitude"])
            .agg(sales=("Sales", "sum"), gross_profit=("Gross Profit", "sum"), products=("Product Name", "nunique"))
            .reset_index()
        )
        if not fac.empty:
            fac["margin_pct"] = fac["gross_profit"] / fac["sales"].replace(0, np.nan) * 100
            fac["profit_share"] = fac["gross_profit"] / fac["gross_profit"].sum() * 100
            # sqrt scale + floor so small factories stay visible next to the big one
            root = np.sqrt(fac["gross_profit"].clip(lower=0))
            fac["bubble"] = np.maximum(root, root.max() * 0.22)
            fac["label"] = fac["Factory"] + " (" + fac["profit_share"].round(1).astype(str) + "%)"

            section("Factory footprint")
            m1, m2 = st.columns([3, 2])
            with m1:
                fig = px.scatter_geo(
                    fac, lat="Factory_Latitude", lon="Factory_Longitude", size="bubble", color="margin_pct",
                    text="label", hover_name="Factory",
                    hover_data={"sales": ":,.0f", "gross_profit": ":,.0f", "products": True, "profit_share": ":.1f",
                                "margin_pct": ":.1f", "bubble": False, "label": False,
                                "Factory_Latitude": False, "Factory_Longitude": False},
                    color_continuous_scale=["#FFD6E0", TEAL, NAVY], size_max=42, scope="usa",
                    labels={"margin_pct": "Margin %", "profit_share": "Profit share %"},
                )
                fig.update_traces(textposition="top center", textfont=dict(size=11),
                                  marker=dict(line=dict(width=1.5, color=GREY)))
                fig.update_geos(
                    scope="usa", bgcolor="rgba(0,0,0,0)", showland=True, landcolor="rgba(127,127,127,.15)",
                    showsubunits=True, subunitcolor="rgba(127,127,127,.3)", subunitwidth=1,
                    showcountries=True, countrycolor="rgba(127,127,127,.5)", showcoastlines=True, coastlinecolor="rgba(127,127,127,.5)",
                    showlakes=False, showframe=False,
                )
                show(fig, height=460)
            with m2:
                fbar = fac.sort_values("gross_profit", ascending=True)
                fig = px.bar(fbar, x="profit_share", y="Factory", orientation="h", text_auto=True,
                             labels={"profit_share": "Share of gross profit (%)"})
                fig.update_traces(marker_color=NAVY, texttemplate="%{x:.1f}")
                fig.update_layout(yaxis_title=None)
                show(fig, height=460)
            st.caption("Bubble size is scaled (square root) so small factories stay visible; colour shows gross margin. "
                       "The bar chart shows each factory's true share of gross profit.")

# ----------------------------------------------------------------------------- tab 5
with tab5:
    monthly = (
        view.assign(Month=view["Order Date"].dt.to_period("M").dt.to_timestamp())
        .groupby("Month").agg(sales=("Sales", "sum"), gross_profit=("Gross Profit", "sum")).reset_index()
    )
    monthly["margin_pct"] = monthly["gross_profit"] / monthly["sales"].replace(0, np.nan) * 100

    section("Monthly sales and gross profit")
    fig = make_subplots(specs=[[{"secondary_y": False}]])
    fig.add_bar(x=monthly["Month"], y=monthly["sales"], name="Sales", marker_color=SALES_BAR, marker_cornerradius=4)
    fig.add_scatter(x=monthly["Month"], y=monthly["gross_profit"], name="Gross profit", mode="lines+markers",
                    line=dict(color=NAVY, width=3))
    fig.update_yaxes(title_text="$")
    show(fig, height=380)

    c1, c2 = st.columns(2)
    with c1:
        section("Monthly gross margin (%)")
        fig = px.line(monthly, x="Month", y="margin_pct", markers=True, labels={"margin_pct": "Gross margin %"})
        fig.update_traces(line=dict(color=AMBER, width=3))
        fig.add_hline(y=margin_threshold, line_dash="dash", line_color=GREY)
        show(fig, height=360)
    with c2:
        section("Margin volatility by product")
        vol = products.sort_values("margin_volatility", ascending=False)
        fig = px.bar(vol, x="Product Name", y="margin_volatility", color="Division", color_discrete_map=DIV_COLORS,
                     labels={"margin_volatility": "Std dev of monthly margin (pts)"})
        fig.update_traces(marker_cornerradius=6)
        fig.update_layout(xaxis_tickangle=-40, xaxis_title=None)
        show(fig, height=360)
    st.caption("Margin volatility = standard deviation of a product's monthly gross margin. Higher means less predictable margins.")

# ----------------------------------------------------------------------------- tab 6
with tab6:
    section("What-if simulator: pricing, cost and volume")
    st.caption("Change price, unit cost or volume for the selected products and see the effect on gross profit. "
               "Changes are applied to all order lines of those products in the current filter.")
    all_names = list(products["Product Name"])
    sel = st.multiselect("Products affected", all_names, default=all_names)
    s1, s2, s3 = st.columns(3)
    pc = s1.slider("Price change (%)", -20, 30, 0) / 100
    cc = s2.slider("Unit cost change (%)", -30, 30, 0) / 100
    vc = s3.slider("Volume change (%)", -30, 30, 0) / 100

    sim = products[["Product Name", "sales", "cost"]].copy()
    aff = sim["Product Name"].isin(sel)
    sim["profit_now"] = sim["sales"] - sim["cost"]
    sim["new_sales"] = np.where(aff, sim["sales"] * (1 + pc) * (1 + vc), sim["sales"])
    sim["new_cost"] = np.where(aff, sim["cost"] * (1 + cc) * (1 + vc), sim["cost"])
    sim["profit_new"] = sim["new_sales"] - sim["new_cost"]

    p0, p1 = sim["profit_now"].sum(), sim["profit_new"].sum()
    m0 = p0 / sim["sales"].sum() * 100 if sim["sales"].sum() else 0.0
    m1 = p1 / sim["new_sales"].sum() * 100 if sim["new_sales"].sum() else 0.0
    delta = p1 - p0
    k1, k2, k3 = st.columns(3)
    k1.markdown(kpi("Gross profit now", money(p0), f"margin {m0:.1f}%", NAVY), unsafe_allow_html=True)
    k2.markdown(kpi("Gross profit after", money(p1), f"margin {m1:.1f}%", TEAL), unsafe_allow_html=True)
    k3.markdown(kpi("Change", f"{'+' if delta >= 0 else '-'}{money(abs(delta))}",
                    f"{delta / p0 * 100:+.1f}% vs now" if p0 else "", TEAL if delta >= 0 else RED), unsafe_allow_html=True)

    if pc == 0 and cc == 0 and vc == 0:
        st.info("All three sliders are at 0%, so 'Now' and 'After change' are identical. Move a slider to see the effect.")
    elif not sel:
        st.warning("No products selected, so nothing changes. Pick at least one product above.")

    only_sel = st.checkbox("Show only the selected products in the charts", value=True)
    chart_sim = sim[sim["Product Name"].isin(sel)] if (only_sel and sel) else sim
    chart_sim = chart_sim.assign(profit_change=chart_sim["profit_new"] - chart_sim["profit_now"])

    st.write("")
    section("Gross profit: now vs after change")
    long = chart_sim.melt(id_vars="Product Name", value_vars=["profit_now", "profit_new"], var_name="Scenario", value_name="Gross profit")
    long["Scenario"] = long["Scenario"].map({"profit_now": "Now", "profit_new": "After change"})
    order = chart_sim.sort_values("profit_new", ascending=False)["Product Name"].tolist()
    fig = px.bar(long, x="Product Name", y="Gross profit", color="Scenario", barmode="group",
                 category_orders={"Product Name": order}, color_discrete_map={"Now": NOW_BAR, "After change": NAVY})
    fig.update_traces(marker_cornerradius=4)
    fig.update_layout(xaxis_tickangle=-40, xaxis_title=None)
    show(fig, height=400)

    section("Change in gross profit by product ($)")
    chg = chart_sim.sort_values("profit_change", ascending=False)
    fig = px.bar(chg, x="Product Name", y="profit_change", text_auto=True, labels={"profit_change": "Change in gross profit ($)"})
    fig.update_traces(marker_color=[TEAL if v >= 0 else RED for v in chg["profit_change"]], marker_cornerradius=4, texttemplate="%{y:.0f}")
    fig.update_layout(xaxis_tickangle=-40, xaxis_title=None)
    show(fig, height=360)
    st.caption("Small products are hard to see in the chart above because Chocolate dominates; this chart shows the exact change for each.")

    section(f"What it takes to reach a {margin_threshold}% margin")
    gap = products[products["margin_pct"] < margin_threshold].copy()
    if gap.empty or margin_threshold >= 100:
        st.success("All products in the current view are at or above the margin threshold.")
    else:
        t = margin_threshold / 100
        gap["Price rise needed (%)"] = (gap["cost"] / (1 - t) / gap["sales"] - 1) * 100
        gap["Or cost cut needed (%)"] = (1 - gap["sales"] * (1 - t) / gap["cost"]) * 100
        out = gap[["Product Name", "Division", "margin_pct", "Price rise needed (%)", "Or cost cut needed (%)"]].rename(
            columns={"margin_pct": "Current margin %"}).sort_values("Current margin %").round(1)
        table_show(out, hide_index=True)
        st.caption("Each column is an alternative: raise price by that % at the same volume, or cut unit cost by that %.")
