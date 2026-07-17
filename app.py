import io
import json
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st

st.set_page_config(page_title="Stock Rationalisation AI", layout="wide")

REQUIRED_COLUMNS = [
    "sku_id", "sku_name", "category", "subcategory", "brand", "store_id", "region",
    "fixture_type", "shelf_level", "facings", "units_sold_30d", "sales_value_30d",
    "gross_margin_pct", "avg_inventory_units", "avg_inventory_cost_value", "current_stock_units",
    "unit_cost", "mrp", "selling_price", "days_since_receipt", "shelf_life_days", "expiry_date",
    "lead_time_days", "min_display_qty", "case_pack", "space_cm", "stockout_days_30d", "return_rate", "promo_flag"
]

# Optional but recommended for true Out of Display tracking.
# If a retailer does not upload these, the app creates safe defaults and still runs.
OPTIONAL_COLUMNS = ["display_qty_units", "backroom_stock_units"]
TEMPLATE_COLUMNS = REQUIRED_COLUMNS + OPTIONAL_COLUMNS

CATEGORY_MAP = {
    "Packaged Foods": ["Atta & Flour", "Rice", "Pulses", "Breakfast Mix", "Ready To Cook"],
    "Snacks": ["Chips", "Namkeen", "Biscuits", "Cookies", "Popcorn"],
    "Beverages": ["Juices", "Carbonated Drinks", "Energy Drinks", "Tea", "Coffee"],
    "Dairy & Chilled": ["Milk", "Curd", "Paneer", "Cheese", "Yoghurt"],
    "Frozen": ["Frozen Snacks", "Frozen Veg", "Ice Cream", "Frozen Meals"],
    "Personal Care": ["Hair Care", "Skin Care", "Oral Care", "Deodorants", "Bath & Body"],
    "Home Care": ["Detergents", "Dishwash", "Surface Cleaners", "Air Fresheners"],
    "Baby Care": ["Diapers", "Baby Food", "Baby Skin Care"],
    "Pet Care": ["Pet Food", "Pet Treats", "Pet Hygiene"],
}

PERISHABLE_CATEGORIES = {"Dairy & Chilled", "Frozen", "Beverages", "Packaged Foods"}

@st.cache_data(show_spinner=False)
def make_template_df() -> pd.DataFrame:
    today = datetime.today().date()
    rows = [
        ["SKU000001", "Example Milk 1L", "Dairy & Chilled", "Milk", "Brand A", "ST001", "North", "Chiller", "Eye", 3, 180, 10800, 18, 95, 4750, 120, 50, 68, 60, 6, 12, today + timedelta(days=6), 2, 6, 12, 45, 0, 0.01, "N", 2, 118],
        ["SKU000002", "Example Shampoo 180ml", "Personal Care", "Hair Care", "Brand B", "ST001", "North", "Gondola", "Eye", 2, 22, 3740, 32, 80, 7200, 240, 90, 199, 170, 65, 720, today + timedelta(days=655), 7, 4, 12, 30, 1, 0.02, "Y", 1, 239],
        ["SKU000003", "Example Chips 50g", "Snacks", "Chips", "Brand C", "ST002", "West", "Gondola", "Hand", 4, 310, 6200, 24, 145, 2175, 190, 15, 25, 20, 14, 120, today + timedelta(days=106), 3, 10, 20, 35, 0, 0.00, "N", 16, 174],
    ]
    return pd.DataFrame(rows, columns=TEMPLATE_COLUMNS)

@st.cache_data(show_spinner=True)
def generate_demo_data(n_rows: int = 100_000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    cats = list(CATEGORY_MAP.keys())
    category = rng.choice(cats, n_rows, p=[.16,.15,.13,.10,.07,.16,.13,.04,.06])
    subcategory = [rng.choice(CATEGORY_MAP[c]) for c in category]
    brand_pool = [f"Brand {chr(65+i)}" for i in range(24)] + ["Private Label"] * 6
    brand = rng.choice(brand_pool, n_rows)
    store_id = rng.choice([f"ST{i:03d}" for i in range(1, 81)], n_rows)
    region = rng.choice(["North", "South", "East", "West", "Central"], n_rows)
    fixture_type = np.where(np.isin(category, ["Dairy & Chilled", "Frozen"]), rng.choice(["Chiller", "Freezer"], n_rows), rng.choice(["Gondola", "Endcap", "Wall Bay", "Checkout"], n_rows))
    shelf_level = rng.choice(["Top", "Eye", "Hand", "Bottom"], n_rows, p=[.18,.32,.34,.16])
    unit_cost = rng.lognormal(mean=3.45, sigma=.55, size=n_rows).round(2)
    margin = rng.uniform(.12, .42, n_rows)
    mrp = (unit_cost / (1 - margin) * rng.uniform(1.02, 1.18, n_rows)).round(2)
    selling_price = (mrp * rng.uniform(.82, 1.0, n_rows)).round(2)
    velocity = rng.gamma(shape=2.1, scale=22, size=n_rows)
    perishable_boost = np.where(np.isin(category, list(PERISHABLE_CATEGORIES)), 1.25, 1.0)
    units_sold = np.maximum(0, (velocity * perishable_boost * rng.uniform(.25, 1.8, n_rows)).astype(int))
    stock_multiplier = rng.choice([.35, .65, 1.1, 2.2, 5.5, 10.0], n_rows, p=[.08,.16,.38,.23,.10,.05])
    current_stock = np.maximum(0, (units_sold * stock_multiplier + rng.normal(8, 16, n_rows)).astype(int))
    avg_inv_units = np.maximum(1, ((current_stock + units_sold * rng.uniform(.7, 1.7, n_rows)) / 2).astype(int))
    # Display stock is intentionally lower than system stock for some SKUs to simulate OOD risk.
    display_ratio = rng.choice([0.0, 0.2, 0.5, 0.8, 1.0], n_rows, p=[.05, .10, .18, .27, .40])
    display_qty_units = np.maximum(0, (current_stock * display_ratio).astype(int))
    backroom_stock_units = np.maximum(0, current_stock - display_qty_units)
    sales_value = (units_sold * selling_price).round(2)
    avg_inv_cost = (avg_inv_units * unit_cost).round(2)
    shelf_life = []
    for c in category:
        if c == "Dairy & Chilled": shelf_life.append(int(rng.integers(5, 35)))
        elif c == "Frozen": shelf_life.append(int(rng.integers(90, 270)))
        elif c == "Beverages": shelf_life.append(int(rng.integers(90, 270)))
        elif c == "Packaged Foods": shelf_life.append(int(rng.integers(90, 365)))
        elif c == "Snacks": shelf_life.append(int(rng.integers(75, 210)))
        else: shelf_life.append(int(rng.integers(365, 1095)))
    shelf_life = np.array(shelf_life)
    days_since = np.minimum(shelf_life - 1, rng.gamma(2.0, 45, n_rows).astype(int))
    today = datetime.today().date()
    expiry_date = [today + timedelta(days=int(max(1, sl - ds))) for sl, ds in zip(shelf_life, days_since)]
    df = pd.DataFrame({
        "sku_id": [f"SKU{i:06d}" for i in range(1, n_rows + 1)],
        "sku_name": [f"{b} {s} {rng.choice(['Small','Regular','Value','Family','Premium'])}" for b, s in zip(brand, subcategory)],
        "category": category,
        "subcategory": subcategory,
        "brand": brand,
        "store_id": store_id,
        "region": region,
        "fixture_type": fixture_type,
        "shelf_level": shelf_level,
        "facings": rng.integers(1, 8, n_rows),
        "units_sold_30d": units_sold,
        "sales_value_30d": sales_value,
        "gross_margin_pct": (margin * 100).round(1),
        "avg_inventory_units": avg_inv_units,
        "avg_inventory_cost_value": avg_inv_cost,
        "current_stock_units": current_stock,
        "display_qty_units": display_qty_units,
        "backroom_stock_units": backroom_stock_units,
        "unit_cost": unit_cost,
        "mrp": mrp,
        "selling_price": selling_price,
        "days_since_receipt": days_since,
        "shelf_life_days": shelf_life,
        "expiry_date": expiry_date,
        "lead_time_days": rng.integers(2, 15, n_rows),
        "min_display_qty": rng.integers(2, 24, n_rows),
        "case_pack": rng.choice([3, 6, 10, 12, 24, 48], n_rows),
        "space_cm": rng.integers(8, 80, n_rows),
        "stockout_days_30d": rng.choice([0,0,0,1,2,3,5,7], n_rows),
        "return_rate": rng.uniform(0, .06, n_rows).round(3),
        "promo_flag": rng.choice(["Y", "N"], n_rows, p=[.22,.78]),
    })
    return df

def to_excel_bytes(df: pd.DataFrame, sheet_name="Upload_Template") -> bytes:
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        df.to_excel(writer, sheet_name=sheet_name, index=False)
        workbook = writer.book
        worksheet = writer.sheets[sheet_name]
        header_fmt = workbook.add_format({"bold": True, "bg_color": "#1F4E78", "font_color": "#FFFFFF", "border": 1})
        for col_num, value in enumerate(df.columns.values):
            worksheet.write(0, col_num, value, header_fmt)
            worksheet.set_column(col_num, col_num, min(max(len(value) + 2, 12), 24))
        worksheet.freeze_panes(1, 0)
    return output.getvalue()

def validate_upload(df: pd.DataFrame):
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        st.error("Missing columns: " + ", ".join(missing))
        st.stop()
    d = df.copy()
    if "display_qty_units" not in d.columns:
        d["display_qty_units"] = np.minimum(d["current_stock_units"], d["min_display_qty"])
    if "backroom_stock_units" not in d.columns:
        d["backroom_stock_units"] = np.maximum(0, d["current_stock_units"] - d["display_qty_units"])
    return d[TEMPLATE_COLUMNS].copy()

def score_data(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["expiry_date"] = pd.to_datetime(d["expiry_date"], errors="coerce")
    d["rate_of_sales_units_per_day"] = d["units_sold_30d"] / 30
    d["rate_of_sales_units_per_week"] = d["rate_of_sales_units_per_day"] * 7
    d["daily_ros"] = d["rate_of_sales_units_per_day"]
    d["out_of_display_units"] = np.maximum(0, d["min_display_qty"] - d["display_qty_units"])
    d["out_of_display_flag"] = np.where((d["current_stock_units"] > 0) & (d["out_of_display_units"] > 0), "YES", "NO")
    d["out_of_display_value"] = d["out_of_display_units"] * d["unit_cost"]
    d["weeks_cover"] = np.where(d["daily_ros"] > 0, d["current_stock_units"] / d["daily_ros"] / 7, 999)
    d["gross_margin_value_30d"] = d["sales_value_30d"] * d["gross_margin_pct"] / 100
    d["gmroi_30d"] = np.where(d["avg_inventory_cost_value"] > 0, d["gross_margin_value_30d"] / d["avg_inventory_cost_value"], 0)
    d["stock_turn_annualised"] = np.where(d["avg_inventory_units"] > 0, (d["units_sold_30d"] * 12) / d["avg_inventory_units"], 0)
    d["days_to_expiry"] = (d["expiry_date"] - pd.Timestamp.today().normalize()).dt.days
    d["sell_through_risk_units"] = np.maximum(0, d["current_stock_units"] - d["daily_ros"] * d["days_to_expiry"].clip(lower=0))
    d["expiry_risk_value"] = d["sell_through_risk_units"] * d["unit_cost"]
    d["reorder_point_units"] = np.ceil(d["daily_ros"] * d["lead_time_days"] + d["min_display_qty"])
    d["reorder_status"] = np.where(d["current_stock_units"] <= d["reorder_point_units"], "REORDER", "HOLD")
    d["markdown_pct"] = np.select(
        [
            (d["days_to_expiry"] <= 7) & (d["weeks_cover"] > 1),
            (d["days_to_expiry"] <= 21) & (d["weeks_cover"] > 3),
            (d["weeks_cover"] > 12) & (d["gmroi_30d"] < .25),
            (d["weeks_cover"] > 8) & (d["stock_turn_annualised"] < 4),
        ],
        [35, 25, 20, 10],
        default=0,
    )
    d["action"] = np.select(
        [
            (d["days_to_expiry"] <= 7) & (d["current_stock_units"] > 0),
            d["out_of_display_flag"].eq("YES") & (d["rate_of_sales_units_per_week"] >= 5),
            d["reorder_status"].eq("REORDER") & (d["gmroi_30d"] >= .6),
            (d["weeks_cover"] > 12) & (d["gmroi_30d"] < .25),
            (d["weeks_cover"] > 8) & (d["gmroi_30d"] >= .25),
            (d["stockout_days_30d"] >= 3),
        ],
        ["URGENT MARKDOWN / LIQUIDATE", "FIX OUT OF DISPLAY", "REORDER FAST MOVER", "RATIONALISE / DELIST", "TRANSFER OR PROMOTE", "INCREASE DEPTH / FIX SUPPLY"],
        default="MAINTAIN"
    )
    d["planogram_priority_score"] = (
        d["gmroi_30d"].clip(0, 3) * 35 +
        np.minimum(d["rate_of_sales_units_per_week"] / d["rate_of_sales_units_per_week"].quantile(.95), 1) * 30 +
        (d["gross_margin_pct"] / 50).clip(0, 1) * 20 -
        np.minimum(d["weeks_cover"] / 20, 1) * 15 +
        np.where(d["out_of_display_flag"].eq("YES"), 10, 0)
    ).round(1)
    d["recommended_facings"] = np.select(
        [d["planogram_priority_score"] >= 60, d["planogram_priority_score"] >= 40, d["planogram_priority_score"] >= 25],
        [5, 3, 2], default=1
    )
    d["recommended_shelf_level"] = np.select(
        [d["planogram_priority_score"] >= 60, d["planogram_priority_score"] >= 35],
        ["Eye", "Hand"], default="Bottom"
    )
    return d

def ollama_explain(summary: dict, rows: pd.DataFrame, model: str) -> str:
    prompt = f"""
You are an FMCG retail inventory and visual merchandising expert.
Explain the stock rationalisation output in plain business language.
Be specific and actionable. No generic textbook answer.

Business summary:
{json.dumps(summary, indent=2, default=str)}

Top risky SKUs:
{rows.head(20).to_csv(index=False)}

Give:
1. What is happening.
2. Which categories need attention.
3. Markdown logic.
4. Planogram action.
5. What buyer/store team should do this week.
"""
    try:
        r = requests.post(
            "http://localhost:11434/api/generate",
            json={"model": model, "prompt": prompt, "stream": False},
            timeout=60,
        )
        if r.ok:
            return r.json().get("response", "No response from Ollama.")
        return f"Ollama error: {r.status_code} {r.text[:300]}"
    except Exception as e:
        return f"Ollama is not reachable. Start Ollama locally and pull the selected model. Error: {e}"

st.title("FMCG Stock Rationalisation AI")
st.caption("100,000 SKU ready. Upload retailer data, calculate GMROI, rate of sales, out of display, markdown risk, expiry risk and planogram actions.")

with st.sidebar:
    st.header("Data")
    mode = st.radio("Choose data source", ["Demo FMCG data", "Upload retailer data"])
    n_rows = st.slider("Demo SKU rows", 10_000, 100_000, 100_000, step=10_000)
    seed = st.number_input("Demo seed", min_value=1, value=42)
    st.download_button("Download upload template CSV", make_template_df().to_csv(index=False).encode("utf-8"), "fmcg_upload_template.csv", "text/csv")
    st.download_button("Download upload template Excel", to_excel_bytes(make_template_df()), "fmcg_upload_template.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    st.header("AI")
    model = st.text_input("Ollama model", value="llama3.1:8b")

if mode == "Upload retailer data":
    file = st.file_uploader("Upload CSV or Excel", type=["csv", "xlsx"])
    if not file:
        st.info("Upload retailer data using the template from the sidebar.")
        st.stop()
    if file.name.endswith(".csv"):
        raw = pd.read_csv(file)
    else:
        raw = pd.read_excel(file)
    raw = validate_upload(raw)
else:
    raw = generate_demo_data(n_rows, int(seed))

scored = score_data(raw)

st.sidebar.header("Filters")
cat_filter = st.sidebar.multiselect("Category", sorted(scored["category"].unique()), default=sorted(scored["category"].unique()))
region_filter = st.sidebar.multiselect("Region", sorted(scored["region"].unique()), default=sorted(scored["region"].unique()))
action_filter = st.sidebar.multiselect("Action", sorted(scored["action"].unique()), default=sorted(scored["action"].unique()))
view = scored[scored["category"].isin(cat_filter) & scored["region"].isin(region_filter) & scored["action"].isin(action_filter)].copy()

k1, k2, k3, k4, k5, k6, k7 = st.columns(7)
k1.metric("SKUs", f"{len(view):,}")
k2.metric("Sales 30D", f"₹{view['sales_value_30d'].sum()/1e7:.2f} Cr")
k3.metric("Inventory Cost", f"₹{view['avg_inventory_cost_value'].sum()/1e7:.2f} Cr")
k4.metric("Avg GMROI 30D", f"{view['gmroi_30d'].mean():.2f}")
k5.metric("Expiry Risk", f"₹{view['expiry_risk_value'].sum()/1e7:.2f} Cr")
k6.metric("Avg ROS / Week", f"{view['rate_of_sales_units_per_week'].mean():.1f}")
k7.metric("OOD SKUs", f"{(view['out_of_display_flag'] == 'YES').sum():,}")

tabs = st.tabs(["Executive Dashboard", "OOD & ROS", "Markdown & Expiry", "Planogram", "SKU Actions", "Explainable AI"])

with tabs[0]:
    c1, c2 = st.columns(2)
    cat = view.groupby("category", as_index=False).agg(
        sales=("sales_value_30d", "sum"), inventory=("avg_inventory_cost_value", "sum"), gmroi=("gmroi_30d", "mean"), expiry_risk=("expiry_risk_value", "sum"), skus=("sku_id", "count")
    )
    c1.plotly_chart(px.bar(cat.sort_values("sales", ascending=False), x="category", y="sales", title="Sales by Category"), use_container_width=True)
    c2.plotly_chart(px.scatter(cat, x="inventory", y="gmroi", size="skus", color="category", title="Inventory vs GMROI"), use_container_width=True)
    action = view.groupby("action", as_index=False).agg(skus=("sku_id", "count"), value=("avg_inventory_cost_value", "sum"))
    st.plotly_chart(px.bar(action.sort_values("skus", ascending=False), x="action", y="skus", title="SKU Count by Recommended Action"), use_container_width=True)

with tabs[1]:
    ood = view[view["out_of_display_flag"] == "YES"].sort_values(["rate_of_sales_units_per_week", "out_of_display_units"], ascending=False)
    c1, c2 = st.columns(2)
    ros = view.groupby("category", as_index=False).agg(avg_ros_week=("rate_of_sales_units_per_week", "mean"), sales=("sales_value_30d", "sum"), ood_skus=("out_of_display_flag", lambda x: (x == "YES").sum()))
    c1.plotly_chart(px.bar(ros.sort_values("avg_ros_week", ascending=False), x="category", y="avg_ros_week", title="Average Rate of Sales per Week by Category"), use_container_width=True)
    c2.plotly_chart(px.bar(ros.sort_values("ood_skus", ascending=False), x="category", y="ood_skus", title="Out of Display SKUs by Category"), use_container_width=True)
    st.subheader("Out of Display Action List")
    st.dataframe(ood[["sku_id", "sku_name", "category", "store_id", "current_stock_units", "display_qty_units", "backroom_stock_units", "min_display_qty", "out_of_display_units", "out_of_display_value", "rate_of_sales_units_per_week", "gmroi_30d", "action"]].head(2000), use_container_width=True)

with tabs[2]:
    risk = view[view["markdown_pct"] > 0].sort_values(["days_to_expiry", "weeks_cover"], ascending=[True, False])
    c1, c2 = st.columns(2)
    c1.plotly_chart(px.histogram(view, x="days_to_expiry", nbins=50, title="Days to Expiry Distribution"), use_container_width=True)
    c2.plotly_chart(px.box(view, x="category", y="weeks_cover", title="Weeks of Cover by Category"), use_container_width=True)
    st.subheader("Markdown Candidates")
    st.dataframe(risk[["sku_id", "sku_name", "category", "store_id", "current_stock_units", "days_to_expiry", "weeks_cover", "gmroi_30d", "markdown_pct", "action"]].head(1000), use_container_width=True)

with tabs[3]:
    pl = view.groupby(["category", "fixture_type", "recommended_shelf_level"], as_index=False).agg(
        skus=("sku_id", "count"), avg_score=("planogram_priority_score", "mean"), sales=("sales_value_30d", "sum"), gmroi=("gmroi_30d", "mean")
    )
    st.plotly_chart(px.treemap(pl, path=["category", "fixture_type", "recommended_shelf_level"], values="sales", color="avg_score", title="Planogram Space Priority Map"), use_container_width=True)
    top_plano = view.sort_values("planogram_priority_score", ascending=False).head(500)
    st.dataframe(top_plano[["sku_id", "sku_name", "category", "subcategory", "fixture_type", "shelf_level", "facings", "recommended_facings", "recommended_shelf_level", "planogram_priority_score", "gmroi_30d", "weeks_cover"]], use_container_width=True)

with tabs[4]:
    cols = ["sku_id", "sku_name", "category", "subcategory", "brand", "store_id", "units_sold_30d", "rate_of_sales_units_per_week", "current_stock_units", "display_qty_units", "out_of_display_flag", "out_of_display_units", "weeks_cover", "gmroi_30d", "stock_turn_annualised", "days_to_expiry", "markdown_pct", "reorder_status", "action"]
    st.dataframe(view.sort_values(["action", "gmroi_30d"], ascending=[True, False])[cols].head(5000), use_container_width=True)
    out = io.BytesIO()
    export_cols = cols + ["recommended_facings", "recommended_shelf_level", "expiry_risk_value", "out_of_display_value", "reorder_point_units"]
    view[export_cols].to_excel(out, index=False)
    st.download_button("Download action file", out.getvalue(), "fmcg_stock_rationalisation_actions.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

with tabs[5]:
    st.subheader("Rule-based Explainability")
    st.markdown("""
- **GMROI 30D** = Gross margin value in last 30 days / average inventory cost value.
- **Rate of Sales** = units sold in 30 days / 30, also shown as weekly ROS.
- **Out of Display** = stock exists in the system, but display quantity is below minimum display quantity.
- **Weeks cover** = current stock / average weekly sales rate.
- **Markdown** is triggered when expiry risk or excess weeks cover is high.
- **Rationalise / delist** is triggered when cover is high and GMROI is weak.
- **Planogram priority** rewards GMROI, sales velocity and margin, and penalises excess cover.
""")
    st.subheader("Ollama AI Explanation")
    if st.button("Generate AI explanation with Ollama"):
        summary = {
            "sku_count": len(view),
            "sales_30d": float(view["sales_value_30d"].sum()),
            "inventory_cost": float(view["avg_inventory_cost_value"].sum()),
            "avg_gmroi_30d": float(view["gmroi_30d"].mean()),
            "expiry_risk_value": float(view["expiry_risk_value"].sum()),
            "avg_rate_of_sales_units_per_week": float(view["rate_of_sales_units_per_week"].mean()),
            "out_of_display_sku_count": int((view["out_of_display_flag"] == "YES").sum()),
            "out_of_display_value": float(view["out_of_display_value"].sum()),
            "top_actions": view["action"].value_counts().head(10).to_dict(),
        }
        top_risk = view.sort_values(["expiry_risk_value", "out_of_display_value", "weeks_cover"], ascending=False)[["sku_id", "sku_name", "category", "store_id", "rate_of_sales_units_per_week", "out_of_display_flag", "out_of_display_units", "weeks_cover", "gmroi_30d", "days_to_expiry", "expiry_risk_value", "action"]]
        st.write(ollama_explain(summary, top_risk, model))
