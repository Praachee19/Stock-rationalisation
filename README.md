# FMCG Stock Rationalisation AI App

Streamlit app for 100,000 SKU FMCG stock rationalisation.

## What it does

- Generates realistic FMCG demo data up to 100,000 SKUs.
- Accepts retailer upload through CSV or Excel.
- Provides downloadable upload template.
- Calculates GMROI, weeks cover, stock turn, expiry risk, markdown percentage and reorder action.
- Calculates Rate of Sales per day and per week.
- Calculates Out of Display using display quantity versus minimum display quantity.
- Creates planogram dashboard and shelf priority recommendations.
- Exports SKU-level action file.
- Uses local Ollama for explainable AI recommendations.

## New metrics added

### Rate of Sales

- `rate_of_sales_units_per_day = units_sold_30d / 30`
- `rate_of_sales_units_per_week = rate_of_sales_units_per_day * 7`

### Out of Display

- `out_of_display_units = max(0, min_display_qty - display_qty_units)`
- `out_of_display_flag = YES when current_stock_units > 0 and display_qty_units < min_display_qty`

This catches the retail problem where stock exists in the system or backroom, but the SKU is not visible enough on shelf.

## How to run

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Ollama setup

```bash
ollama serve
ollama pull llama3.1:8b
```

Then use the Explainable AI tab in the app.
