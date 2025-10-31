import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="Smart Supplier Advisor — France (Agro)", layout="wide")
st.title("🇫🇷 Smart Supplier Advisor — Agro‑Food (France)")
st.write("B2B tool to explore French agro‑food suppliers (sample dataset). Scores are simulated — verify before use.")

@st.cache_data
def load_suppliers(path="suppliers_france_agro.csv"):
    df = pd.read_csv(path)
    # Basic cleaning: remove obviously invalid rows
    df = df[df['name'].notna()]
    # Ensure numeric types
    num_cols = ['estimated_price_eur','reliability_score','delivery_time_days','sustainability_score','certification_quality_score']
    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0)
    return df

try:
    suppliers = load_suppliers()
except FileNotFoundError:
    st.error("File suppliers_france_agro.csv not found. Put it in the same folder as app.py.")
    st.stop()

# Sidebar filters
st.sidebar.header("Filters")
categories = ["All"] + sorted(suppliers['product_category'].dropna().unique().tolist())
category = st.sidebar.selectbox("Product category", categories)
min_reliability = st.sidebar.slider("Min reliability", 0, 10, 0)
max_price = st.sidebar.number_input("Max estimated price (EUR)", min_value=0.0, value=100.0)

# Weight sliders
st.subheader("Set criteria importance (0 = low, 10 = high)")
col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    w_price = st.slider("Price", 0, 10, 5, key="w_price")
with col2:
    w_reliability = st.slider("Reliability", 0, 10, 5, key="w_rel")
with col3:
    w_delivery = st.slider("Delivery time", 0, 10, 5, key="w_del")
with col4:
    w_sustainability = st.slider("Sustainability", 0, 10, 5, key="w_sus")
with col5:
    w_cert = st.slider("Certification / Quality", 0, 10, 5, key="w_cert")

st.markdown("---")
st.write("Optional: search by partial supplier name (case insensitive)")
name_search = st.text_input("Supplier name contains:")

# Apply basic filters
df = suppliers.copy()
if category != "All":
    df = df[df['product_category'].str.contains(category, case=False, na=False)]
df = df[df['reliability_score'] >= min_reliability]
df = df[df['estimated_price_eur'] <= max_price]
if name_search.strip():
    df = df[df['name'].str.contains(name_search.strip(), case=False, na=False)]

if df.empty:
    st.info("No suppliers match your filters. Try widening them.")
    st.stop()

# Score computation
# We normalize each column to 0..1 then apply weights. Note: for price and delivery lower is better.
def normalize_series(s, invert=False):
    s = s.astype(float)
    if s.max() == s.min():
        return pd.Series(0.5, index=s.index)  # flat
    if invert:
        # lower is better -> invert after min-max
        norm = (s - s.min()) / (s.max() - s.min())
        return 1 - norm
    else:
        return (s - s.min()) / (s.max() - s.min())

df = df.copy()
df['n_price'] = normalize_series(df['estimated_price_eur'], invert=True)
df['n_reliability'] = normalize_series(df['reliability_score'], invert=False)
df['n_delivery'] = normalize_series(df['delivery_time_days'], invert=True)
df['n_sustainability'] = normalize_series(df['sustainability_score'], invert=False)
df['n_cert'] = normalize_series(df['certification_quality_score'], invert=False)

# Weighted sum
total_weight = w_price + w_reliability + w_delivery + w_sustainability + w_cert
if total_weight == 0:
    st.warning("All weights are zero — please increase at least one weight to compute ranking.")
    st.stop()

df['score'] = (
    df['n_price'] * w_price +
    df['n_reliability'] * w_reliability +
    df['n_delivery'] * w_delivery +
    df['n_sustainability'] * w_sustainability +
    df['n_cert'] * w_cert
) / total_weight

df_sorted = df.sort_values(by='score', ascending=False)

# Display top results
st.subheader("🏆 Top suppliers (by computed score)")
top_n = st.slider("How many top suppliers to show", 3, 20, 10)
st.dataframe(df_sorted.head(top_n)[['name','product_category','estimated_price_eur','reliability_score','delivery_time_days','sustainability_score','certification_quality_score','score']].reset_index(drop=True))

# Simple bar chart of top 10 scores
st.subheader("Score breakdown for top suppliers")
import matplotlib.pyplot as plt

top_plot = df_sorted.head(10).set_index('name')
fig, ax = plt.subplots(figsize=(10,5))
top_plot['score'].plot.bar(ax=ax)
ax.set_ylabel("Aggregated score (0–1 scaled weighted)")
ax.set_xlabel("")
ax.set_ylim(0,1)
plt.xticks(rotation=45, ha='right')
st.pyplot(fig)

# Show full dataset download option
st.markdown("---")
st.markdown("### Dataset")
st.write("You can download the filtered dataset (CSV):")
csv = df_sorted.to_csv(index=False).encode('utf-8')
st.download_button("Download filtered suppliers (CSV)", data=csv, file_name="suppliers_france_agro_filtered.csv", mime='text/csv')

st.info("Reminder: numeric attributes in this sample dataset are simulated. Replace by verified data before operational use.")

