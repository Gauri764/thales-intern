import streamlit as st
import pandas as pd

st.set_page_config(page_title="License Management Software", layout="wide")

st.title('License Management Software')
st.info('License Management Software - Thales')

# Load data and parse dates
df = pd.read_csv('synthetic_licensing_dataset.csv')
df['purchase_date'] = pd.to_datetime(df['purchase_date'])

# --- Sidebar Filters ---
st.sidebar.header("Filter Options")

# 1. Customer ID filter
customer_ids = df['customer_id'].dropna().unique()
selected_customers = st.sidebar.multiselect(
    "Select Customer ID",
    options=sorted(customer_ids),
    placeholder="All customers"
)

# 2. Product ID filter
product_ids = df['product_id'].dropna().unique()
selected_products = st.sidebar.multiselect(
    "Select Product ID",
    options=sorted(product_ids),
    placeholder="All products"
)

# 3. Purchase Date Range filter
date_range = st.sidebar.slider(
    "Select Purchase Date Range",
    min_value=min_date,
    max_value=max_date,
    value=(min_date, max_date),
    format="YYYY-MM-DD"
)

# --- Apply Filters Dynamically ---
filtered_df = df.copy()

# Apply filters only if something is selected
if selected_customers:
    filtered_df = filtered_df[filtered_df['customer_id'].isin(selected_customers)]

if selected_products:
    filtered_df = filtered_df[filtered_df['product_id'].isin(selected_products)]

# Ensure date_range is a valid range
if isinstance(date_range, list) and len(date_range) == 2:
    start_date, end_date = pd.to_datetime(date_range)
    filtered_df = filtered_df[
        (filtered_df['purchase_date'] >= start_date) &
        (filtered_df['purchase_date'] <= end_date)
    ]

# --- Display Final Table ---
st.subheader("📄 Licensing Data")
st.write(f"Showing {len(filtered_df)} rows")
st.dataframe(filtered_df, use_container_width=True)
