import streamlit as st
import pandas as pd

st.title('License Management Software')
st.info('License Managament Software Thales')

with st.expander('Data'):
  st.write('Raw Data')
  df = pd.read_csv('synthetic_licensing_dataset.csv')
  st.dataframe(df)

# --- Sidebar Filters ---
st.sidebar.header("Filter Options")

# 1. Company Name filter
customer = df['customer_id'].dropna().unique()
selected_companies = st.sidebar.multiselect("Select Customer ID", customer, default=customer)

# 2. Product Name filter
product = df['product_id'].dropna().unique()
selected_product = st.sidebar.multiselect("Select Product ID", product, default=product)

# 3. Date Range Filter
min_date = df['purchase_date'].min()
max_date = df['purchase_date'].max()
start_date, end_date = st.sidebar.date_input(
    "Select Date Range",
    [min_date, max_date],
    min_value=min_date,
    max_value=max_date
)

# --- Apply Filters ---
filtered_df = df[
    (df['customer_id'].isin(selected_companies)) &
    (df['product_id'].isin(selected_products)) &
    (df['purchase_date'] >= pd.to_datetime(start_date)) &
    (df['purchase_date'] <= pd.to_datetime(end_date))
]

# --- Display ---
st.subheader("🔍 Filtered Data")
st.write(f"Showing {len(filtered_df)} rows")
st.dataframe(filtered_df)
