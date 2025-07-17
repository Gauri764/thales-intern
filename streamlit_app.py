import streamlit as st
import pandas as pd

# Page config
st.set_page_config(page_title="License Management Software", layout="wide")

# Load data
df = pd.read_csv('sample_modified.csv')
df['purchase_date'] = pd.to_datetime(df['purchase_date'])

# --- Session state to toggle filter popup ---
if 'show_filters' not in st.session_state:
    st.session_state.show_filters = False

# Title and subtitle
st.title('License Management Software')
st.info('License Management Software - Thales')

# --- Apply Filters (default or user-selected) ---
# Default selections (if filter not visible)
selected_customers = []
selected_products = []
date_range = (df['purchase_date'].min(), df['purchase_date'].max())

# --- Filter Popup Simulation ---
if st.session_state.show_filters:
    with st.container():
        st.markdown("### 🧰 Apply Filters")
        st.markdown("Use the options below to filter the licensing data:")

        # Customer ID filter
        customer_ids = df['customer_id'].dropna().unique()
        selected_customers = st.multiselect(
            "Select Customer ID",
            options=sorted(customer_ids),
            placeholder="All customers"
        )

        # Product ID filter
        product_ids = df['product_id'].dropna().unique()
        selected_products = st.multiselect(
            "Select Product ID",
            options=sorted(product_ids),
            placeholder="All products"
        )

        # Date Range filter
        min_date = df['purchase_date'].min()
        max_date = df['purchase_date'].max()
        date_range = st.date_input(
            "Select Purchase Date Range",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date,
            key="purchase_date_range"
        )

# --- Apply Filters to DataFrame ---
filtered_df = df.copy()

if selected_customers:
    filtered_df = filtered_df[filtered_df['customer_id'].isin(selected_customers)]

if selected_products:
    filtered_df = filtered_df[filtered_df['product_id'].isin(selected_products)]

if isinstance(date_range, tuple) and len(date_range) == 2:
    start_date, end_date = pd.to_datetime(date_range)
    filtered_df = filtered_df[
        (filtered_df['purchase_date'] >= start_date) &
        (filtered_df['purchase_date'] <= end_date)
    ]

# --- Licensing Data Header and Filter Button ---
col1, col2 = st.columns([8, 2])
with col1:
    st.subheader("📄 Licensing Data")
with col2:
    if st.button("🔍 Filters", use_container_width=True):
        st.session_state.show_filters = not st.session_state.show_filters

# --- Show Filtered Table ---
st.write(f"Showing {len(filtered_df)} rows")
st.dataframe(filtered_df, use_container_width=True)
