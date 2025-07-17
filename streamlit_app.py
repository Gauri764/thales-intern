import streamlit as st
import pandas as pd

# Page config
st.set_page_config(page_title="License Management Software", layout="wide")

# Load data
df = pd.read_csv('sample_modified.csv')
df['purchase_date'] = pd.to_datetime(df['purchase_date'])

# --- Initialize session state ---
if 'show_filters' not in st.session_state:
    st.session_state.show_filters = False

if 'selected_customers' not in st.session_state:
    st.session_state.selected_customers = []

if 'selected_products' not in st.session_state:
    st.session_state.selected_products = []

if 'date_range' not in st.session_state:
    st.session_state.date_range = (df['purchase_date'].min(), df['purchase_date'].max())

# --- Title and description ---
st.title('License Management Software')
st.info('License Management Software - Thales')

# --- Licensing Data Header and Filter Button ---
col1, col2 = st.columns([8, 2])
with col1:
    st.subheader("📄 Licensing Data")
with col2:
    if st.button("🔍 Filters", use_container_width=True):
        st.session_state.show_filters = not st.session_state.show_filters

# --- Filter Popup ---
if st.session_state.show_filters:
    with st.container():
        st.markdown("### 🧰 Apply Filters")
        st.markdown("Use the options below to filter the licensing data:")

        # Customer ID filter
        customer_ids = df['customer_id'].dropna().unique()
        selected_customers = st.multiselect(
            "Select Customer ID",
            options=sorted(customer_ids),
            default=st.session_state.selected_customers,
            placeholder="All customers"
        )
        st.session_state.selected_customers = selected_customers

        # Product ID filter
        product_ids = df['product_id'].dropna().unique()
        selected_products = st.multiselect(
            "Select Product ID",
            options=sorted(product_ids),
            default=st.session_state.selected_products,
            placeholder="All products"
        )
        st.session_state.selected_products = selected_products

        # Date Range filter
        min_date = df['purchase_date'].min()
        max_date = df['purchase_date'].max()
        date_range = st.date_input(
            "Select Purchase Date Range",
            value=st.session_state.date_range,
            min_value=min_date,
            max_value=max_date,
            key="purchase_date_range"
        )
        st.session_state.date_range = date_range

# --- Apply Filters ---
filtered_df = df.copy()

# Apply customer filter
if st.session_state.selected_customers:
    filtered_df = filtered_df[filtered_df['customer_id'].isin(st.session_state.selected_customers)]

# Apply product filter
if st.session_state.selected_products:
    filtered_df = filtered_df[filtered_df['product_id'].isin(st.session_state.selected_products)]

# Apply date filter
if isinstance(st.session_state.date_range, tuple) and len(st.session_state.date_range) == 2:
    start_date, end_date = pd.to_datetime(st.session_state.date_range)
    filtered_df = filtered_df[
        (filtered_df['purchase_date'] >= start_date) &
        (filtered_df['purchase_date'] <= end_date)
    ]

# --- Show Filtered Table ---
st.write(f"Showing {len(filtered_df)} rows")
st.dataframe(filtered_df, use_container_width=True)
