import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Page config
st.set_page_config(page_title="License Management Software", layout="wide")

# Load data
df = pd.read_csv('sample_modified.csv')
df['purchase_date'] = pd.to_datetime(df['purchase_date'])

# --- Initialize session state ---
for key, default in {
    'show_filters': False,
    'selected_customers': [],
    'selected_products': [],
    'date_range': (df['purchase_date'].min(), df['purchase_date'].max()),
    'show_table': True
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# --- Title and description ---
st.title('License Management Software')
st.info('License Management Software - Thales')

# --- Licensing Data Header with Filters ---
col1, col2 = st.columns([8, 2])
with col1:
    st.subheader("Licensing Data")
with col2:
    if st.button("🔍 Filters", use_container_width=True):
        st.session_state.show_filters = not st.session_state.show_filters

# --- Filter Popup ---
if st.session_state.show_filters:
    with st.container():
        st.markdown("### 🧰 Apply Filters")

        selected_customers = st.multiselect(
            "Select Customer ID",
            options=sorted(df['customer_id'].dropna().unique()),
            default=st.session_state.selected_customers,
            placeholder="All customers"
        )
        st.session_state.selected_customers = selected_customers

        selected_products = st.multiselect(
            "Select Product ID",
            options=sorted(df['product_id'].dropna().unique()),
            default=st.session_state.selected_products,
            placeholder="All products"
        )
        st.session_state.selected_products = selected_products

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

if st.session_state.selected_customers:
    filtered_df = filtered_df[filtered_df['customer_id'].isin(st.session_state.selected_customers)]

if st.session_state.selected_products:
    filtered_df = filtered_df[filtered_df['product_id'].isin(st.session_state.selected_products)]

if isinstance(st.session_state.date_range, tuple) and len(st.session_state.date_range) == 2:
    start_date, end_date = pd.to_datetime(st.session_state.date_range)
    filtered_df = filtered_df[
        (filtered_df['purchase_date'] >= start_date) &
        (filtered_df['purchase_date'] <= end_date)
    ]

# --- Show Data Table ---
with st.expander(f"📄 Show Data Table ({len(filtered_df)} rows)", expanded=False):
    st.dataframe(filtered_df, use_container_width=True)

# --- Charts Section ---
st.markdown("## 📊 Analytics")

# 1. Top 10 sold products over the years
product_sales = (
    filtered_df.groupby(['product_id', filtered_df['purchase_date'].dt.year])['licenses_purchased']
    .sum()
    .reset_index()
)
top_10_products = (
    product_sales.groupby('product_id')['licenses_purchased'].sum().nlargest(10).index
)
filtered_product_sales = product_sales[product_sales['product_id'].isin(top_10_products)]

fig1 = px.line(
    filtered_product_sales,
    x='purchase_date',
    y='licenses_purchased',
    color='product_id',
    markers=True,
    title="📈 Top 10 Sold Products Over the Years"
)
st.plotly_chart(fig1, use_container_width=True)

# 2. Pie chart: Top 10 customers and Others
customer_total = filtered_df.groupby('customer_id')['licenses_purchased'].sum()
top_customers = customer_total.nlargest(10)
others_sum = customer_total.sum() - top_customers.sum()

labels = list(top_customers.index) + ['Others']
values = list(top_customers.values) + [others_sum]

fig2 = go.Figure(data=[
    go.Pie(labels=labels, values=values, hole=0.4)
])
fig2.update_layout(title_text="🧑‍💼 Top 10 Customers by License Share")
st.plotly_chart(fig2, use_container_width=True)

# --- Chart 3: Licenses Purchased vs Activated vs Used (Top 10 Customers) ---
license_columns = ['licenses_purchased', 'licenses_activated', 'licenses_used']
if all(col in filtered_df.columns for col in license_columns):
    license_stats = (
        filtered_df.groupby('customer_id')[license_columns]
        .sum()
        .sort_values(by='licenses_purchased', ascending=False)
        .head(10)
    )

    if not license_stats.empty:
        license_stats = license_stats.reset_index().melt(id_vars='customer_id', var_name='License Type', value_name='Count')
        bar_fig = px.bar(
            license_stats,
            x='customer_id',
            y='Count',
            color='License Type',
            barmode='group',
            title='Licenses Purchased vs Activated vs Used (Top 10 Customers)'
        )
        st.plotly_chart(bar_fig, use_container_width=True)
    else:
        st.warning("No data available for top customers' license stats based on the current filters.")
else:
    st.warning("Required columns for license stats not found in dataset.")

