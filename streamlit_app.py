import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import datetime

# --- Page Configuration ---
st.set_page_config(
    page_title="Licensing Analytics Dashboard",
    page_icon="🔑",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Caching Function for Data Loading ---
@st.cache_data
def load_data(filepath):
    """
    Loads and preprocesses the licensing data from a CSV file.
    Caches the data to avoid reloading on every interaction.
    """
    try:
        df = pd.read_csv(filepath)
        # --- Data Cleaning and Feature Engineering ---
        # Convert date columns to datetime objects
        df['purchase_date'] = pd.to_datetime(df['purchase_date'])
        df['activation_date'] = pd.to_datetime(df['activation_date'])
        df['expiration_date'] = pd.to_datetime(df['expiration_date'])

        # Engineer a 'price' column for monetary analysis, as it's not in the original data.
        # We'll create a consistent, random price for each product.
        np.random.seed(42)
        product_prices = pd.DataFrame({
            'product_id': df['product_id'].unique(),
            'price': np.random.randint(50, 500, size=df['product_id'].nunique())
        })
        df = df.merge(product_prices, on='product_id', how='left')
        df['revenue'] = df['licenses_purchased'] * df['price']

        # Determine license status
        today = datetime.now()
        df['status'] = np.where(df['expiration_date'] > today, 'Active', 'Expired')

        return df
    except FileNotFoundError:
        st.error(f"Error: The file '{filepath}' was not found.")
        st.info("Please make sure you have generated the `synthetic_licensing_dataset.csv` file and it is in the same directory as this script.")
        return None

# --- Load Data ---
df = load_data("synthetic_licensing_dataset.csv")

# --- Sidebar Navigation ---
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Dashboard Overview", "Customer RFM Analysis", "Product Insights", "Predictive Analytics"])

st.sidebar.markdown("---")
st.sidebar.info("This dashboard provides analytics on software licensing data, helping to identify trends in sales, customer behavior, and product performance.")

# --- Main Application ---
if df is not None:
    # --- Page 1: Dashboard Overview ---
    if page == "Dashboard Overview":
        st.title("🔑 Licensing Dashboard Overview")
        st.markdown("A high-level view of key business metrics.")

        # Key Performance Indicators (KPIs)
        total_revenue = df['revenue'].sum()
        total_licenses_sold = df['licenses_purchased'].sum()
        unique_customers = df['customer_id'].nunique()
        unique_products = df['product_id'].nunique()

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Revenue", f"${total_revenue:,.0f}")
        col2.metric("Total Licenses Sold", f"{total_licenses_sold:,}")
        col3.metric("Total Customers", f"{unique_customers:,}")
        col4.metric("Total Products", f"{unique_products:,}")

        st.markdown("---")

        # Charts
        col1, col2 = st.columns((2, 1))

        with col1:
            st.subheader("Revenue Over Time")
            revenue_over_time = df.set_index('purchase_date').resample('Y')['revenue'].sum().reset_index()
            fig_rev_time = px.line(revenue_over_time, x='purchase_date', y='revenue', title="Annual Revenue Trend", markers=True)
            fig_rev_time.update_layout(xaxis_title="Year", yaxis_title="Total Revenue")
            st.plotly_chart(fig_rev_time, use_container_width=True)

        with col2:
            st.subheader("License Status")
            status_counts = df['status'].value_counts()
            fig_status_pie = px.pie(status_counts, values=status_counts.values, names=status_counts.index, title="Active vs. Expired Licenses", hole=0.3)
            st.plotly_chart(fig_status_pie, use_container_width=True)


    # --- Page 2: Customer RFM Analysis ---
    elif page == "Customer RFM Analysis":
        st.title("👥 Customer RFM Analysis")
        st.markdown("Segmenting customers based on Recency, Frequency, and Monetary value.")

        # Calculate RFM metrics
        snapshot_date = df['purchase_date'].max() + pd.Timedelta(days=1)
        rfm_df = df.groupby('customer_id').agg({
            'purchase_date': lambda date: (snapshot_date - date.max()).days,
            'customer_id': 'count',
            'revenue': 'sum'
        })
        rfm_df.rename(columns={'purchase_date': 'Recency', 'customer_id': 'Frequency', 'revenue': 'Monetary'}, inplace=True)

        # Create RFM segments
        r_labels = range(4, 0, -1)
        f_labels = range(1, 5)
        m_labels = range(1, 5)
        rfm_df['R_Score'] = pd.qcut(rfm_df['Recency'], 4, labels=r_labels, duplicates='drop')
        rfm_df['F_Score'] = pd.qcut(rfm_df['Frequency'].rank(method='first'), 4, labels=f_labels, duplicates='drop')
        rfm_df['M_Score'] = pd.qcut(rfm_df['Monetary'], 4, labels=m_labels, duplicates='drop')
        
        rfm_df['RFM_Segment'] = rfm_df['R_Score'].astype(str) + rfm_df['F_Score'].astype(str) + rfm_df['M_Score'].astype(str)
        rfm_df['RFM_Score'] = rfm_df[['R_Score', 'F_Score', 'M_Score']].sum(axis=1)

        def assign_segment_name(row):
            if row['RFM_Score'] >= 11:
                return 'Champions'
            elif row['RFM_Score'] >= 9:
                return 'Loyal Customers'
            elif row['RFM_Score'] >= 6:
                return 'Potential Loyalists'
            elif row['RFM_Score'] >= 5:
                return 'At-Risk Customers'
            else:
                return 'Lost Customers'

        rfm_df['Segment_Name'] = rfm_df.apply(assign_segment_name, axis=1)

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("RFM Segment Distribution")
            segment_counts = rfm_df['Segment_Name'].value_counts()
            fig_rfm_bar = px.bar(segment_counts, x=segment_counts.index, y=segment_counts.values, title="Number of Customers by RFM Segment")
            fig_rfm_bar.update_layout(xaxis_title="Segment", yaxis_title="Number of Customers")
            st.plotly_chart(fig_rfm_bar, use_container_width=True)

        with col2:
            st.subheader("Recency vs. Frequency Scatter Plot")
            fig_rfm_scatter = px.scatter(rfm_df, x='Recency', y='Frequency', color='Segment_Name',
                                         title="Customer Segments",
                                         labels={'Recency': 'Recency (Days)', 'Frequency': 'Frequency (Purchases)'})
            st.plotly_chart(fig_rfm_scatter, use_container_width=True)

        st.subheader("Customer Data with RFM Segments")
        st.dataframe(rfm_df.sort_values(by='RFM_Score', ascending=False))


    # --- Page 3: Product Insights ---
    elif page == "Product Insights":
        st.title("📦 Product Insights")
        st.markdown("Analysis of product performance and usage.")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Top 10 Products by Revenue")
            top_products_rev = df.groupby('product_id')['revenue'].sum().nlargest(10).sort_values(ascending=True)
            fig_top_prod_rev = px.bar(top_products_rev, x=top_products_rev.values, y=top_products_rev.index, orientation='h', title="Top 10 Products by Revenue")
            fig_top_prod_rev.update_layout(xaxis_title="Total Revenue", yaxis_title="Product ID")
            st.plotly_chart(fig_top_prod_rev, use_container_width=True)

        with col2:
            st.subheader("Top 10 Products by Licenses Sold")
            top_products_lic = df.groupby('product_id')['licenses_purchased'].sum().nlargest(10).sort_values(ascending=True)
            fig_top_prod_lic = px.bar(top_products_lic, x=top_products_lic.values, y=top_products_lic.index, orientation='h', title="Top 10 Products by Licenses Sold")
            fig_top_prod_lic.update_layout(xaxis_title="Number of Licenses", yaxis_title="Product ID")
            st.plotly_chart(fig_top_prod_lic, use_container_width=True)

        st.subheader("Product Usage Analysis")
        usage_df = df.groupby('product_id').agg({
            'licenses_purchased': 'sum',
            'licenses_activated': 'sum',
            'usage_last_year': 'mean'
        }).reset_index()
        usage_df['activation_rate'] = (usage_df['licenses_activated'] / usage_df['licenses_purchased']) * 100
        usage_df['activation_rate'] = usage_df['activation_rate'].fillna(0)
        
        st.dataframe(usage_df.sort_values(by='activation_rate', ascending=False))


    # --- Page 4: Predictive Analytics ---
    elif page == "Predictive Analytics":
        st.title("🔮 Predictive Analytics")
        st.warning("This section is a placeholder for the machine learning model integration.", icon="⚠️")
        st.markdown("""
        Once the prediction models are built, this is where you will interact with them.
        
        ### Planned Features:
        1.  **Churn Prediction:**
            -   Select a `customer_id`.
            -   The model will predict the likelihood of churn (e.g., not renewing their subscription).
            -   It will also list the key factors influencing this prediction (e.g., low usage, expiring license).

        2.  **Cross-Sell/Up-Sell Recommendations:**
            -   Select a `customer_id`.
            -   The model will recommend other products the customer is likely to be interested in.
            -   Recommendations will be based on their purchase history and the behavior of similar customers.
        
        Below is a conceptual mock-up of how the results might be displayed.
        """)

        st.subheader("Select a Customer to Analyze")
        customer_id_to_predict = st.selectbox("Customer ID", options=df['customer_id'].unique())

        if st.button("Run Predictions"):
            # This is where you would call your trained ML models
            # For now, we'll show mock results
            st.success(f"Running predictions for Customer ID: {customer_id_to_predict}")
            
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Churn Prediction")
                churn_prob = np.random.uniform(0.05, 0.95)
                if churn_prob > 0.6:
                    st.error(f"High Churn Risk: {churn_prob:.1%}", icon="🔥")
                    st.write("**Reason:** Low product usage in the last quarter.")
                else:
                    st.success(f"Low Churn Risk: {churn_prob:.1%}", icon="✅")
                    st.write("**Reason:** Consistent product activation and usage.")

            with col2:
                st.subheader("Cross-Sell Recommendations")
                # Get products the customer already owns
                owned_products = df[df['customer_id'] == customer_id_to_predict]['product_id'].unique()
                # Recommend products they don't own
                all_products = df['product_id'].unique()
                potential_recs = np.setdiff1d(all_products, owned_products)
                
                if len(potential_recs) > 2:
                    recommended_products = np.random.choice(potential_recs, 2, replace=False)
                    st.info(f"Recommended Product 1: **{recommended_products[0]}**", icon="🛍️")
                    st.info(f"Recommended Product 2: **{recommended_products[1]}**", icon="🛍️")
                else:
                    st.write("No new products to recommend at this time.")

