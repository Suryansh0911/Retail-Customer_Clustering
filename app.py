import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from model_utils import get_rfm_data, apply_pca_clustering, get_cluster_stats

sample_data = pd.DataFrame({
    'InvoiceNo': ['536365', '536366'],
    'StockCode': ['85123A', '22633'],
    'Description': ['WHITE HANGING HEART T-LIGHT HOLDER', 'HAND WARMER UNION JACK'],
    'Quantity': [6, 12],
    'InvoiceDate': ['2010-12-01 08:26:00', '2010-12-01 08:28:00'],
    'UnitPrice': [2.55, 1.85],
    'CustomerID': [17850, 17850],
    'Country': ['United Kingdom', 'United Kingdom']
})


csv_sample = sample_data.to_csv(index=False).encode('utf-8')

st.sidebar.download_button(
    label="📥 Download Sample Template",
    data=csv_sample,
    file_name="retail_sample_template.csv",
    mime="text/csv",
)

st.set_page_config(page_title="Retail Customer Segmentation", layout="wide")

st.title("🛍️ Retail Store Customer Clustering System")
st.markdown("""
Segment your customers based on **Recency, Frequency, and Monetary (RFM)** patterns using 
Unsupervised Machine Learning (**K-Means**) and Dimensionality Reduction (**PCA**).
""")


st.sidebar.header("1. Upload Data")
uploaded_file = st.sidebar.file_uploader("Upload Online Retail CSV or XLSX", type=['csv', 'xlsx'])

if uploaded_file:
    try:
        
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file, encoding='unicode_escape')
        else:
            df = pd.read_excel(uploaded_file)
        
        st.success("File uploaded successfully!")
        
        
        with st.spinner("Calculating RFM metrics..."):
            
            df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
            rfm = get_rfm_data(df)
        
        
        with st.spinner("Running PCA and K-Means..."):
            
            rfm_with_clusters, pca_df = apply_pca_clustering(rfm, k=3)
            stats = get_cluster_stats(rfm_with_clusters)

        
        st.divider()
        col1, col2 = st.columns([2, 1])

        with col1:
            st.subheader("📍 Customer Segments in PCA Space")
            fig, ax = plt.subplots(figsize=(10, 7))
            sns.scatterplot(
                x='PC1', y='PC2', hue='Cluster', 
                data=pca_df, palette='viridis', s=60, ax=ax
            )
            plt.title("2D Projection of Customer Behavior")
            st.pyplot(fig)

        with col2:
            st.subheader("📊 Segment Profiles (Averages)")
            st.dataframe(stats, use_container_width=True)
            
            st.info("""
            **Interpreting the Clusters:**
            * **High Spend/Freq:** Your 'Champions' (VIPs).
            * **Mid Range:** Loyal, steady customers.
            * **High Recency:** 'At-Risk' customers who haven't shopped in a while.
            """)

        
        with st.expander("View Segmented Customer List"):
            st.write(rfm_with_clusters.head(100))
            
    except Exception as e:
        st.error(f"An error occurred: {e}")
        st.info("Check if your dataset has the correct columns: InvoiceNo, StockCode, Quantity, InvoiceDate, UnitPrice, CustomerID.")

else:
    st.info("Please upload a dataset to begin the analysis.")