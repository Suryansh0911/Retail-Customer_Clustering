import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import datetime as dt

def get_rfm_data(df):
    """
    Transforms raw transaction data into Customer-level RFM metrics.
    """
    
    df = df.dropna(subset=['CustomerID'])
    df['CustomerID'] = df['CustomerID'].astype(int)
    
    
    df = df[~df['InvoiceNo'].astype(str).str.contains('C', na=False)]
    df = df[(df['Quantity'] > 0) & (df['UnitPrice'] > 0)]
    
    
    df['TotalSum'] = df['Quantity'] * df['UnitPrice']
    
    
    snapshot_date = df['InvoiceDate'].max() + dt.timedelta(days=1)
    
    rfm = df.groupby('CustomerID').agg({
        'InvoiceDate': lambda x: (snapshot_date - x.max()).days,
        'InvoiceNo': 'count',
        'TotalSum': 'sum'
    })
    
    rfm.rename(columns={
        'InvoiceDate': 'Recency',
        'InvoiceNo': 'Frequency',
        'TotalSum': 'Total Sum'
    }, inplace=True)
    
    return rfm

def apply_pca_clustering(rfm_df, k=3):
    """
    Handles Log Transformation, Scaling, PCA, and K-Means clustering with NaN safety checks.
    """
    
    rfm_log = np.log1p(rfm_df)
    
    
    rfm_log = rfm_log.replace([np.inf, -np.inf], np.nan).dropna()
    
    
    rfm_df = rfm_df.loc[rfm_log.index].copy()
    
    
    
    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm_log)
    
   
    pca = PCA(n_components=2)
    rfm_pca = pca.fit_transform(rfm_scaled)
    pca_df = pd.DataFrame(rfm_pca, columns=['PC1', 'PC2'], index=rfm_df.index)
    
    
    kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42)
    clusters = kmeans.fit_predict(rfm_pca)
    
    
    rfm_df['Cluster'] = clusters
    pca_df['Cluster'] = clusters
    
    return rfm_df, pca_df

def get_cluster_stats(rfm_df):
    """
    Returns the mean RFM values for each cluster to identify segments.
    """
    return rfm_df.groupby('Cluster').agg({
        'Recency': 'mean',
        'Frequency': 'mean',
        'Total Sum': 'mean'
    }).round(2)