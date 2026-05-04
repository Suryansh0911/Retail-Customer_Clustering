import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import datetime as dt
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.neighbors import NearestNeighbors
from scipy.cluster.hierarchy import dendrogram, linkage

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="Retail Customer Segmentation", layout="wide")

# ── Sample template ───────────────────────────────────────────────────────────
sample_data = pd.DataFrame({
    'InvoiceNo':   ['536365', '536366'],
    'StockCode':   ['85123A', '22633'],
    'Description': ['WHITE HANGING HEART T-LIGHT HOLDER', 'HAND WARMER UNION JACK'],
    'Quantity':    [6, 12],
    'InvoiceDate': ['2010-12-01 08:26:00', '2010-12-01 08:28:00'],
    'UnitPrice':   [2.55, 1.85],
    'CustomerID':  [17850, 17850],
    'Country':     ['United Kingdom', 'United Kingdom']
})
csv_sample = sample_data.to_csv(index=False).encode('utf-8')

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.download_button(
    label="📥 Download Sample Template",
    data=csv_sample,
    file_name="retail_sample_template.csv",
    mime="text/csv",
)

st.sidebar.header("1. Upload Data")
uploaded_file = st.sidebar.file_uploader(
    "Upload Online Retail CSV or XLSX", type=['csv', 'xlsx']
)

# ── Title ─────────────────────────────────────────────────────────────────────
st.title("🛍️ Retail Store Customer Clustering System")
st.markdown("""
Segment your customers based on **Recency, Frequency, and Monetary (RFM)** patterns using
three unsupervised ML models: **K-Means · DBSCAN · Agglomerative Hierarchical Clustering**
""")

# ── Main logic ────────────────────────────────────────────────────────────────
if uploaded_file:
    try:
        # ── Section 2: Load & Clean ───────────────────────────────────────────
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file, encoding='unicode_escape')
        else:
            df = pd.read_excel(uploaded_file)

        with st.spinner("Cleaning data..."):
            df = df.dropna(subset=['CustomerID'])
            df['CustomerID'] = df['CustomerID'].astype(int)
            df = df[~df['InvoiceNo'].astype(str).str.startswith('C')]
            df = df[(df['Quantity'] > 0) & (df['UnitPrice'] > 0)]

        st.success(f"✅ Clean dataset: {df.shape[0]:,} rows | {df['CustomerID'].nunique():,} unique customers")

        # ── Section 3: RFM Engineering ────────────────────────────────────────
        with st.spinner("Calculating RFM metrics..."):
            df['TotalSum'] = df['Quantity'] * df['UnitPrice']
            df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'], dayfirst=True)
            snapshot_date = df['InvoiceDate'].max() + dt.timedelta(days=1)

            rfm = df.groupby('CustomerID').agg(
                Recency   = ('InvoiceDate', lambda x: (snapshot_date - x.max()).days),
                Frequency = ('InvoiceNo',   'count'),
                Monetary  = ('TotalSum',    'sum')
            ).reset_index()

        # ── Section 4: Log Transform + Scaling + PCA ─────────────────────────
        with st.spinner("Preprocessing: log transform → scaling → PCA..."):
            rfm_log = np.log1p(rfm[['Recency', 'Frequency', 'Monetary']])

            scaler = StandardScaler()
            rfm_scaled = scaler.fit_transform(rfm_log)
            rfm_scaled_df = pd.DataFrame(
                rfm_scaled, columns=['Recency', 'Frequency', 'Monetary'], index=rfm.index
            )

            pca = PCA(n_components=2, random_state=42)
            rfm_pca = pca.fit_transform(rfm_scaled_df)
            pca_df = pd.DataFrame(rfm_pca, columns=['PC1', 'PC2'], index=rfm.index)

        st.divider()

        # ══════════════════════════════════════════════════════════════════════
        # SECTION 5 — K-MEANS
        # ══════════════════════════════════════════════════════════════════════
        st.header("📌 Model 1: K-Means Clustering")

        with st.spinner("Running K-Means..."):
            inertia, silhouette_scores_km = [], []
            k_range = range(2, 11)
            for k in k_range:
                km = KMeans(n_clusters=k, init='k-means++', random_state=42, n_init=10)
                km.fit(rfm_scaled_df)
                inertia.append(km.inertia_)
                silhouette_scores_km.append(silhouette_score(rfm_scaled_df, km.labels_))

            kmeans = KMeans(n_clusters=3, init='k-means++', random_state=42, n_init=10)
            kmeans_labels = kmeans.fit_predict(rfm_scaled_df)
            rfm['KMeans_Cluster'] = kmeans_labels
            pca_df['KMeans_Cluster'] = kmeans_labels
            rfm_scaled_df['KMeans_Cluster'] = kmeans_labels

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Elbow Method & Silhouette Score")
            fig, axes = plt.subplots(1, 2, figsize=(12, 4))
            axes[0].plot(k_range, inertia, marker='o', color='steelblue', linewidth=2)
            axes[0].axvline(x=3, color='red', linestyle='--', alpha=0.7, label='K=3 (chosen)')
            axes[0].set_xlabel('Number of Clusters (K)')
            axes[0].set_ylabel('Inertia')
            axes[0].set_title('Elbow Method')
            axes[0].legend()
            axes[1].plot(list(k_range), silhouette_scores_km, marker='s', color='darkorange', linewidth=2)
            axes[1].axvline(x=3, color='red', linestyle='--', alpha=0.7, label='K=3 (chosen)')
            axes[1].set_xlabel('Number of Clusters (K)')
            axes[1].set_ylabel('Silhouette Score')
            axes[1].set_title('Silhouette Score vs K')
            axes[1].legend()
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        with col2:
            st.subheader("Customer Segments in PCA Space")
            fig, ax = plt.subplots(figsize=(8, 5))
            sns.scatterplot(x='PC1', y='PC2', hue='KMeans_Cluster',
                            data=pca_df, palette='viridis', s=50, alpha=0.7, ax=ax)
            ax.set_title('K-Means: Customer Segments in PCA Space')
            st.pyplot(fig)
            plt.close()

        col3, col4 = st.columns(2)

        with col3:
            st.subheader("Cluster Profiles")
            km_profile = rfm.groupby('KMeans_Cluster').agg(
                Count=('Monetary', 'count'),
                Avg_Recency=('Recency', 'mean'),
                Avg_Frequency=('Frequency', 'mean'),
                Avg_Monetary=('Monetary', 'mean')
            ).round(2)
            st.dataframe(km_profile, use_container_width=True)

        with col4:
            st.subheader("Snake Plot — RFM Profile per Cluster")
            rfm_melt_km = pd.melt(
                rfm_scaled_df.reset_index(),
                id_vars=['index', 'KMeans_Cluster'],
                value_vars=['Recency', 'Frequency', 'Monetary'],
                var_name='Metric', value_name='Value'
            )
            fig, ax = plt.subplots(figsize=(8, 4))
            sns.lineplot(x='Metric', y='Value', hue='KMeans_Cluster',
                         data=rfm_melt_km, palette='viridis', linewidth=2.5, ax=ax)
            ax.axhline(0, color='grey', linestyle='--', alpha=0.5)
            ax.set_title('K-Means Snake Plot')
            st.pyplot(fig)
            plt.close()

        st.divider()

        # ══════════════════════════════════════════════════════════════════════
        # SECTION 6 — DBSCAN
        # ══════════════════════════════════════════════════════════════════════
        st.header("📌 Model 2: DBSCAN Clustering")

        with st.spinner("Running DBSCAN..."):
            min_samples = 2 * rfm_scaled_df.shape[1] - 1  # exclude KMeans_Cluster col
            min_samples = 6
            nbrs = NearestNeighbors(n_neighbors=min_samples).fit(
                rfm_scaled_df[['Recency', 'Frequency', 'Monetary']]
            )
            distances, _ = nbrs.kneighbors(rfm_scaled_df[['Recency', 'Frequency', 'Monetary']])
            k_distances = np.sort(distances[:, min_samples - 1])

            dbscan = DBSCAN(eps=0.8, min_samples=6)
            dbscan_labels = dbscan.fit_predict(rfm_scaled_df[['Recency', 'Frequency', 'Monetary']])
            rfm['DBSCAN_Cluster'] = dbscan_labels
            pca_df['DBSCAN_Cluster'] = dbscan_labels

            n_clusters_db = len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)
            n_noise_db    = list(dbscan_labels).count(-1)
            n_noise_pct   = n_noise_db / len(dbscan_labels) * 100

        col5, col6 = st.columns(2)

        with col5:
            st.subheader("k-NN Distance Plot (eps selection)")
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.plot(k_distances, color='steelblue', linewidth=1.5)
            ax.axhline(y=0.8, color='red', linestyle='--', label='eps ≈ 0.8 (chosen)')
            ax.set_xlabel('Points (sorted by distance)')
            ax.set_ylabel(f'{min_samples}-NN Distance')
            ax.set_title('k-NN Distance Plot — Finding Optimal eps')
            ax.legend()
            st.pyplot(fig)
            plt.close()

        with col6:
            st.subheader("DBSCAN Segments in PCA Space")
            fig, ax = plt.subplots(figsize=(8, 5))
            palette = {-1: '#ff4444'}
            unique_labels = sorted(set(dbscan_labels))
            for lbl in unique_labels:
                if lbl != -1:
                    palette[lbl] = cm.viridis(lbl / max(1, n_clusters_db - 1))
            sns.scatterplot(x='PC1', y='PC2', hue='DBSCAN_Cluster',
                            data=pca_df, palette=palette, s=40, alpha=0.6, ax=ax)
            ax.set_title(f'DBSCAN: {n_clusters_db} Clusters + {n_noise_db} Noise Points (red)')
            ax.legend(title='Cluster', bbox_to_anchor=(1.01, 1), loc='upper left')
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        st.metric("Clusters Found", n_clusters_db)
        col_n1, col_n2 = st.columns(2)
        with col_n1:
            st.metric("Noise / Outlier Points", f"{n_noise_db:,} ({n_noise_pct:.1f}%)")
        with col_n2:
            st.subheader("Outlier Customer Profile")
            noise_customers = rfm[rfm['DBSCAN_Cluster'] == -1][['Recency', 'Frequency', 'Monetary']]
            st.dataframe(noise_customers.describe().round(2), use_container_width=True)

        st.divider()

        # ══════════════════════════════════════════════════════════════════════
        # SECTION 7 — AGGLOMERATIVE HIERARCHICAL
        # ══════════════════════════════════════════════════════════════════════
        st.header("📌 Model 3: Agglomerative Hierarchical Clustering")

        with st.spinner("Running Agglomerative Clustering..."):
            agg = AgglomerativeClustering(n_clusters=3, linkage='ward')
            agg_labels = agg.fit_predict(rfm_scaled_df[['Recency', 'Frequency', 'Monetary']])
            rfm['Agg_Cluster'] = agg_labels
            pca_df['Agg_Cluster'] = agg_labels

            # Dendrogram on a sample of 300
            np.random.seed(42)
            sample_idx = np.random.choice(
                len(rfm_scaled_df), size=min(300, len(rfm_scaled_df)), replace=False
            )
            rfm_sample = rfm_scaled_df[['Recency', 'Frequency', 'Monetary']].iloc[sample_idx]
            linked = linkage(rfm_sample, method='ward')

        col7, col8 = st.columns(2)

        with col7:
            st.subheader("Dendrogram (Sample of 300 Customers)")
            fig, ax = plt.subplots(figsize=(10, 5))
            dendrogram(linked, orientation='top', distance_sort='descending',
                       show_leaf_counts=False, no_labels=True,
                       color_threshold=8, above_threshold_color='grey', ax=ax)
            ax.axhline(y=8, color='red', linestyle='--', linewidth=1.5,
                       label='Cut height = 8 (→ 3 clusters)')
            ax.set_title('Dendrogram — Ward Linkage')
            ax.set_xlabel('Customer Index')
            ax.set_ylabel('Ward Linkage Distance')
            ax.legend()
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        with col8:
            st.subheader("Agglomerative Segments in PCA Space")
            fig, ax = plt.subplots(figsize=(8, 5))
            sns.scatterplot(x='PC1', y='PC2', hue='Agg_Cluster',
                            data=pca_df, palette='plasma', s=50, alpha=0.7, ax=ax)
            ax.set_title('Agglomerative: Customer Segments in PCA Space')
            ax.legend(title='Cluster')
            st.pyplot(fig)
            plt.close()

        st.subheader("Agglomerative Cluster Profiles")
        agg_profile = rfm.groupby('Agg_Cluster').agg(
            Count=('Monetary', 'count'),
            Avg_Recency=('Recency', 'mean'),
            Avg_Frequency=('Frequency', 'mean'),
            Avg_Monetary=('Monetary', 'mean')
        ).round(2)
        st.dataframe(agg_profile, use_container_width=True)

        st.divider()

        # ══════════════════════════════════════════════════════════════════════
        # SECTION 8 — MODEL COMPARISON
        # ══════════════════════════════════════════════════════════════════════
        st.header("📊 Model Comparison & Evaluation")

        results = {}

        results['K-Means'] = {
            'Silhouette':        round(silhouette_score(rfm_scaled_df[['Recency','Frequency','Monetary']], kmeans_labels), 4),
            'Davies-Bouldin':    round(davies_bouldin_score(rfm_scaled_df[['Recency','Frequency','Monetary']], kmeans_labels), 4),
            'Calinski-Harabasz': round(calinski_harabasz_score(rfm_scaled_df[['Recency','Frequency','Monetary']], kmeans_labels), 1),
            'N Clusters': len(set(kmeans_labels)),
            'Notes': 'Baseline model'
        }

        dbscan_valid = len(set(dbscan_labels) - {-1}) >= 2
        if dbscan_valid:
            mask = dbscan_labels != -1
            results['DBSCAN'] = {
                'Silhouette':        round(silhouette_score(rfm_scaled_df[['Recency','Frequency','Monetary']][mask], dbscan_labels[mask]), 4),
                'Davies-Bouldin':    round(davies_bouldin_score(rfm_scaled_df[['Recency','Frequency','Monetary']][mask], dbscan_labels[mask]), 4),
                'Calinski-Harabasz': round(calinski_harabasz_score(rfm_scaled_df[['Recency','Frequency','Monetary']][mask], dbscan_labels[mask]), 1),
                'N Clusters': len(set(dbscan_labels) - {-1}),
                'Notes': f'{n_noise_db} noise/outlier points'
            }
        else:
            results['DBSCAN'] = {
                'Silhouette': 'N/A', 'Davies-Bouldin': 'N/A', 'Calinski-Harabasz': 'N/A',
                'N Clusters': len(set(dbscan_labels) - {-1}),
                'Notes': 'Tune eps — try lower value for more clusters'
            }

        results['Agglomerative'] = {
            'Silhouette':        round(silhouette_score(rfm_scaled_df[['Recency','Frequency','Monetary']], agg_labels), 4),
            'Davies-Bouldin':    round(davies_bouldin_score(rfm_scaled_df[['Recency','Frequency','Monetary']], agg_labels), 4),
            'Calinski-Harabasz': round(calinski_harabasz_score(rfm_scaled_df[['Recency','Frequency','Monetary']], agg_labels), 1),
            'N Clusters': len(set(agg_labels)),
            'Notes': 'Ward linkage'
        }

        comparison_df = pd.DataFrame(results).T
        st.dataframe(comparison_df, use_container_width=True)
        st.caption("Silhouette ↑ Higher is better · Davies-Bouldin ↓ Lower is better · Calinski-Harabasz ↑ Higher is better")

        # Visual comparison
        numeric_cols = ['Silhouette', 'Davies-Bouldin', 'Calinski-Harabasz']
        plot_df = comparison_df[numeric_cols].copy()
        plot_df = plot_df[plot_df['Silhouette'] != 'N/A'].astype(float)

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        colors = ['#4c72b0', '#dd8452', '#55a868', '#c44e52']
        for ax, metric, better in zip(axes, numeric_cols,
                                       ['↑ Higher Better', '↓ Lower Better', '↑ Higher Better']):
            vals = plot_df[metric]
            bars = ax.bar(vals.index, vals.values, color=colors[:len(vals)],
                          edgecolor='white', linewidth=0.8)
            ax.set_title(f'{metric}\n({better})', fontsize=11)
            ax.set_ylabel(metric)
            ax.tick_params(axis='x', rotation=15)
            for bar, val in zip(bars, vals.values):
                ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.01,
                        f'{val:.3f}', ha='center', va='bottom', fontsize=9)
        plt.suptitle('Model Performance Comparison', fontsize=14, fontweight='bold')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        st.divider()

        # ══════════════════════════════════════════════════════════════════════
        # SECTION 9 — FINAL BUSINESS SEGMENTATION
        # ══════════════════════════════════════════════════════════════════════
        st.header("🏷️ Final Business Segmentation (K-Means)")

        profile = rfm.groupby('KMeans_Cluster')[['Recency', 'Frequency', 'Monetary']].mean()
        champion_cluster = profile['Monetary'].idxmax()
        atrisk_cluster   = profile['Recency'].idxmax()
        loyalist_cluster = [c for c in [0, 1, 2]
                            if c not in [champion_cluster, atrisk_cluster]][0]

        label_map = {
            champion_cluster: '🏆 Champions',
            loyalist_cluster: '💛 Loyalists',
            atrisk_cluster:   '⚠️  At-Risk'
        }
        rfm['Segment'] = rfm['KMeans_Cluster'].map(label_map)
        pca_df['Segment'] = rfm['Segment'].values

        segment_summary = rfm.groupby('Segment').agg(
            Customer_Count=('Monetary', 'count'),
            Avg_Recency_Days=('Recency', 'mean'),
            Avg_Orders=('Frequency', 'mean'),
            Avg_Revenue_GBP=('Monetary', 'mean'),
            Total_Revenue_GBP=('Monetary', 'sum')
        ).round(2)
        segment_summary['% of Base'] = (
            segment_summary['Customer_Count'] / len(rfm) * 100
        ).round(1)

        st.dataframe(segment_summary, use_container_width=True)

        palette_seg = {
            '🏆 Champions': '#2ecc71',
            '💛 Loyalists': '#f39c12',
            '⚠️  At-Risk':  '#e74c3c'
        }
        fig, axes = plt.subplots(1, 3, figsize=(16, 6))

        sns.scatterplot(x='PC1', y='PC2', hue='Segment', data=pca_df,
                        palette=palette_seg, s=50, alpha=0.7, ax=axes[0])
        axes[0].set_title('Final Segments in PCA Space')
        axes[0].legend(fontsize=8)

        counts = segment_summary['Customer_Count']
        axes[1].pie(counts, labels=counts.index, autopct='%1.1f%%',
                    colors=['#2ecc71', '#f39c12', '#e74c3c'],
                    startangle=140, wedgeprops={'edgecolor': 'white', 'linewidth': 2})
        axes[1].set_title('Customer Distribution by Segment')

        rev = segment_summary['Total_Revenue_GBP']
        bars = axes[2].bar(rev.index, rev.values / 1e6,
                           color=['#2ecc71', '#f39c12', '#e74c3c'], edgecolor='white')
        axes[2].set_ylabel('Total Revenue (£M)')
        axes[2].set_title('Revenue Contribution by Segment')
        axes[2].tick_params(axis='x', rotation=10)
        for bar in bars:
            axes[2].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.001,
                         f'£{bar.get_height():.2f}M', ha='center', va='bottom', fontsize=9)

        plt.suptitle('Customer Segmentation — Final Business View',
                     fontsize=14, fontweight='bold')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        with st.expander("View Full Segmented Customer List"):
            st.dataframe(
                rfm[['CustomerID', 'Recency', 'Frequency', 'Monetary',
                      'KMeans_Cluster', 'DBSCAN_Cluster', 'Agg_Cluster', 'Segment']].head(100),
                use_container_width=True
            )

    except Exception as e:
        st.error(f"An error occurred: {e}")
        st.info("Check your dataset has the correct columns: "
                "InvoiceNo, StockCode, Quantity, InvoiceDate, UnitPrice, CustomerID.")

else:
    st.info("Please upload a dataset to begin the analysis.")