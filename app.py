import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import joblib
import io

# Gaya sidebar
st.markdown("""
    <style>
        section[data-testid="stSidebar"] {
            background-color: #F0F8FF !important;
            color: black !important;
        }
        section[data-testid="stSidebar"] * {
            color: black !important;
        }
    </style>
""", unsafe_allow_html=True)

sns.set_style("whitegrid")

# ------------------ SIDEBAR ------------------ #
st.sidebar.title("👨‍💻 Dashboard Performa Maskapai")
menu = st.sidebar.radio("Pilih Menu", ["📊 Visualisasi", "📄 Tabel Detail Data"])
uploaded_file = st.sidebar.file_uploader("🗂️ Upload file CSV", type=["csv"])

# ------------------ MUAT MODEL ------------------ #
# Cek jika model KMeans sudah disimpan sebelumnya
try:
    kmeans = joblib.load('kmeans_model.pkl')  # Memuat model yang sudah disimpan
    st.session_state['model_loaded'] = True
    st.warning("Model Berhasil Dimuat.")
except:
    st.session_state['model_loaded'] = False
    st.warning("Model KMeans belum dimuat. Pastikan model sudah disimpan.")

# Load file jika ada
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    # Grouping berdasarkan OPERATOR, menjumlahkan nilai numerik kecuali BULAN
    if 'BULAN' in df.columns:
    # Buang kolom BULAN sebelum pengelompokan
        df_grouped = df.drop(columns=['BULAN','NO'], errors='ignore').groupby('OPERATOR', as_index=False).sum(numeric_only=True)
    else:
        df_grouped = df.groupby('OPERATOR', as_index=False).sum(numeric_only=True)

    # Hitung kolom persentase
    df_grouped['PERCENT_ON_TIME'] = (df_grouped['ON TIME'] / df_grouped['REAL']) * 100
    df_grouped['PERCENT_DELAY'] = (df_grouped['DELAY'] / df_grouped['REAL']) * 100
    df_grouped['PERCENT_CANCEL'] = (df_grouped['CANCEL'] / df_grouped['REAL']) * 100

    # Ganti inf dan -inf menjadi NaN
    df_grouped[['PERCENT_ON_TIME', 'PERCENT_DELAY', 'PERCENT_CANCEL']] = df_grouped[[
        'PERCENT_ON_TIME', 'PERCENT_DELAY', 'PERCENT_CANCEL'
    ]].replace([np.inf, -np.inf], np.nan)

    # Imputasi NaN
    imputer = SimpleImputer(strategy='mean')
    features = df_grouped[['PERCENT_ON_TIME', 'PERCENT_DELAY', 'PERCENT_CANCEL']]
    features_imputed = imputer.fit_transform(features)

    # Scaling
    scaler = MinMaxScaler()
    features_scaled = scaler.fit_transform(features_imputed)

    # Jika model sudah diload, gunakan model tersebut untuk mengklaster
    if st.session_state['model_loaded']:
        df_grouped['Cluster'] = kmeans.predict(features_scaled)  # Gunakan model yang sudah ada untuk mengklaster
        silhouette_avg = silhouette_score(features_scaled, df_grouped['Cluster'])
        st.session_state['data'] = df_grouped
    else:
        # Clustering menggunakan model baru jika model belum ada
        kmeans = KMeans(n_clusters=3, random_state=0, n_init=10)
        df_grouped['Cluster'] = kmeans.fit_predict(features_scaled)
        silhouette_avg = silhouette_score(features_scaled, df_grouped['Cluster'])
        joblib.dump(kmeans, 'kmeans_model.pkl')  # Simpan model yang sudah dilatih
        st.session_state['data'] = df_grouped

    # Penentuan kategori berdasarkan rata-rata performa cluster
    cluster_means = df_grouped.groupby('Cluster')[['PERCENT_ON_TIME']].mean()
    cluster_order = cluster_means['PERCENT_ON_TIME'].sort_values(ascending=False).index.tolist()

    kategori_mapping = {}
    for i, cluster_id in enumerate(cluster_order):
        if i == 0:
            kategori_mapping[cluster_id] = 'BAIK'
        elif i == len(cluster_order) - 1:
            kategori_mapping[cluster_id] = 'BURUK'
        else:
            kategori_mapping[cluster_id] = 'SEDANG'

    df_grouped['Kategori'] = df_grouped['Cluster'].map(kategori_mapping)

else:
    df_grouped = st.session_state.get('data', None)

# ------------------ MENU 1 ------------------ #
if menu == "📊 Visualisasi":
    st.title("📊 Visualisasi Performa Maskapai")

    if df_grouped is not None:
        if 'Cluster' in df_grouped.columns and 'PERCENT_ON_TIME' in df_grouped.columns and 'PERCENT_DELAY' in df_grouped.columns:
            st.markdown("### 🔍 Persebaran Cluster Maskapai")
            plt.figure(figsize=(8, 6))
            sns.scatterplot(
                data=df_grouped,
                x='PERCENT_ON_TIME',
                y='PERCENT_DELAY',
                hue='Cluster',
                palette='Set1',
                s=100,
                alpha=0.7,
                legend='full'
            )
            plt.xlabel("ON TIME")
            plt.ylabel("DELAY")
            plt.title("Distribusi Maskapai Berdasarkan Cluster")
            plt.legend(title="Cluster", loc='upper right')
            st.pyplot(plt.gcf())

            st.markdown(f"**ℹ️ Silhouette Score untuk evaluasi cluster: {silhouette_avg:.3f}**")

        else:
            st.warning("Kolom 'Cluster', 'PERCENT_ON_TIME', atau 'PERCENT_DELAY' tidak ditemukan.")

        if 'Kategori' in df_grouped.columns:
            if 'Cluster' in df_grouped.columns:
                st.markdown("### 📊 Jumlah Maskapai Berdasarkan Cluster dan Kategori")
                cluster_kategori = df_grouped.groupby(['Cluster', 'Kategori']).size().reset_index(name='Jumlah')
                st.dataframe(cluster_kategori)

            kategori_counts = df_grouped['Kategori'].value_counts().reset_index()
            kategori_counts.columns = ['Kategori', 'Jumlah']

            plt.figure(figsize=(8, 5))
            sns.barplot(data=kategori_counts, x='Kategori', y='Jumlah', palette='Set2')
            plt.title("Total Maskapai Berdasarkan Kategori Performa", fontsize=14)
            st.pyplot(plt.gcf())

            if 'OPERATOR' in df_grouped.columns:
                st.markdown("---")
                st.markdown("### 🏆 Ranking Operator Maskapai")
                col1, col2 = st.columns(2)

                with col1:
                    st.subheader("✅ Kategori 'BAIK'")
                    top_baik = df_grouped[df_grouped['Kategori'] == 'BAIK']['OPERATOR'].value_counts().head(5).reset_index()
                    top_baik.columns = ['Operator', 'Jumlah']
                    plt.figure(figsize=(5, 4))
                    ax = sns.barplot(data=top_baik, y='Operator', x='Jumlah', palette='Blues_r')
                    plt.title("5 Operator Terbaik", fontsize=12)
                    for i, v in enumerate(top_baik['Jumlah']):
                        ax.text(v + 0.3, i, str(v), color='black', va='center')
                    st.pyplot(plt.gcf())

                with col2:
                    st.subheader("❌ Kategori 'BURUK'")
                    top_buruk = df_grouped[df_grouped['Kategori'] == 'BURUK']['OPERATOR'].value_counts().head(5).reset_index()
                    top_buruk.columns = ['Operator', 'Jumlah']
                    plt.figure(figsize=(5, 4))
                    ax = sns.barplot(data=top_buruk, y='Operator', x='Jumlah', palette='Reds_r')
                    plt.title("5 Operator Terburuk", fontsize=12)
                    for i, v in enumerate(top_buruk['Jumlah']):
                        ax.text(v + 0.3, i, str(v), color='black', va='center')
                    st.pyplot(plt.gcf())
        else:
            st.warning("Kolom 'Kategori' tidak tersedia.")
    else:
        st.info("📤 Silakan upload file CSV terlebih dahulu.")

# ------------------ MENU 2 ------------------ #
elif menu == "📄 Tabel Detail Data":
    st.title("📄 Tabel Detail Data Maskapai")
    

    # Tombol download untuk seluruh data yang diupload
    if df_grouped is not None:
        df_uploaded_copy = df_grouped.copy()
        csv_uploaded = df_uploaded_copy.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Hasil Clustering",
            data=csv_uploaded,
            file_name="data_clustering.csv",
            mime='text/csv'
        )


    if df_grouped is not None and 'Kategori' in df_grouped.columns and 'OPERATOR' in df_grouped.columns:
        # Hapus kolom 'BULAN' jika ada
        if 'BULAN' in df_grouped.columns:
            df_grouped = df_grouped.drop(columns=['BULAN'])

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### ✅ 5 Operator Kategori BAIK")
            detail_baik = df_grouped[df_grouped['Kategori'] == 'BAIK']['OPERATOR'].value_counts().head(5).reset_index()
            detail_baik.columns = ['Operator', 'Jumlah']
            detail_baik_data = df_grouped[df_grouped['OPERATOR'].isin(detail_baik['Operator']) & (df_grouped['Kategori'] == 'BAIK')]
            st.dataframe(detail_baik_data.reset_index(drop=True))

        with col2:
            st.markdown("#### ❌ 5 Operator Kategori BURUK")
            detail_buruk = df_grouped[df_grouped['Kategori'] == 'BURUK']['OPERATOR'].value_counts().head(5).reset_index()
            detail_buruk.columns = ['Operator', 'Jumlah']
            detail_buruk_data = df_grouped[df_grouped['OPERATOR'].isin(detail_buruk['Operator']) & (df_grouped['Kategori'] == 'BURUK')]
            st.dataframe(detail_buruk_data.reset_index(drop=True))

        st.markdown("---")
        st.markdown("### 📦 Hasil Clustering")
        st.dataframe(df_grouped)

        
    else:
        st.warning("Pastikan file CSV memiliki kolom 'Kategori' dan 'OPERATOR'.")

# ------------------ FOOTER ------------------ #
st.markdown("---")
st.markdown("*Website ini dibuat untuk membantu memvisualisasikan performa maskapai berdasarkan data yang diupload.*")
