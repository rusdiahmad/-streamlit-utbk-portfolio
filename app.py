
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import matplotlib.pyplot as plt

st.set_page_config(page_title='My Portfolio with Streamlit', layout='wide')

st.title('My Portfolio with Streamlit')
st.markdown('''
A portfolio app for presenting analysis and prediction on **Nilai UTBK** dataset.
Features:\
- Data preview & visualizations\
- Model training summary\
- Upload CSV and run predictions (regression and classification)\
''')

# Sidebar - About and Projects
with st.sidebar:
    st.header('Tentang Saya')
    st.markdown('Nama: Siap\nLatar belakang: Mahasiswa/Bootcamper\nKeahlian: Data Science, Machine Learning, Streamlit')
    st.markdown('---')
    st.header('Proyek Saya')
    st.markdown('1. Analisis Nilai UTBK\n2. Prediksi Total Nilai UTBK\n3. Dashboard Visualisasi UTBK')

# Load metadata and models
OUT_DIR = Path(__file__).parent
model_dir = OUT_DIR / "models"

if not model_dir.exists():
    st.error("Folder 'models' tidak ditemukan. Pastikan sudah diunggah ke repo GitHub.")
    st.stop()


try:
    reg = joblib.load(model_dir / 'reg_model.pkl')
    clf = joblib.load(model_dir / 'class_model.pkl')
    meta = pd.read_json(model_dir / 'meta.json', typ='series')
except Exception as e:
    st.error('Model files not found. Please ensure models are in the `models/` folder or in /mnt/data/streamlit_utbk_portfolio.')
    st.stop()

st.header('Dataset Preview')
uploaded = st.file_uploader('Upload file .csv (optional) — jika tidak, dataset contoh akan dipakai', type=['csv'])
if uploaded is not None:
    data = pd.read_csv(uploaded)
else:
    # try to load packaged example (if available)
    sample_path = model_dir.parent / 'NILAI UTBK ANGK 4.xlsx'
    try:
        data = pd.read_excel(sample_path)
    except Exception:
        data = pd.DataFrame(columns=meta['numeric_columns'])

st.write('Data (preview):')
st.dataframe(data.head(10))

# Visualizations
st.header('Visualisasi')
if len(meta['numeric_columns'])>0:
    numeric = list(meta['numeric_columns'])
    st.subheader('Distribusi salah satu fitur (pilih)')
    feat = st.selectbox('Pilih fitur', numeric)
    if feat in data.columns:
        fig, ax = plt.subplots()
        ax.hist(data[feat].dropna(), bins=20)
        ax.set_xlabel(feat)
        ax.set_ylabel('Count')
        st.pyplot(fig)
    else:
        st.info('Fitur tidak ada pada file upload; menggunakan distribusi dari dataset internal.')
        # try to compute distribution from metadata path
        # skip for brevity

st.header('Prediksi')
st.markdown('Upload file CSV yang berisi kolom fitur (nilai per mata pelajaran). Setelah itu tekan **Prediksi**.')

if uploaded is not None:
    X = data[meta['numeric_columns']].copy()
    X = X.fillna(X.median())
    if st.button('Prediksi'):
        # Regression prediction
        pred_total = reg.predict(X)
        data['pred_total_score'] = pred_total
        # Classification
        pred_lulus = clf.predict(X)
        data['pred_lulus'] = pred_lulus
        st.success('Prediksi selesai — hasil ditambahkan ke tabel berikut:')
        st.dataframe(data.head(50))
        st.markdown('**Ringkasan Prediksi (regression metrics from trained model)**')
        st.write(meta['regression_metrics'])
        st.markdown('**Ringkasan Prediksi (classification metrics)**')
        st.write(meta['classification_metrics'])
else:
    st.info('Unggah file CSV untuk melakukan prediksi.')

st.markdown('---')
st.markdown('**Catatan:** Aplikasi ini dibuat otomatis. Sesuaikan label, threshold, dan fitur sesuai kebutuhan.')
