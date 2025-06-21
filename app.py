import streamlit as st
import numpy as np
import pandas as pd
import joblib, cloudpickle
from scipy.sparse import hstack
import os, gdown

# Download .pkl dari Drive
def download_file(file_id, output):
    if not os.path.exists(output):
        gdown.download(f'https://drive.google.com/uc?id={file_id}', output, quiet=True)

download_file("1ArgvQ6v1fYTwM17QRIUzfYdQ8HwyIk90", "model_rf.pkl")
download_file("1_C0nq9LM7tOg1uoqdK8aM_rWr2uhR74h", "tfidf_vectorizer.pkl")
download_file("13JZiUIBcSuXB9YGtkjmgBVrsc9CQqqLF", "label_encoder.pkl")

with open("model_rf.pkl","rb") as f:
    model = cloudpickle.load(f)
tfidf = joblib.load("tfidf_vectorizer.pkl")
le = joblib.load("label_encoder.pkl")

st.title("Prediksi Kategori Kontrak – PT PLN Batam")
uraian = st.text_area("Uraian pekerjaan:")
pelaksana = st.text_input("Pelaksana pekerjaan:")
nilai_kontrak = st.number_input("Nilai kontrak (Rp):", min_value=0.0)
jangka_waktu = st.number_input("Jangka waktu (hari):", min_value=1)

if st.button("Prediksi"):
    uraian_vec = tfidf.transform([uraian])
    pelaksana_enc = le.transform([pelaksana])[0] if pelaksana in le.classes_ else -1
    fitur = np.array([[nilai_kontrak, pelaksana_enc, jangka_waktu]])
    df_f = pd.DataFrame(fitur, columns=["nilai_kontrak","pelaksana","jangka_waktu"])
    X_in = hstack([uraian_vec, df_f])
    pred = model.predict(X_in)[0]
    label = {0:"Kecil (<100 juta)",1:"Menengah (100–500 juta)",2:"Besar (>500 juta)"}
    st.success(f"✅ Prediksi: {label[pred]}")
