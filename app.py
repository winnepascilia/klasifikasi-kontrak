import streamlit as st
import numpy as np
import pandas as pd
import joblib
import cloudpickle
from scipy.sparse import hstack
import os
import gdown

# ======== Download Model Files ========
def download_file(file_id, output):
    if not os.path.exists(output):
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, output, quiet=False)

download_file("1ArgvQ6v1fYTwM17QRIUzfYdQ8HwyIk90", "model_rf.pkl")
download_file("1_C0nq9LM7tOg1uoqdK8aM_rWr2uhR74h", "tfidf_vectorizer.pkl")
download_file("13JZiUIBcSuXB9YGtkjmgBVrsc9CQqqLF", "label_encoder.pkl")

# ======== Load Model ========
with open("model_rf.pkl", "rb") as f:
    model = cloudpickle.load(f)

tfidf = joblib.load("tfidf_vectorizer.pkl")
le = joblib.load("label_encoder.pkl")

# ======== Streamlit UI ========
st.title("Prediksi Kategori Nilai Kontrak – PT PLN Batam")

uraian = st.text_area("Masukkan uraian pekerjaan:")
pelaksana = st.text_input("Masukkan nama pelaksana:")
nilai_kontrak = st.number_input("Masukkan nilai kontrak (Rp):", min_value=0.0)
jangka_waktu = st.number_input("Masukkan jangka waktu (hari):", min_value=1)

if st.button("Prediksi"):
    # TF-IDF uraian
    uraian_vec = tfidf.transform([uraian])

    # Encode pelaksana
    if pelaksana in le.classes_:
        pelaksana_enc = le.transform([pelaksana])[0]
    else:
        pelaksana_enc = -1

    fitur_lain = np.array([[nilai_kontrak, pelaksana_enc, jangka_waktu]])
    fitur_df = pd.DataFrame(fitur_lain, columns=["nilai_kontrak", "pelaksana", "jangka_waktu"])
    X_input = hstack([uraian_vec, fitur_df])
    pred = model.predict(X_input)[0]

    label_kategori = {
        0: "Kontrak Kecil (< 100 juta)",
        1: "Kontrak Menengah (100 - 500 juta)",
        2: "Kontrak Besar (> 500 juta)"
    }

    st.success(f"? Kategori Kontrak: {label_kategori[pred]}")
