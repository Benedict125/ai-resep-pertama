import streamlit as st
import joblib
import numpy as np
import pandas as pd

# -----------------
# Fungsi Pemuatan (dijalankan sekali saja)
# -----------------
@st.cache_resource  # Ini "mantra" Streamlit agar model & data di-load sekali saja
def muat_model_dan_data():
    try:
        # Muat 1 Model dan 3 Penerjemah
        model = joblib.load('model_penebak_resep.pkl')
        le_utama = joblib.load('penerjemah_utama.pkl')
        le_pelengkap = joblib.load('penerjemah_pelengkap.pkl')
        le_hasil = joblib.load('penerjemah_hasil.pkl')
        
        return model, le_utama, le_pelengkap, le_hasil
        
    except FileNotFoundError:
        st.error("ERROR: File .pkl tidak ditemukan.")
        st.error("Pastikan Anda sudah menjalankan 'latih_ai.py' yang baru.")
        return None, None, None, None

# Muat model dan penerjemah
model, le_utama, le_pelengkap, le_hasil = muat_model_dan_data()

# -----------------
# Tampilan Aplikasi Web
# -----------------
st.title("🤖 AI Penebak Resep (Versi 2.0)")
st.caption("Sekarang sudah tidak error! Dibuat oleh Anda.")

# Tampilkan bahan yang dikenal (opsional tapi bagus)
if le_utama is not None and le_pelengkap is not None:
    st.info(f"**Bahan Utama yang saya kenal:** {list(le_utama.classes_)}")
    st.info(f"**Bahan Pelengkap yang saya kenal:** {list(le_pelengkap.classes_)}")

# Inisialisasi "chat history" di session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Tampilkan chat history yang sudah ada
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# -----------------
# Logika Chat Interaktif (Sederhana)
# -----------------

# Kita gunakan form agar user bisa input 2 bahan sekaligus
with st.form(key='resep_form'):
    bahan_utama_input = st.text_input("Masukkan Bahan Utama:")
    bahan_pelengkap_input = st.text_input("Masukkan Bahan Pelengkap:")
    tombol_prediksi = st.form_submit_button("Prediksi Resep Ini!")

# Proses HANYA jika tombol ditekan DAN model sudah siap
if tombol_prediksi and model is not None:
    
    # Tampilkan input user di chat
    pesan_user = f"**Bahan Utama:** {bahan_utama_input}, **Bahan Pelengkap:** {bahan_pelengkap_input}"
    st.chat_message("user").markdown(pesan_user)
    st.session_state.messages.append({"role": "user", "content": pesan_user})

    # Tampilkan balasan AI
    with st.chat_message("assistant"):
        try:
            # 1. Terjemahkan input user ke angka (pakai penerjemah yang benar)
            bahan_utama_encoded = le_utama.transform([bahan_utama_input])
            bahan_pelengkap_encoded = le_pelengkap.transform([bahan_pelengkap_input])
            
            resep_encoded = np.array([bahan_utama_encoded[0], bahan_pelengkap_encoded[0]]).reshape(1, -1)
            
            # 2. Prediksi pakai model
            prediksi_encoded = model.predict(resep_encoded)
            
            # 3. Terjemahkan hasil prediksi ke teks (pakai penerjemah hasil)
            hasil_teks = le_hasil.inverse_transform(prediksi_encoded)
            
            # Tampilkan jawaban
            jawaban_ai = f"Resep dengan **{bahan_utama_input}** dan **{bahan_pelengkap_input}** kemungkinan akan... **{hasil_teks[0]}**!"
            st.markdown(jawaban_ai)
            
        except ValueError:
            # Jika user memasukkan bahan yang tidak dikenal OLEH PENERJEMAH YANG TEPAT
            jawaban_ai = f"Maaf, saya tidak kenal bahan '{bahan_utama_input}' atau '{bahan_pelengkap_input}'. Coba lagi ya."
            st.warning(jawaban_ai)
        
        # Simpan balasan AI ke history
        st.session_state.messages.append({"role": "assistant", "content": jawaban_ai})