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
        model = joblib.load('model_penebak_resep.pkl')
        data_latih = pd.read_csv('data_resep.csv')
        
        # "Ajari" LabelEncoder (penerjemah)
        le_bahan = joblib.load('penerjemah_label.pkl')
        le_bahan_utama = le_bahan.fit(data_latih['BahanUtama'])
        le_bahan_pelengkap = le_bahan.fit(data_latih['BahanPelengkap'])
        le_hasil = le_bahan.fit(data_latih['Hasil'])
        
        return model, le_bahan_utama, le_bahan_pelengkap, le_hasil
    except FileNotFoundError:
        st.error("ERROR: File model (.pkl) atau data (.csv) tidak ditemukan.")
        st.error("Pastikan file ada di repository GitHub Anda & .gitignore sudah benar.")
        return None, None, None, None

# Muat model dan penerjemah
model, le_utama, le_pelengkap, le_hasil = muat_model_dan_data()

# -----------------
# Tampilan Aplikasi Web
# -----------------
st.title("🤖 AI Penebak Resep")
st.caption("Dibuat dengan Python & Streamlit oleh Anda!")

# Inisialisasi "chat history" di session state
# Ini agar obrolan tidak hilang setiap kali user menekan tombol
if "messages" not in st.session_state:
    st.session_state.messages = []

# Tampilkan chat history yang sudah ada
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# -----------------
# Logika Chat Interaktif
# -----------------

# Dapatkan input dari user (akan muncul sebagai kotak chat di bawah)
prompt_utama = st.chat_input("Masukkan Bahan Utama...")
prompt_pelengkap = st.chat_input("Masukkan Bahan Pelengkap...")

# Kita akan proses jika *kedua* input sudah diisi
# (Ini cara sederhana, idealnya pakai form)
# Mari kita sederhanakan: kita minta satu per satu
if prompt_utama:
    # Tampilkan pesan user di chat
    st.chat_message("user").markdown(f"**Bahan Utama:** {prompt_utama}")
    
    # Simpan bahan utama di session, tunggu bahan pelengkap
    st.session_state.bahan_utama_sementara = prompt_utama
    
    # Simpan pesan user ke history
    st.session_state.messages.append({"role": "user", "content": f"Bahan Utama: {prompt_utama}"})


if prompt_pelengkap and "bahan_utama_sementara" in st.session_state:
    # Ambil bahan utama yang disimpan
    bahan_utama = st.session_state.bahan_utama_sementara
    bahan_pelengkap = prompt_pelengkap
    
    # Tampilkan pesan user di chat
    st.chat_message("user").markdown(f"**Bahan Pelengkap:** {bahan_pelengkap}")
    st.session_state.messages.append({"role": "user", "content": f"Bahan Pelengkap: {bahan_pelengkap}"})

    # Tampilkan balasan AI
    with st.chat_message("assistant"):
        try:
            # 1. Terjemahkan input user ke angka
            bahan_utama_encoded = le_utama.transform([bahan_utama])
            bahan_pelengkap_encoded = le_pelengkap.transform([bahan_pelengkap])
            
            resep_encoded = np.array([bahan_utama_encoded[0], bahan_pelengkap_encoded[0]]).reshape(1, -1)
            
            # 2. Prediksi pakai model
            prediksi_encoded = model.predict(resep_encoded)
            
            # 3. Terjemahkan hasil prediksi ke teks
            hasil_teks = le_hasil.inverse_transform(prediksi_encoded)
            
            # Tampilkan jawaban
            jawaban_ai = f"Resep dengan **{bahan_utama}** dan **{bahan_pelengkap}** kemungkinan akan... **{hasil_teks[0]}**!"
            st.markdown(jawaban_ai)
            
        except ValueError:
            # Jika user memasukkan bahan yang tidak dikenal
            jawaban_ai = f"Maaf, saya tidak kenal bahan '{bahan_utama}' atau '{bahan_pelengkap}'. Coba lagi ya."
            st.warning(jawaban_ai)
        
        # Simpan balasan AI ke history
        st.session_state.messages.append({"role": "assistant", "content": jawaban_ai})
    
    # Hapus bahan utama sementara agar bisa mulai dari awal
    del st.session_state.bahan_utama_sementara