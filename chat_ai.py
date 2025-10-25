import joblib
import numpy as np
import pandas as pd

print("Memuat AI Penebak Resep... Mohon tunggu.")

# 1. Muat Model dan Data (HANYA SEKALI di awal)
try:
    model = joblib.load('model_penebak_resep.pkl')
    le_bahan = joblib.load('penerjemah_label.pkl') # Kita akan pakai ini untuk semua
    data_latih = pd.read_csv('data_resep.csv')
except FileNotFoundError:
    print("ERROR: File model (.pkl) atau data (.csv) tidak ditemukan.")
    print("Pastikan Anda sudah menjalankan 'latih_ai.py' terlebih dahulu.")
    exit() # Keluar dari program jika file tidak ada

# 2. "Ajari" LabelEncoder (penerjemah) SEMUA kata yang dia tahu
# Ini jauh lebih efisien daripada meng-fit ulang di dalam loop
le_bahan_utama = le_bahan.fit(data_latih['BahanUtama'])
le_bahan_pelengkap = le_bahan.fit(data_latih['BahanPelengkap'])
le_hasil = le_bahan.fit(data_latih['Hasil'])

# Beri tahu user bahan apa saja yang AI kenal
kata_dikenal_utama = list(le_bahan_utama.classes_)
kata_dikenal_pelengkap = list(le_bahan_pelengkap.classes_)

print("===== APLIKASI CHAT PENEBAK RESEP =====")
print("AI siap! Ketik 'keluar' kapan saja untuk berhenti.")
print("------------------------------------------")
print(f"Bahan utama yang saya kenal: {kata_dikenal_utama}")
print(f"Bahan pelengkap yang saya kenal: {kata_dikenal_pelengkap}")
print("------------------------------------------")


# 3. Buat Loop Chatting (WHILE TRUE)
# Ini adalah "obrolan" yang akan berjalan terus
while True:
    
    # 4. Minta input dari user
    resep_baru_bahan_utama = input("Anda (Bahan Utama): ")
    
    # 5. Cek kondisi keluar
    if resep_baru_bahan_utama.lower() == 'keluar':
        break # Hentikan loop obrolan

    resep_baru_bahan_pelengkap = input("Anda (Bahan Pelengkap): ")
    if resep_baru_bahan_pelengkap.lower() == 'keluar':
        break # Hentikan loop obrolan

    # 6. Error Handling (TRY...EXCEPT)
    # Kita "coba" (try) terjemahkan. Jika gagal (except), kita tangani
    try:
        # 7. "Menerjemahkan" input user menjadi angka
        bahan_utama_encoded = le_bahan_utama.transform([resep_baru_bahan_utama])
        bahan_pelengkap_encoded = le_bahan_pelengkap.transform([resep_baru_bahan_pelengkap])
        
        # Gabungkan jadi satu resep (array 2D)
        resep_baru_encoded = np.array([bahan_utama_encoded[0], bahan_pelengkap_encoded[0]]).reshape(1, -1)
        
        # 8. TANYA AI (Prediction)
        hasil_prediksi_encoded = model.predict(resep_baru_encoded)
        
        # 9. Tampilkan Hasil (terjemahkan balik ke teks)
        hasil_prediksi_teks = le_hasil.inverse_transform(hasil_prediksi_encoded)
        
        print(f"   -> AI: Resep ini kemungkinan akan... {hasil_prediksi_teks[0]}!")

    except ValueError:
        # Ini terjadi jika user memasukkan kata yang tidak ada di data_resep.csv
        print(f"   -> AI: ! Maaf, saya tidak kenal bahan '{resep_baru_bahan_utama}' atau '{resep_baru_bahan_pelengkap}'. Coba lagi ya.")
    
    print("------------------------------------------") # Pemisah antar chat

# 10. Pesan Penutup (Saat loop berhenti)
print("===== Obrolan Selesai =====")
print("Terima kasih sudah chatting! Sampai jumpa.")