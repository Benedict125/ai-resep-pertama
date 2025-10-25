import joblib
import numpy as np # Library untuk mengelola array angka
import pandas as pd # <-- TAMBAHKAN BARIS INI

print("===== APLIKASI PENEBAK RESEP =====")

# 1. Muat "Otak AI" dan "Penerjemah" yang sudah disimpan
model = joblib.load('model_penebak_resep.pkl')
le = joblib.load('penerjemah_label.pkl')

# 2. Siapkan "Resep Baru" dari User
# Kita pura-pura user memasukkan resep baru
resep_baru_bahan_utama = 'Cokelat'
resep_baru_bahan_pelengkap = 'Keju'

print(f"Resep Baru: {resep_baru_bahan_utama} dan {resep_baru_bahan_pelengkap}")

# 3. "Menerjemahkan" resep baru
# AI hanya mengerti angka, jadi kita ubah 'Cokelat' dan 'Keju' jadi angka
# PENTING: Kita harus menggunakan 'penerjemah' (le) yang SAMA dengan saat latihan

# Muat data asli untuk 'mengajari' ulang LabelEncoder
data_latih = pd.read_csv('data_resep.csv')

# Ubah BahanUtama
le_bahan_utama = le.fit(data_latih['BahanUtama'])
bahan_utama_encoded = le_bahan_utama.transform([resep_baru_bahan_utama])

# Ubah BahanPelengkap
le_bahan_pelengkap = le.fit(data_latih['BahanPelengkap'])
bahan_pelengkap_encoded = le_bahan_pelengkap.transform([resep_baru_bahan_pelengkap])


# Gabungkan jadi satu resep (array 2D)
resep_baru_encoded = np.array([bahan_utama_encoded[0], bahan_pelengkap_encoded[0]]).reshape(1, -1)


# 4. TANYA AI (Prediction)
# Kita masukkan resep baru (yang sudah jadi angka) ke otak AI
hasil_prediksi_encoded = model.predict(resep_baru_encoded)

# 5. Tampilkan Hasil
# AI akan mengembalikan angka (misal 1). Kita ubah lagi jadi teks "Laku"
# Kita 'inverse_transform' data target (Hasil)
le_hasil = le.fit(data_latih['Hasil'])
hasil_prediksi_teks = le_hasil.inverse_transform(hasil_prediksi_encoded)


print(f"Hasil Prediksi AI: Resep ini kemungkinan akan... {hasil_prediksi_teks[0]}!")
print("========================================")