import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
import joblib # Ini adalah alat untuk menyimpan "otak" AI

print("Mulai proses latihan AI...")

# 1. Baca "Buku Resep Lama" (data_resep.csv)
data = pd.read_csv('data_resep.csv')

# 2. "Menerjemahkan" data
# AI tidak mengerti "Cokelat" atau "Laku". Kita ubah jadi angka.
# Cokelat=1, Stroberi=2, dst. Laku=1, Tidak Laku=0.
le = LabelEncoder()

# Kita buat salinan data agar data asli tidak berubah
data_encoded = data.copy()
data_encoded['BahanUtama'] = le.fit_transform(data['BahanUtama'])
data_encoded['BahanPelengkap'] = le.fit_transform(data['BahanPelengkap'])
data_encoded['Hasil'] = le.fit_transform(data['Hasil'])

# 3. Pisahkan antara "Soal" (fitur) dan "Kunci Jawaban" (target)
# Fitur (X) = BahanUtama, BahanPelengkap
# Target (y) = Hasil
fitur = ['BahanUtama', 'BahanPelengkap']
target = 'Hasil'

X = data_encoded[fitur]
y = data_encoded[target]

# 4. PROSES BELAJAR (Training)
# Kita panggil Asistennya (DecisionTreeClassifier)
model_ai = DecisionTreeClassifier()

# Kita suruh dia belajar dari data X dan y
# Ini adalah saat si Asisten mencari pola (membangun flowchart)
model_ai.fit(X, y)

print("AI selesai belajar!")

# 5. Simpan "Otak Pintar" si Asisten
# Kita simpan model yang sudah pintar ke dalam sebuah file
joblib.dump(model_ai, 'model_penebak_resep.pkl')
# Kita juga simpan 'penerjemah'-nya agar nanti bisa dipakai lagi
joblib.dump(le, 'penerjemah_label.pkl')


print("Model AI sudah disimpan sebagai 'model_penebak_resep.pkl'")