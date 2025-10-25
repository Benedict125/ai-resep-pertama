import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
import joblib 

print("Mulai proses latihan AI...")

# 1. Baca "Buku Resep Lama" (data_resep.csv)
data = pd.read_csv('data_resep.csv')

# 2. "Menerjemahkan" data
# KITA BUAT 3 PENERJEMAH BERBEDA
le_utama = LabelEncoder()
le_pelengkap = LabelEncoder()
le_hasil = LabelEncoder()

# Kita buat salinan data agar data asli tidak berubah
data_encoded = data.copy()

# Gunakan penerjemah yang tepat untuk tiap kolom
data_encoded['BahanUtama'] = le_utama.fit_transform(data['BahanUtama'])
data_encoded['BahanPelengkap'] = le_pelengkap.fit_transform(data['BahanPelengkap'])
data_encoded['Hasil'] = le_hasil.fit_transform(data['Hasil'])


# 3. Pisahkan antara "Soal" (fitur) dan "Kunci Jawaban" (target)
fitur = ['BahanUtama', 'BahanPelengkap']
target = 'Hasil'

X = data_encoded[fitur]
y = data_encoded[target]

# 4. PROSES BELAJAR (Training)
model_ai = DecisionTreeClassifier()
model_ai.fit(X, y)

print("AI selesai belajar!")

# 5. Simpan "Otak Pintar" DAN KETIGA PENERJEMAH-nya
joblib.dump(model_ai, 'model_penebak_resep.pkl')
joblib.dump(le_utama, 'penerjemah_utama.pkl')       # Simpan penerjemah 1
joblib.dump(le_pelengkap, 'penerjemah_pelengkap.pkl') # Simpan penerjemah 2
joblib.dump(le_hasil, 'penerjemah_hasil.pkl')     # Simpan penerjemah 3


print("Model AI dan 3 Penerjemah berhasil disimpan!")