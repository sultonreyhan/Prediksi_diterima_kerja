# Employability Prediction Dashboard

Dashboard Streamlit untuk eksplorasi dataset tracer study dan prediksi peluang lulusan diterima kerja.

## Fitur

- Home dashboard dengan ringkasan akurasi, jumlah data, jumlah fitur, dan workflow ML.
- Dataset overview dengan preview data, statistik deskriptif, missing values, pie chart, bar chart, histogram, dan correlation heatmap.
- Penjelasan preprocessing step-by-step.
- Form prediksi kandidat realtime dengan probability gauge dan confidence score.
- Batch prediction dari CSV dan download hasil prediksi.
- Evaluasi model: accuracy, precision, recall, F1-score, confusion matrix, ROC curve, model comparison, dan feature importance.

## Struktur Project

```text
.
├── app.py
├── requirements.txt
├── data/
│   └── Kuesioner Faktor Akademik dan Aktivitas Organisasi terhadap Kecepatan Diterima Kerja Lulusan Mahasiswa  (Responses) - Form responses 1.csv
├── notebooks/
│   └── eksperimen.ipynb
├── saved_models/
│   └── model.pkl
├── src/
│   ├── __init__.py
│   ├── model.py
│   ├── predict.py
│   ├── preprocessing.py
│   └── visualization.py
└── assets/
```

## Cara Menjalankan

```bash
pip install -r requirements.txt
python -m src.model
streamlit run app.py
```

## Catatan Modeling

Target prediksi adalah `Status Pekerjaan`, dengan kelas positif `Sudah bekerja`.
Kolom hasil setelah bekerja seperti `Waktu Tunggu Kerja`, `Kesesuaian Pekerjaan`, dan `Kisaran Gaji` tidak digunakan sebagai fitur untuk mengurangi data leakage.

## Deployment Streamlit Cloud

1. Push project ke GitHub.
2. Pastikan `requirements.txt`, `app.py`, folder `src`, folder `data`, dan `saved_models` ikut ter-upload.
3. Di Streamlit Cloud, pilih repository dan set main file ke `app.py`.
4. Deploy. Jika `saved_models/model.pkl` tidak ada, aplikasi akan melatih model otomatis saat pertama dijalankan.
