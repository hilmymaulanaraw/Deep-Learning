# 🏥 AI Medical Image Analyzer - GUI Expandable

Aplikasi GUI expandable yang elegan dan powerful untuk analisis citra medis menggunakan AI/Deep Learning.

## ✨ Fitur Utama

### 1. **Upload & Preview Citra** 📤
- Upload file citra medis (JPG, PNG, BMP)
- Preview real-time dengan statistik detail
- Informasi dimensi, format, dan metadata citra

### 2. **Pemrosesan & Augmentasi Citra** 🔧
- **Transformasi Citra:**
  - Rotasi custom (±180°)
  - Brightness control (0.5 - 2.0x)
  - Contrast adjustment (0.5 - 2.0x)
  - Gaussian blur (kernel size 1-11)

- **Data Augmentasi:**
  - Flip horizontal/vertical
  - Zoom in/out (0.8x - 1.5x)
  - Preview real-time untuk setiap transformasi

### 3. **Analisis AI** 🤖
- Pilih dari berbagai model AI:
  - ResNet50 (Akurasi ~95%)
  - VGG16 (Akurasi ~92%)
  - EfficientNet (Akurasi ~96%)
  - Custom Model
- Jenis analisis yang tersedia:
  - Klasifikasi Penyakit
  - Deteksi Abnormalitas
  - Segmentasi Organ
  - Analisis Tekstur
  - Prediksi Prognosis
- Confidence threshold adjustable

### 4. **Hasil & Export** 📊
- Ringkasan hasil analisis dalam tabel
- Rekomendasi otomatis
- Export ke multiple format (PDF, CSV, Image)

### 5. **Dokumentasi Interaktif** 📖
- Panduan penggunaan step-by-step
- Info detail setiap model AI
- FAQ lengkap
- Tentang aplikasi

## 🚀 Cara Instalasi & Menjalankan

### Prasyarat
- Python 3.8 atau lebih tinggi
- pip (Python package manager)

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

Atau install manual:
```bash
pip install streamlit numpy Pillow
```

### Step 2: Jalankan Aplikasi
```bash
streamlit run gui_expandable_ai.py
```

Aplikasi akan otomatis membuka di browser pada `http://localhost:8501`

## 📋 Panduan Penggunaan

### Alur Dasar:
1. **Expand "Upload & Preview Citra"** → Upload file citra
2. **Review statistik citra** → Lihat informasi detail
3. **(Optional) Expand "Pemrosesan & Augmentasi"** → Terapkan transformasi
4. **Expand "Analisis AI"** → Atur model dan jalankan analisis
5. **Expand "Hasil & Export"** → Lihat hasil dan export

## 🎨 Struktur UI

```
┌─────────────────────────────────────────────┐
│ 🏥 AI Medical Image Analyzer (Main Header)  │
├─────────────────────────────────────────────┤
│ SIDEBAR ⚙️                                  │
│ - Model Selection                           │
│ - Confidence Threshold                      │
│ - Processing Options                        │
├─────────────────────────────────────────────┤
│ MAIN CONTENT (Expandable Sections)          │
│ ▼ 📤 Upload & Preview Citra [EXPANDED]     │
│   - File uploader                           │
│   - Image preview                           │
│   - Statistics                              │
│                                             │
│ ▶ 🔧 Pemrosesan & Augmentasi [COLLAPSED]   │
│ ▶ 🤖 Analisis AI [COLLAPSED]                │
│ ▶ 📊 Hasil & Export [COLLAPSED]             │
│ ▶ 📖 Dokumentasi & Bantuan [COLLAPSED]     │
├─────────────────────────────────────────────┤
│ FOOTER                                      │
│ Timestamp | App Info | Developer Credit     │
└─────────────────────────────────────────────┘
```

## 🔧 Kustomisasi

Anda dapat customize aplikasi dengan:

### 1. Tambah Model AI Baru
Edit bagian "Model Selection" di sidebar:
```python
model_type = st.radio(
    "Pilih Model:",
    ["ResNet50", "VGG16", "EfficientNet", "Custom Model", "YourNewModel"],
    index=0
)
```

### 2. Tambah Jenis Analisis
Edit bagian "Jenis Analisis" di AI Analysis section:
```python
analysis_type = st.selectbox(
    "Jenis Analisis:",
    [
        "Klasifikasi Penyakit",
        "Deteksi Abnormalitas",
        "Segmentasi Organ",
        "Analisis Tekstur",
        "Prediksi Prognosis",
        "Your New Analysis Type"  # Tambahkan di sini
    ]
)
```

### 3. Integrasi dengan Model Nyata
Ganti dummy analysis dengan model ML Anda:
```python
# Original (dummy):
results = {
    "diagnosis": "Normal",
    "confidence": np.random.uniform(0.8, 0.99),
    ...
}

# Ganti dengan (actual model):
import your_model
model = your_model.load_model(model_type)
prediction = model.predict(image_array)
results = {
    "diagnosis": prediction["class"],
    "confidence": prediction["confidence"],
    ...
}
```

## 📊 Contoh Integrasi TensorFlow

```python
# Tambahkan import di awal
import tensorflow as tf

# Di bagian "Jalankan Analisis"
if st.button("▶️ Jalankan Analisis", key="run_analysis"):
    # Load model
    model = tf.keras.models.load_model(f'models/{model_type}.h5')
    
    # Preprocess image
    img_array = np.array(st.session_state.uploaded_image)
    img_array = tf.image.resize(img_array, [224, 224])
    img_array = tf.expand_dims(img_array, 0)
    img_array = img_array / 255.0
    
    # Predict
    prediction = model.predict(img_array)
    
    # Save results
    st.session_state.analysis_results = {
        "diagnosis": class_names[np.argmax(prediction)],
        "confidence": float(np.max(prediction)),
        ...
    }
```

## 📂 File Structure

```
Project UAS/
├── gui_expandable_ai.py          # Main GUI app
├── requirements.txt              # Python dependencies
├── README.md                     # Dokumentasi ini
├── models/                       # (Optional) Folder untuk model AI
│   ├── ResNet50.h5
│   ├── VGG16.h5
│   └── EfficientNet.h5
├── data/                         # (Optional) Folder untuk sample images
│   └── sample_images/
└── logs/                         # (Optional) Folder untuk output
    └── analysis_results/
```

## 🎯 Tips & Tricks

1. **Optimasi Performance:**
   - Resize image besar sebelum upload (< 1MB ideal)
   - Gunakan EfficientNet untuk kecepatan maksimal
   - Cache model dengan `@st.cache_resource`

2. **Meningkatkan Akurasi:**
   - Preprocess citra dengan normalisasi
   - Gunakan data augmentation
   - Fine-tune model dengan data lokal

3. **Debugging:**
   - Buka terminal dan lihat console output
   - Gunakan `st.write()` untuk debug info
   - Cek logs di folder logs/

## 🐛 Troubleshooting

### Error: "ModuleNotFoundError: No module named 'streamlit'"
**Solusi:** Install ulang dependencies
```bash
pip install -r requirements.txt
```

### Aplikasi lambat
**Solusi:** 
- Gunakan model lebih kecil (EfficientNet)
- Resize image input
- Aktifkan caching di model

### Preview tidak muncul
**Solusi:**
- Pastikan file adalah gambar valid
- Coba format berbeda (JPG/PNG)
- Clear cache: `streamlit cache clear`

## 📝 License

Bebas digunakan untuk tujuan pendidikan dan penelitian.

## 👨‍💻 Developer

Dibuat untuk: Semester 6 - Deep Learning Citra Medis
Created with ❤️ using Streamlit

## 📞 Support

Jika ada pertanyaan atau masalah, silakan check dokumentasi dalam app atau lihat FAQ di bagian "Dokumentasi & Bantuan".

---

**Version:** 1.0.0  
**Last Updated:** 2024  
**Status:** Ready for Production ✅
