# 🚀 Quick Start Guide - Integrated AI Medical Image Analyzer

## ⚡ Setup Cepat (5 Menit)

### 1️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

Tunggu ~2-3 menit untuk download dan install semua packages.

### 2️⃣ Jalankan Aplikasi
```bash
streamlit run ai_medical_integrated.py
```

Aplikasi akan otomatis buka di `http://localhost:8501`

---

## 📋 Checklist Persiapan Model

Sebelum jalankan aplikasi, siapkan:

- [ ] **Model File** (.h5 atau .keras)
  - Contoh: `cancer_detection_model.h5`
  - Harus trained dengan CBIS-DDSM atau dataset serupa
  - Input: (224, 224, 3) normalized [0, 1]
  - Output: (batch_size, 2) probabilities

- [ ] **Test Image** (JPG/PNG)
  - Contoh: `mammography_sample.jpg`
  - Resolusi: min 256x256
  - Format: mammography (CC atau MLO view)

- [ ] **Requirements installed**
  - ✅ Streamlit
  - ✅ TensorFlow
  - ✅ OpenCV
  - ✅ NumPy, Pillow, Pandas

---

## 🎯 First Run Workflow

### Step 1: Buka Sidebar (di kiri layar)
- Lihat section "⚙️ Pengaturan"
- Scroll down ke "Model Upload"

### Step 2: Upload Model Anda
- Click "Browse files"
- Select file `.h5` atau `.keras` model Anda
- Tunggu konfirmasi "✅ Model loaded"

### Step 3: Upload Test Image
- Expand section "📤 Upload & Preview Citra"
- Click "Choose file"
- Select mammo image
- Lihat preview dan statistics

### Step 4: Jalankan Preprocessing
- Expand "🔧 Preprocessing & Enhancement"
- Pastikan checkbox dicheck:
  - ☑️ Ekstrak ROI
  - ☑️ Apply CLAHE
- Click "▶️ Jalankan Preprocessing"
- Bandingkan before-after

### Step 5: Jalankan Analisis
- Expand "🤖 Analisis AI"
- Click "▶️ Jalankan Analisis AI"
- Tunggu hasil prediksi (2-5 detik)

### Step 6: Lihat Hasil
- Expand "📊 Hasil & Report"
- Review diagnosis, confidence, risk score
- Lihat clinical recommendations

---

## 📊 Expected Output

Setelah analisis, Anda akan dapat:

```
✅ BENIGN (atau ⚠️ MALIGNANT)
├─ Diagnosis: Benign (Jinak) / Malignant (Ganas)
├─ Confidence: 95.32% (sesuai model Anda)
├─ Risk Score: 0.047 (untuk malignant prediction)
└─ Recommendations:
   1. Follow-up dalam 3-6 bulan
   2. Konsultasi dengan radiolog
   3. Pertimbangkan biopsi jika perlu
```

---

## 🔧 Troubleshooting

### ❌ "ModuleNotFoundError"
```bash
# Cek & install missing packages
pip install -r requirements.txt --upgrade
```

### ❌ "Could not load model"
- Pastikan format `.h5` atau `.keras`
- File tidak corrupted
- Model saved dengan benar: `model.save('mymodel.h5')`

### ❌ Aplikasi lambat/hang
- Close aplikasi: Ctrl+C
- Restart: `streamlit run ai_medical_integrated.py`
- Cek memory & CPU usage

### ❌ Preprocessing error
- Unchecked preprocessing options di sidebar
- Atau gunakan image resolusi lebih tinggi

---

## 📁 File Structure

```
Project UAS/
├── ai_medical_integrated.py      ← JALANKAN INI
├── requirements.txt              ← Dependencies
├── INTEGRATED_README.md          ← Full documentation
├── QUICK_START.md                ← File ini
├── gui_expandable_ai.py          ← Original GUI (backup)
├── aidept.py                     ← Data prep script
│
├── models/                       ← Folder untuk model (create jika perlu)
│   └── my_cancer_model.h5       ← Model Anda
│
└── sample_images/               ← Folder untuk test images (create jika perlu)
    └── mammography_sample.jpg   ← Test image
```

---

## 💡 Tips & Tricks

### 1. Upload Model Sekali
- Model di-load di memory, tidak perlu re-upload
- Bisa analyze multiple images dengan model yang sama

### 2. Batch Analysis (Manual)
- Upload dan analyze satu image
- Review hasil
- Ulangi dengan image lain

### 3. Preprocessing Optimization
- ROI extraction: Penting untuk menghilangkan background
- CLAHE: Penting untuk visibility microcalcifications
- Unchecked jika sudah preprocess di training

### 4. Model Performance
- Lihat confidence score
- Risk score hanya untuk malignant prediction
- Below 70% confidence = uncertain, perlu expert review

---

## 🎓 Learning Resources

Baca file dokumentasi:
- **INTEGRATED_README.md** → Full technical documentation
- **README.md** → Feature overview
- Help tab dalam aplikasi → Built-in FAQ

---

## ✨ What's Next

Setelah berhasil jalankan:

1. **Train Model Yourself**
   - Use CBIS-DDSM dataset dari Kaggle
   - Train ResNet50 atau EfficientNet
   - Evaluate dengan test set

2. **Improve Preprocessing**
   - Adjust CLAHE parameters
   - Add histogram normalization
   - Test different ROI extraction methods

3. **Add More Features**
   - Batch processing
   - Export reports (PDF/CSV)
   - Multi-class classification
   - Saliency map visualization

---

## 📞 Quick Help

### Model tidak bisa predict?
1. Check model format (.h5 / .keras)
2. Check model input shape (harus 224x224x3)
3. Check model output shape (harus 2 classes)

### Image tidak bisa diupload?
1. Check format (JPG/PNG/BMP)
2. Check file size (< 200MB)
3. Check if corrupted

### Preprocessing terlalu lambat?
1. Unchecked preprocessing di sidebar
2. Upload image dengan resolusi lebih kecil
3. Use komputer dengan spec lebih tinggi

---

## 🚀 You're Ready!

Sekarang Anda siap:
```bash
streamlit run ai_medical_integrated.py
```

Selamat menganalisis! 🏥✨

---

**Version:** 1.0 Quick Start  
**Updated:** 2024  
**Status:** Ready to Use ✅
