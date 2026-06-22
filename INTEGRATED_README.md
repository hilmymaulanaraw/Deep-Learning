# 🏥 AI Medical Image Analyzer - Integrated Version

**Versi Terintegrasi dengan CBIS-DDSM Dataset & Preprocessing Pipeline**

## 📋 Perubahan dari Versi Sebelumnya

### ✨ Fitur Baru:
1. **Integrated Preprocessing Pipeline** (dari `aidept.py`)
   - ROI Extraction: Menghilangkan background artifact
   - CLAHE Enhancement: Peningkatan kontras adaptif
   - Normalisasi otomatis

2. **Model Upload Integration**
   - Upload trained model (.h5 / .keras) langsung di sidebar
   - Support untuk TensorFlow/Keras models
   - Real-time prediction dengan model Anda

3. **Medical-Specific Analysis**
   - Klasifikasi Benign vs Malignant
   - Risk assessment score
   - Clinical recommendations
   - Clinical report generation

4. **Enhanced Preprocessing Visualization**
   - Before-after comparison
   - Real-time preprocessing preview
   - Detailed statistics per stage

## 🚀 Cara Instalasi

### Langkah 1: Install Dependencies
```bash
pip install -r requirements.txt
```

Atau install manual:
```bash
pip install streamlit numpy Pillow opencv-python tensorflow pandas
```

### Langkah 2: Siapkan Model
1. Train model Anda menggunakan dataset CBIS-DDSM
2. Save model dalam format `.h5` atau `.keras`
3. Contoh: `model.h5` atau `model.keras`

### Langkah 3: Jalankan Aplikasi
```bash
streamlit run ai_medical_integrated.py
```

## 📖 Panduan Penggunaan

### Step 1: Upload Model (PENTING!)
- Buka sidebar di sebelah kiri
- Scroll ke bagian "Model Upload"
- Click "Browse files" dan pilih model `.h5` atau `.keras` Anda
- Tunggu konfirmasi "Model loaded"

### Step 2: Upload Citra Medis
- Buka section "📤 Upload & Preview Citra"
- Upload file mammografi (JPG, PNG, BMP)
- Review preview dan statistik citra

### Step 3: Preprocessing (Opsional)
- Buka section "🔧 Preprocessing & Enhancement"
- Pilih opsi preprocessing di sidebar:
  - ☑️ Ekstrak ROI (Remove Artifacts)
  - ☑️ Apply CLAHE Enhancement
- Click "▶️ Jalankan Preprocessing"
- Lihat perbandingan before-after

### Step 4: Jalankan Analisis
- Buka section "🤖 Analisis AI"
- Click "▶️ Jalankan Analisis AI"
- Tunggu hingga model selesai prediksi (~2-5 detik)

### Step 5: Review Hasil
- Buka section "📊 Hasil & Report"
- Lihat ringkasan hasil dan rekomendasi klinis
- Export laporan (fitur sedang dikembangkan)

## 🔧 Preprocessing Pipeline Detail

### ROI Extraction (Ekstrak Region of Interest)
```
Input: Original mammography image
↓
1. Convert ke grayscale
2. Thresholding (binary segmentation)
3. Find contours (breast boundary detection)
4. Extract largest contour (breast region)
5. Create mask
6. Apply bitwise AND dengan original
7. Crop ke bounding box
8. Resize ke 224x224
↓
Output: ROI-extracted image
```

**Fungsi:** Menghilangkan background dan noise, fokus pada area payudara

### CLAHE Enhancement (Contrast Limited Adaptive Histogram Equalization)
```
Input: ROI-extracted image
↓
1. Apply median blur (denoising)
2. Split image ke channels (R, G, B atau grayscale)
3. Apply CLAHE ke setiap channel
   - Clip limit: 2.0
   - Tile grid size: 8x8
4. Merge channels kembali
5. Normalize ke [0, 1]
↓
Output: Enhanced image dengan contrast lebih baik
```

**Fungsi:** Meningkatkan visibility of microcalcifications dan suspicious regions

## 🤖 Model Integration Guide

### Format Model yang Didukung:
- ✅ `.h5` (TensorFlow/Keras SavedModel format)
- ✅ `.keras` (TensorFlow Keras v3 format)
- ❌ `.pt` atau `.pth` (PyTorch - belum didukung)
- ❌ ONNX format (sedang dikembangkan)

### Cara Menyimpan Model:
```python
# TensorFlow/Keras
model.save('model.h5')
# atau
model.save('model.keras')

# PyTorch users: konversi ke ONNX terlebih dahulu
```

### Input Requirements Model:
- **Input Shape:** (batch_size, 224, 224, 3)
- **Input Type:** float32
- **Input Range:** [0.0, 1.0] (normalized)

### Output Requirements:
- **Output Shape:** (batch_size, num_classes)
- **Output Type:** float32
- **Output Range:** [0.0, 1.0] (probabilities)

### Contoh: Training & Saving Model
```python
import tensorflow as tf
from tensorflow.keras import layers

# Build model
model = tf.keras.Sequential([
    layers.Conv2D(32, 3, activation='relu', input_shape=(224, 224, 3)),
    layers.MaxPooling2D(),
    # ... more layers
    layers.Dense(128, activation='relu'),
    layers.Dense(2, activation='softmax')  # Binary classification
])

# Compile
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Train
model.fit(train_ds, validation_data=val_ds, epochs=20)

# Save
model.save('my_cancer_model.h5')  # atau .keras
```

## 📊 Hasil Analisis

### Output Metrics:
- **Diagnosis:** "Benign (Jinak)" atau "Malignant (Ganas)"
- **Confidence:** Probabilitas prediksi (0-100%)
- **Risk Score:** Skor risiko untuk malignancy (0-1)
- **Class:** Numeric class (0=Benign, 1=Malignant)

### Clinical Recommendations:
Berdasarkan prediksi, aplikasi memberikan rekomendasi:
- Follow-up timeline
- Consultation requirements
- Additional tests consideration

## 🎨 UI Components

```
┌─────────────────────────────────────────────────┐
│ 🏥 AI Medical Image Analyzer - Integrated v1.1 │
├─────────────────────────────────────────────────┤
│                                                 │
│ SIDEBAR ⚙️                                      │
│ ├─ Model Selection (ResNet50/VGG16/etc)       │
│ ├─ Confidence Threshold (slider)               │
│ ├─ Processing Options (checkboxes)             │
│ └─ 📤 Model Upload (.h5/.keras)               │
│                                                 │
├─────────────────────────────────────────────────┤
│                                                 │
│ MAIN CONTENT (Expandable Sections)             │
│ ▼ 📤 Upload & Preview Citra [EXPANDED]        │
│   ├─ File uploader                             │
│   ├─ Image preview                             │
│   └─ Statistics (dimensions, values, etc)     │
│                                                 │
│ ▶ 🔧 Preprocessing & Enhancement              │
│   ├─ ROI extraction option                     │
│   ├─ CLAHE enhancement option                  │
│   ├─ Before/After comparison                   │
│   └─ Stage-wise visualization                  │
│                                                 │
│ ▶ 🎨 Transformasi & Data Augmentasi           │
│   ├─ Geometric transformations                 │
│   └─ Augmentation preview                      │
│                                                 │
│ ▶ 🤖 Analisis AI - Deteksi Kanker             │
│   ├─ Analysis options                          │
│   ├─ Run button                                │
│   └─ Results display (right column)            │
│                                                 │
│ ▶ 📊 Hasil & Report                           │
│   ├─ Summary table                             │
│   ├─ Clinical recommendations                  │
│   └─ Export options (PDF/CSV/Image)           │
│                                                 │
│ ▶ 📖 Dokumentasi & Bantuan                    │
│   ├─ Usage guide                               │
│   ├─ Model info                                │
│   ├─ Dataset info                              │
│   └─ FAQ                                       │
│                                                 │
├─────────────────────────────────────────────────┤
│ FOOTER                                          │
│ Timestamp | App Version | Dataset Info          │
└─────────────────────────────────────────────────┘
```

## 🐛 Troubleshooting

### Error: "No module named 'cv2'"
**Solusi:**
```bash
pip install opencv-python
```

### Error: "No module named 'tensorflow'"
**Solusi:**
```bash
pip install tensorflow
```

### Model tidak bisa diupload
**Cek:**
- File format harus `.h5` atau `.keras`
- File tidak corrupted
- File size < 500MB (limit Streamlit)

### Preprocessing sangat lambat
**Solusi:**
- Unchecked "Ekstrak ROI" jika tidak perlu
- Resize image lebih kecil sebelum upload
- Gunakan komputer dengan spesifikasi lebih tinggi

### Prediksi tidak akurat
**Cek:**
- Model dilatih dengan CBIS-DDSM atau dataset serupa
- Input preprocessing sama dengan saat training
- Confidence threshold pas

## 📝 Dataset Information

### CBIS-DDSM (Curated Breast Imaging-DDSM)
- **Total Images:** 6,000+
- **Resolution:** 4,000 x 6,000 pixels
- **File Format:** DICOM → converted to JPEG
- **Classes:** 2 (Benign, Malignant)
- **Views:** CC (Cranio-Caudal), MLO (Mediolateral-Oblique)
- **Download:** https://www.kaggle.com/datasets/awsaf49/cbis-ddsm-breast-cancer-image-dataset

## 📚 References

- CBIS-DDSM Dataset: Lee et al. (2017)
- CLAHE Algorithm: Zuiderveld (1994)
- ROI Extraction: OpenCV contour detection
- Model Architectures: TensorFlow documentation

## 🎓 Educational Purpose

Aplikasi ini dibuat untuk:
- Semester 6: Deep Learning Citra Medis
- Research & development
- Educational demonstrations

**Disclaimer:** Ini adalah tool educational. Jangan gunakan untuk diagnosis klinis tanpa approval dari ahli medis.

## 📞 Support

Untuk issues atau pertanyaan, lihat FAQ di dalam aplikasi atau dokumentasi lengkap di README.md.

---

**Version:** 1.1 (Integrated)  
**Last Updated:** 2024  
**Status:** Production Ready ✅  
**Model Integration:** ✅ Ready
