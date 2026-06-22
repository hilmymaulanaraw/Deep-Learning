# ✅ GUI Expandable AI - Terintegrasi dengan aidept.py

## 📋 Integrasi yang Dilakukan

Saya telah mengintegrasikan file `aidept.py` langsung ke dalam `gui_expandable_ai.py`. Berikut adalah detailnya:

---

## 🔗 Apa yang Diintegrasikan

### 1. **Import Tambahan**
```python
import cv2
import pandas as pd
```
Library OpenCV untuk preprocessing medical images.

### 2. **Preprocessing Functions** (dari aidept.py)

#### a) `remove_artifacts_and_extract_roi()`
- Menghilangkan background artifact
- Mengekstrak ROI (Region of Interest - area payudara)
- Menggunakan contour detection OpenCV
- Return: Image dengan background dihapus (224x224)

#### b) `apply_clahe()`
- Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
- Meningkatkan kontras lokal
- Denoising dengan median blur
- Return: Enhanced image dengan kontras lebih baik

#### c) `preprocess_pipeline()`
- Menggabungkan kedua fungsi di atas
- Parameters:
  - `apply_roi`: Boolean untuk ROI extraction
  - `apply_clahe_enh`: Boolean untuk CLAHE
- Return: Fully preprocessed image

### 3. **Session State Update**
```python
st.session_state.preprocessed_image = None  # BARU
```
Menyimpan hasil preprocessing untuk digunakan di analisis.

### 4. **Sidebar Updates**
```python
apply_roi_extraction = st.checkbox("ROI Extraction (aidept.py)", value=True)
apply_clahe = st.checkbox("CLAHE Enhancement (aidept.py)", value=True)
```
User bisa control preprocessing options langsung dari sidebar.

### 5. **Section 2: Preprocessing**
- Diubah dari "Transformasi & Augmentasi" menjadi 2 tabs:
  - **Tab 1: Preprocessing** - ROI Extraction + CLAHE (dari aidept.py)
  - **Tab 2: Augmentasi** - Transformasi geometris (original)

---

## 🎯 Workflow Baru

```
1. Upload Citra
   ↓
2. [PILIH OPSI]
   a) Jalankan Preprocessing (NEW - dari aidept.py)
      - ROI Extraction
      - CLAHE Enhancement
      - Lihat before-after comparison
   b) Atau langsung Augmentasi (original)
   ↓
3. Jalankan Analisis
   - Jika sudah preprocess → gunakan preprocessed image
   - Jika belum → gunakan original image
   ↓
4. Lihat Hasil
```

---

## 📂 File Structure

```
gui_expandable_ai.py (DIUPDATE)
├── Import: cv2, pandas (BARU)
├── Preprocessing Functions (dari aidept.py):
│   ├── remove_artifacts_and_extract_roi()
│   ├── apply_clahe()
│   └── preprocess_pipeline()
├── Session State: +preprocessed_image
├── Sidebar: +ROI & CLAHE checkboxes
└── Section 2: Preprocessing + Augmentasi (dengan tabs)
```

---

## 🚀 Cara Menggunakan

### Install Dependencies
```bash
pip install -r requirements.txt
```
Pastikan `opencv-python` terinstall.

### Jalankan Aplikasi
```bash
streamlit run gui_expandable_ai.py
```

### Workflow Step-by-Step

#### Step 1: Upload Citra
- Expand "📤 Upload & Preview Citra"
- Upload file mammografi (.jpg, .png, .bmp)
- Lihat preview dan statistik

#### Step 2: Jalankan Preprocessing
- Expand "🔧 Pemrosesan & Augmentasi Citra"
- Click tab "Preprocessing (dari aidept.py)"
- Pastikan checkboxes dicheck:
  - ☑️ ROI Extraction
  - ☑️ CLAHE Enhancement
- Click "▶️ Jalankan Preprocessing"
- Tunggu processing selesai
- Lihat perbandingan Before vs After

#### Step 3: Jalankan Analisis
- Expand "🤖 Analisis AI"
- Click "▶️ Jalankan Analisis"
- Model akan otomatis gunakan preprocessed image

---

## 📊 Visual Flow

```
┌─────────────────────────────────────┐
│ SIDEBAR - Preprocessing Options     │
├─────────────────────────────────────┤
│ ☑️ ROI Extraction (aidept.py)       │
│ ☑️ CLAHE Enhancement (aidept.py)    │
└─────────────────────────────────────┘
              ↓
┌──────────────────────────────────────┐
│ MAIN CONTENT                         │
├──────────────────────────────────────┤
│ Section 2: Pemrosesan               │
│ ├─ Tab 1: Preprocessing             │
│ │  ├─ ROI Extraction (aidept)       │
│ │  ├─ CLAHE Enhancement (aidept)    │
│ │  └─ Before-After Comparison       │
│ │                                   │
│ └─ Tab 2: Augmentasi                │
│    ├─ Rotasi, Brightness, Contrast  │
│    ├─ Flip, Zoom                    │
│    └─ Preview                       │
└──────────────────────────────────────┘
              ↓
┌──────────────────────────────────────┐
│ Section 3: Analisis AI              │
│ ├─ Input: Preprocessed Image (jika) │
│ └─ Output: Prediction Results       │
└──────────────────────────────────────┘
```

---

## 🔍 Code Details

### ROI Extraction Pipeline
```python
def remove_artifacts_and_extract_roi(image_array):
    1. Convert to grayscale
    2. Thresholding (binary segmentation)
    3. Find contours (breast boundary)
    4. Extract largest contour
    5. Create and apply mask
    6. Crop to bounding box
    7. Resize to 224x224
    → Output: ROI image
```

### CLAHE Pipeline
```python
def apply_clahe(image_array):
    1. Convert to uint8
    2. Median blur (denoising)
    3. Apply CLAHE:
       - Clip limit: 2.0
       - Tile grid: 8x8
    4. Merge channels
    5. Normalize to [0, 1]
    → Output: Enhanced image
```

### Complete Preprocessing
```python
def preprocess_pipeline(image_array, apply_roi=True, apply_clahe_enh=True):
    if apply_roi:
        image_array = remove_artifacts_and_extract_roi(image_array)
    if apply_clahe_enh:
        image_array = apply_clahe(image_array)
    return image_array
```

---

## ⚙️ Configuration

### Dalam Sidebar
```python
apply_roi_extraction = st.checkbox("ROI Extraction (aidept.py)", value=True)
apply_clahe = st.checkbox("CLAHE Enhancement (aidept.py)", value=True)
```

### CLAHE Parameters (bisa diubah di code)
```python
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
# Adjust clipLimit untuk kontras lebih/kurang
# Adjust tileGridSize untuk detail lebih/kurang
```

---

## ✨ Features yang Ditambahkan

| Feature | Status |
|---------|--------|
| ROI Extraction | ✅ NEW - dari aidept.py |
| CLAHE Enhancement | ✅ NEW - dari aidept.py |
| Preprocessing Preview | ✅ NEW |
| Before-After Comparison | ✅ NEW |
| Auto-use preprocessed in analysis | ✅ NEW |
| Sidebar preprocessing toggle | ✅ NEW |

---

## 🎓 Educational Value

Integrasi ini menunjukkan:
- Bagaimana medical image preprocessing dilakukan
- ROI extraction untuk menghilangkan background
- CLAHE untuk peningkatan kontras
- Integration best practices
- Streamlit app architecture

---

## 📝 Notes

1. **Preprocessing bersifat opsional** - User bisa skip jika ingin
2. **Preprocessed image disimpan** di session state untuk performa
3. **Before-After comparison** membantu visualisasi perubahan
4. **CLAHE parameters** bisa disesuaikan di code jika perlu

---

## 🔧 Troubleshooting

### Error: "No module named 'cv2'"
```bash
pip install opencv-python
```

### Preprocessing terlalu lambat
- ROI extraction paling slow
- Unchecked jika tidak perlu
- Atau gunakan image lebih kecil

### Hasil preprocessing tidak bagus
- Adjust CLAHE parameters
- Try different ROI extraction thresholds
- Check image quality

---

## 📚 References

**ROI Extraction:**
- OpenCV contour detection
- Thresholding techniques
- Binary morphology

**CLAHE:**
- Adaptive histogram equalization
- Contrast limited enhancement
- Local contrast improvement

---

## ✅ Summary

✅ **aidept.py terintegrasi penuh ke gui_expandable_ai.py**
✅ **Preprocessing pipeline siap digunakan**
✅ **User interface intuitif dengan tabs**
✅ **Before-after visualization tersedia**
✅ **Ready for medical image analysis**

**Status:** Production Ready 🎉

---

**Last Updated:** 2024  
**Integration Version:** 1.0  
**Compatibility:** OpenCV + Streamlit
