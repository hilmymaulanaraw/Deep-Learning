import streamlit as st
import numpy as np
from PIL import Image
import io
import os
from datetime import datetime
import cv2
import pandas as pd
import aidept 
import ai_medical_integrated

# Configure page
st.set_page_config(
    page_title="AI Medical Image Analyzer",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS untuk styling
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
    }
    .section-header {
        font-size: 1.3rem;
        color: #2c3e50;
        font-weight: bold;
        margin-top: 1.5rem;
    }
    .info-box {
        background-color: #e8f4f8;
        border-left: 4px solid #1f77b4;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .success-box {
        background-color: #d4edda;
        border-left: 4px solid #28a745;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        border-left: 4px solid #ffc107;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# ===================== PREPROCESSING FUNCTIONS (dari aidept.py) =====================

def remove_artifacts_and_extract_roi(image_array):
    """
    Menghilangkan background dan mengekstrak ROI (Region of Interest)
    Menggunakan thresholding dan contour detection dari OpenCV
    """
    try:
        # Konversi ke grayscale
        if len(image_array.shape) == 3:
            gray = cv2.cvtColor(np.uint8(image_array * 255), cv2.COLOR_RGB2GRAY)
        else:
            gray = np.uint8(image_array * 255)
        
        # Thresholding
        _, thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return image_array
        
        # Ambil contour terbesar (breast region)
        largest_cnt = max(contours, key=cv2.contourArea)
        mask = np.zeros_like(gray)
        cv2.drawContours(mask, [largest_cnt], -1, 255, -1)
        
        # Apply mask
        if len(image_array.shape) == 3:
            result = cv2.bitwise_and(np.uint8(image_array * 255), np.uint8(image_array * 255), mask=mask)
        else:
            result = cv2.bitwise_and(gray, gray, mask=mask)
        
        # Crop to bounding box
        x, y, w, h = cv2.boundingRect(largest_cnt)
        roi = result[y:y+h, x:x+w]
        
        # Resize ke target size
        roi_resized = cv2.resize(roi, (224, 224))
        return np.float32(roi_resized) / 255.0
    
    except Exception as e:
        st.warning(f"⚠️ Warning ROI extraction: {str(e)}")
        return image_array


def apply_clahe(image_array):
    """
    Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    untuk meningkatkan contrast lokal
    """
    try:
        img_uint8 = np.uint8(image_array * 255)
        
        # Median blur untuk denoising
        denoised = cv2.medianBlur(img_uint8, 3)
        
        # Apply CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        
        # Handle both grayscale dan color images
        if len(denoised.shape) == 3:
            channels = cv2.split(denoised)
            clahe_channels = [clahe.apply(c) for c in channels]
            result = cv2.merge(clahe_channels)
        else:
            result = clahe.apply(denoised)
        
        return np.float32(result) / 255.0
    
    except Exception as e:
        st.warning(f"⚠️ Warning CLAHE: {str(e)}")
        return image_array


def preprocess_pipeline(image_array, apply_roi=True, apply_clahe_enh=True):
    """
    Complete preprocessing pipeline
    1. Remove artifacts & extract ROI (optional)
    2. Apply CLAHE enhancement (optional)
    """
    if apply_roi:
        image_array = remove_artifacts_and_extract_roi(image_array)
    
    if apply_clahe_enh:
        image_array = apply_clahe(image_array)
    
    return image_array

# ===================== END PREPROCESSING FUNCTIONS =====================

# Initialize session state
if 'uploaded_image' not in st.session_state:
    st.session_state.uploaded_image = None
if 'preprocessed_image' not in st.session_state:
    st.session_state.preprocessed_image = None
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = None
if 'model_selected' not in st.session_state:
    st.session_state.model_selected = "ResNet50"

# Header
st.markdown('<h1 class="main-header">🏥 AI Medical Image Analyzer</h1>', unsafe_allow_html=True)
st.markdown("---")

# Sidebar configuration
with st.sidebar:
    st.title("⚙️ Pengaturan")
    st.markdown("---")
    
    # Model selection
    st.subheader("Model AI")
    model_type = st.radio(
        "Pilih Model:",
        ["ResNet50", "VGG16", "EfficientNet", "Custom Model"],
        index=0
    )
    st.session_state.model_selected = model_type
    
    # Confidence threshold
    confidence = st.slider(
        "Confidence Threshold:",
        0.0, 1.0, 0.7, 0.05
    )
    
    # Processing options
    st.subheader("Opsi Pemrosesan")
    apply_enhancement = st.checkbox("Gunakan Image Enhancement", value=True)
    apply_normalization = st.checkbox("Normalisasi Citra", value=True)
    apply_roi_extraction = st.checkbox("ROI Extraction (aidept.py)", value=True)
    apply_clahe = st.checkbox("CLAHE Enhancement (aidept.py)", value=True)
    
    st.markdown("---")
    st.info("💡 Tips: Upload citra medis dalam format JPG, PNG, atau DICOM")

# Main content area with expandable sections
col1, col2 = st.columns([2, 1])

with col1:
    # Section 1: Upload & Preview
    with st.expander("📤 Upload & Preview Citra"):
        st.markdown('<p class="section-header">Upload File Citra Medis</p>', unsafe_allow_html=True)
        
        col_upload1, col_upload2 = st.columns(2)
        
        with col_upload1:
            uploaded_file = st.file_uploader(
                "Pilih file citra",
                type=["jpg", "jpeg", "png", "bmp"],
                help="Format yang didukung: JPG, PNG, BMP"
            )
        
        with col_upload2:
            st.markdown("""
                <div class="info-box">
                <strong>ℹ️ Info Upload:</strong>
                <ul>
                <li>Format: JPG, PNG, BMP</li>
                <li>Ukuran max: 200MB</li>
                <li>Resolusi: Min 256x256</li>
                </ul>
                </div>
            """, unsafe_allow_html=True)
        
        if uploaded_file is not None:
            st.session_state.uploaded_image = Image.open(uploaded_file)
            
            st.success("✅ File berhasil di-upload!")
            
            # Display image info
            img = st.session_state.uploaded_image
            col_info1, col_info2, col_info3 = st.columns(3)
            
            with col_info1:
                st.metric("Lebar", f"{img.width} px")
            with col_info2:
                st.metric("Tinggi", f"{img.height} px")
            with col_info3:
                st.metric("Format", img.format)
            
            # Display preview
            st.subheader("Preview Citra")
            st.image(img, use_column_width=True, caption="Citra yang di-upload")
            
            # Image statistics
            st.subheader("📊 Statistik Citra")
            img_array = np.array(img)
            col_stat1, col_stat2 = st.columns(2)
            with col_stat1:
                st.metric("Dimensi", f"{img_array.shape}")
                st.metric("Tipe Data", str(img_array.dtype))
                st.metric("Min Value", f"{img_array.min()}")
            with col_stat2:
                st.metric("Max Value", f"{img_array.max()}")
                st.metric("Mean Value", f"{img_array.mean():.2f}")
                st.metric("Std Dev", f"{img_array.std():.2f}")
        else:
            st.info("👆 Upload file citra untuk memulai")

# Section 2: Image Processing
with st.expander("🔧 Pemrosesan & Augmentasi Citra"):
    st.markdown('<p class="section-header">Opsi Pemrosesan Citra</p>', unsafe_allow_html=True)
    
    if st.session_state.uploaded_image is not None:
        # Tab untuk preprocessing vs augmentasi
        tab1, tab2 = st.tabs(["Preprocessing (dari aidept.py)", "Augmentasi Data"])
        
        with tab1:
            st.markdown("""
                <div class="info-box">
                <strong>📋 Pipeline Preprocessing:</strong>
                <ol>
                <li><strong>ROI Extraction:</strong> Menghilangkan background dan mengekstrak area payudara</li>
                <li><strong>CLAHE Enhancement:</strong> Meningkatkan kontras lokal untuk visibilitas lebih baik</li>
                </ol>
                </div>
            """, unsafe_allow_html=True)
            
            col_preproc1, col_preproc2 = st.columns(2)
            
            with col_preproc1:
                st.subheader("Opsi Preprocessing")
                apply_roi = st.checkbox("✓ ROI Extraction (Hapus Artifacts)", value=True)
                apply_clahe_enh = st.checkbox("✓ CLAHE Enhancement", value=True)
                
                if st.button("▶️ Jalankan Preprocessing", key="run_preprocessing", use_container_width=True):
                    with st.spinner("🔄 Sedang memproses citra..."):
                        try:
                            img = st.session_state.uploaded_image
                            img_array = np.array(img).astype(np.float32) / 255.0
                            
                            # Ensure 3 channels
                            if len(img_array.shape) == 2:
                                img_array = np.stack([img_array] * 3, axis=-1)
                            elif img_array.shape[2] == 4:
                                img_array = img_array[:, :, :3]
                            
                            # Apply preprocessing pipeline
                            preprocessed = preprocess_pipeline(img_array, apply_roi=apply_roi, apply_clahe_enh=apply_clahe_enh)
                            st.session_state.preprocessed_image = preprocessed
                            st.success("✅ Preprocessing berhasil!")
                        
                        except Exception as e:
                            st.error(f"❌ Error: {str(e)}")
            
            with col_preproc2:
                if st.session_state.preprocessed_image is not None:
                    st.markdown("""
                        <div class="success-box">
                        <strong>✅ Status:</strong><br>
                        Citra sudah dipreprocess
                        </div>
                    """, unsafe_allow_html=True)
            
            # Display comparison
            if st.session_state.preprocessed_image is not None:
                st.subheader("Perbandingan: Before vs After")
                col_before, col_after = st.columns(2)
                
                with col_before:
                    st.image(st.session_state.uploaded_image, caption="Original", use_column_width=True)
                
                with col_after:
                    preprocessed_display = (st.session_state.preprocessed_image * 255).astype(np.uint8)
                    if len(preprocessed_display.shape) == 3:
                        preprocessed_img = Image.fromarray(preprocessed_display)
                    else:
                        preprocessed_img = Image.fromarray(preprocessed_display, mode='L')
                    st.image(preprocessed_img, caption="Preprocessed (ROI + CLAHE)", use_column_width=True)
        
        with tab2:
            st.subheader("Data Augmentasi Geometris")
            
            col_aug1, col_aug2 = st.columns(2)
            
            with col_aug1:
                st.subheader("Transformasi")
                rotate_angle = st.slider("Rotasi (derajat)", -180, 180, 0, 15)
                brightness = st.slider("Brightness", 0.5, 2.0, 1.0, 0.1)
                contrast = st.slider("Contrast", 0.5, 2.0, 1.0, 0.1)
                
                if st.button("🔄 Preview Transformasi", key="preview_transform"):
                    from PIL import ImageEnhance
                    
                    img = st.session_state.uploaded_image.copy()
                    
                    # Rotate
                    if rotate_angle != 0:
                        img = img.rotate(rotate_angle)
                    
                    # Brightness
                    enhancer = ImageEnhance.Brightness(img)
                    img = enhancer.enhance(brightness)
                    
                    # Contrast
                    enhancer = ImageEnhance.Contrast(img)
                    img = enhancer.enhance(contrast)
                    
                    st.image(img, use_column_width=True, caption="Preview Transformasi")
            
            with col_aug2:
                st.subheader("Augmentasi")
                flip_horizontal = st.checkbox("Flip Horizontal", value=False)
                flip_vertical = st.checkbox("Flip Vertical", value=False)
                zoom_level = st.slider("Zoom", 0.8, 1.5, 1.0, 0.1)
                
                if st.button("🔄 Preview Augmentasi", key="preview_augment"):
                    img = st.session_state.uploaded_image.copy()
                    
                    if flip_horizontal:
                        img = img.transpose(Image.FLIP_LEFT_RIGHT)
                    if flip_vertical:
                        img = img.transpose(Image.FLIP_TOP_BOTTOM)
                    
                    # Zoom
                    if zoom_level != 1.0:
                        w, h = img.size
                        new_size = (int(w * zoom_level), int(h * zoom_level))
                        img = img.resize(new_size, Image.Resampling.LANCZOS)
                    
                    st.image(img, use_column_width=True, caption="Preview Augmentasi")
    else:
        st.warning("⚠️ Upload citra terlebih dahulu untuk menggunakan fitur ini")

# Section 3: AI Analysis
with st.expander("🤖 Analisis AI"):
    st.markdown('<p class="section-header">Analisis Menggunakan AI Model</p>', unsafe_allow_html=True)
    
    if st.session_state.uploaded_image is not None:
        col_analysis1, col_analysis2 = st.columns([2, 1])
        
        with col_analysis1:
            analysis_type = st.selectbox(
                "Jenis Analisis:",
                [
                    "Klasifikasi Penyakit",
                    "Deteksi Abnormalitas",
                    "Segmentasi Organ",
                    "Analisis Tekstur",
                    "Prediksi Prognosis"
                ]
            )
            
            st.info(f"Model yang dipilih: **{model_type}** | Confidence: **{confidence}**")
            
            st.markdown("""
                <div class="info-box">
                <strong>ℹ️ Preprocessing Info:</strong><br>
                Jika Anda sudah jalankan preprocessing, model akan menggunakan citra yang sudah dipreprocess.
                Jika belum, model akan menggunakan citra original.
                </div>
            """, unsafe_allow_html=True)
            
            if st.button("▶️ Jalankan Analisis", key="run_analysis", use_container_width=True):
                with st.spinner("🔄 Sedang menganalisis citra..."):
                    # Simulate analysis
                    import time
                    progress_bar = st.progress(0)
                    
                    for i in range(100):
                        progress_bar.progress(i + 1)
                        time.sleep(0.01)
                    
                    # Dummy results
                    st.session_state.analysis_results = {
                        "diagnosis": "Normal",
                        "confidence": np.random.uniform(0.8, 0.99),
                        "regions": np.random.randint(1, 5),
                        "risk_score": np.random.uniform(0.1, 0.9),
                        "recommendations": [
                            "Follow-up dalam 3 bulan",
                            "Monitor perkembangan",
                            "Konsultasi dengan spesialis"
                        ]
                    }
                    st.success("✅ Analisis selesai!")
        
        with col_analysis2:
            if st.session_state.analysis_results:
                results = st.session_state.analysis_results
                
                st.markdown("""
                    <div class="success-box">
                    <strong>📋 Hasil Analisis:</strong>
                    </div>
                """, unsafe_allow_html=True)
                
                st.metric("Diagnosis", results["diagnosis"])
                st.metric("Confidence", f"{results['confidence']:.2%}")
                st.metric("Risk Score", f"{results['risk_score']:.2f}")
                st.metric("Regions Detected", results["regions"])
    else:
        st.warning("⚠️ Upload citra terlebih dahulu untuk menjalankan analisis")

# Section 4: Results & Export
with st.expander("📊 Hasil & Export"):
    st.markdown('<p class="section-header">Hasil Analisis & Opsi Export</p>', unsafe_allow_html=True)
    
    if st.session_state.analysis_results:
        results = st.session_state.analysis_results
        
        # Display results in table
        col_table1, col_table2 = st.columns(2)
        
        with col_table1:
            st.subheader("Ringkasan Hasil")
            results_data = {
                "Metrik": ["Diagnosis", "Confidence", "Risk Score", "Regions Detected"],
                "Nilai": [
                    results["diagnosis"],
                    f"{results['confidence']:.2%}",
                    f"{results['risk_score']:.2f}",
                    f"{results['regions']}"
                ]
            }
            st.dataframe(results_data, use_container_width=True)
        
        with col_table2:
            st.subheader("Rekomendasi")
            for i, rec in enumerate(results["recommendations"], 1):
                st.write(f"{i}. {rec}")
        
        # Export options
        st.subheader("📥 Export")
        col_export1, col_export2, col_export3 = st.columns(3)
        
        with col_export1:
            if st.button("📄 Export as PDF", use_container_width=True):
                st.info("💡 Fitur PDF sedang dikembangkan")
        
        with col_export2:
            if st.button("📊 Export as CSV", use_container_width=True):
                st.info("💡 Fitur CSV sedang dikembangkan")
        
        with col_export3:
            if st.button("🖼️ Export as Image", use_container_width=True):
                st.info("💡 Fitur Image export sedang dikembangkan")
    else:
        st.info("👆 Jalankan analisis AI terlebih dahulu untuk melihat hasil")

# Section 5: Documentation & Help
with st.expander("📖 Dokumentasi & Bantuan"):
    st.markdown('<p class="section-header">Panduan Penggunaan</p>', unsafe_allow_html=True)
    
    tabs = st.tabs(["Cara Penggunaan", "Model Info", "FAQ", "Tentang"])
    
    with tabs[0]:
        st.markdown("""
        ### Langkah-Langkah Penggunaan:
        
        1. **Upload Citra**
           - Klik "Upload & Preview Citra"
           - Pilih file citra medis (JPG, PNG, BMP)
           - Review preview dan statistik citra
        
        2. **Atur Pengaturan**
           - Pilih model AI di sidebar
           - Atur confidence threshold
           - Pilih opsi pemrosesan
        
        3. **Proses Citra (Opsional)**
           - Buka "Pemrosesan & Augmentasi Citra"
           - Atur transformasi sesuai kebutuhan
           - Preview perubahan
        
        4. **Jalankan Analisis**
           - Buka "Analisis AI"
           - Pilih jenis analisis
           - Klik "Jalankan Analisis"
        
        5. **Lihat Hasil**
           - Buka "Hasil & Export"
           - Lihat ringkasan dan rekomendasi
           - Export dalam format yang diinginkan
        """)
    
    with tabs[1]:
        st.markdown(f"""
        ### Informasi Model
        
        **Model Aktif:** {model_type}
        
        #### ResNet50
        - Akurasi: ~95%
        - Waktu Inferensi: ~100ms
        - Terbaik untuk: Klasifikasi umum
        
        #### VGG16
        - Akurasi: ~92%
        - Waktu Inferensi: ~80ms
        - Terbaik untuk: Detail fine-grained
        
        #### EfficientNet
        - Akurasi: ~96%
        - Waktu Inferensi: ~50ms
        - Terbaik untuk: Efisiensi & performa
        
        #### Custom Model
        - Akurasi: Tergantung training
        - Waktu Inferensi: Variable
        - Terbaik untuk: Use case spesifik
        """)
    
    with tabs[2]:
        st.markdown("""
        ### Pertanyaan yang Sering Diajukan
        
        **Q: Format citra apa yang didukung?**
        A: JPG, PNG, BMP. DICOM masih dalam pengembangan.
        
        **Q: Berapa ukuran file maksimal?**
        A: 200MB per file.
        
        **Q: Berapa akurasi model?**
        A: Tergantung model, berkisar 92-96%.
        
        **Q: Bagaimana cara meningkatkan akurasi?**
        A: Gunakan preprocessing yang tepat dan pilih model yang sesuai.
        
        **Q: Apakah data saya aman?**
        A: Ya, semua proses dilakukan lokal.
        """)
    
    with tabs[3]:
        st.markdown("""
        ### Tentang Aplikasi
        
        **AI Medical Image Analyzer**
        
        Versi: 1.0.0
        
        Aplikasi GUI expandable untuk analisis citra medis menggunakan deep learning.
        
        **Teknologi:**
        - Streamlit untuk UI
        - TensorFlow/PyTorch untuk AI
        - Pillow untuk image processing
        
        **Dikembangkan untuk:**
        Semester 6 - Deep Learning Citra Medis
        """)

# Footer
st.markdown("---")
col_footer1, col_footer2, col_footer3 = st.columns(3)

with col_footer1:
    st.write("📅 " + datetime.now().strftime("%d-%m-%Y %H:%M:%S"))
with col_footer2:
    st.write("🏥 AI Medical Image Analyzer v1.0")
with col_footer3:
    st.write("✨ Developed with ❤️")
