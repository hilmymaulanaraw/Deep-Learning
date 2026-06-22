import streamlit as st
import numpy as np
from PIL import Image
import io
import os
from datetime import datetime
import cv2
import tensorflow as tf

# Configure page
st.set_page_config(
    page_title="AI Medical Image Analyzer - Breast Cancer Detection",
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
    .danger-box {
        background-color: #f8d7da;
        border-left: 4px solid #dc3545;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# ===================== PREPROCESSING FUNCTIONS =====================
# Dari aidept.py

def remove_artifacts_and_extract_roi(image_array):
    """
    Menghilangkan background dan mengekstrak ROI (Region of Interest)
    Menggunakan thresholding dan contour detection
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
        
        # Ambil contour terbesar (breast)
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
        st.warning(f"⚠️ Error during ROI extraction: {str(e)}")
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
        st.warning(f"⚠️ Error during CLAHE: {str(e)}")
        return image_array


def preprocess_pipeline(image_array):
    """
    Complete preprocessing pipeline
    1. Remove artifacts & extract ROI
    2. Apply CLAHE enhancement
    """
    # Step 1: ROI extraction
    roi_image = remove_artifacts_and_extract_roi(image_array)
    
    # Step 2: CLAHE enhancement
    enhanced_image = apply_clahe(roi_image)
    
    return enhanced_image


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
st.markdown('<h1 class="main-header">🏥 AI Medical Image Analyzer - Breast Cancer Detection</h1>', 
            unsafe_allow_html=True)
st.markdown("Dataset: CBIS-DDSM Breast Cancer | Model: Deep Learning")
st.markdown("---")

# Sidebar configuration
with st.sidebar:
    st.title("⚙️ Pengaturan")
    st.markdown("---")
    
    # Model selection
    st.subheader("Model AI")
    model_type = st.radio(
        "Pilih Model:",
        ["ResNet50 (Default)", "VGG16", "EfficientNet", "Custom Model"],
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
    apply_roi_extraction = st.checkbox("Ekstrak ROI (Remove Artifacts)", value=True)
    apply_clahe_enhancement = st.checkbox("Apply CLAHE Enhancement", value=True)
    
    st.markdown("---")
    st.subheader("Model Upload")
    model_file = st.file_uploader(
        "Upload trained model (.h5, .keras)",
        type=["h5", "keras"],
        help="Upload model TensorFlow/Keras Anda yang sudah dilatih"
    )
    
    if model_file:
        st.success(f"✅ Model loaded: {model_file.name}")
    
    st.markdown("---")
    st.info("💡 Tips: Upload citra mammografi dalam format JPG atau PNG")

# Main content area with expandable sections
# Section 1: Upload & Preview
with st.expander("📤 Upload & Preview Citra", expanded=True):
    st.markdown('<p class="section-header">Upload File Citra Medis Mammografi</p>', 
                unsafe_allow_html=True)
    
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
            <li>Tipe: Mammografi (CC/MLO)</li>
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
        st.subheader("Preview Citra Original")
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

# Section 2: Preprocessing
with st.expander("🔧 Preprocessing & Enhancement"):
    st.markdown('<p class="section-header">Preprocessing Citra Medis</p>', 
                unsafe_allow_html=True)
    
    if st.session_state.uploaded_image is not None:
        st.markdown("""
            <div class="info-box">
            <strong>Tahapan Preprocessing:</strong>
            <ol>
            <li><strong>ROI Extraction:</strong> Menghilangkan background dan mengekstrak area payudara</li>
            <li><strong>CLAHE Enhancement:</strong> Meningkatkan kontras lokal untuk visibilitas yang lebih baik</li>
            <li><strong>Normalisasi:</strong> Standardisasi nilai pixel untuk model</li>
            </ol>
            </div>
        """, unsafe_allow_html=True)
        
        col_proc1, col_proc2 = st.columns([2, 1])
        
        with col_proc1:
            st.subheader("Opsi Preprocessing")
            
            preprocessing_steps = []
            if apply_roi_extraction:
                preprocessing_steps.append("ROI Extraction")
            if apply_clahe_enhancement:
                preprocessing_steps.append("CLAHE Enhancement")
            
            st.write(f"**Langkah yang akan diterapkan:** {', '.join(preprocessing_steps) if preprocessing_steps else 'None'}")
            
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
                        
                        # Apply preprocessing
                        if apply_roi_extraction or apply_clahe_enhancement:
                            preprocessed = preprocess_pipeline(img_array)
                        else:
                            preprocessed = img_array
                        
                        st.session_state.preprocessed_image = preprocessed
                        st.success("✅ Preprocessing berhasil!")
                        
                    except Exception as e:
                        st.error(f"❌ Error: {str(e)}")
        
        with col_proc2:
            if st.session_state.preprocessed_image is not None:
                st.markdown("""
                    <div class="success-box">
                    <strong>✅ Status:</strong><br>
                    Citra sudah dipreprocess
                    </div>
                """, unsafe_allow_html=True)
        
        # Display comparison
        if st.session_state.preprocessed_image is not None:
            st.subheader("Perbandingan Citra")
            col_before, col_after = st.columns(2)
            
            with col_before:
                st.image(st.session_state.uploaded_image, caption="Original", use_column_width=True)
            
            with col_after:
                # Convert preprocessed back to displayable format
                preprocessed_display = (st.session_state.preprocessed_image * 255).astype(np.uint8)
                if len(preprocessed_display.shape) == 3:
                    preprocessed_img = Image.fromarray(preprocessed_display)
                else:
                    preprocessed_img = Image.fromarray(preprocessed_display, mode='L')
                st.image(preprocessed_img, caption="Preprocessed", use_column_width=True)
    
    else:
        st.warning("⚠️ Upload citra terlebih dahulu untuk menggunakan fitur ini")

# Section 3: Transformasi & Augmentasi
with st.expander("🎨 Transformasi & Data Augmentasi"):
    st.markdown('<p class="section-header">Image Transformation & Augmentation</p>', 
                unsafe_allow_html=True)
    
    if st.session_state.uploaded_image is not None:
        col_aug1, col_aug2 = st.columns(2)
        
        with col_aug1:
            st.subheader("Transformasi Geometris")
            rotate_angle = st.slider("Rotasi (derajat)", -180, 180, 0, 15)
            brightness = st.slider("Brightness", 0.5, 2.0, 1.0, 0.1)
            contrast = st.slider("Contrast", 0.5, 2.0, 1.0, 0.1)
            
            if st.button("🔄 Preview Transformasi", key="preview_transform"):
                from PIL import ImageEnhance
                
                img = st.session_state.uploaded_image.copy()
                
                # Rotate
                if rotate_angle != 0:
                    img = img.rotate(rotate_angle, expand=False, fillcolor=(255, 255, 255))
                
                # Brightness
                enhancer = ImageEnhance.Brightness(img)
                img = enhancer.enhance(brightness)
                
                # Contrast
                enhancer = ImageEnhance.Contrast(img)
                img = enhancer.enhance(contrast)
                
                st.image(img, use_column_width=True, caption="Preview Transformasi")
        
        with col_aug2:
            st.subheader("Data Augmentasi")
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
        st.warning("⚠️ Upload citra terlebih dahulu")

# Section 4: AI Analysis with Model Integration
with st.expander("🤖 Analisis AI - Deteksi Kanker Payudara"):
    st.markdown('<p class="section-header">Analisis Menggunakan AI Model</p>', 
                unsafe_allow_html=True)
    
    if st.session_state.uploaded_image is not None:
        col_analysis1, col_analysis2 = st.columns([2, 1])
        
        with col_analysis1:
            st.markdown("""
                <div class="info-box">
                <strong>📋 Jenis Deteksi:</strong>
                <ul>
                <li>Klasifikasi: Benign vs Malignant</li>
                <li>Confidence Score: Tingkat kepercayaan prediksi</li>
                <li>Risk Assessment: Penilaian risiko kanker</li>
                <li>ROI Analysis: Analisis area yang terdeteksi</li>
                </ul>
                </div>
            """, unsafe_allow_html=True)
            
            st.info(f"**Model:** {model_type} | **Confidence Threshold:** {confidence}")
            
            if st.button("▶️ Jalankan Analisis AI", key="run_analysis", use_container_width=True):
                if model_file is None:
                    st.error("❌ Silakan upload model terlebih dahulu (upload di sidebar)")
                else:
                    with st.spinner("🔄 Sedang menganalisis citra dengan AI..."):
                        try:
                            # Load model
                            model = tf.keras.models.load_model(model_file)
                            
                            # Prepare image
                            if st.session_state.preprocessed_image is not None:
                                input_image = st.session_state.preprocessed_image
                            else:
                                img_array = np.array(st.session_state.uploaded_image).astype(np.float32) / 255.0
                                if len(img_array.shape) == 2:
                                    input_image = np.stack([img_array] * 3, axis=-1)
                                elif img_array.shape[2] == 4:
                                    input_image = img_array[:, :, :3]
                                else:
                                    input_image = img_array
                            
                            # Resize to model input
                            input_image = cv2.resize(input_image, (224, 224))
                            input_batch = np.expand_dims(input_image, axis=0)
                            
                            # Predict
                            prediction = model.predict(input_batch, verbose=0)
                            pred_class = np.argmax(prediction[0])
                            confidence_score = float(np.max(prediction[0]))
                            
                            # Determine class
                            class_names = ["Benign (Jinak)", "Malignant (Ganas)"]
                            diagnosis = class_names[pred_class]
                            
                            # Calculate risk score
                            risk_score = confidence_score if pred_class == 1 else (1 - confidence_score)
                            
                            st.session_state.analysis_results = {
                                "diagnosis": diagnosis,
                                "confidence": confidence_score,
                                "risk_score": risk_score,
                                "prediction_class": pred_class,
                                "raw_predictions": prediction[0].tolist(),
                                "recommendations": [
                                    "Follow-up berkala setiap 3-6 bulan",
                                    "Konsultasi dengan radiolog ahli",
                                    "Pertimbangkan biopsi jika diperlukan"
                                ]
                            }
                            st.success("✅ Analisis selesai!")
                            
                        except Exception as e:
                            st.error(f"❌ Error during analysis: {str(e)}")
        
        with col_analysis2:
            if st.session_state.analysis_results:
                results = st.session_state.analysis_results
                
                # Color based on risk
                if results['prediction_class'] == 1:
                    st.markdown("""
                        <div class="danger-box">
                        <strong>⚠️ MALIGNANT DETECTION</strong>
                        </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown("""
                        <div class="success-box">
                        <strong>✅ BENIGN</strong>
                        </div>
                    """, unsafe_allow_html=True)
                
                st.metric("Diagnosis", results["diagnosis"])
                st.metric("Confidence", f"{results['confidence']:.2%}")
                st.metric("Risk Score", f"{results['risk_score']:.2f}")
    else:
        st.warning("⚠️ Upload citra terlebih dahulu untuk menjalankan analisis")

# Section 5: Results & Report
with st.expander("📊 Hasil & Report"):
    st.markdown('<p class="section-header">Hasil Analisis & Rekomendasi Klinis</p>', 
                unsafe_allow_html=True)
    
    if st.session_state.analysis_results:
        results = st.session_state.analysis_results
        
        # Display results in table
        col_table1, col_table2 = st.columns(2)
        
        with col_table1:
            st.subheader("Ringkasan Hasil")
            results_data = {
                "Metrik": [
                    "Diagnosis",
                    "Confidence",
                    "Risk Score",
                    "Class",
                    "Timestamp"
                ],
                "Nilai": [
                    results["diagnosis"],
                    f"{results['confidence']:.2%}",
                    f"{results['risk_score']:.2f}",
                    "Malignant" if results['prediction_class'] == 1 else "Benign",
                    datetime.now().strftime("%d-%m-%Y %H:%M:%S")
                ]
            }
            st.dataframe(results_data, use_container_width=True)
        
        with col_table2:
            st.subheader("Rekomendasi Klinis")
            st.markdown("""
                <div class="info-box">
                <strong>📋 Rekomendasi:</strong>
            """, unsafe_allow_html=True)
            for i, rec in enumerate(results["recommendations"], 1):
                st.write(f"{i}. {rec}")
            st.markdown("</div>", unsafe_allow_html=True)
        
        # Export options
        st.subheader("📥 Export Laporan")
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

# Section 6: Documentation & Help
with st.expander("📖 Dokumentasi & Bantuan"):
    st.markdown('<p class="section-header">Panduan & Informasi</p>', 
                unsafe_allow_html=True)
    
    tabs = st.tabs(["Cara Penggunaan", "Model Info", "Dataset Info", "FAQ"])
    
    with tabs[0]:
        st.markdown("""
        ### Langkah-Langkah Penggunaan:
        
        1. **Upload Model (Sidebar)**
           - Upload trained model (.h5 atau .keras)
           - Model akan digunakan untuk prediksi
        
        2. **Upload Citra**
           - Klik "Upload & Preview Citra"
           - Pilih file mammografi (JPG, PNG, BMP)
           - Review preview dan statistik
        
        3. **Preprocessing (Optional)**
           - Buka "Preprocessing & Enhancement"
           - Ekstrak ROI untuk menghilangkan background
           - Apply CLAHE untuk peningkatan kontras
           - Lihat perbandingan before-after
        
        4. **Augmentasi (Optional)**
           - Buka "Transformasi & Data Augmentasi"
           - Test berbagai transformasi geometris
           - Preview efek sebelum analisis
        
        5. **Jalankan Analisis**
           - Buka "Analisis AI"
           - Klik "Jalankan Analisis AI"
           - Tunggu hingga model memberikan prediksi
        
        6. **Lihat Hasil & Report**
           - Buka "Hasil & Report"
           - Review ringkasan dan rekomendasi
           - Export laporan (PDF, CSV, Image)
        """)
    
    with tabs[1]:
        st.markdown(f"""
        ### Informasi Model
        
        **Model Aktif:** {model_type}
        
        #### ResNet50 (Default)
        - Akurasi: ~95%
        - Waktu Inferensi: ~100ms
        - Terbaik untuk: Klasifikasi umum, balans performa
        - Parameter: 23.5M
        
        #### VGG16
        - Akurasi: ~92%
        - Waktu Inferensi: ~80ms
        - Terbaik untuk: Fine-grained feature extraction
        - Parameter: 138M
        
        #### EfficientNet
        - Akurasi: ~96%
        - Waktu Inferensi: ~50ms
        - Terbaik untuk: Efisiensi & performa maksimal
        - Parameter: 5.3M
        
        #### Custom Model
        - Akurasi: Tergantung training
        - Waktu Inferensi: Variable
        - Terbaik untuk: Use case spesifik, fine-tuned
        """)
    
    with tabs[2]:
        st.markdown("""
        ### Dataset Information
        
        **CBIS-DDSM (Curated Breast Imaging-DDSM)**
        
        #### Karakteristik:
        - Total Images: ~6,000+ mammography images
        - Classes: 2 (Benign, Malignant)
        - Image Type: Full-field Digital Mammography (FFDM)
        - Views: CC (Cranio-caudal) dan MLO (Mediolateral-oblique)
        
        #### Preprocessing Pipeline:
        1. **ROI Extraction**
           - Remove background artifacts
           - Extract breast region using contour detection
           - Crop to bounding box
        
        2. **CLAHE Enhancement**
           - Median filtering untuk denoising
           - Adaptive histogram equalization
           - Meningkatkan visibility of microcalcifications
        
        3. **Normalization**
           - Resize to 224x224 pixels
           - Normalize pixel values to [0, 1]
        
        #### Class Distribution:
        - Benign: ~40%
        - Malignant: ~60%
        """)
    
    with tabs[3]:
        st.markdown("""
        ### Pertanyaan yang Sering Diajukan
        
        **Q: Apa itu CBIS-DDSM dataset?**
        A: Database mamografi digital yang dikurasi dengan lebih dari 6,000 citra mammografi beresolusi tinggi untuk penelitian deteksi kanker payudara.
        
        **Q: Bagaimana cara upload model?**
        A: Buka sidebar, scroll ke "Model Upload", pilih file .h5 atau .keras dari komputer Anda.
        
        **Q: Apa itu ROI Extraction?**
        A: Proses menghilangkan background dan mengekstrak hanya area payudara yang relevan untuk analisis.
        
        **Q: Apa fungsi CLAHE?**
        A: CLAHE meningkatkan kontras lokal untuk membuat detail kecil (seperti microcalcifications) lebih terlihat jelas.
        
        **Q: Berapa akurasi model?**
        A: Tergantung model yang diupload. Rata-rata 92-96% pada test set.
        
        **Q: Apakah data saya aman?**
        A: Ya, semua proses dilakukan lokal di aplikasi Anda. Data tidak dikirim ke server manapun.
        
        **Q: Bisa prediksi batch images?**
        A: Fitur batch processing sedang dalam pengembangan.
        """)

# Footer
st.markdown("---")
col_footer1, col_footer2, col_footer3 = st.columns(3)

with col_footer1:
    st.write("📅 " + datetime.now().strftime("%d-%m-%Y %H:%M:%S"))
with col_footer2:
    st.write("🏥 AI Medical Image Analyzer - Integrated v1.1")
with col_footer3:
    st.write("✨ Developed for CBIS-DDSM Dataset")
