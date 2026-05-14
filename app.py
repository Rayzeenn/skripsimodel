import streamlit as st
import cv2
import numpy as np
import torch
from PIL import Image
import sys
import os

# Konfigurasi Halaman Streamlit
st.set_page_config(page_title="Deteksi Wajah YOLOv5 CLAHE", layout="wide")

# Konfigurasi Device (CPU/GPU)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ---------------------------------------------------------
# 1. Modul CLAHE Enhancer
# ---------------------------------------------------------
class CLAHEEnhancer:
    def __init__(self, clip_limit=1.0, tile_size=(8, 8)):
        self.clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_size)

    def enhance(self, image_bgr: np.ndarray) -> np.ndarray:
        lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l_enhanced = self.clahe.apply(l)
        lab_enhanced = cv2.merge([l_enhanced, a, b])
        return cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)

    def simulate_low_light(self, image_bgr: np.ndarray, gamma: float = 0.5) -> np.ndarray:
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)]).astype('uint8')
        return cv2.LUT(image_bgr, table)

# ---------------------------------------------------------
# 2. Modul YOLOv5-Face Detector (dengan Caching)
# ---------------------------------------------------------
@st.cache_resource
def load_yolo_model(yolov5_dir, weights_path, conf_threshold):
    # Memasukkan folder yolov5-face ke sys.path agar bisa import modul
    if yolov5_dir not in sys.path:
        sys.path.insert(0, yolov5_dir)
    
    try:
        from models.experimental import attempt_load
        model = attempt_load(weights_path, map_location=DEVICE)
        model.conf = conf_threshold
        model.iou = 0.45
        model.eval()
        return model
    except Exception as e:
        st.error(f"Gagal memuat model YOLOv5: {e}")
        return None

class YOLOv5FaceDetector:
    def __init__(self, model, conf_threshold=0.4):
        self.model = model
        self.conf_threshold = conf_threshold

    def detect(self, image_bgr: np.ndarray) -> list:
        if self.model is None:
            return []
            
        from utils.general import non_max_suppression
        img_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        h, w = img_rgb.shape[:2]
        
        # Resize ke ukuran standar YOLOv5 (640x640)
        img_resized = cv2.resize(img_rgb, (640, 640))
        img_array = img_resized.transpose(2, 0, 1).astype('float32') / 255.0
        img_tensor = torch.tensor(img_array, dtype=torch.float32).unsqueeze(0).to(DEVICE)
        
        with torch.no_grad():
            results = self.model(img_tensor)
            
        pred = non_max_suppression(results[0], conf_thres=self.conf_threshold, iou_thres=0.45)
        detections = pred[0]
        
        faces = []
        if detections is not None and len(detections):
            for det in detections.cpu().tolist():
                x1, y1 = int(det[0] * w / 640), int(det[1] * h / 640)
                x2, y2 = int(det[2] * w / 640), int(det[3] * h / 640)
                conf = float(det[4])
                if conf >= self.conf_threshold:
                    faces.append([x1, y1, x2, y2, conf])
                    
        return sorted(faces, key=lambda x: x[4], reverse=True)

# ---------------------------------------------------------
# 3. Fungsi Helper untuk Menggambar Bounding Box
# ---------------------------------------------------------
def draw_boxes(img: np.ndarray, detector: YOLOv5FaceDetector):
    dets = detector.detect(img)
    out = img.copy()
    for d in dets:
        # Bounding box hijau
        cv2.rectangle(out, (d[0], d[1]), (d[2], d[3]), (0, 255, 0), 2)
        # Teks confidence score
        cv2.putText(out, f'{d[4]:.2f}', (d[0], d[1] - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    return out, len(dets)

# ---------------------------------------------------------
# 4. Antarmuka Streamlit Utama
# ---------------------------------------------------------
def main():
    st.title("Peningkatan Deteksi Wajah YOLOv5 Pada Cahaya Rendah")
    st.markdown("**Pipeline:** Gambar Original → Simulasi Cahaya Rendah → CLAHE Enhancement → YOLOv5 Face Detection")

    # Konfigurasi Sidebar
    st.sidebar.header("Konfigurasi Sistem")
    
    # Path Direktori dan Weights
    yolov5_dir = st.sidebar.text_input("Path Folder YOLOv5-Face", value="./yolov5-face")
    weights_path = st.sidebar.text_input("Path Weights (.pt)", value="best.pt")
    
    # Parameter Model dan Enhancer
    conf_thresh = st.sidebar.slider("Confidence Threshold YOLO", min_value=0.1, max_value=1.0, value=0.4, step=0.05)
    gamma_val = st.sidebar.slider("Gamma (Simulasi Low-Light)", min_value=0.1, max_value=1.0, value=0.5, step=0.1)
    clip_limit = st.sidebar.slider("CLAHE Clip Limit", min_value=1.0, max_value=5.0, value=1.0, step=0.5)

    # Inisialisasi Modul
    clahe_enhancer = CLAHEEnhancer(clip_limit=clip_limit, tile_size=(8, 8))
    
    # Coba muat model
    if os.path.exists(yolov5_dir) and os.path.exists(weights_path):
        model = load_yolo_model(yolov5_dir, weights_path, conf_thresh)
        yolo_detector = YOLOv5FaceDetector(model, conf_threshold=conf_thresh)
    else:
        st.warning("Silakan pastikan direktori `yolov5-face` dan file `weights` tersedia untuk melanjutkan deteksi.")
        yolo_detector = None

    st.write("---")
    uploaded_file = st.file_uploader("Unggah Gambar", type=['jpg', 'jpeg', 'png'])

    if uploaded_file is not None and yolo_detector is not None:
        # Konversi file unggahan menjadi array OpenCV BGR
        image = Image.open(uploaded_file).convert('RGB')
        image_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

        with st.spinner("Memproses pipeline gambar..."):
            # 1. Terapkan efek low-light
            low_light_bgr = clahe_enhancer.simulate_low_light(image_bgr, gamma=gamma_val)
            
            # 2. Lakukan enhancement CLAHE pada gambar low-light
            enhanced_bgr = clahe_enhancer.enhance(low_light_bgr)

            # 3. Hitung deteksi dan gambar box
            orig_box, count_orig = draw_boxes(image_bgr, yolo_detector)
            dark_box, count_dark = draw_boxes(low_light_bgr, yolo_detector)
            enh_box, count_enh = draw_boxes(enhanced_bgr, yolo_detector)

        # Menampilkan perbandingan secara berdampingan
        col1, col2, col3 = st.columns(3)

        with col1:
            st.image(cv2.cvtColor(orig_box, cv2.COLOR_BGR2RGB), use_container_width=True)
            st.info(f"**Original**\n\nWajah terdeteksi: {count_orig}")

        with col2:
            st.image(cv2.cvtColor(dark_box, cv2.COLOR_BGR2RGB), use_container_width=True)
            st.warning(f"**Simulasi Low-Light (γ={gamma_val})**\n\nWajah terdeteksi: {count_dark}")

        with col3:
            st.image(cv2.cvtColor(enh_box, cv2.COLOR_BGR2RGB), use_container_width=True)
            st.success(f"**Hasil CLAHE**\n\nWajah terdeteksi: {count_enh}")

if __name__ == '__main__':
    main()
