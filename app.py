import os
import sys
import cv2
import numpy as np
import streamlit as st
import torch
from PIL import Image
import pathlib
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration
import av

# ─── FIX BUG PYTORCH 2.6+ (SAFE GLOBALS & MONKEY PATCH) ───────────────────────
# 1. Paksa override fungsi torch.load sebagai lapis keamanan utama
_original_torch_load = torch.load
def _patched_load(*args, **kwargs):
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False
    return _original_torch_load(*args, **kwargs)
torch.load = _patched_load

os.environ['TORCH_FORCE_WEIGHTS_ONLY_LOAD'] = '0'

# 2. Masukkan folder YOLO ke sys.path
YOLO_DIR = "yolov5-face"
if os.path.exists(YOLO_DIR) and YOLO_DIR not in sys.path:
    sys.path.insert(0, YOLO_DIR)

# 3. Daftarkan safe globals
try:
    from torch.serialization import add_safe_globals
    from models.yolo import Model
    from models.common import Conv, Bottleneck, SPP, Focus, BottleneckCSP, Concat
    
    add_safe_globals([
        np.core.multiarray._reconstruct,
        np.ndarray,
        np.dtype,
        np.core.multiarray.scalar,
        type(np.dtype('float64')),
        pathlib.PosixPath,
        pathlib.WindowsPath,
        Model, Conv, Bottleneck, SPP, Focus, BottleneckCSP, Concat
    ])
except Exception as e:
    pass
# ──────────────────────────────────────────────────────────────────────────────

# ─── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="FaceVision CLAHE",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;600&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
.stApp { background: #0a0a0f; color: #e8e8f0; }
h1, h2, h3 { font-family: 'Space Mono', monospace !important; letter-spacing: -0.03em; }
.main-title { font-family: 'Space Mono', monospace; font-size: 2.4rem; font-weight: 700; background: linear-gradient(135deg, #7DF9FF 0%, #89CFF0 50%, #B8A9FA 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text; margin-bottom: 0.2rem; }
.sub-title { color: #6b7280; font-size: 0.95rem; margin-bottom: 2rem; font-family: 'Space Mono', monospace; }
.metric-card { background: #13131a; border: 1px solid #2a2a3a; border-radius: 12px; padding: 1.2rem 1.5rem; margin: 0.4rem 0; }
.metric-label { color: #6b7280; font-size: 0.75rem; font-family: 'Space Mono', monospace; text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 0.3rem; }
.metric-value { color: #7DF9FF; font-size: 1.6rem; font-weight: 700; font-family: 'Space Mono', monospace; }
.result-card { background: linear-gradient(135deg, #13131a, #1a1a2e); border: 1px solid #2a2a3a; border-radius: 16px; padding: 1.5rem; margin: 1rem 0; }
.badge { display: inline-block; padding: 0.25rem 0.75rem; border-radius: 999px; font-size: 0.75rem; font-family: 'Space Mono', monospace; font-weight: 700; margin: 0.2rem; }
.badge-cyan { background: rgba(125, 249, 255, 0.15); color: #7DF9FF; border: 1px solid rgba(125, 249, 255, 0.3); }
.badge-green { background: rgba(74, 222, 128, 0.15); color: #4ade80; border: 1px solid rgba(74, 222, 128, 0.3); }
.badge-yellow { background: rgba(250, 204, 21, 0.15); color: #facc15; border: 1px solid rgba(250, 204, 21, 0.3); }
.pipeline-step { display: flex; align-items: center; gap: 0.75rem; padding: 0.6rem 0; border-bottom: 1px solid #1e1e2e; font-size: 0.85rem; color: #9ca3af; }
.step-dot { width: 8px; height: 8px; border-radius: 50%; background: #7DF9FF; flex-shrink: 0; }
.info-box { background: rgba(125, 249, 255, 0.05); border: 1px solid rgba(125, 249, 255, 0.2); border-radius: 10px; padding: 1rem; font-size: 0.85rem; color: #9ca3af; margin: 0.75rem 0; }
.stButton > button { background: linear-gradient(135deg, #7DF9FF, #89CFF0) !important; color: #0a0a0f !important; border: none !important; font-family: 'Space Mono', monospace !important; font-weight: 700 !important; border-radius: 8px !important; padding: 0.5rem 1.5rem !important; }
.stFileUploader label { color: #9ca3af !important; font-size: 0.85rem !important; }
div[data-testid="stImage"] img { border-radius: 12px; }
</style>
""", unsafe_allow_html=True)

# ─── Device Setup ─────────────────────────────────────────────────────────────
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ─── CLAHE Enhancer ───────────────────────────────────────────────────────────
class CLAHEEnhancer:
    def __init__(self, clip_limit=2.0, tile_size=(8, 8)):
        self.clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_size)
        self.clip_limit = clip_limit
        self.tile_size = tile_size

    def enhance(self, image_bgr: np.ndarray) -> np.ndarray:
        lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l_enhanced = self.clahe.apply(l)
        lab_enhanced = cv2.merge([l_enhanced, a, b])
        return cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)

    def update_params(self, clip_limit, tile_size):
        self.clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_size)
        self.clip_limit = clip_limit
        self.tile_size = tile_size

# ─── YOLOv5-Face Detector ─────────────────────────────────────────────────────
class YOLOv5FaceDetector:
    def __init__(self, weights_path: str, conf_threshold=0.4):
        self.conf_threshold = conf_threshold
        self.loaded = False
        self.model = None

        if not os.path.exists(weights_path):
            st.sidebar.error(f"❌ YOLOv5 weights tidak ditemukan:\n`{weights_path}`")
            return

        yolo_dir = os.path.dirname(os.path.dirname(weights_path))
        if yolo_dir not in sys.path:
            sys.path.insert(0, yolo_dir)

        try:
            from models.experimental import attempt_load
            self.model = attempt_load(weights_path, map_location=DEVICE)
            self.model.conf = conf_threshold
            self.model.iou = 0.45
            self.model.eval()
            self.loaded = True
        except Exception as e:
            st.sidebar.error(f"❌ Gagal memuat YOLOv5: {e}")

    def detect(self, image_bgr: np.ndarray) -> list:
        if not self.loaded:
            return []
        try:
            from utils.general import scale_coords
        except ImportError:
            return []

        img_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        shape = img_rgb.shape[:2]
        
        # Letterbox padding
        r = min(640 / shape[0], 640 / shape[1])
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = 640 - new_unpad[0], 640 - new_unpad[1]
        dw, dh = np.mod(dw, 32) / 2, np.mod(dh, 32) / 2
        
        img_resized = cv2.resize(img_rgb, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        img_padded = cv2.copyMakeBorder(img_resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))
        
        img_array = img_padded.transpose(2, 0, 1).astype('float32') / 255.0
        img_tensor = torch.tensor(img_array, dtype=torch.float32).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            results = self.model(img_tensor)

        pred = results[0]

        # FIX BUG NMS 11 WAJAH MENUMPUK
        try:
            from utils.general import non_max_suppression_face
            pred = non_max_suppression_face(pred, conf_thres=self.conf_threshold, iou_thres=0.45)
        except ImportError:
            from utils.general import non_max_suppression
            if pred.shape[-1] > 6:
                clean_pred = torch.zeros((pred.shape[0], pred.shape[1], 6), device=pred.device)
                clean_pred[..., :5] = pred[..., :5]
                clean_pred[..., 5] = 1.0 # Paksa menjadi 1 kelas (wajah) untuk abaikan landmarks
                pred = clean_pred
            pred = non_max_suppression(pred, conf_thres=self.conf_threshold, iou_thres=0.45)

        detections = pred[0]
        faces = []
        if detections is not None and len(detections):
            detections[:, :4] = scale_coords(img_tensor.shape[2:], detections[:, :4], img_rgb.shape).round()
            for det in detections.cpu().tolist():
                x1, y1, x2, y2 = int(det[0]), int(det[1]), int(det[2]), int(det[3])
                conf = float(det[4])
                if conf >= self.conf_threshold:
                    faces.append([x1, y1, x2, y2, conf])

        return sorted(faces, key=lambda x: x[4], reverse=True)

    def crop_face(self, image_bgr: np.ndarray, det: list, target_size=(160, 160)):
        x1, y1, x2, y2 = int(det[0]), int(det[1]), int(det[2]), int(det[3])
        h, w = image_bgr.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 <= x1 or y2 <= y1:
            return None
        face = image_bgr[y1:y2, x1:x2]
        return cv2.resize(face, target_size)

# ─── FaceNet Embedder ─────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Memuat FaceNet model...")
def load_facenet():
    try:
        from facenet_pytorch import InceptionResnetV1
        model = InceptionResnetV1(pretrained='vggface2').eval().to(DEVICE)
        return model
    except Exception as e:
        st.error(f"Gagal memuat FaceNet: {e}")
        return None

def get_embedding(face_bgr: np.ndarray, facenet_model) -> np.ndarray:
    face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(face_rgb).resize((160, 160))
    tensor = torch.tensor(np.array(img_pil), dtype=torch.float32)
    tensor = (tensor - 127.5) / 128.0
    tensor = tensor.permute(2, 0, 1).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        emb_tensor = facenet_model(tensor).cpu().detach()
        emb = np.array(emb_tensor.tolist(), dtype='float32')[0]
    emb = emb / (np.linalg.norm(emb) + 1e-10)
    return emb

# ─── Draw detections on image ─────────────────────────────────────────────────
def draw_detections(image_bgr: np.ndarray, detections: list, labels_map: dict = None) -> np.ndarray:
    img = image_bgr.copy()
    for det in detections:
        x1, y1, x2, y2, conf = int(det[0]), int(det[1]), int(det[2]), int(det[3]), det[4]
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 180), 2)
        label_text = f"{conf:.2f}"
        if labels_map and (x1, y1) in labels_map:
            label_text = labels_map[(x1, y1)]
        cv2.putText(img, label_text, (x1, max(y1 - 8, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 180), 2)
    return img

# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="main-title" style="font-size:1.3rem;">⚙️ Konfigurasi</div>', unsafe_allow_html=True)
    st.markdown("### 📁 Model Path")
    yolo_weights = st.text_input("YOLOv5-Face Weights", value="best.pt")
    yolo_repo_path = st.text_input("YOLOv5-Face Repo Path", value="yolov5-face")

    st.markdown("---")
    st.markdown("### 🎛️ CLAHE Parameters")
    clip_limit = st.slider("Clip Limit", 0.5, 10.0, 2.0, 0.5)
    tile_w = st.select_slider("Tile Width", options=[4, 8, 16, 32], value=8)
    tile_h = st.select_slider("Tile Height", options=[4, 8, 16, 32], value=8)

    st.markdown("---")
    st.markdown("### 🔍 Detection Settings")
    conf_thresh = st.slider("Confidence Threshold", 0.1, 0.9, 0.4, 0.05)
    similarity_threshold = st.slider("Similarity Threshold", 0.3, 0.9, 0.6, 0.05)
    
    device_badge = "🟢 GPU (CUDA)" if torch.cuda.is_available() else "🟡 CPU"
    st.markdown(f'<br><span class="badge badge-cyan">{device_badge}</span>', unsafe_allow_html=True)

# ─── Load Models ──────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Memuat YOLOv5-Face...")
def load_yolo(weights, repo, conf):
    if repo not in sys.path:
        sys.path.insert(0, repo)
    return YOLOv5FaceDetector(weights, conf)

facenet = load_facenet()
clahe_enhancer = CLAHEEnhancer(clip_limit, (tile_w, tile_h))

# ─── Main UI ──────────────────────────────────────────────────────────────────
st.markdown('<div class="main-title">👁️ FaceVision CLAHE</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Deteksi Wajah Real-Time · YOLOv5-Face + CLAHE Enhancement · FaceNet VGGFace2</div>', unsafe_allow_html=True)

if "face_db" not in st.session_state:
    st.session_state.face_db = {
        "embeddings": np.zeros((0, 512), dtype="float32"),
        "names": [],
        "face_images": [],
    }

tab1, tab2, tab3 = st.tabs(["🖼️ Deteksi & Enhance", "🗄️ Database Wajah", "📊 Evaluasi CLAHE"])

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1: Detection
# ═══════════════════════════════════════════════════════════════════════════════
with tab1:
    col_left, col_right = st.columns([1, 1], gap="large")

   with col_left:
        st.markdown("#### 📤 Input Gambar")
        input_mode = st.radio("Sumber Input", ["Upload File", "Kamera (Foto)", "Live Video (WebRTC)"], horizontal=True)
        apply_clahe_toggle = st.checkbox("Terapkan CLAHE Enhancement", value=True)

        image_input = None
        if input_mode == "Upload File":
            uploaded = st.file_uploader("Pilih gambar", type=["jpg", "jpeg", "png", "bmp", "webp"])
            if uploaded:
                from PIL import ImageOps
                # Membaca EXIF agar gambar tegak secara otomatis
                pil_img = ImageOps.exif_transpose(Image.open(uploaded)).convert("RGB")
                image_input = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
                
        elif input_mode == "Kamera (Foto)":
            camera_img = st.camera_input("Ambil foto")
            if camera_img:
                image_input = cv2.cvtColor(np.array(Image.open(camera_img).convert("RGB")), cv2.COLOR_RGB2BGR)
                
        elif input_mode == "Live Video (WebRTC)":
            st.markdown('<div class="info-box">🔴 <b>Live Camera Aktif.</b> Izinkan akses kamera pada browser Anda.</div>', unsafe_allow_html=True)
            
            # Konfigurasi Server STUN untuk koneksi WebRTC yang stabil di Cloud
            RTC_CONFIGURATION = RTCConfiguration(
                {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
            )

            # Class Transformer untuk memproses setiap frame video secara real-time
            class FaceVideoProcessor(VideoTransformerBase):
                def __init__(self):
                    self.detector = load_yolo(yolo_weights, yolo_repo_path, conf_thresh)
                    self.enhancer = CLAHEEnhancer(clip_limit, (tile_w, tile_h))
                    self.facenet = load_facenet()

                def transform(self, frame):
                    # Ubah frame video menjadi array gambar OpenCV
                    img = frame.to_ndarray(format="bgr24")
                    
                    # 1. CLAHE Enhancement
                    if apply_clahe_toggle:
                        self.enhancer.update_params(clip_limit, (tile_w, tile_h))
                        processed = self.enhancer.enhance(img)
                    else:
                        processed = img.copy()

                    # 2. Deteksi YOLOv5
                    detections = self.detector.detect(processed)
                    
                    # 3. Pengenalan Wajah (FaceNet)
                    labels_map = {}
                    # Mengambil database dari global karena session_state tidak bisa diakses dalam thread WebRTC
                    db = st.session_state.face_db 
                    if self.facenet and len(detections) > 0 and len(db["embeddings"]) > 0:
                        for det in detections:
                            face_crop = self.detector.crop_face(processed, det)
                            if face_crop is not None:
                                emb = get_embedding(face_crop, self.facenet)
                                sims = np.dot(db["embeddings"], emb)
                                best_idx = np.argmax(sims)
                                best_sim = float(sims[best_idx])
                                if best_sim >= similarity_threshold:
                                    labels_map[(int(det[0]), int(det[1]))] = db["names"][best_idx]
                    
                    # 4. Gambar Bounding Box
                    result_img = draw_detections(processed, detections, labels_map)
                    
                    return result_img

            # Menjalankan pemutar WebRTC
            webrtc_streamer(
                key="face-recognition",
                mode=1, # Mode SENDRECV (Kirim kamera, terima video hasil)
                rtc_configuration=RTC_CONFIGURATION,
                video_transformer_factory=FaceVideoProcessor,
                media_stream_constraints={"video": True, "audio": False},
                async_transform=True
            )

    with col_right:
        st.markdown("#### 🔍 Hasil Deteksi")

        if image_input is not None:
            clahe_enhancer.update_params(clip_limit, (tile_w, tile_h))
            processed = clahe_enhancer.enhance(image_input) if apply_clahe_toggle else image_input.copy()
            detector = load_yolo(yolo_weights, yolo_repo_path, conf_thresh)

            with st.spinner("Mendeteksi wajah..."):
                detections = detector.detect(processed)

            labels_map = {}
            if facenet and len(detections) > 0 and len(st.session_state.face_db["embeddings"]) > 0:
                st.markdown("---")
                st.markdown("#### 🧠 Pengenalan Wajah")
                db = st.session_state.face_db
                for i, det in enumerate(detections):
                    face_crop = detector.crop_face(processed, det)
                    if face_crop is None: continue
                    
                    emb = get_embedding(face_crop, facenet)
                    sims = np.dot(db["embeddings"], emb)
                    
                    best_idx = np.argmax(sims)
                    best_sim = float(sims[best_idx])
                    best_name = db["names"][best_idx]
                    
                    # Top 3 Logic
                    top_k = min(3, len(sims))
                    top_indices = np.argsort(sims)[-top_k:][::-1]
                    top3_str = ", ".join([f"{db['names'][idx]}({sims[idx]:.2f})" for idx in top_indices])

                    if best_sim >= similarity_threshold:
                        labels_map[(int(det[0]), int(det[1]))] = best_name
                        badge_class, status = "badge-green", f"✅ {best_name}"
                    else:
                        badge_class, status = "badge-yellow", "❓ Tidak dikenal"

                    st.markdown(f"""
                    <div class="result-card">
                        <span class="badge {badge_class}">Wajah #{i+1}</span>
                        <div style="margin-top:0.5rem;font-size:1rem;font-weight:600;">{status}</div>
                        <div style="color:#6b7280;font-size:0.8rem;margin-top:0.2rem;">Sim: {best_sim:.3f} | Top3: {top3_str}</div>
                    </div>
                    """, unsafe_allow_html=True)

            result_img = draw_detections(processed, detections, labels_map)
            st.image(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB), caption=f"{'CLAHE Enhanced' if apply_clahe_toggle else 'Original'} — {len(detections)} wajah terdeteksi", use_container_width=True)

            m1, m2, m3 = st.columns(3)
            with m1:
                st.markdown(f'<div class="metric-card"><div class="metric-label">Wajah Terdeteksi</div><div class="metric-value">{len(detections)}</div></div>', unsafe_allow_html=True)
            with m2:
                st.markdown(f'<div class="metric-card"><div class="metric-label">Avg Brightness</div><div class="metric-value">{np.mean(cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)):.0f}</div></div>', unsafe_allow_html=True)
            with m3:
                best_conf = max([d[4] for d in detections], default=0)
                st.markdown(f'<div class="metric-card"><div class="metric-label">Best Confidence</div><div class="metric-value">{best_conf:.2f}</div></div>', unsafe_allow_html=True)
        else:
            st.info("📷 Upload gambar atau aktifkan kamera untuk memulai deteksi.")

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2: Database Management
# ═══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("#### 🗄️ Database Wajah (Gallery)")
    col_add, col_list = st.columns([1, 1], gap="large")

    with col_add:
        st.markdown("**➕ Tambah Wajah Baru**")
        person_name = st.text_input("Nama Orang", placeholder="Contoh: Budi Santoso")
        face_upload = st.file_uploader("Upload foto wajah", type=["jpg", "jpeg", "png"], key="db_upload")

        if st.button("💾 Tambah ke Database", disabled=(not person_name or face_upload is None)):
            if facenet and face_upload and person_name.strip():
                img_bgr = cv2.cvtColor(np.array(Image.open(face_upload).convert("RGB")), cv2.COLOR_RGB2BGR)
                clahe_enhancer.update_params(clip_limit, (tile_w, tile_h))
                enhanced = clahe_enhancer.enhance(img_bgr)
                
                detector = load_yolo(yolo_weights, yolo_repo_path, conf_thresh)
                dets = detector.detect(enhanced)

                if dets:
                    face_crop = detector.crop_face(enhanced, dets[0])
                    if face_crop is None: face_crop = cv2.resize(enhanced, (160, 160))
                else:
                    face_crop = cv2.resize(enhanced, (160, 160))

                emb = get_embedding(face_crop, facenet)
                db = st.session_state.face_db
                db["embeddings"] = np.vstack([db["embeddings"], emb[np.newaxis, :]]) if len(db["embeddings"]) > 0 else emb[np.newaxis, :]
                db["names"].append(person_name.strip())
                db["face_images"].append(face_crop.copy())
                st.success(f"✅ {person_name} berhasil ditambahkan!")

        # --- FITUR NPZ YANG DIKEMBALIKAN ---
        st.markdown("---")
        st.markdown("**📂 Load dari .npz File**")
        npz_file = st.file_uploader("Upload file .npz (dari Kaggle preprocessing)", type=["npz"], key="npz_upload")
        if st.button("📥 Load NPZ Database", disabled=(npz_file is None)):
            if npz_file:
                with st.spinner("Memuat database..."):
                    try:
                        import tempfile
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".npz") as tmp:
                            tmp.write(npz_file.read())
                            tmp_path = tmp.name

                        data = np.load(tmp_path, allow_pickle=True)
                        faces_arr = data["faces"]
                        labels_arr = data["labels"]

                        if facenet:
                            all_embs = []
                            BATCH = 32
                            progress = st.progress(0)
                            for start in range(0, len(faces_arr), BATCH):
                                batch = faces_arr[start:start+BATCH]
                                tensors = []
                                for f in batch:
                                    rgb = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
                                    t = torch.tensor(rgb, dtype=torch.float32)
                                    t = (t - 127.5) / 128.0
                                    tensors.append(t.permute(2, 0, 1))
                                with torch.no_grad():
                                    e = facenet(torch.stack(tensors).to(DEVICE)).cpu().detach().numpy()
                                norms = np.linalg.norm(e, axis=1, keepdims=True) + 1e-10
                                all_embs.append(e / norms)
                                progress.progress(min((start+BATCH)/len(faces_arr), 1.0))

                            all_embs = np.vstack(all_embs).astype("float32")
                            st.session_state.face_db = {
                                "embeddings": all_embs,
                                "names": list(labels_arr),
                                "face_images": list(faces_arr),
                            }
                            st.success(f"✅ Database dimuat: {len(faces_arr)} wajah, {len(set(labels_arr))} orang")
                        os.unlink(tmp_path)
                    except Exception as e:
                        st.error(f"Gagal memuat NPZ: {e}")
        # ------------------------------------

    with col_list:
        st.markdown("**📋 Isi Database Saat Ini**")
        db = st.session_state.face_db
        if len(db["names"]) > 0:
            unique_names = list(set(db["names"]))[:12]
            cols = st.columns(4)
            for idx, name in enumerate(unique_names):
                name_idx = db["names"].index(name)
                face_rgb = cv2.cvtColor(db["face_images"][name_idx], cv2.COLOR_BGR2RGB)
                with cols[idx % 4]: st.image(face_rgb, caption=name[:12], use_container_width=True)
            if st.button("🗑️ Hapus Semua Data"):
                st.session_state.face_db = { "embeddings": np.zeros((0, 512), dtype="float32"), "names": [], "face_images": [] }
                st.rerun()
with tab3:
    st.markdown("#### 📊 Evaluasi Efek CLAHE")
    eval_files = st.file_uploader("Upload gambar untuk evaluasi", type=["jpg", "jpeg", "png"], accept_multiple_files=True, key="eval_upload")

    if eval_files and st.button("▶️ Jalankan Evaluasi"):
        detector = load_yolo(yolo_weights, yolo_repo_path, conf_thresh)
        clahe_enhancer.update_params(clip_limit, (tile_w, tile_h))
        results = []
        progress = st.progress(0)

        for i, f in enumerate(eval_files):
            img_bgr = cv2.cvtColor(np.array(Image.open(f).convert("RGB")), cv2.COLOR_RGB2BGR)
            enhanced = clahe_enhancer.enhance(img_bgr)
            dets_orig = detector.detect(img_bgr)
            dets_enh = detector.detect(enhanced)
            results.append({
                "Nama File": f.name,
                "Deteksi Original": len(dets_orig),
                "Deteksi CLAHE": len(dets_enh),
                "Δ Deteksi": len(dets_enh) - len(dets_orig)
            })
            progress.progress((i + 1) / len(eval_files))
            
        import pandas as pd
        st.dataframe(pd.DataFrame(results), use_container_width=True)
        # Visual comparison for first image
        if eval_files:
            st.markdown("**Visual Perbandingan (gambar pertama):**")
            pil = Image.open(eval_files[0]).convert("RGB")
            img_bgr = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
            
            # Update params & Enhance
            clahe_enhancer.update_params(clip_limit, (tile_w, tile_h))
            enhanced = clahe_enhancer.enhance(img_bgr)
            
            # Detect
            dets_orig = detector.detect(img_bgr)
            dets_enh = detector.detect(enhanced)

            c1, c2 = st.columns(2)
            with c1:
                drawn_orig = draw_detections(img_bgr, dets_orig)
                st.image(cv2.cvtColor(drawn_orig, cv2.COLOR_BGR2RGB),
                         caption=f"Original — {len(dets_orig)} wajah", use_container_width=True)
            with c2:
                drawn_enh = draw_detections(enhanced, dets_enh)
                st.image(cv2.cvtColor(drawn_enh, cv2.COLOR_BGR2RGB),
                         caption=f"CLAHE Enhanced — {len(dets_enh)} wajah", use_container_width=True)

# ─── Footer ───────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style="text-align:center;color:#374151;font-size:0.75rem;font-family:'Space Mono',monospace;padding:1rem 0;">
    FaceVision CLAHE · YOLOv5-Face + FaceNet VGGFace2 · Real Input — No Simulation
</div>
""", unsafe_allow_html=True)
