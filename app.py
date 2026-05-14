
import os
import sys
import cv2
import numpy as np
import streamlit as st
import torch
from PIL import Image
from collections import defaultdict

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

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

.stApp {
    background: #0a0a0f;
    color: #e8e8f0;
}

h1, h2, h3 {
    font-family: 'Space Mono', monospace !important;
    letter-spacing: -0.03em;
}

.main-title {
    font-family: 'Space Mono', monospace;
    font-size: 2.4rem;
    font-weight: 700;
    background: linear-gradient(135deg, #7DF9FF 0%, #89CFF0 50%, #B8A9FA 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 0.2rem;
}

.sub-title {
    color: #6b7280;
    font-size: 0.95rem;
    margin-bottom: 2rem;
    font-family: 'Space Mono', monospace;
}

.metric-card {
    background: #13131a;
    border: 1px solid #2a2a3a;
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    margin: 0.4rem 0;
}

.metric-label {
    color: #6b7280;
    font-size: 0.75rem;
    font-family: 'Space Mono', monospace;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-bottom: 0.3rem;
}

.metric-value {
    color: #7DF9FF;
    font-size: 1.6rem;
    font-weight: 700;
    font-family: 'Space Mono', monospace;
}

.result-card {
    background: linear-gradient(135deg, #13131a, #1a1a2e);
    border: 1px solid #2a2a3a;
    border-radius: 16px;
    padding: 1.5rem;
    margin: 1rem 0;
}

.match-correct {
    color: #4ade80;
    font-size: 1.1rem;
    font-weight: 600;
}

.match-unknown {
    color: #facc15;
    font-size: 1.1rem;
    font-weight: 600;
}

.badge {
    display: inline-block;
    padding: 0.25rem 0.75rem;
    border-radius: 999px;
    font-size: 0.75rem;
    font-family: 'Space Mono', monospace;
    font-weight: 700;
    margin: 0.2rem;
}

.badge-cyan {
    background: rgba(125, 249, 255, 0.15);
    color: #7DF9FF;
    border: 1px solid rgba(125, 249, 255, 0.3);
}

.badge-green {
    background: rgba(74, 222, 128, 0.15);
    color: #4ade80;
    border: 1px solid rgba(74, 222, 128, 0.3);
}

.badge-yellow {
    background: rgba(250, 204, 21, 0.15);
    color: #facc15;
    border: 1px solid rgba(250, 204, 21, 0.3);
}

.pipeline-step {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    padding: 0.6rem 0;
    border-bottom: 1px solid #1e1e2e;
    font-size: 0.85rem;
    color: #9ca3af;
}

.step-dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: #7DF9FF;
    flex-shrink: 0;
}

.sidebar-section {
    background: #13131a;
    border: 1px solid #2a2a3a;
    border-radius: 10px;
    padding: 1rem;
    margin: 0.75rem 0;
}

.info-box {
    background: rgba(125, 249, 255, 0.05);
    border: 1px solid rgba(125, 249, 255, 0.2);
    border-radius: 10px;
    padding: 1rem;
    font-size: 0.85rem;
    color: #9ca3af;
    margin: 0.75rem 0;
}

/* Streamlit widget overrides */
.stButton > button {
    background: linear-gradient(135deg, #7DF9FF, #89CFF0) !important;
    color: #0a0a0f !important;
    border: none !important;
    font-family: 'Space Mono', monospace !important;
    font-weight: 700 !important;
    border-radius: 8px !important;
    padding: 0.5rem 1.5rem !important;
}

.stFileUploader label {
    color: #9ca3af !important;
    font-size: 0.85rem !important;
}

div[data-testid="stImage"] img {
    border-radius: 12px;
}

.stSlider > div {
    color: #9ca3af;
}

hr {
    border-color: #2a2a3a !important;
}
</style>
""", unsafe_allow_html=True)

# ─── Device Setup ─────────────────────────────────────────────────────────────
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ─── CLAHE Enhancer ───────────────────────────────────────────────────────────
class CLAHEEnhancer:
    def __init__(self, clip_limit=2.0, tile_size=(8, 8)):
        self.clahe = cv2.createCLAHE(
            clipLimit=clip_limit,
            tileGridSize=tile_size
        )
        self.clip_limit = clip_limit
        self.tile_size = tile_size

    def enhance(self, image_bgr: np.ndarray) -> np.ndarray:
        """Terapkan CLAHE pada gambar input asli — tanpa simulasi gelap."""
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
            st.sidebar.error(f"❌ YOLOv5 weights tidak ditemukan:\n`{weights_path}`\n\nUbah path di sidebar.")
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
            from utils.general import non_max_suppression
        except ImportError:
            return []

        img_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        h, w = img_rgb.shape[:2]
        img_resized = cv2.resize(img_rgb, (640, 640))
        img_array = img_resized.transpose(2, 0, 1).astype('float32') / 255.0
        img_tensor = torch.tensor(img_array, dtype=torch.float32).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            results = self.model(img_tensor)

        pred = results[0]
        pred = non_max_suppression(pred, conf_thres=self.conf_threshold, iou_thres=0.45)
        detections = pred[0]

        faces = []
        if detections is not None and len(detections):
            for det in detections.cpu().tolist():
                x1 = int(det[0] * w / 640)
                y1 = int(det[1] * h / 640)
                x2 = int(det[2] * w / 640)
                y2 = int(det[3] * h / 640)
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
        cv2.putText(img, label_text, (x1, max(y1 - 8, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 180), 2)
    return img


# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="main-title" style="font-size:1.3rem;">⚙️ Konfigurasi</div>', unsafe_allow_html=True)

    st.markdown("### 📁 Model Path")
    yolo_weights = st.text_input(
        "YOLOv5-Face Weights",
        value="best.pt",
        help="Path ke file best.pt dari YOLOv5-face"
    )
    yolo_repo_path = st.text_input(
        "YOLOv5-Face Repo Path",
        value="yolov5-face",
        help="Folder repo YOLOv5-face (berisi models/, utils/)"
    )

    st.markdown("---")
    st.markdown("### 🎛️ CLAHE Parameters")
    clip_limit = st.slider("Clip Limit", 0.5, 10.0, 2.0, 0.5,
                           help="Batas kontras CLAHE. Lebih tinggi = lebih agresif.")
    tile_w = st.select_slider("Tile Width", options=[4, 8, 16, 32], value=8)
    tile_h = st.select_slider("Tile Height", options=[4, 8, 16, 32], value=8)

    st.markdown("---")
    st.markdown("### 🔍 Detection Settings")
    conf_thresh = st.slider("Confidence Threshold", 0.1, 0.9, 0.4, 0.05)
    similarity_threshold = st.slider("Similarity Threshold", 0.3, 0.9, 0.6, 0.05,
                                     help="Cosine similarity minimum untuk match wajah ke database")

    st.markdown("---")
    st.markdown('<div class="info-box">💡 <b>Catatan:</b><br>App ini memproses foto/kamera <b>asli</b> — tidak ada simulasi low-light. CLAHE diterapkan langsung untuk meningkatkan kontras gambar input.</div>', unsafe_allow_html=True)

    device_badge = "🟢 GPU (CUDA)" if torch.cuda.is_available() else "🟡 CPU"
    st.markdown(f'<span class="badge badge-cyan">{device_badge}</span>', unsafe_allow_html=True)


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

# Pipeline steps
with st.expander("📋 Pipeline Aktif", expanded=False):
    steps = [
        ("Input Gambar", "Upload foto atau ambil dari kamera"),
        ("CLAHE Enhancement", f"clip_limit={clip_limit}, tile=({tile_w}×{tile_h})"),
        ("YOLOv5-Face Detection", f"conf≥{conf_thresh}"),
        ("FaceNet Embedding", "InceptionResnetV1 pretrained vggface2 → 512-dim"),
        ("Cosine Similarity Match", f"threshold={similarity_threshold}"),
    ]
    for name, desc in steps:
        st.markdown(f"""
        <div class="pipeline-step">
            <div class="step-dot"></div>
            <span style="color:#e8e8f0;font-weight:600;">{name}</span>
            <span style="margin-left:auto;font-size:0.8rem;">{desc}</span>
        </div>""", unsafe_allow_html=True)

st.markdown("---")

# Tabs
tab1, tab2, tab3 = st.tabs(["🖼️ Deteksi & Enhance", "🗄️ Database Wajah", "📊 Evaluasi CLAHE"])

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1: Detection
# ═══════════════════════════════════════════════════════════════════════════════
with tab1:
    col_left, col_right = st.columns([1, 1], gap="large")

    with col_left:
        st.markdown("#### 📤 Input Gambar")
        input_mode = st.radio("Sumber Input", ["Upload File", "Kamera"], horizontal=True)

        image_input = None
        if input_mode == "Upload File":
            uploaded = st.file_uploader("Pilih gambar", type=["jpg", "jpeg", "png", "bmp", "webp"])
            if uploaded:
                pil_img = Image.open(uploaded).convert("RGB")
                image_input = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        else:
            camera_img = st.camera_input("Ambil foto")
            if camera_img:
                pil_img = Image.open(camera_img).convert("RGB")
                image_input = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

        apply_clahe_toggle = st.checkbox("Terapkan CLAHE Enhancement", value=True)

    with col_right:
        st.markdown("#### 🔍 Hasil Deteksi")

        if image_input is not None:
            # Update CLAHE params
            clahe_enhancer.update_params(clip_limit, (tile_w, tile_h))

            # Step 1: Enhance (jika aktif)
            if apply_clahe_toggle:
                processed = clahe_enhancer.enhance(image_input)
            else:
                processed = image_input.copy()

            # Step 2: Load YOLOv5
            detector = load_yolo(yolo_weights, yolo_repo_path, conf_thresh)

            # Step 3: Detect
            with st.spinner("Mendeteksi wajah..."):
                detections = detector.detect(processed)

            # Draw boxes
            result_img = draw_detections(processed, detections)
            result_rgb = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
            st.image(result_rgb, caption=f"{'CLAHE Enhanced' if apply_clahe_toggle else 'Original'} — {len(detections)} wajah terdeteksi", use_column_width=True)

            # Metrics
            m1, m2, m3 = st.columns(3)
            with m1:
                st.markdown(f'<div class="metric-card"><div class="metric-label">Wajah Terdeteksi</div><div class="metric-value">{len(detections)}</div></div>', unsafe_allow_html=True)
            with m2:
                h, w = image_input.shape[:2]
                gray = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)
                brightness = np.mean(gray)
                st.markdown(f'<div class="metric-card"><div class="metric-label">Avg Brightness</div><div class="metric-value">{brightness:.0f}</div></div>', unsafe_allow_html=True)
            with m3:
                best_conf = max([d[4] for d in detections], default=0)
                st.markdown(f'<div class="metric-card"><div class="metric-label">Best Confidence</div><div class="metric-value">{best_conf:.2f}</div></div>', unsafe_allow_html=True)

            # Original vs Enhanced comparison
            if apply_clahe_toggle:
                st.markdown("**Perbandingan Original vs CLAHE:**")
                c1, c2 = st.columns(2)
                with c1:
                    gray_orig = cv2.cvtColor(image_input, cv2.COLOR_BGR2GRAY)
                    st.image(cv2.cvtColor(image_input, cv2.COLOR_BGR2RGB),
                             caption=f"Original (brightness={np.mean(gray_orig):.0f})", use_column_width=True)
                with c2:
                    st.image(result_rgb,
                             caption=f"CLAHE (brightness={brightness:.0f})", use_column_width=True)

            # Face Recognition (if database exists)
            if facenet and len(detections) > 0 and "face_db" in st.session_state and len(st.session_state.face_db["embeddings"]) > 0:
                st.markdown("---")
                st.markdown("#### 🧠 Pengenalan Wajah")
                db = st.session_state.face_db

                for i, det in enumerate(detections):
                    face_crop = detector.crop_face(processed, det)
                    if face_crop is None:
                        continue
                    emb = get_embedding(face_crop, facenet)
                    sims = np.dot(db["embeddings"], emb)
                    best_idx = np.argmax(sims)
                    best_sim = float(sims[best_idx])
                    best_name = db["names"][best_idx]

                    if best_sim >= similarity_threshold:
                        badge_class = "badge-green"
                        status = f"✅ {best_name}"
                        note = f"Sim: {best_sim:.3f}"
                    else:
                        badge_class = "badge-yellow"
                        status = "❓ Tidak dikenal"
                        note = f"Sim terbaik: {best_sim:.3f} (di bawah threshold)"

                    st.markdown(f"""
                    <div class="result-card">
                        <span class="badge {badge_class}">Wajah #{i+1}</span>
                        <div style="margin-top:0.5rem;font-size:1rem;font-weight:600;">{status}</div>
                        <div style="color:#6b7280;font-size:0.8rem;margin-top:0.2rem;">{note}</div>
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="text-align:center;padding:4rem 2rem;color:#4b5563;border:2px dashed #2a2a3a;border-radius:12px;">
                <div style="font-size:3rem;margin-bottom:1rem;">📷</div>
                <div style="font-family:'Space Mono',monospace;">Upload gambar atau aktifkan kamera</div>
                <div style="font-size:0.8rem;margin-top:0.5rem;">untuk memulai deteksi</div>
            </div>
            """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2: Database Management
# ═══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("#### 🗄️ Database Wajah (Gallery)")
    st.markdown('<div class="info-box">Tambahkan wajah ke database untuk fitur pengenalan. Upload foto beserta nama orang. Sistem akan menyimpan embedding 512-dimensi dari FaceNet.</div>', unsafe_allow_html=True)

    # Initialize DB in session state
    if "face_db" not in st.session_state:
        st.session_state.face_db = {
            "embeddings": np.zeros((0, 512), dtype="float32"),
            "names": [],
            "face_images": [],
        }

    col_add, col_list = st.columns([1, 1], gap="large")

    with col_add:
        st.markdown("**➕ Tambah Wajah Baru**")
        person_name = st.text_input("Nama Orang", placeholder="Contoh: Budi Santoso")
        face_upload = st.file_uploader("Upload foto wajah", type=["jpg", "jpeg", "png"], key="db_upload")

        if st.button("💾 Tambah ke Database", disabled=(not person_name or face_upload is None)):
            if facenet and face_upload and person_name.strip():
                pil_img = Image.open(face_upload).convert("RGB")
                img_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

                # Enhance before embedding
                clahe_enhancer.update_params(clip_limit, (tile_w, tile_h))
                enhanced = clahe_enhancer.enhance(img_bgr)

                # Detect face
                detector = load_yolo(yolo_weights, yolo_repo_path, conf_thresh)
                dets = detector.detect(enhanced)

                if dets:
                    face_crop = detector.crop_face(enhanced, dets[0])
                else:
                    face_crop = cv2.resize(enhanced, (160, 160))

                emb = get_embedding(face_crop, facenet)

                db = st.session_state.face_db
                db["embeddings"] = np.vstack([db["embeddings"], emb[np.newaxis, :]]) if len(db["embeddings"]) > 0 else emb[np.newaxis, :]
                db["names"].append(person_name.strip())
                db["face_images"].append(face_crop.copy())

                st.success(f"✅ {person_name} berhasil ditambahkan ke database!")
            else:
                st.warning("Pastikan nama diisi, gambar diupload, dan FaceNet sudah termuat.")

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

    with col_list:
        st.markdown("**📋 Isi Database Saat Ini**")
        db = st.session_state.face_db
        n_faces = len(db["names"])
        n_people = len(set(db["names"]))

        c1, c2 = st.columns(2)
        with c1:
            st.markdown(f'<div class="metric-card"><div class="metric-label">Total Wajah</div><div class="metric-value">{n_faces}</div></div>', unsafe_allow_html=True)
        with c2:
            st.markdown(f'<div class="metric-card"><div class="metric-label">Unik Orang</div><div class="metric-value">{n_people}</div></div>', unsafe_allow_html=True)

        if n_faces > 0:
            # Show preview
            unique_names = list(set(db["names"]))[:12]
            cols = st.columns(4)
            for idx, name in enumerate(unique_names):
                name_idx = db["names"].index(name)
                face_img = db["face_images"][name_idx]
                face_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
                with cols[idx % 4]:
                    st.image(face_rgb, caption=name[:12], use_column_width=True)

            if st.button("🗑️ Hapus Semua Data"):
                st.session_state.face_db = {
                    "embeddings": np.zeros((0, 512), dtype="float32"),
                    "names": [],
                    "face_images": [],
                }
                st.rerun()
        else:
            st.markdown('<div style="color:#4b5563;text-align:center;padding:2rem;">Database kosong. Tambahkan wajah di panel kiri.</div>', unsafe_allow_html=True)



with tab3:
    st.markdown("#### 📊 Evaluasi Efek CLAHE")
    st.markdown('<div class="info-box">Upload beberapa gambar untuk membandingkan performa deteksi sebelum dan sesudah CLAHE diterapkan.</div>', unsafe_allow_html=True)

    eval_files = st.file_uploader("Upload gambar untuk evaluasi", type=["jpg", "jpeg", "png"],
                                  accept_multiple_files=True, key="eval_upload")

    if eval_files and st.button("▶️ Jalankan Evaluasi"):
        detector = load_yolo(yolo_weights, yolo_repo_path, conf_thresh)
        clahe_enhancer.update_params(clip_limit, (tile_w, tile_h))

        results = []
        progress = st.progress(0)

        for i, f in enumerate(eval_files):
            pil = Image.open(f).convert("RGB")
            img_bgr = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
            enhanced = clahe_enhancer.enhance(img_bgr)

            dets_orig = detector.detect(img_bgr)
            dets_enh = detector.detect(enhanced)

            gray_orig = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
            gray_enh = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)

            results.append({
                "Nama File": f.name,
                "Deteksi Original": len(dets_orig),
                "Deteksi CLAHE": len(dets_enh),
                "Δ Deteksi": len(dets_enh) - len(dets_orig),
                "Brightness Original": f"{np.mean(gray_orig):.1f}",
                "Brightness CLAHE": f"{np.mean(gray_enh):.1f}",
                "Conf Terbaik (Orig)": f"{max([d[4] for d in dets_orig], default=0):.3f}",
                "Conf Terbaik (CLAHE)": f"{max([d[4] for d in dets_enh], default=0):.3f}",
            })
            progress.progress((i + 1) / len(eval_files))

        # Summary
        total_orig = sum(r["Deteksi Original"] for r in results)
        total_enh = sum(r["Deteksi CLAHE"] for r in results)
        improvement = total_enh - total_orig

        m1, m2, m3 = st.columns(3)
        with m1:
            st.markdown(f'<div class="metric-card"><div class="metric-label">Total Deteksi Original</div><div class="metric-value">{total_orig}</div></div>', unsafe_allow_html=True)
        with m2:
            st.markdown(f'<div class="metric-card"><div class="metric-label">Total Deteksi CLAHE</div><div class="metric-value">{total_enh}</div></div>', unsafe_allow_html=True)
        with m3:
            color = "#4ade80" if improvement >= 0 else "#f87171"
            sign = "+" if improvement >= 0 else ""
            st.markdown(f'<div class="metric-card"><div class="metric-label">Perubahan Deteksi</div><div class="metric-value" style="color:{color};">{sign}{improvement}</div></div>', unsafe_allow_html=True)

        # Table
        import pandas as pd
        df = pd.DataFrame(results)
        st.dataframe(df, use_container_width=True)

        # Visual comparison for first image
        if eval_files:
            st.markdown("**Visual Perbandingan (gambar pertama):**")
            pil = Image.open(eval_files[0]).convert("RGB")
            img_bgr = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
            enhanced = clahe_enhancer.enhance(img_bgr)
            dets_orig = detector.detect(img_bgr)
            dets_enh = detector.detect(enhanced)

            c1, c2 = st.columns(2)
            with c1:
                drawn_orig = draw_detections(img_bgr, dets_orig)
                st.image(cv2.cvtColor(drawn_orig, cv2.COLOR_BGR2RGB),
                         caption=f"Original — {len(dets_orig)} wajah", use_column_width=True)
            with c2:
                drawn_enh = draw_detections(enhanced, dets_enh)
                st.image(cv2.cvtColor(drawn_enh, cv2.COLOR_BGR2RGB),
                         caption=f"CLAHE Enhanced — {len(dets_enh)} wajah", use_column_width=True)

# ─── Footer ───────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style="text-align:center;color:#374151;font-size:0.75rem;font-family:'Space Mono',monospace;padding:1rem 0;">
    FaceVision CLAHE · YOLOv5-Face + FaceNet VGGFace2 · Real Input — No Simulation
</div>
""", unsafe_allow_html=True)
