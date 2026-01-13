import os
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration
from src import inference, train_model

st.set_page_config(page_title="Face Recognition Attendance", layout="centered")

st.title("Face Recognition Attendance System")

st.markdown(
    "This Streamlit app performs face recognition from the browser webcam and logs attendance to `Attendance.csv`."
)

MODEL_DIR = "models"
SVM_PATH = os.path.join(MODEL_DIR, "svm_model.pkl")
LE_PATH = os.path.join(MODEL_DIR, "label_encoder.pkl")
PCA_PATH = os.path.join(MODEL_DIR, "pca.pkl")

os.makedirs(MODEL_DIR, exist_ok=True)

status_placeholder = st.empty()

@st.cache_resource
def load_models_if_exist():
    return inference.load_models_if_exist(SVM_PATH, LE_PATH, PCA_PATH)

models_present, models = load_models_if_exist()

if models_present:
    status_placeholder.info("Loaded existing models.")
else:
    status_placeholder.warning("No trained models found — training will start now. This may take several minutes.")

retrain = st.button("Retrain models (force)")

if retrain or (not models_present):
    with st.spinner("Training models — downloading dataset and computing embeddings..."):
        try:
            train_model.train_and_persist(models_dir=MODEL_DIR)
            status_placeholder.success("Training completed and models saved.")
            models_present, models = load_models_if_exist()
        except Exception as e:
            status_placeholder.error(f"Training failed: {e}")

if models_present:
    st.success("Models ready. Start webcam below.")

st.sidebar.header("Controls")
min_confidence = st.sidebar.slider("Minimum confidence to show name", 0.1, 0.99, 0.6)

RTC_CONFIGURATION = RTCConfiguration({
    "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
})

class VideoProcessor(VideoTransformerBase):
    def __init__(self):
        self.models_present, self.models = load_models_if_exist()

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        if self.models_present:
            img = inference.process_frame(img, self.models, min_confidence=min_confidence)
        return img


webrtc_streamer(
    key="face-recognition",
    rtc_configuration=RTC_CONFIGURATION,
    video_transformer_factory=VideoProcessor,
    media_stream_constraints={"video": True, "audio": False},
)

st.markdown("---")
st.write("Attendance log:`Attendance.csv`")
if st.button("Show Attendance (today)"):
    st.write(inference.read_attendance())
