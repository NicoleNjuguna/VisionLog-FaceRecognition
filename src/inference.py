import os
import cv2
import joblib
import numpy as np
from datetime import datetime, date
import pandas as pd
from . import preprocessing


CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")


def load_models_if_exist(svm_path, le_path, pca_path=None):
    if os.path.exists(svm_path) and os.path.exists(le_path):
        svm = joblib.load(svm_path)
        le = joblib.load(le_path)
        pca = joblib.load(pca_path) if (pca_path and os.path.exists(pca_path)) else None
        models = {"svm": svm, "le": le, "pca": pca}
        return True, models
    return False, None


def detect_faces_bboxes(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    rects = CASCADE.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))
    return rects


def recognize_face(face_img, models):
    # face_img: RGB image of face region
    arr = cv2.resize(face_img, (160, 160))
    arr = arr.astype("float32") / 127.5 - 1.0
    try:
        emb = preprocessing.get_embedding(arr)
    except ModuleNotFoundError as e:
        # Propagate a clear error so callers can handle absence of TF at runtime
        raise
    X = emb.reshape(1, -1)
    if models.get("pca") is not None:
        X = models["pca"].transform(X)
    probs = models["svm"].predict_proba(X)[0]
    idx = np.argmax(probs)
    label = models["le"].inverse_transform([idx])[0]
    confidence = float(probs[idx])
    return label, confidence


def process_frame(frame_bgr, models, min_confidence=0.6):
    # detect faces and annotate frame
    rects = detect_faces_bboxes(frame_bgr)
    for (x, y, w, h) in rects:
        # extract face ROI in RGB
        face = frame_bgr[y : y + h, x : x + w]
        face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        try:
            name, conf = recognize_face(face_rgb, models)
        except ModuleNotFoundError:
            # TensorFlow / keras-facenet not installed in this environment.
            # Draw a red rectangle and text to indicate recognition is unavailable.
            cv2.rectangle(frame_bgr, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.putText(frame_bgr, "Model missing", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            continue
        except Exception:
            # For any other error during recognition, skip this face.
            continue

        label_text = f"{name}: {conf:.2f}"
        # overlay
        if conf >= min_confidence:
            cv2.rectangle(frame_bgr, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame_bgr, label_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            log_attendance(name)
        else:
            cv2.rectangle(frame_bgr, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.putText(frame_bgr, f"Unknown: {conf:.2f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    return frame_bgr


ATTENDANCE_CSV = os.path.join(os.path.dirname(os.path.dirname(__file__)), "Attendance.csv")


def ensure_attendance_csv():
    if not os.path.exists(ATTENDANCE_CSV):
        pd.DataFrame(columns=["Name", "Date", "Time"]).to_csv(ATTENDANCE_CSV, index=False)


def log_attendance(name):
    ensure_attendance_csv()
    df = pd.read_csv(ATTENDANCE_CSV)
    today_str = date.today().isoformat()
    # check duplicates for today
    already = ((df["Name"] == name) & (df["Date"] == today_str)).any()
    if not already:
        now = datetime.now()
        row = {"Name": name, "Date": today_str, "Time": now.strftime("%H:%M:%S")}
        df = df.append(row, ignore_index=True)
        df.to_csv(ATTENDANCE_CSV, index=False)


def read_attendance():
    ensure_attendance_csv()
    df = pd.read_csv(ATTENDANCE_CSV)
    today_str = date.today().isoformat()
    return df[df["Date"] == today_str]
