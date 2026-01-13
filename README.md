# Face Recognition Attendance System (Streamlit)

This is a single-file Streamlit application (with helper modules) that performs face recognition in the browser and logs attendance to `Attendance.csv`.

Features
- Downloads dataset via `kagglehub` (dataset: `ziya07/face-based-attendance-dataset`) when training is required.
- Preprocesses images: resize to `160x160`, convert to RGB, normalize pixels to [-1, 1].
- Extracts face embeddings with `keras-facenet`.
- Reduces embeddings to 128 dimensions (PCA) if needed.
- Trains a scikit-learn SVM classifier and persists `models/svm_model.pkl`, `models/label_encoder.pkl`, and optionally `models/pca.pkl`.
- Uses `streamlit-webrtc` to access the browser webcam and performs real-time recognition.
- Logs attendance (`Name`, `Date`, `Time`) to `Attendance.csv`, avoiding duplicate entries for the same person on the same day.

Run locally
1. Create a Python virtual environment and activate it.

```powershell
python -m venv .venv; .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

2. (Optional) Configure Kaggle credentials if you want the app to auto-download the dataset during training:

- In Streamlit Cloud you can add `KAGGLE_USERNAME` and `KAGGLE_KEY` as secrets.
- Locally, ensure `~/.kaggle/kaggle.json` or environment variables are set as required by your kaggle client. This project uses `kagglehub` to download via the Kaggle API.

3. Run the app:

```powershell
streamlit run app.py
```

Deploy to Streamlit Cloud
1. Push this repository to GitHub (include all files above).
2. In Streamlit Cloud create a new app and point it to the repository and branch.
3. Add the following secrets in Streamlit Cloud (if you want automatic dataset download):
   - `KAGGLE_USERNAME`
   - `KAGGLE_KEY`
4. The app's entry point is `app.py` and Streamlit Cloud will install `requirements.txt` automatically.

Browser permissions
- The app uses your browser webcam through `streamlit-webrtc`. You must grant webcam permission when the browser prompts you.
- No OS-level camera permissions are required beyond the browser.

Notes and troubleshooting
- If `kagglehub` cannot download the dataset, manually download and place dataset files under a `data/` directory (the training script looks recursively inside `data/`).
- Training may be slow on Streamlit Cloud depending on plan limits. Models are persisted in `/models` so retraining happens only when models are missing or when you click "Retrain models (force)".
