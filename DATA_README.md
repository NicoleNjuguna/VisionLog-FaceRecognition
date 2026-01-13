# Dataset README

This file documents the dataset used for the Face Recognition Attendance project and how to prepare models locally.

Dataset source
- Kaggle dataset: https://www.kaggle.com/ziya07/face-based-attendance-dataset

Expected folder structure (for local training)
- `data/`
  - `<person_name_1>/image1.jpg`
  - `<person_name_1>/image2.jpg`
  - `<person_name_2>/...`

Notes
- The repository's default `.gitignore` excludes `data/` because datasets can be large. Do NOT commit large datasets unless you know how to use Git LFS.

How I prepared models locally
1. Ensure you have a Python virtual environment and the project dependencies installed. If you need TensorFlow for embedding extraction/train, install a CPU wheel (example):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
pip install tensorflow-cpu==2.20.0
```

2. Place the unzipped dataset under `data/` following the folder structure above. You can download and unzip with the Kaggle CLI:

```powershell
kaggle datasets download -d ziya07/face-based-attendance-dataset -p data --unzip
```

3. Run the training helper to extract embeddings, train PCA (if needed) and the SVM, and save models into `models/`:

```powershell
python scripts/prepare_models_locally.py --data-dir data --models-dir models
```

4. After verifying `models/` contains `svm_model.pkl` and `label_encoder.pkl` (and optionally `pca.pkl`), you can commit them to your branch if you want the app to run without TensorFlow on Streamlit Cloud.

```powershell
# (optional) commit models
git add models
git commit -m "Add pretrained models (local)"
git push origin streamlit-deploy
```

If you do not commit the models to the repo, the Streamlit app will attempt to train on first run (if configured) or will run in model-less mode. See `app.py` and `src/preprocessing.py` for lazy-loading details.
