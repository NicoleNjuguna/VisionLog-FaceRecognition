import os
import glob
import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.decomposition import PCA
import numpy as np
from tqdm import tqdm
from . import preprocessing


def download_dataset_via_kagglehub(dest="data"):
    try:
        import kagglehub
    except Exception as e:
        raise RuntimeError("kagglehub not available; please install and configure Kaggle credentials") from e

    os.makedirs(dest, exist_ok=True)
    # This API unzips in place when unzip=True
    # kagglehub vX exposes `dataset_download`; use that to download and unzip
    kagglehub.dataset_download("ziya07/face-based-attendance-dataset", path=dest, unzip=True)
    return dest


def collect_image_paths(base_dir):
    exts = ("*.jpg", "*.jpeg", "*.png")
    paths = []
    for ext in exts:
        paths.extend(glob.glob(os.path.join(base_dir, "**", ext), recursive=True))
    return paths


def train_and_persist(models_dir="models", data_dir="data"):
    # ensure data exists
    if not os.path.exists(data_dir) or not any(os.scandir(data_dir)):
        download_dataset_via_kagglehub(dest=data_dir)

    image_paths = collect_image_paths(data_dir)
    if not image_paths:
        raise RuntimeError("No images found in dataset directory: %s" % data_dir)

    embeddings = []
    labels = []

    for p in tqdm(image_paths, desc="Processing images"):
        try:
            arr = preprocessing.load_and_preprocess_image(p)
            emb = preprocessing.get_embedding(arr)
            embeddings.append(emb)
            # label is parent folder name
            label = os.path.basename(os.path.dirname(p))
            labels.append(label)
        except Exception:
            # skip unreadable images
            continue

    X = np.vstack(embeddings)
    y = np.array(labels)

    # Reduce to 128 dimensions if embeddings are larger
    pca = None
    if X.shape[1] > 128:
        pca = PCA(n_components=128)
        X = pca.fit_transform(X)

    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    clf = SVC(kernel="linear", probability=True)
    clf.fit(X, y_enc)

    os.makedirs(models_dir, exist_ok=True)
    joblib.dump(clf, os.path.join(models_dir, "svm_model.pkl"))
    joblib.dump(le, os.path.join(models_dir, "label_encoder.pkl"))
    if pca is not None:
        joblib.dump(pca, os.path.join(models_dir, "pca.pkl"))
