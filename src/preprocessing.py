import numpy as np
from PIL import Image

# Lazily import keras-facenet (TensorFlow) only when embeddings are required.
_embedder = None


def _get_embedder():
    global _embedder
    if _embedder is None:
        try:
            from keras_facenet import FaceNet
        except Exception as e:
            raise ModuleNotFoundError("keras-facenet (and TensorFlow) is required for embedding extraction: " + str(e))
        _embedder = FaceNet()
    return _embedder


def load_and_preprocess_image(path, size=(160, 160)):
    """Load an image, convert to RGB, resize to `size`, and normalize to [-1, 1]."""
    img = Image.open(path).convert("RGB")
    img = img.resize(size)
    arr = np.asarray(img).astype("float32")
    # normalize to [-1, 1]
    arr = arr / 127.5 - 1.0
    return arr


def get_embedding(img_array):
    """Return embedding for a single preprocessed image array.

    Expects `img_array` shaped (160,160,3) and normalized as above.
    This will attempt to import the FaceNet embedder lazily and will raise
    ModuleNotFoundError if TensorFlow/keras-facenet are not installed.
    """
    embedder = _get_embedder()
    emb = embedder.embeddings(np.expand_dims(img_array, axis=0))
    return emb[0]
