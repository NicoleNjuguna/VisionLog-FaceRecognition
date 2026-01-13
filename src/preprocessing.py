import numpy as np
from PIL import Image
from keras_facenet import FaceNet

# Initialize FaceNet embedder (loads model once)
embedder = FaceNet()


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
    """
    # FaceNet embedder expects batch of images
    emb = embedder.embeddings(np.expand_dims(img_array, axis=0))
    return emb[0]
