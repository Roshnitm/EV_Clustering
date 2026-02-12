import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import logging
import warnings

# Reduce noisy warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

import cv2  # type: ignore
import numpy as np  # type: ignore
import tensorflow as tf  # type: ignore
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input  # type: ignore

# Silence TensorFlow logs
tf.get_logger().setLevel('ERROR')
logging.getLogger('tensorflow').setLevel(logging.ERROR)


feature_extractor = MobileNetV2(
    weights="imagenet",
    include_top=False,
    pooling="avg"
)

# Feature Extraction Function
def extract_features(folder):

    features = []
    image_paths = []

    for root, dirs, files in os.walk(folder):

        for file in files:

            img_path = os.path.join(root, file)

            if not os.path.isfile(img_path):
                continue

            img = cv2.imread(img_path)

            if img is None:
                continue

            # Convert BGR → RGB
            try:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            except Exception:
                continue

            # Resize
            img = cv2.resize(img, (224, 224))
            img = img.astype(np.float32)

            # Preprocess
            img = preprocess_input(img)
            img = np.expand_dims(img, axis=0)

            try:
                feature = feature_extractor.predict(img, verbose=0)
            except Exception:
                continue

            features.append(feature.flatten())
            image_paths.append(img_path)

    return np.array(features), image_paths
