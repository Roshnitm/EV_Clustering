import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from flask import Flask, render_template, request
import joblib, cv2, numpy as np
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from utils.feature_extractor import feature_extractor

app = Flask(__name__)
model = joblib.load("models/best_clustering_model.pkl")

def predict_cluster(path):
    img = cv2.imread(path)
    img = cv2.resize(img,(224,224))
    img = preprocess_input(img)
    img = np.expand_dims(img,0)

    feature = feature_extractor.predict(img, verbose=0)
    cluster = model.predict(feature)[0]

    return f"Cluster {cluster}"

@app.route("/", methods=["GET","POST"])
def home():
    if request.method=="POST":
        file = request.files["image"]
        path = "static/test.jpg"
        file.save(path)

        result = predict_cluster(path)
        return render_template("index.html", result=result)

    return render_template("index.html", result="Upload an EV image")

app.run(debug=True)
