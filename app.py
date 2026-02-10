from flask import Flask, render_template, request
import numpy as np
import cv2
import os
from tensorflow.keras.models import load_model

app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"
MODEL_PATH = "tb_attention_hybrid_model.h5"

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load model once
model = load_model(MODEL_PATH)

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    confidence = None
    img_path = None
    error = None

    if request.method == "POST":

        # SAFE FILE CHECK
        file = request.files.get("image")
        if file is None or file.filename == "":
            error = "Please upload a chest X-ray image"
            return render_template("index.html", error=error)

        # Save uploaded image
        image_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
        file.save(image_path)
        img_path = image_path

        # ===============================
        # IMAGE PREPROCESSING
        # ===============================
        img = cv2.imread(image_path)              # BGR (3 channels)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (224, 224))
        img = img / 255.0
        img = np.expand_dims(img, axis=0)         # (1, 224, 224, 3)

        # ===============================
        # MODEL PREDICTION
        # ===============================
        prediction = model.predict(img)[0][0]

        tb_prob = prediction * 100
        normal_prob = (1 - prediction) * 100

        if prediction > 0.5:
            result = "🟥 Tuberculosis Detected"
        else:
            result = "🟩 Normal"

        confidence = {
            "tb": tb_prob,
            "normal": normal_prob
        }

    return render_template(
        "index.html",
        result=result,
        confidence=confidence,
        img_path=img_path,
        error=error
    )

if __name__ == "__main__":
    app.run(debug=True)
