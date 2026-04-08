from flask import Flask, render_template, request
import numpy as np
import cv2
import os
import tensorflow as tf
from tensorflow.keras.models import load_model

app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"
HEATMAP_FOLDER = "static/heatmaps"
MODEL_PATH = "tb_attention_hybrid_model.h5"

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(HEATMAP_FOLDER, exist_ok=True)

model = load_model(MODEL_PATH)

# ✅ GRAD-CAM FUNCTION (FIXED)
def generate_gradcam(model, img_array, image_path, filename):

    last_conv_layer = None

    # Find last convolution layer
    for layer in reversed(model.layers):
        if len(layer.output.shape) == 4:
            last_conv_layer = layer.name
            break

    grad_model = tf.keras.models.Model(
        inputs=model.input,
        outputs=[model.get_layer(last_conv_layer).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[:, 0]

    grads = tape.gradient(loss, conv_outputs)

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]

    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # Convert safely to numpy
    if hasattr(heatmap, "numpy"):
        heatmap = heatmap.numpy()

    # Normalize
    heatmap = np.maximum(heatmap, 0)
    if np.max(heatmap) != 0:
        heatmap = heatmap / np.max(heatmap)

    # Load original image
    img = cv2.imread(image_path)

    # Resize heatmap (FIXED)
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)

    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    superimposed = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)

    heatmap_path = os.path.join(HEATMAP_FOLDER, filename)
    cv2.imwrite(heatmap_path, superimposed)

    return heatmap_path


@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    confidence = None
    img_path = None
    heatmap_path = None
    error = None
    severity = None
    message = None
    suggestion = None

    if request.method == "POST":

        file = request.files.get("image")
        if file is None or file.filename == "":
            error = "Please upload a chest X-ray image"
            return render_template("index.html", error=error)

        filename = file.filename
        image_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(image_path)
        img_path = image_path

        # Preprocessing
        img = cv2.imread(image_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, (224, 224))
        img_norm = img_resized / 255.0
        img_array = np.expand_dims(img_norm, axis=0)

        # Prediction
        prediction = model.predict(img_array)[0][0]

        tb_prob = prediction * 100
        normal_prob = (1 - prediction) * 100

        # Grad-CAM
        heatmap_path = generate_gradcam(model, img_array, image_path, filename)

        # Result + Severity
        if prediction < 0.4:
            result = "🟩 Normal"
            severity = "No TB Detected"
            message = "Lungs appear normal."
            suggestion = "No immediate action required."
        elif prediction < 0.7:
            result = "🟥 Tuberculosis Detected"
            severity = "Mild TB"
            message = "Low to moderate probability detected."
            suggestion = "Consult a doctor."
        elif prediction < 0.9:
            result = "🟥 Tuberculosis Detected"
            severity = "Moderate TB"
            message = "High probability detected."
            suggestion = "Medical attention recommended."
        else:
            result = "🟥 Tuberculosis Detected"
            severity = "Severe TB"
            message = "Very high probability detected."
            suggestion = "Immediate medical attention required."

        confidence = {
            "tb": tb_prob,
            "normal": normal_prob
        }

    return render_template(
        "index.html",
        result=result,
        confidence=confidence,
        img_path=img_path,
        heatmap_path=heatmap_path,
        error=error,
        severity=severity,
        message=message,
        suggestion=suggestion
    )


if __name__ == "__main__":
    app.run(debug=True)