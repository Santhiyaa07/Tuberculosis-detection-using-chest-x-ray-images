import numpy as np
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt

# ==============================
# Load Trained Model
# ==============================
model = tf.keras.models.load_model("tb_attention_hybrid_model.h5", compile=False)

IMG_SIZE = 224

# ==============================
# Image Preprocessing
# ==============================
def preprocess_image(img_path):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return img

# ==============================
# Grad-CAM Function
# ==============================
def make_gradcam_heatmap(img_array, model, last_conv_layer_name):

    # Create model that outputs last conv layer + predictions
    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[model.get_layer(last_conv_layer_name).output, model.output]
    )

    # Record gradients
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[:, 0]   # binary class

    # Compute gradients
    grads = tape.gradient(loss, conv_outputs)

    # Global average pooling of gradients
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]

    # Weight feature maps
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # Normalize
    heatmap = tf.maximum(heatmap, 0) / tf.reduce_max(heatmap)

    return heatmap.numpy()

# ==============================
# Display Grad-CAM
# ==============================
def display_gradcam(img_path, heatmap, alpha=0.4):

    img = cv2.imread(img_path)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

    # Resize heatmap
    heatmap = cv2.resize(heatmap, (IMG_SIZE, IMG_SIZE))
    heatmap = np.uint8(255 * heatmap)

    # Apply color map
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    # Superimpose
    superimposed_img = heatmap * alpha + img

    # Show using matplotlib (better than cv2)
    plt.imshow(cv2.cvtColor(superimposed_img.astype('uint8'), cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.title("Grad-CAM Visualization")
    plt.show()

# ==============================
# MAIN EXECUTION
# ==============================
if __name__ == "__main__":

    # 👉 Give your test image path
    img_path = "datasets/combined_dataset/test/Tuberculosis/CHNCXR_0329_1_7881.png"

    # Preprocess image
    img_array = preprocess_image(img_path)

    # 👉 IMPORTANT: Last conv layer name (DenseNet121)
    last_conv_layer_name = "conv5_block16_concat"

    # Generate heatmap
    heatmap = make_gradcam_heatmap(
        img_array,
        model,
        last_conv_layer_name
    )

    # Display result
    display_gradcam(img_path, heatmap)