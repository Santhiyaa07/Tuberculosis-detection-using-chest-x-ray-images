import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras import layers
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc, accuracy_score
import matplotlib.pyplot as plt

# ------------------ CONFIG ------------------
IMG_SIZE = 224
BATCH_SIZE = 16

test_dir = r"E:\TB-Project\datasets\combined_dataset\test"
# --------------------------------------------

print("\nLoading Test Dataset...\n")

test_ds = tf.keras.preprocessing.image_dataset_from_directory(
    test_dir,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    label_mode="binary",
    shuffle=False
)

normalization = layers.Rescaling(1./255)

# ------------------ LOAD MODELS ------------------
print("\nLoading WITH Attention model...\n")
model_att = load_model("tb_attention_model.h5")

print("Loading WITHOUT Attention model...\n")
model_no_att = load_model("tb_no_attention_model.h5")


# ------------------ EVALUATION FUNCTION ------------------
def evaluate_model(model, name):

    y_true = []
    y_pred = []
    y_scores = []

    for images, labels in test_ds:
        images = normalization(images)
        preds = model.predict(images, verbose=0)

        y_scores.extend(preds.flatten())
        preds_binary = (preds > 0.5).astype(int)

        y_true.extend(labels.numpy())
        y_pred.extend(preds_binary.flatten())

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_scores = np.array(y_scores)

    # -------- Accuracy --------
    accuracy = accuracy_score(y_true, y_pred)
    print(f"\n{name} Accuracy: {accuracy:.4f}")

    # -------- Confusion Matrix --------
    cm = confusion_matrix(y_true, y_pred)
    print(f"\n{name} Confusion Matrix:")
    print(cm)

    # -------- Classification Report --------
    print(f"\n{name} Classification Report:")
    print(classification_report(
        y_true,
        y_pred,
        target_names=["Normal", "Tuberculosis"]
    ))

    # -------- ROC --------
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    print(f"{name} AUC Score: {roc_auc:.4f}")

    return accuracy, fpr, tpr, roc_auc


# ------------------ EVALUATE WITH ATTENTION ------------------
acc_att, fpr_att, tpr_att, auc_att = evaluate_model(model_att, "WITH Attention")

# ------------------ ACCURACY GRAPH (WITH ATTENTION) ------------------
plt.figure()
plt.bar(["WITH Attention"], [acc_att])
plt.ylim(0, 1)
plt.ylabel("Accuracy")
plt.title("Accuracy - WITH Attention Model")
plt.show()


# ------------------ EVALUATE WITHOUT ATTENTION ------------------
acc_no, fpr_no, tpr_no, auc_no = evaluate_model(model_no_att, "WITHOUT Attention")

# ------------------ ACCURACY GRAPH (WITHOUT ATTENTION) ------------------
plt.figure()
plt.bar(["WITHOUT Attention"], [acc_no])
plt.ylim(0, 1)
plt.ylabel("Accuracy")
plt.title("Accuracy - WITHOUT Attention Model")
plt.show()


# ------------------ ROC GRAPH (SEPARATE) ------------------
plt.figure()
plt.plot(fpr_att, tpr_att, label=f"With Attention (AUC = {auc_att:.2f})")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - WITH Attention")
plt.legend()
plt.show()

plt.figure()
plt.plot(fpr_no, tpr_no, label=f"Without Attention (AUC = {auc_no:.2f})")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - WITHOUT Attention")
plt.legend()
plt.show()


# ------------------ FINAL SUMMARY ------------------
print("\nFinal Comparison:")
print(f"Attention Model Accuracy: {acc_att:.4f}")
print(f"No Attention Model Accuracy: {acc_no:.4f}")
print(f"Attention Model AUC: {auc_att:.4f}")
print(f"No Attention Model AUC: {auc_no:.4f}")
