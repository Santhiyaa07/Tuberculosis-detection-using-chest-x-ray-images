import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import DenseNet121

# ------------------ CONFIG ------------------
IMG_SIZE = 224
BATCH_SIZE = 16
EPOCHS = 20

train_dir = r"E:\TB-Project\datasets\Tuberculosis Detection.v3-dataset_v1.folder\train"
val_dir   = r"E:\TB-Project\datasets\Tuberculosis Detection.v3-dataset_v1.folder\valid"
test_dir  = r"E:\TB-Project\datasets\Tuberculosis Detection.v3-dataset_v1.folder\test"
# --------------------------------------------

# Load datasets
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    train_dir,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    label_mode="binary"
)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    val_dir,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    label_mode="binary"
)

test_ds = tf.keras.preprocessing.image_dataset_from_directory(
    test_dir,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    label_mode="binary",
    shuffle=False
)

# Normalize
normalization = layers.Rescaling(1./255)

# Attention block
def attention_block(x):
    att = layers.Dense(x.shape[-1], activation="sigmoid")(x)
    return layers.Multiply()([x, att])

# Base Hybrid CNN (DenseNet)
base_model = DenseNet121(
    weights="imagenet",
    include_top=False,
    input_shape=(IMG_SIZE, IMG_SIZE, 3)
)

base_model.trainable = False  # transfer learning

# Model architecture
x = base_model.output
x = layers.GlobalAveragePooling2D()(x)
x = attention_block(x)
x = layers.Dense(128, activation="relu")(x)
output = layers.Dense(1, activation="sigmoid")(x)

model = models.Model(inputs=base_model.input, outputs=output)

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# Train
history = model.fit(
    train_ds.map(lambda x, y: (normalization(x), y)),
    validation_data=val_ds.map(lambda x, y: (normalization(x), y)),
    epochs=EPOCHS
)

# Save model
model.save("tb_attention_hybrid_model.h5")

# Evaluate on test set
test_loss, test_acc = model.evaluate(
    test_ds.map(lambda x, y: (normalization(x), y))
)

print("✅ Test Accuracy:", test_acc)
