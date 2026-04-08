import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import numpy as np

IMG_SIZE = 224
BATCH_SIZE = 16
EPOCHS = 25  

train_dir = r"E:\TB-Project\datasets\combined_dataset\train"
val_dir   = r"E:\TB-Project\datasets\combined_dataset\valid"

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

normalization = layers.Rescaling(1./255)

train_ds = train_ds.map(lambda x, y: (normalization(x), y))
val_ds   = val_ds.map(lambda x, y: (normalization(x), y))

# ------------------ SE ATTENTION BLOCK ------------------
def se_block(x, reduction=16):
    channels = x.shape[-1]

    se = layers.Dense(channels // reduction, activation='relu')(x)
    se = layers.Dense(channels, activation='sigmoid')(se)

    return layers.Multiply()([x, se])

# ------------------ MODEL BUILDER ------------------
def build_attention_model():

    base_model = DenseNet121(
        weights="imagenet",
        include_top=False,
        input_shape=(IMG_SIZE, IMG_SIZE, 3)
    )

    # Fine-tuning: unfreeze last 30 layers
    base_model.trainable = True
    for layer in base_model.layers[:-30]:
        layer.trainable = False

    x = base_model.output
    x = layers.GlobalAveragePooling2D()(x)

    # Apply SE Attention
    x = se_block(x)

    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.4)(x)
    output = layers.Dense(1, activation="sigmoid")(x)

    model = models.Model(inputs=base_model.input, outputs=output)

    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-5)

    model.compile(
        optimizer=optimizer,
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    return model


# ------------------ CALLBACKS ------------------
early_stop = EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True
)

checkpoint = ModelCheckpoint(
    "tb_attention_model.h5",
    monitor="val_accuracy",
    save_best_only=True,
    mode="max"
)

# ------------------ TRAIN ATTENTION MODEL ------------------
print("\nTraining Improved Attention Model (Fine-tuned + SE)...\n")

model_att = build_attention_model()

history_att = model_att.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=[early_stop, checkpoint]
)

print("\nTraining Completed. Best model saved as tb_attention_model.h5\n")
