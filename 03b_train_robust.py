import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks

PROCESSED_DIR = "./data/processed_v2"
MODEL_SAVE_PATH = "./robust_pi_murmur_model.keras"

def build_robust_edge_model():
    # --- Branch A (Spectrogram) ---
    input_spec = layers.Input(shape=(64, 150, 1), name="Input_A_Spectrogram")

    x = layers.DepthwiseConv2D((3, 3), padding="same")(input_spec)
    x = layers.Conv2D(32, (1, 1), padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu6")(x)
    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.DepthwiseConv2D((3, 3), padding="same")(x)
    x = layers.Conv2D(32, (1, 1), padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu6")(x)
    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.DepthwiseConv2D((3, 3), padding="same")(x)
    x = layers.Conv2D(48, (1, 1), padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu6")(x)

    x = layers.DepthwiseConv2D((3, 3), padding="same")(x)
    x = layers.Conv2D(64, (1, 1), padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu6")(x)
    x = layers.MaxPooling2D((2, 2))(x)

    # GAP limits parameter explosion on edge devices
    spec_features = layers.GlobalAveragePooling2D(name="GAP_2D")(x)

    # --- Branch B (Temporal) ---
    input_temp = layers.Input(shape=(16, 1), name="Input_B_PeakInterval")

    y = layers.Conv1D(8, 3, padding="same", dilation_rate=1)(input_temp)
    y = layers.BatchNormalization()(y)
    y = layers.Activation("relu")(y)

    y = layers.Conv1D(16, 3, padding="same", dilation_rate=2)(y)
    y = layers.BatchNormalization()(y)
    y = layers.Activation("relu")(y)

    temp_features = layers.GlobalAveragePooling1D(name="GAP_1D")(y)

    # --- Fusion Head ---
    fused = layers.concatenate([spec_features, temp_features])

    z = layers.Dense(32)(fused)
    z = layers.BatchNormalization()(z)
    z = layers.Activation("relu")(z)
    z = layers.Dropout(0.4)(z) # Increased dropout for anti-overfitting

    output = layers.Dense(1, activation="sigmoid")(z)

    return models.Model(inputs=[input_spec, input_temp], outputs=output)

def main():
    X_train_spec = np.load(os.path.join(PROCESSED_DIR, "X_train_spec.npy"))
    X_train_temp = np.load(os.path.join(PROCESSED_DIR, "X_train_temp.npy"))
    Y_train = np.load(os.path.join(PROCESSED_DIR, "Y_train.npy"))

    X_test_spec = np.load(os.path.join(PROCESSED_DIR, "X_test_spec.npy"))
    X_test_temp = np.load(os.path.join(PROCESSED_DIR, "X_test_temp.npy"))
    Y_test = np.load(os.path.join(PROCESSED_DIR, "Y_test.npy"))

    model = build_robust_edge_model()
    
    # AdamW incorporates direct weight decay to restrict overfitting parameters
    optimizer = tf.keras.optimizers.AdamW(
        learning_rate=tf.keras.optimizers.schedules.CosineDecay(1e-3, decay_steps=1000),
        weight_decay=1e-4
    )

    model.compile(
        optimizer=optimizer,
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")]
    )

    # Callbacks modified to save the best model without abruptly stopping execution
    cb = [
        callbacks.ModelCheckpoint(MODEL_SAVE_PATH, monitor="val_auc", mode="max", save_best_only=True)
    ]

    model.fit(
        x={"Input_A_Spectrogram": X_train_spec, "Input_B_PeakInterval": X_train_temp},
        y=Y_train,
        validation_data=({"Input_A_Spectrogram": X_test_spec, "Input_B_PeakInterval": X_test_temp}, Y_test),
        epochs=50,
        batch_size=32,
        callbacks=cb
    )

if __name__ == "__main__":
    main()