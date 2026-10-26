# PROJECT 2: IMAGE RECONSTRUCTION USING AUTOENCODER
# Dataset: MNIST | Framework: TensorFlow / Keras

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, Model

# 1. Load the MNIST dataset
(x_train, _), (x_test, _) = tf.keras.datasets.mnist.load_data()

# 2. Normalize pixel values and flatten images
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

x_train = x_train.reshape(-1, 784)
x_test = x_test.reshape(-1, 784)

print("Training images:", x_train.shape)
print("Testing images:", x_test.shape)

# 3. Build the encoder
input_img = layers.Input(shape=(784,), name="input_image")
encoded = layers.Dense(128, activation="relu")(input_img)
latent = layers.Dense(32, activation="relu",
                      name="latent_representation")(encoded)

# 4. Build the decoder
decoded = layers.Dense(128, activation="relu")(latent)
output_img = layers.Dense(784, activation="sigmoid")(decoded)

# 5. Create and compile the autoencoder
autoencoder = Model(input_img, output_img, name="Autoencoder")
autoencoder.compile(optimizer="adam", loss="mse")

autoencoder.summary()

# 6. Train the model
history = autoencoder.fit(
    x_train, x_train,
    epochs=5,
    batch_size=256,
    shuffle=True,
    validation_data=(x_test, x_test),
    verbose=1
)

# 7. Reconstruct test images
reconstructed = autoencoder.predict(x_test[:10], verbose=0)

# 8. Display original and reconstructed images
plt.figure(figsize=(15, 4))

for i in range(10):
    # Original image
    plt.subplot(2, 10, i + 1)
    plt.imshow(x_test[i].reshape(28, 28), cmap="gray")
    plt.axis("off")
    if i == 0:
        plt.ylabel("Original", fontsize=10)

    # Reconstructed image
    plt.subplot(2, 10, i + 11)
    plt.imshow(reconstructed[i].reshape(28, 28), cmap="gray")
    plt.axis("off")
    if i == 0:
        plt.ylabel("Reconstructed", fontsize=10)

plt.suptitle("Original vs Reconstructed MNIST Images")
plt.tight_layout()
plt.show()

# 9. Plot training and validation loss
plt.figure(figsize=(8, 5))
plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Mean Squared Error")
plt.title("Autoencoder Training Performance")
plt.legend()
plt.grid(True)
plt.show()

# 10. Evaluate reconstruction quality
predictions = autoencoder.predict(x_test, verbose=0)
mse = np.mean((x_test - predictions) ** 2)

print(f"Test Reconstruction MSE: {mse:.6f}")

# 11. Save the trained model
autoencoder.save("mnist_autoencoder.keras")
print("Model saved as mnist_autoencoder.keras")
