# Image Reconstruction Using Autoencoder

## Objective
Train an autoencoder on MNIST handwritten digits, compress 28×28 images into a 32-dimensional latent representation, reconstruct them, and evaluate reconstruction error.

## Dataset
MNIST is loaded through `tf.keras.datasets.mnist`. It contains 60,000 training images and 10,000 test images. Images are normalized to [0, 1] and flattened to 784 features.

## Architecture
- Encoder: 784 → 128 → 32
- Decoder: 32 → 128 → 784
- Hidden activation: ReLU; output activation: sigmoid
- Optimizer: Adam; loss: Mean Squared Error (MSE)
- Training: 5 epochs, batch size 256

## Run
```bash
pip install -r requirements.txt
python autoencoder.py
```
The script trains the model, plots original/reconstructed images and training/validation loss, prints test MSE, and saves `mnist_autoencoder.keras`.

## Recorded result
The uploaded Colab notebook reports **Test Reconstruction MSE: 0.013447**. Results can vary across runs.

## Files
`autoencoder.py` is the standalone implementation; `autoencoder.ipynb` preserves the executed notebook and outputs.
