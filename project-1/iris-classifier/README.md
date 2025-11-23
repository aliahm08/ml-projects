# Simple MNIST Logistic Regression

This project follows the classic scikit-learn workflow to train a multinomial logistic regression classifier on the TensorFlow MNIST digits dataset, evaluate it, and persist the artifacts with standard `pickle`. When scikit-learn is unavailable (e.g., offline environments), a lightweight `mini_sklearn` fallback mimics the handful of APIs we rely on so the script still runs end-to-end.

## Repository layout

```
iris-classifier/
├── main.py          # Training script
├── mini_sklearn/    # Tiny fallback implementations of the sklearn APIs we use
├── model.pkl        # Saved LogisticRegression model (created when you run main.py)
├── scaler.pkl       # Saved StandardScaler instance (created when you run main.py)
├── data/            # Local cache for the TensorFlow MNIST archive
├── README.md        # You are here
└── requirements.txt # Python dependencies
```

## Step-by-step workflow

1. **Load data** – `tensorflow.keras.datasets.mnist` is used when TensorFlow is installed. Otherwise the script downloads the official TensorFlow `mnist.npz` archive into `data/` and loads it with NumPy.
2. **Train/test split** – the TensorFlow dataset already includes dedicated train (60k) and test (10k) partitions, so no extra splitting is needed. Optional environment variables (`TRAIN_FRACTION`, `TEST_FRACTION`) allow you to randomly subsample each split for quicker experiments.
3. **Scale data** – pixel vectors are flattened to length 784 and standardized with `sklearn.preprocessing.StandardScaler` (or the bundled fallback) so each feature has zero mean and unit variance.
4. **Train model** – a multinomial `LogisticRegression` (LBFGS solver, 200 iterations by default) fits on the scaled training set. If real scikit-learn is unavailable the bundled implementation performs mini-batch gradient descent to approximate the same behaviour.
5. **Evaluate** – accuracy plus the full `classification_report` are printed using the hold-out MNIST test set.
6. **Save artifacts** – both the fitted model (`model.pkl`) and the scaler (`scaler.pkl`) are serialized with `pickle.dump` for future inference.

## Getting started

1. Create/activate a virtual environment (recommended).
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
   Only NumPy is strictly required. If you later gain network access, installing `scikit-learn` will make training faster and fully align with the official estimators.
   3. Run the trainer:
   ```
   python main.py
   ```

To speed up experimentation you can subsample by exporting fractions before running:

```
TRAIN_FRACTION=0.1 TEST_FRACTION=0.1 python main.py
```

Both values must be in `(0, 1]` and large enough to yield at least one sample (the script enforces this).

## Reusing the artifacts

```
import pickle

with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# Scale new 28×28 images (already flattened) before predicting.
# example_pixels = scaler.transform(example_pixels)
# preds = model.predict(example_pixels)
```

Keep `model.pkl` and `scaler.pkl` together—the logistic regression expects inputs scaled using the same statistics captured during training.
