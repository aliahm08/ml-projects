# Iris-Handwriting Classifier

A multinomial logistic regression that started life as an "iris classifier" pattern and was extended to recognize handwritten digits. The project documents how the classic iris workflow (tabular features → standardization → logistic regression) was transferred to pixel-based handwriting recognition.

## Repository layout

```
iris-handwriting-classifier/
├── main.py          # Training and evaluation script
├── mini_sklearn/    # Lightweight fallbacks for scikit-learn APIs
├── model.pkl        # Saved LogisticRegression model (created when you run main.py)
├── scaler.pkl       # Saved StandardScaler instance (created when you run main.py)
├── data/            # Local cache for the TensorFlow MNIST archive
├── README.md        # You are here
└── requirements.txt # Python dependencies
```

## From iris classifier to handwriting recognizer

1. **Reuse the iris recipe** – The original iris classifier used a simple pipeline: feature scaling with `StandardScaler` followed by `LogisticRegression` for multi-class prediction. That same recipe is preserved here to keep the math interpretable.
2. **Translate features** – Instead of four iris measurements, MNIST provides 28×28 pixel grids. `main.py` flattens each image into 784 numeric features so the scaler and classifier operate just like they did on iris features.
3. **Standardize the pixels** – As with iris sepal/petal lengths, pixel intensities are standardized to zero mean and unit variance so the logistic regression can converge reliably.
4. **Tune for more classes** – MNIST has 10 digit classes vs. iris's three flower species. The solver is configured for multinomial training (`lbfgs`, 200 iterations) to handle the expanded label space.
5. **Persist artifacts for reuse** – The trained `model.pkl` and `scaler.pkl` are saved together so the same preprocessing/weights can be applied when serving handwritten digit predictions.

## Project walkthrough

Follow these steps to reproduce the handwriting classifier end to end.

1. **Set up the environment**
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

2. **Train and evaluate the model**
   ```bash
   python main.py
   ```
   The script will download the TensorFlow MNIST archive if TensorFlow is unavailable, flatten and scale the pixels, fit the logistic regression model, print accuracy plus a classification report, and save both the model and scaler artifacts locally.

3. **Speed up experimentation (optional)**
   To run faster while experimenting, subsample the dataset using environment variables:
   ```bash
   TRAIN_FRACTION=0.1 TEST_FRACTION=0.1 python main.py
   ```
   Fractions must be in `(0, 1]` and large enough to keep at least one example of each digit.

4. **Use the saved artifacts for inference**
   ```python
   import pickle

   with open("model.pkl", "rb") as model_file:
       model = pickle.load(model_file)

   with open("scaler.pkl", "rb") as scaler_file:
       scaler = pickle.load(scaler_file)

   # new_pixels is an array of shape (n_samples, 784) with float pixel values
   new_pixels_scaled = scaler.transform(new_pixels)
   predictions = model.predict(new_pixels_scaled)
   ```

## Why logistic regression works here

- **Interpretable baseline** – Keeps the same linear decision boundaries as the iris project, making it easy to inspect weights per pixel/class.
- **Strong benchmark** – Despite its simplicity, multinomial logistic regression reaches competitive accuracy on MNIST and establishes a baseline for future CNN upgrades.
- **Minimal dependencies** – If scikit-learn is unavailable, the bundled `mini_sklearn` fallback mirrors just the APIs needed so the workflow remains runnable offline.
