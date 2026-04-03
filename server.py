"""
Flask inference server for the MNIST drawing UI.

Loads best_model.onnx (produced by mnist_pipeline.py) and exposes two endpoints:
  POST /predict   – accepts a 28×28 grayscale pixel array, returns JSON predictions
  GET  /          – serves the UI (ui.html from the same directory)

Usage:
  pip install flask onnxruntime numpy pillow
  python server.py
Then open http://localhost:5000 in your browser.
"""

import io
import base64
import json
import os

import numpy as np
from flask import Flask, request, jsonify, send_from_directory
from PIL import Image
import onnxruntime as ort

# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────
MODEL_PATH = os.environ.get("MODEL_PATH", "best_model.onnx")
HOST       = os.environ.get("HOST", "0.0.0.0")
PORT       = int(os.environ.get("PORT", 5000))
MEAN, STD  = 0.1307, 0.3081

app = Flask(__name__, static_folder=".")

# ─────────────────────────────────────────────
# Load ONNX model at startup
# ─────────────────────────────────────────────
print(f"Loading model from {MODEL_PATH} …")
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"Model not found at '{MODEL_PATH}'. "
        "Run mnist_pipeline.py first to train and export the model."
    )

sess_options = ort.SessionOptions()
sess_options.intra_op_num_threads = 2
session = ort.InferenceSession(MODEL_PATH, sess_options=sess_options,
                                providers=["CPUExecutionProvider"])
input_name  = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name
print(f"Model ready — input: '{input_name}', output: '{output_name}'")


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────
def softmax(logits: np.ndarray) -> np.ndarray:
    e = np.exp(logits - logits.max())
    return e / e.sum()


def preprocess_pixels(pixels: list[float]) -> np.ndarray:
    """
    Accept a flat list of 784 grayscale values in [0, 1] (white digit on black),
    normalise with MNIST stats, and return a (1, 1, 28, 28) float32 array.
    """
    arr = np.array(pixels, dtype=np.float32).reshape(1, 1, 28, 28)
    arr = (arr - MEAN) / STD
    return arr


def preprocess_image_bytes(img_bytes: bytes) -> np.ndarray:
    """
    Accept raw PNG/JPEG bytes (e.g. from a data-URL), resize to 28×28 greyscale,
    normalise, and return a (1, 1, 28, 28) float32 array.
    """
    img = Image.open(io.BytesIO(img_bytes)).convert("L").resize((28, 28), Image.LANCZOS)
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = (arr - MEAN) / STD
    return arr.reshape(1, 1, 28, 28)


# ─────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────
@app.route("/")
def index():
    return send_from_directory(".", "ui.html")


@app.route("/predict", methods=["POST"])
def predict():
    """
    Accepts JSON with one of:
      { "pixels": [784 floats in 0-1, white digit on black bg] }
      { "image":  "<base64-encoded PNG/JPEG>" }

    Returns:
      {
        "prediction": int,
        "confidence": float (0-1),
        "probabilities": [10 floats]
      }
    """
    try:
        data = request.get_json(force=True)

        if "pixels" in data:
            pixels = data["pixels"]
            if len(pixels) != 784:
                return jsonify({"error": f"Expected 784 pixels, got {len(pixels)}"}), 400
            tensor = preprocess_pixels(pixels)

        elif "image" in data:
            img_bytes = base64.b64decode(data["image"])
            tensor = preprocess_image_bytes(img_bytes)

        else:
            return jsonify({"error": "Provide 'pixels' or 'image' in request body"}), 400

        logits = session.run([output_name], {input_name: tensor})[0][0]
        probs  = softmax(logits)
        pred   = int(probs.argmax())

        return jsonify({
            "prediction":   pred,
            "confidence":   float(probs[pred]),
            "probabilities": probs.tolist(),
        })

    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/health")
def health():
    return jsonify({"status": "ok", "model": MODEL_PATH})


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print(f"Starting server on http://{HOST}:{PORT}")
    app.run(host=HOST, port=PORT, debug=False)
