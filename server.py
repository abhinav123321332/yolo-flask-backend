import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from ultralytics import YOLO
import torch

app = Flask(__name__)
CORS(app)

# --- Railway config ---
PORT = int(os.environ.get("PORT", 8080))

# --- Force CPU (Railway has NO GPU) ---
torch.set_default_device("cpu")

# --- Load model ---
MODEL_PATH = "yolov8n-cls.pt"

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

model = YOLO(MODEL_PATH)

@app.route("/", methods=["GET"])
def health():
    return {"status": "ok", "message": "YOLO Flask backend running"}

@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image file"}), 400

    image = request.files["image"]
    result = model(image, verbose=False)[0]

    top1 = result.probs.top1
    confidence = float(result.probs.top1conf)
    label = model.names[top1]

    return jsonify({
        "label": label,
        "confidence": confidence
    })

if __name__ == "__main__":
    print(f"Starting server on port {PORT}")
    app.run(host="0.0.0.0", port=PORT)
