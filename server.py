import os
import torch
from flask import Flask, request, jsonify
from flask_cors import CORS
from ultralytics import YOLO

# ---------- App ----------
app = Flask(__name__)
CORS(app)

# ---------- Config ----------
PORT = int(os.environ.get("PORT", 8080))
MODEL_PATH = "yolov8n-cls.pt"

# ---------- Load model (CPU only) ----------
try:
    device = "cpu"
    model = YOLO(MODEL_PATH)
    model.to(device)
    print("✅ YOLO model loaded successfully on CPU")
except Exception as e:
    print("❌ Model load failed:", e)
    raise e

# ---------- Routes ----------
@app.route("/", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "message": "YOLO Flask backend running"
    })

@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    image = request.files["image"]
    results = model(image, verbose=False)[0]

    probs = results.probs
    class_id = int(probs.top1)
    confidence = float(probs.top1conf)
    label = model.names[class_id]

    return jsonify({
        "label": label,
        "confidence": round(confidence, 4)
    })

# ---------- Start server ----------
if __name__ == "__main__":
    print(f"🚀 Starting server on port {PORT}")
    app.run(host="0.0.0.0", port=PORT)
