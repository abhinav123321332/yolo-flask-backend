from flask import Flask, request, jsonify
from flask_cors import CORS
from ultralytics import YOLO
from PIL import Image
import io
import os
import traceback

app = Flask(__name__)
CORS(app)

# ----------------------------
# Load YOLO classification model ONCE
# ----------------------------
MODEL_PATH = "yolov8n-cls.pt"

try:
    model = YOLO(MODEL_PATH)
    print("✅ YOLO model loaded successfully on CPU")
except Exception as e:
    print("❌ YOLO model load failed:", e)
    raise e


# ----------------------------
# Health check
# ----------------------------
@app.route("/", methods=["GET"])
def home():
    return jsonify({"status": "ok"}), 200


# ----------------------------
# Prediction endpoint
# ----------------------------
@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Accept multiple possible keys (robust)
        file = (
            request.files.get("image")
            or request.files.get("file")
            or request.files.get("img")
        )

        if not file:
            return jsonify({"error": "No image uploaded"}), 400

        # Read image safely
        image_bytes = file.read()
        if not image_bytes:
            return jsonify({"error": "Empty image file"}), 400

        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        # Run YOLO classification
        results = model(image)

        if not results or not results[0].probs:
            return jsonify({"error": "Inference failed"}), 500

        probs = results[0].probs
        class_id = int(probs.top1)
        class_name = results[0].names[class_id]
        confidence = float(probs.top1conf)

        return jsonify({
            "class": class_name,
            "confidence": round(confidence, 4)
        })

    except Exception as e:
        # IMPORTANT: expose error for debugging
        traceback.print_exc()
        return jsonify({
            "error": str(e),
            "type": type(e).__name__
        }), 500


# ----------------------------
# Render / Gunicorn entry
# ----------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
