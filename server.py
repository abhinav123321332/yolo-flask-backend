from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from ultralytics import YOLO
from PIL import Image
import io
import os
import uuid
import traceback

app = Flask(__name__)
CORS(app)

# ----------------------------
# Config
# ----------------------------
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

MODEL_PATH = "yolov8n-cls.pt"

# ----------------------------
# Load model once
# ----------------------------
model = YOLO(MODEL_PATH)
print("✅ YOLO model loaded")

# ----------------------------
# Health check
# ----------------------------
@app.route("/", methods=["GET"])
def home():
    return jsonify({"status": "ok"}), 200


# ----------------------------
# Serve uploaded images (preview)
# ----------------------------
@app.route("/uploads/<filename>")
def get_uploaded_file(filename):
    return send_from_directory(UPLOAD_DIR, filename)


# ----------------------------
# API: upload + classify
# ----------------------------
@app.route("/api/upload", methods=["POST"])
def upload_image():
    try:
        # 1️⃣ Validate inputs
        image_file = request.files.get("image")
        machine_name = request.form.get("machineName")

        if not image_file:
            return jsonify({"success": False, "error": "Image missing"}), 400

        if not machine_name:
            return jsonify({"success": False, "error": "machineName missing"}), 400

        # 2️⃣ Save image for preview
        ext = os.path.splitext(image_file.filename)[1].lower() or ".jpg"
        filename = f"{uuid.uuid4().hex}{ext}"
        filepath = os.path.join(UPLOAD_DIR, filename)
        image_file.save(filepath)

        image_url = f"/uploads/{filename}"

        # 3️⃣ Run YOLO classification
        image = Image.open(filepath).convert("RGB")
        results = model(image)

        probs = results[0].probs
        class_id = int(probs.top1)
        raw_label = results[0].names[class_id]
        confidence = float(probs.top1conf)

        # 4️⃣ Map model label → material (TEMP LOGIC)
        label = raw_label.lower()

        if any(k in label for k in ["plastic", "bottle", "bag", "container"]):
            classification = "plastic"
        elif any(k in label for k in ["metal", "can", "chain", "steel", "iron"]):
            classification = "metal"
     

        # 5️⃣ Response matches frontend
        return jsonify({
            "success": True,
            "machineName": machine_name,
            "imageUrl": image_url,
            "classification": classification,
            "confidence": round(confidence, 4)
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({
            "success": False,
            "error": str(e),
            "type": type(e).__name__
        }), 500


# ----------------------------
# Entry (Render / local)
# ----------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
