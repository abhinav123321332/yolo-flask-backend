from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from ultralytics import YOLO
from PIL import Image
import os
import uuid
import traceback

app = Flask(__name__)
CORS(app)

# ----------------------------
# Configuration
# ----------------------------
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

MODEL_PATH = "yolov8n-cls.pt"

# ----------------------------
# Load YOLO model once
# ----------------------------
model = YOLO(MODEL_PATH)
print("✅ YOLO model loaded successfully")

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
def serve_uploaded_image(filename):
    return send_from_directory(UPLOAD_DIR, filename)

# ----------------------------
# Upload + classify (canteen-focused)
# ----------------------------
@app.route("/api/upload", methods=["POST"])
def upload_image():
    try:
        # ---- Validate input ----
        image_file = request.files.get("image")
        machine_name = request.form.get("machineName")

        if not image_file:
            return jsonify({"success": False, "error": "Image missing"}), 400

        if not machine_name:
            return jsonify({"success": False, "error": "machineName missing"}), 400

        # ---- Save image ----
        ext = os.path.splitext(image_file.filename)[1].lower() or ".jpg"
        filename = f"{uuid.uuid4().hex}{ext}"
        filepath = os.path.join(UPLOAD_DIR, filename)
        image_file.save(filepath)

        image_url = f"/uploads/{filename}"

        # ---- Load image ----
        try:
            image = Image.open(filepath).convert("RGB")
        except Exception as e:
            return jsonify({
                "success": False,
                "error": "Image decode failed",
                "details": str(e)
            }), 400

        # ---- YOLO inference ----
        results = model(image)

        if not results or not results[0].probs:
            return jsonify({
                "success": False,
                "error": "Model returned no probabilities"
            }), 500

        probs = results[0].probs
        names = results[0].names

        # ----------------------------
        # CANTEEN-OPTIMIZED LOGIC
        # ----------------------------
        plastic_keywords = [
            "bottle", "water_bottle", "beer_bottle", "pop_bottle",
            "plastic", "container", "cup", "wrapper",
            "packet", "bag", "sachet", "lid"
        ]

        metal_keywords = [
            "can", "tin", "aluminum", "steel",
            "bottlecap", "cap",
            "foil", "tray",
            "spoon", "ladle", "strainer",
            "screw", "nut", "washer"
        ]

        plastic_score = 0.0
        metal_score = 0.0

        # ✅ Use TOP-5 only (less noise)
        for idx, score in zip(probs.top5, probs.top5conf):
            label = names[int(idx)].lower()

            if any(k in label for k in plastic_keywords):
                plastic_score += score * 1.0

            if any(k in label for k in metal_keywords):
                metal_score += score * 2.5   # 🔥 strong metal bias

            # 🔒 HARD RULE: cans & caps are always metal
            if any(k in label for k in ["can", "bottlecap", "cap"]):
                metal_score += 1.0

        # ---- Final forced decision ----
        if metal_score > plastic_score:
            classification = "metal"
            confidence = metal_score
        else:
            classification = "plastic"
            confidence = plastic_score

        return jsonify({
            "success": True,
            "machineName": machine_name,
            "imageUrl": image_url,
            "classification": classification,
            "confidence": round(float(confidence), 4)
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({
            "success": False,
            "error": str(e),
            "type": type(e).__name__
        }), 500

# ----------------------------
# Entry point (Render / local)
# ----------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
