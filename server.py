from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from ultralytics import YOLO
from PIL import Image
import os, uuid, traceback

app = Flask(__name__)
CORS(app)

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

model = YOLO("yolov8n-cls.pt")
print("✅ YOLO loaded")

# --------------------
# Health check
# --------------------
@app.route("/", methods=["GET"])
def home():
    return jsonify({"status": "ok"}), 200

# --------------------
# Serve uploaded images
# --------------------
@app.route("/uploads/<filename>")
def serve_image(filename):
    return send_from_directory(UPLOAD_DIR, filename)

# --------------------
# Upload + classify
# --------------------
@app.route("/api/upload", methods=["POST"])
def upload():
    try:
        image_file = request.files.get("image")
        machine = request.form.get("machineName")

        if not image_file or not machine:
            return jsonify({"success": False, "error": "missing data"}), 400

        filename = f"{uuid.uuid4().hex}.jpg"
        path = os.path.join(UPLOAD_DIR, filename)
        image_file.save(path)

        image = Image.open(path).convert("RGB")
        results = model(image)

        probs = results[0].probs
        names = results[0].names

        top5 = [names[int(i)].lower() for i in probs.top5]

        # --------------------
        # Beverage can override
        # --------------------
        can_keywords = [
            "can", "cola", "soda", "beer", "drink",
            "coca", "pepsi", "redbull", "mirinda",
            "fanta", "sprite", "7up"
        ]

        if any(k in " ".join(top5) for k in can_keywords):
            return jsonify({
                "success": True,
                "machineName": machine,
                "classification": "metal",
                "confidence": 0.95,
                "imageUrl": f"/uploads/{filename}",
                "note": "beverage can override"
            })

        # --------------------
        # Plastic vs metal scoring
        # --------------------
        plastic_keys = ["bottle", "plastic", "packet", "wrapper", "cup", "bag"]
        metal_keys = ["metal", "steel", "aluminum", "foil", "tray", "cap", "ladle", "spoon"]

        plastic_score = 0
        metal_score = 0

        for i, conf in zip(probs.top5, probs.top5conf):
            label = names[int(i)].lower()
            if any(k in label for k in plastic_keys):
                plastic_score += conf
            if any(k in label for k in metal_keys):
                metal_score += conf * 2.0

        classification = "metal" if metal_score > plastic_score else "plastic"

        return jsonify({
            "success": True,
            "machineName": machine,
            "classification": classification,
            "confidence": round(float(max(plastic_score, metal_score)), 4),
            "imageUrl": f"/uploads/{filename}"
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 10000)))