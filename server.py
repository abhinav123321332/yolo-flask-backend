from flask import Flask, request, jsonify
from ultralytics import YOLO
from PIL import Image

app = Flask(__name__)

# Load trained YOLO classification model
model = YOLO("runs/classify/train2/weights/best.pt")

@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    img = Image.open(request.files["image"]).convert("RGB")
    result = model(img)[0]

    cls = result.names[result.probs.top1]
    conf = float(result.probs.top1conf)

    return jsonify({
        "class": cls,
        "confidence": round(conf, 4)
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
