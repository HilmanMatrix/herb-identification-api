import io
import requests
from flask import Flask, request, jsonify
from PIL import Image
from ultralytics import YOLO

app = Flask(__name__)

# Load your YOLO model (ensure best.pt is in your project folder)
model = YOLO("best.pt")

# Define the six herb classes (must match your training)
HERB_CLASSES = [
    "Variegated Mexican Mint",
    "Java Pennywort",
    "Mexican Mint",
    "Green Chiretta",
    "Java Tea",
    "Chinese Gynura"
]

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    
    if not data or "image_url" not in data:
        return jsonify({"error": "No image URL provided"}), 400

    image_url = data["image_url"]
    
    try:
        # Download the image from the URL
        response = requests.get(image_url, stream=True)
        
        # Check if the request was successful
        if response.status_code != 200:
            return jsonify({"error": "Failed to download image"}), 400

        # Try to open the image
        img = Image.open(response.raw).convert("RGB")

    except Exception as e:
        return jsonify({"error": f"Invalid image format: {str(e)}"}), 400

    # Run inference with YOLO
    results = model(img)

    # If no detections, return "Not a Herb"
    if not results[0].boxes:
        return jsonify({"herb_name": "Not a Herb", "confidence": 0.0}), 200

    # Get the top detection
    top_box = results[0].boxes[0]
    class_id = int(top_box.cls[0].item())
    conf = float(top_box.conf[0].item())

    # Use a confidence threshold (e.g., 0.5)
    if conf < 0.5:
        return jsonify({"herb_name": "Not a Herb", "confidence": conf}), 200

    herb_name = HERB_CLASSES[class_id]
    return jsonify({"herb_name": herb_name, "confidence": conf}), 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
