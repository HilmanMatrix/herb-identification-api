import io
import os
import requests
from flask import Flask, request, jsonify
from PIL import Image
from ultralytics import YOLO

app = Flask(__name__)

# Function to download best.pt from Google Drive if not found
def download_model():
    model_path = "best.pt"
    if not os.path.exists(model_path):
        print("Downloading best.pt from Google Drive...")
        file_id = "107Egyp0zJih7XTlNq2pJMFsb1JSiSPK2"  # ✅ Updated with your correct file ID
        url = f"https://drive.google.com/uc?id={file_id}"
        response = requests.get(url, stream=True)

        if response.status_code == 200:
            with open(model_path, "wb") as f:
                for chunk in response.iter_content(1024):
                    f.write(chunk)
            print("Download complete.")
        else:
            print("Failed to download best.pt")

# Ensure the model is downloaded
download_model()

# Load the YOLO model
model = YOLO("best.pt")

# Optional: Set a lower threshold to allow weaker detections (your logic will control actual filtering)
model.conf = 0.25

# Define your 6 herb classes
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
        response = requests.get(image_url, stream=True)
        if response.status_code != 200:
            return jsonify({"error": "Failed to download image"}), 400

        img = Image.open(io.BytesIO(response.content)).convert("RGB")
        img = img.resize((640, 640))  # Resize for YOLOv8 training format

        results = model(img)

        if not results[0].boxes:
            return jsonify({"herb_name": "Not a Herb", "confidence": 0.0}), 200

        # Sort detections by confidence
        sorted_boxes = sorted(results[0].boxes, key=lambda b: b.conf[0].item(), reverse=True)
        top_box = sorted_boxes[0]
        class_id = int(top_box.cls[0].item())
        conf = float(top_box.conf[0].item())

        # If confidence is less than 0.8, return "Not a Herb"
        if conf < 0.8:
            return jsonify({"herb_name": "Not a Herb", "confidence": conf}), 200

        herb_name = HERB_CLASSES[class_id]
        return jsonify({"herb_name": herb_name, "confidence": conf}), 200

    except Exception as e:
        return jsonify({"error": f"Could not process image: {str(e)}"}), 400

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
