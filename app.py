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
        file_id = "1-sTy5PN78zZR3M1AjMzjBlMMtBYeGabP"  # Your Google Drive file ID
        url = f"https://drive.google.com/uc?id={file_id}"
        response = requests.get(url, stream=True)

        if response.status_code == 200:
            with open(model_path, "wb") as f:
                for chunk in response.iter_content(1024):
                    f.write(chunk)
            print("Download complete.")
        else:
            print("Failed to download best.pt")

# Ensure the model is available
download_model()

# Load YOLO model
model = YOLO("best.pt")

# ✅ Lower Confidence Threshold
model.conf = 0.25  # Reduce from 0.5 to 0.25 for more detections

# Define herb classes (must match YOLO training)
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
        # ✅ Step 1: Download and Open Image
        response = requests.get(image_url, stream=True)
        if response.status_code != 200:
            return jsonify({"error": "Failed to download image"}), 400

        img = Image.open(io.BytesIO(response.content)).convert("RGB")

        # ✅ Step 2: Resize Image to 640x640 (Fixes Mismatched Size Issue)
        img = img.resize((640, 640))  # Resize to match YOLO training size

        # ✅ Step 3: Run YOLO Inference
        results = model(img)

        # ✅ Debugging: Print Detection Results
        print("Detection Results:", results)

        # ✅ Step 4: Handle No Detection Case
        if not results[0].boxes:  # If no objects detected
            return jsonify({"herb_name": "Not a Herb", "confidence": 0.0}), 200

        # ✅ Step 5: Sort Detections by Confidence and Get the Most Confident One
        sorted_boxes = sorted(results[0].boxes, key=lambda b: b.conf[0].item(), reverse=True)
        top_box = sorted_boxes[0]  # Get the highest confidence detection

        class_id = int(top_box.cls[0].item())
        conf = float(top_box.conf[0].item())

        # ✅ Step 6: Ensure Confidence is Above Threshold
        if conf < 0.25:  # Lowered from 0.5 to 0.25
            return jsonify({"herb_name": "Not a Herb", "confidence": conf}), 200

        # ✅ Step 7: Map Class ID to Herb Name
        herb_name = HERB_CLASSES[class_id]

        return jsonify({"herb_name": herb_name, "confidence": conf}), 200

    except Exception as e:
        return jsonify({"error": f"Could not process image: {str(e)}"}), 400

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
