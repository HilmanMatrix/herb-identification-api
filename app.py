import io
import os
import requests
from flask import Flask, request, Response
from PIL import Image
from ultralytics import YOLO

# Initialize Flask app
app = Flask(__name__)

def download_model():
    """
    Downloads the YOLO model weights (best.pt) from Google Drive
    if they are not already present on disk.
    """
    model_path = "best.pt"
    if not os.path.exists(model_path):
        print("Downloading best.pt from Google Drive…")
        file_id = "107Egyp0zJih7XTlNq2pJMFsb1JSiSPK2"
        url = f"https://drive.google.com/uc?id={file_id}"
        resp = requests.get(url, stream=True)
        if resp.status_code == 200:
            with open(model_path, "wb") as f:
                for chunk in resp.iter_content(1024):
                    f.write(chunk)
            print("Download complete.")
        else:
            print(f"❌ Failed to download best.pt (status {resp.status_code})")

# Ensure model is downloaded
download_model()

# Load YOLO model
model = YOLO("best.pt")
# Lower built-in threshold; we'll enforce our own 0.8 cutoff below
model.conf = 0.25

# Class names matching training order
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
    """
    Expects JSON: { "image_url": "<public_url>" }
    Returns plain-text herb name or "Not a Herb".
    """
    payload = request.get_json(silent=True)
    if not payload or "image_url" not in payload:
        msg = "Error: no image_url provided"
        print(msg)
        return Response(msg, status=400, mimetype="text/plain")

    image_url = payload["image_url"]
    try:
        # Download image
        resp = requests.get(image_url, stream=True, timeout=10)
        if resp.status_code != 200:
            msg = f"Error: failed to download image ({resp.status_code})"
            print(msg)
            return Response(msg, status=400, mimetype="text/plain")

        # Preprocess image
        img = Image.open(io.BytesIO(resp.content)).convert("RGB")
        img = img.resize((640, 640))

        # Inference
        results = model(img)

        # Decision logic
        if not results[0].boxes:
            decision = "Not a Herb"
        else:
            # Sort by confidence (highest first)
            sorted_boxes = sorted(
                results[0].boxes,
                key=lambda b: b.conf[0].item(),
                reverse=True
            )
            top = sorted_boxes[0]
            conf = float(top.conf[0].item())
            if conf < 0.8:
                decision = "Not a Herb"
            else:
                class_id = int(top.cls[0].item())
                decision = HERB_CLASSES[class_id]

        # Log decision to Render
        print(f"🎯 Prediction result: {decision}")
        # Return plain text (just the name)
        return Response(decision, status=200, mimetype="text/plain")

    except Exception as e:
        err = f"Error: exception during prediction: {e}"
        print(err)
        return Response(err, status=500, mimetype="text/plain")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
