import io
import os
import requests
from flask import Flask, request, Response
from PIL import Image
from ultralytics import YOLO

# ─── App & Model Initialization ────────────────────────────────────────────────
app = Flask(__name__)

def download_model():
    """
    If best.pt isn’t on disk, fetch it from Google Drive.
    """
    model_path = "best.pt"
    if not os.path.exists(model_path):
        print("Downloading best.pt from Google Drive…")
        # The file_id below should match your Model file on Drive:
        file_id = "107Egyp0zJih7XTlNq2pJMFsb1JSiSPK2"
        url = f"https://drive.google.com/uc?id={file_id}"
        resp = requests.get(url, stream=True)
        if resp.status_code == 200:
            # Write it out in 1KB chunks:
            with open(model_path, "wb") as f:
                for chunk in resp.iter_content(1024):
                    f.write(chunk)
            print("Download complete.")
        else:
            print(f"❌ Download failed: status {resp.status_code}")

# Ensure the model weights are present:
download_model()

# Load YOLO (this reads best.pt from disk)
model = YOLO("best.pt")
# Lower the built-in conf threshold; we'll enforce our own 0.8 cutoff manually.
model.conf = 0.25

# ─── Herb Classes ───────────────────────────────────────────────────────────────
# Make sure this list matches the order you trained your model with:
HERB_CLASSES = [
    "Variegated Mexican Mint",
    "Java Pennywort",
    "Mexican Mint",
    "Green Chiretta",
    "Java Tea",
    "Chinese Gynura"
]

# ─── /predict Endpoint ───────────────────────────────────────────────────────────
@app.route("/predict", methods=["POST"])
def predict():
    """
    Expects JSON: { "image_url": "<publicly-accessible-url>" }
    Returns plain text: either the herb name or “Not a Herb”.
    """
    payload = request.get_json(silent=True)
    # 1) Validate input
    if not payload or "image_url" not in payload:
        err = "Error: no image_url provided"
        print(err)
        return Response(err, status=400, mimetype="text/plain")

    image_url = payload["image_url"]

    try:
        # 2) Download the image bytes
        resp = requests.get(image_url, stream=True, timeout=10)
        if resp.status_code != 200:
            err = f"Error: could not download image ({resp.status_code})"
            print(err)
            return Response(err, status=400, mimetype="text/plain")

        # 3) Open & preprocess image for YOLO
        img = Image.open(io.BytesIO(resp.content)).convert("RGB")
        img = img.resize((640, 640))

        # 4) Run inference
        results = model(img)

        # 5) If no detections at all → “Not a Herb”
        if not results[0].boxes:
            decision = "Not a Herb"
        else:
            # 6) Sort all detections by confidence (highest first)
            boxes = sorted(
                results[0].boxes,
                key=lambda b: b.conf[0].item(),
                reverse=True
            )
            top_box = boxes[0]
            class_id = int(top_box.cls[0].item())
            conf     = float(top_box.conf[0].item())

            # 7) Apply threshold: below 0.8 → “Not a Herb”
            if conf < 0.8:
                decision = "Not a Herb"
            else:
                # 8) Otherwise pick the class name
                decision = HERB_CLASSES[class_id]

        # 9) Log & return the single decision (plain text)
        print("Final decision:", decision)
        return Response(decision, status=200, mimetype="text/plain")

    except Exception as e:
        # Catch-all for any unexpected error
        err = f"Error: exception during prediction: {e}"
        print(err)
        return Response(err, status=500, mimetype="text/plain")

# ─── Run Locally ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Listens on 0.0.0.0:5000 by default
    app.run(host="0.0.0.0", port=5000)
