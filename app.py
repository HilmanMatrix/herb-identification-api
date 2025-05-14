import io
import os
import requests
from flask import Flask, request, Response
from PIL import Image
from ultralytics import YOLO

app = Flask(__name__)

# --- CONFIGURATION -----------------------------------------------------------

# Where to store the model on disk
MODEL_PATH = "best.pt"
# Your six classes, in exactly the same order you trained YOLO with them:
HERB_CLASSES = [
    "Variegated Mexican Mint",
    "Java Pennywort",
    "Mexican Mint",
    "Green Chiretta",
    "Java Tea",
    "Chinese Gynura"
]
# Minimum confidence threshold to call it “a herb”
CONF_THRESHOLD = 0.8
# Detection threshold passed to YOLO (only boxes ≥ .25 will ever appear in results)
YOLO_CONF = 0.25

# ------------------------------------------------------------------------------

def download_model():
    """Download best.pt from Google Drive if it’s not already here."""
    if not os.path.exists(MODEL_PATH):
        print("Downloading best.pt from Google Drive…")
        file_id = "107Egyp0zJih7XTlNq2pJMFsb1JSiSPK2"
        url = f"https://drive.google.com/uc?id={file_id}"
        resp = requests.get(url, stream=True)
        resp.raise_for_status()
        with open(MODEL_PATH, "wb") as f:
            for chunk in resp.iter_content(1024):
                f.write(chunk)
        print("Download complete.")

# Ensure model is present before loading
download_model()

# Load YOLO model and set its internal confidence threshold
model = YOLO(MODEL_PATH)
model.conf = YOLO_CONF

# --- END CONFIGURATION -------------------------------------------------------

@app.route("/predict", methods=["POST"])
def predict():
    # 1) Validate input
    payload = request.get_json(force=True)
    if not payload or "image_url" not in payload:
        print("❌ No image_url in request")
        return Response("Error: No image URL provided", status=400)

    image_url = payload["image_url"]
    print("Fetching image from:", image_url)

    # 2) Download the image
    try:
        resp = requests.get(image_url, stream=True, timeout=10)
        resp.raise_for_status()
    except Exception as e:
        print("❌ Failed to download image:", e)
        return Response("Error: Failed to download image", status=400)

    # 3) Open & preprocess
    try:
        img = Image.open(io.BytesIO(resp.content)).convert("RGB")
        img = img.resize((640, 640))
    except Exception as e:
        print("❌ Failed to open/process image:", e)
        return Response("Error: Could not process image", status=400)

    # 4) Run YOLO
    results = model(img)

    # 5) If no detections at all → “Not a Herb”
    if not results[0].boxes:
        final = "Not a Herb"
    else:
        # 6) Sort boxes by descending confidence
        boxes = sorted(
            results[0].boxes,
            key=lambda b: b.conf[0].item(),
            reverse=True
        )
        top = boxes[0]
        class_id = int(top.cls[0].item())
        conf = float(top.conf[0].item())

        # 7) Decide: below threshold → Not a Herb; otherwise pick class name
        if conf < CONF_THRESHOLD:
            final = "Not a Herb"
        else:
            final = HERB_CLASSES[class_id]

    # 8) Log the final decision
    print(f"Final prediction: {final}")

    # 9) Return *only* the plain‐text herb name (no JSON, no quotes, no confidence)
    return Response(final, mimetype="text/plain")


if __name__ == "__main__":
    # In Render this will pick up the $PORT env var automatically
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
