import io
import os
import requests
from flask import Flask, request, make_response
from PIL import Image
from ultralytics import YOLO

app = Flask(__name__)

# ──────────────────── CONFIGURATION ────────────────────
MODEL_PATH   = "best.pt"
GOOGLE_DRIVE_FILE_ID = "107Egyp0zJih7XTlNq2pJMFsb1JSiSPK2"
YOLO_CONF    = 0.25   # Pre‐filter threshold (boxes won't appear under this, if it were detection)
CONF_THRESHOLD = 0.8  # Final cutoff for “is it a herb?”
HERB_CLASSES = [
    "Variegated Mexican Mint",
    "Java Pennywort",
    "Mexican Mint",
    "Green Chiretta",
    "Java Tea",
    "Chinese Gynura"
]
# ────────────────────────────────────────────────────────

def download_model():
    """Download best.pt from Google Drive if it’s not already here."""
    if not os.path.exists(MODEL_PATH):
        print("Downloading best.pt from Google Drive…")
        url = f"https://drive.google.com/uc?id={GOOGLE_DRIVE_FILE_ID}"
        resp = requests.get(url, stream=True)
        resp.raise_for_status()
        with open(MODEL_PATH, "wb") as f:
            for chunk in resp.iter_content(1024):
                f.write(chunk)
        print("Download complete.")

# Ensure the model is present, then load it
download_model()
model = YOLO(MODEL_PATH)
model.conf = YOLO_CONF

@app.route("/predict", methods=["POST"])
def predict():
    payload = request.get_json(silent=True)
    if not payload or "image_url" not in payload:
        msg = "Error: No image URL provided"
        print(msg)
        return make_response(msg, 400)

    image_url = payload["image_url"]
    print("Fetching image from:", image_url)
    resp = requests.get(image_url, stream=True, timeout=10)
    if resp.status_code != 200:
        msg = f"Error: Failed to download image ({resp.status_code})"
        print(msg)
        return make_response(msg, 400)

    # Load & resize
    img = Image.open(io.BytesIO(resp.content)).convert("RGB")
    img = img.resize((640, 640))

    # Run classification
    results = model(img)  

    # ultralytics classification models put their scores in .probs
    if not hasattr(results[0], "probs"):
        # Unexpected: treat as no‐detection as fallback
        decision = "Not a Herb"
        print("⚠️  No .probs on results, falling back to:", decision)
        return make_response(decision, 200, {"Content-Type": "text/plain"})

    # Extract per‐class probabilities (a Tensor of length 6)
    probs = results[0].probs.cpu().numpy()  # e.g. [1.0, 0.0, 0.0, ...]
    # Log the full vector so you still see it in your Render logs:
    print("Raw confidences:", 
          ", ".join(f"{HERB_CLASSES[i]} {probs[i]:.2f}" for i in range(len(probs))))

    # Find the top class
    top_idx  = int(probs.argmax())
    top_conf = float(probs[top_idx])

    # Apply your cutoff
    if top_conf < CONF_THRESHOLD:
        decision = "Not a Herb"
    else:
        decision = HERB_CLASSES[top_idx]

    # Print & return _only_ the decision (no quotes, no confidence)
    print("Final decision:", decision)
    return make_response(decision, 200, {"Content-Type": "text/plain"})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    # Bind both localhost & internal IP so Render can route traffic correctly
    app.run(host="0.0.0.0", port=port)
