import io, os, requests
from flask import Flask, request, make_response
from PIL import Image
from ultralytics import YOLO
import torch
import numpy as np

app = Flask(__name__)

# ───────────────── CONFIGURATION ────────────────────────
MODEL_PATH      = "best.pt"
DRIVE_FILE_ID   = "107Egyp0zJih7XTlNq2pJMFsb1JSiSPK2"
HERB_CLASSES    = [
    "Variegated Mexican Mint",
    "Java Pennywort",
    "Mexican Mint",
    "Green Chiretta",
    "Java Tea",
    "Chinese Gynura"
]
CONF_THRESHOLD  = 0.8  # must exceed this to be “a herb”
# ────────────────────────────────────────────────────────

def download_model():
    """Download best.pt from Google Drive if missing."""
    if not os.path.exists(MODEL_PATH):
        url = f"https://drive.google.com/uc?export=download&id={DRIVE_FILE_ID}"
        resp = requests.get(url, stream=True, timeout=30)
        resp.raise_for_status()
        with open(MODEL_PATH, "wb") as f:
            for chunk in resp.iter_content(1024):
                f.write(chunk)
        app.logger.info("Downloaded best.pt from Google Drive")

# ensure we have the model file
download_model()

# load it _as a classifier_ so results[0].probs exists
model = YOLO(MODEL_PATH, task="classify")

@app.route("/predict", methods=["POST"])
def predict():
    payload = request.get_json(silent=True)
    if not payload or "image_url" not in payload:
        return make_response("Error: No image URL provided", 400)

    image_url = payload["image_url"]
    app.logger.info(f"Fetching image from: {image_url}")
    try:
        resp = requests.get(image_url, stream=True, timeout=10)
        resp.raise_for_status()
    except Exception as e:
        app.logger.error(f"Failed to download image: {e}")
        return make_response("Error: Failed to download image", 400)

    # open & resize
    img = Image.open(io.BytesIO(resp.content)).convert("RGB")
    img = img.resize((640, 640))

    # run the classifier
    results = model.predict(img, verbose=False)

    # pull out the numpy array of 6 confidences
    try:
        probs = results[0].probs
        if isinstance(probs, torch.Tensor):
            probs = probs.cpu().numpy()
        elif isinstance(probs, list):
            probs = np.array(probs)
        else:
            raise ValueError(f"Unexpected probs type: {type(probs)}")
    except Exception as e:
        app.logger.error(f"Could not read classification scores: {e}")
        decision = "Not a Herb"
        return make_response(decision, 200, {"Content-Type": "text/plain"})

    # log the full vector for debugging
    app.logger.info("Raw confidences: " +
        ", ".join(f"{HERB_CLASSES[i]} {probs[i]:.2f}" for i in range(len(probs)))
    )

    # pick the top class
    top_idx  = int(np.argmax(probs))
    top_conf = float(probs[top_idx])

    # apply your 0.8 cutoff
    if top_conf < CONF_THRESHOLD:
        decision = "Not a Herb"
    else:
        decision = HERB_CLASSES[top_idx]

    app.logger.info(f"Final decision: {decision}")
    # return _only_ the name, plain‐text
    return make_response(decision, 200, {"Content-Type": "text/plain"})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
