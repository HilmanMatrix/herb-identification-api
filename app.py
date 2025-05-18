import io
import os
import requests
from flask import Flask, request, make_response
from PIL import Image
from ultralytics import YOLO

app = Flask(__name__)

# ─────────────── CONFIG ───────────────
MODEL_PATH       = "best.pt"
DRIVE_FILE_ID    = "107Egyp0zJih7XTlNq2pJMFsb1JSiSPK2"
YOLO_CONF        = 0.25    # pre-filter detection threshold (unused for pure classification but left here)
MIN_CONF         = 0.8     # your cutoff to call it “a herb”
HERB_CLASSES     = [
    "Variegated Mexican Mint",
    "Java Pennywort",
    "Mexican Mint",
    "Green Chiretta",
    "Java Tea",
    "Chinese Gynura"
]
# ─────────────────────────────────────────

def download_model():
    """Download best.pt from Google Drive if it’s not already here."""
    if not os.path.exists(MODEL_PATH):
        print("Downloading best.pt from Google Drive…")
        url = f"https://drive.google.com/uc?id={DRIVE_FILE_ID}"
        resp = requests.get(url, stream=True)
        resp.raise_for_status()
        with open(MODEL_PATH, "wb") as f:
            for chunk in resp.iter_content(1024):
                f.write(chunk)
        print("Download complete.")

# 1) Ensure the model binary is present
download_model()

# 2) Load the YOLOv8 model (classification)
model = YOLO(MODEL_PATH)
model.conf = YOLO_CONF

@app.route("/predict", methods=["POST"])
def predict():
    payload = request.get_json(silent=True)
    if not payload or "image_url" not in payload:
        return make_response("Error: No image URL provided", 400)

    image_url = payload["image_url"]
    print("Fetching image from:", image_url)
    resp = requests.get(image_url, stream=True, timeout=10)
    if resp.status_code != 200:
        return make_response(f"Error: Failed to download image ({resp.status_code})", 400)

    # 3) Open & resize
    img = Image.open(io.BytesIO(resp.content)).convert("RGB")
    img = img.resize((640, 640))

    # 4) Run the classification model
    results = model(img)

    # 5) Extract the raw per-class probabilities tensor
    if not hasattr(results[0], "probs"):
        # if something’s gone wrong, just bail out
        decision = "Not a Herb"
        print("⚠️  No .probs found; falling back to:", decision)
        return make_response(decision, 200, {"Content-Type": "text/plain"})

    raw_probs = results[0].probs            # an Ultralytics container
    # 6) Convert to a real NumPy array so we can iterate/index it
    np_probs  = raw_probs.cpu().numpy()     # now a true np.ndarray

    # 7) Log the full vector so you see e.g. "Mexican Mint 1.00, Java Tea 0.00, …" in Render
    print("Raw confidences:",
          ", ".join(f"{HERB_CLASSES[i]} {np_probs[i]:.2f}"
                    for i in range(len(np_probs))))

    # 8) Pick the top class + confidence
    top_idx  = int(np_probs.argmax())
    top_conf = float(np_probs[top_idx])

    # 9) Apply your 0.8 cutoff
    if top_conf < MIN_CONF:
        decision = "Not a Herb"
    else:
        decision = HERB_CLASSES[top_idx]

    # 10) Print it (for Render logs) and return just the plain text
    print("Final decision:", decision)
    return make_response(decision, 200, {"Content-Type": "text/plain"})


if __name__ == "__main__":
    # In Render it will pick up $PORT; locally defaults to 5000
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
