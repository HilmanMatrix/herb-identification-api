import io
import os
import requests
from flask import Flask, request, make_response
from PIL import Image
from ultralytics import YOLO

app = Flask(__name__)

# ──────── CONFIGURATION ────────
MODEL_PATH       = "best.pt"
GOOGLE_DRIVE_ID  = "107Egyp0zJih7XTlNq2pJMFsb1JSiSPK2"
YOLO_CONF        = 0.25     # internal YOLO pre-filter (unused for pure classification)
MIN_CONF         = 0.8      # minimum “herb” confidence
HERB_CLASSES     = [
    "Variegated Mexican Mint",
    "Java Pennywort",
    "Mexican Mint",
    "Green Chiretta",
    "Java Tea",
    "Chinese Gynura"
]
# ───────────────────────────────

def download_model():
    """Grab best.pt from Google Drive if missing."""
    if not os.path.exists(MODEL_PATH):
        print("Downloading best.pt from Google Drive…")
        url = f"https://drive.google.com/uc?id={GOOGLE_DRIVE_ID}"
        resp = requests.get(url, stream=True)
        resp.raise_for_status()
        with open(MODEL_PATH, "wb") as f:
            for chunk in resp.iter_content(1024):
                f.write(chunk)
        print("Download complete.")

# Ensure we have the weights, then load the YOLO classification model
download_model()
model = YOLO(MODEL_PATH)
model.conf = YOLO_CONF

@app.route("/predict", methods=["POST"])
def predict():
    # 1) Validate JSON + URL
    payload = request.get_json(silent=True)
    if not payload or "image_url" not in payload:
        return make_response("Error: No image URL provided", 400)

    image_url = payload["image_url"]
    print("Fetching image from:", image_url)

    # 2) Download + open
    try:
        resp = requests.get(image_url, stream=True, timeout=10)
        resp.raise_for_status()
    except Exception as e:
        print("❌ Failed to download image:", e)
        return make_response("Error: Failed to download image", 400)

    img = Image.open(io.BytesIO(resp.content)).convert("RGB")
    img = img.resize((640, 640))

    # 3) Run classification
    results = model(img)

    # 4) Make sure we got a classifier result
    if not hasattr(results[0], "probs"):
        print("⚠️  No .probs on results; falling back to Not a Herb")
        return make_response("Not a Herb", 200, {"Content-Type": "text/plain"})

    # 5) Extract the raw tensor, move to CPU, convert to numpy
    #    .data is the actual torch.Tensor inside the Ultralytics Probs container
    probs_tensor = results[0].probs.data
    np_probs     = probs_tensor.cpu().numpy()  # now a true numpy array

    # 6) Log full vector in your Render logs
    print("Raw confidences:",
          ", ".join(f"{HERB_CLASSES[i]} {np_probs[i]:.2f}"
                    for i in range(len(np_probs))))

    # 7) Pick top index + its confidence
    top_idx  = int(np_probs.argmax())
    top_conf = float(np_probs[top_idx])

    # 8) Apply your cutoff
    decision = HERB_CLASSES[top_idx] if top_conf >= MIN_CONF else "Not a Herb"

    # 9) Print & return just the plain‐text name (no quotes, no confidence)
    print("Final decision:", decision)
    return make_response(decision, 200, {"Content-Type": "text/plain"})


if __name__ == "__main__":
    # Pick up PORT from Render or default to 5000 locally
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
