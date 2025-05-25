import io
import os
import requests
from flask import Flask, request, make_response
from PIL import Image
from ultralytics import YOLO

app = Flask(__name__)

# ───────── CONFIG ────────────
MODEL_PATH    = "best.pt"
GOOGLE_ID     = "107Egyp0zJih7XTlNq2pJMFsb1JSiSPK2"
YOLO_CONF     = 0.25    # unused in classify but harmless
CONF_CUTOFF   = 0.8

HERB_CLASSES = [
    "Variegated Mexican Mint",
    "Mexican Mint",
    "Java Tea",
    "Java Pennywort",
    "Green Chiretta",
    "Chinese Gynura"
]
# ─────────────────────────────

def download_model():
    if not os.path.exists(MODEL_PATH):
        print("Downloading best.pt from Google Drive…")
        url = f"https://drive.google.com/uc?id={GOOGLE_ID}"
        resp = requests.get(url, stream=True)
        resp.raise_for_status()
        with open(MODEL_PATH, "wb") as f:
            for chunk in resp.iter_content(1024):
                f.write(chunk)
        print("Download complete.")

download_model()
model = YOLO(MODEL_PATH)          # load as classification model
model.conf = YOLO_CONF            # no effect on .probs but OK

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True)
    if not data or "image_url" not in data:
        return make_response("Error: No image URL provided", 400)

    url = data["image_url"]
    print("Fetching image from:", url)
    resp = requests.get(url, timeout=10)
    if resp.status_code != 200:
        return make_response(f"Error: Failed to download image ({resp.status_code})", 400)

    img = Image.open(io.BytesIO(resp.content)).convert("RGB")
    img = img.resize((640, 640))

    results = model(img)

    # classification only → results[0].probs exists
    p = results[0].probs
    np_p = p.cpu().numpy()  # shape (6,)

    # Log the full vector in correct order:
    print("Raw confidences:",
          ", ".join(f"{HERB_CLASSES[i]} {np_p[i]:.2f}"
                    for i in range(len(np_p))))

    top = int(np_p.argmax())
    top_conf = float(np_p[top])
    decision = HERB_CLASSES[top] if top_conf >= CONF_CUTOFF else "Not a Herb"

    print("Final decision:", decision)
    return make_response(decision, 200, {"Content-Type": "text/plain"})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
