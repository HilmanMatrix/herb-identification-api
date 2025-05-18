import io, os, requests
import numpy as np
from flask import Flask, request, make_response
from PIL import Image
from ultralytics import YOLO

app = Flask(__name__)

# ──────────── CONFIG ────────────
MODEL_PATH    = "best.pt"
DRIVE_FILE_ID = "107Egyp0zJih7XTlNq2pJMFsb1JSiSPK2"
MIN_CONF      = 0.8
HERB_CLASSES  = [
    "Variegated Mexican Mint",
    "Java Pennywort",
    "Mexican Mint",
    "Green Chiretta",
    "Java Tea",
    "Chinese Gynura"
]
# ────────────────────────────────

def download_model():
    if not os.path.exists(MODEL_PATH):
        url = f"https://drive.google.com/uc?id={DRIVE_FILE_ID}"
        resp = requests.get(url, stream=True)
        resp.raise_for_status()
        with open(MODEL_PATH, "wb") as f:
            for chunk in resp.iter_content(1024):
                f.write(chunk)
        print("Downloaded best.pt")

download_model()
model = YOLO(MODEL_PATH)

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(force=True) or {}
    url  = data.get("image_url")
    if not url:
        return make_response("Error: No image URL provided", 400)

    # 1) fetch & preprocess
    resp = requests.get(url, timeout=10, stream=True)
    if resp.status_code != 200:
        return make_response(f"Error downloading image ({resp.status_code})", 400)
    img = Image.open(io.BytesIO(resp.content)).convert("RGB").resize((640, 640))

    # 2) run classification
    results = model(img)

    # 3) unwrap .probs to raw tensor
    raw = results[0].probs
    if hasattr(raw, "tensor"):
        tensor = raw.tensor
    else:
        tensor = raw
    np_probs = tensor.cpu().numpy()

    # 4) pick top
    top_idx  = int(np_probs.argmax())
    top_conf = float(np_probs[top_idx])

    # log all confidences for your Render logs
    print("Raw confidences:",
          ", ".join(f"{HERB_CLASSES[i]} {np_probs[i]:.2f}"
                    for i in range(len(np_probs))))

    # 5) decision
    decision = HERB_CLASSES[top_idx] if top_conf >= MIN_CONF else "Not a Herb"
    print("Final decision:", decision)

    # 6) return just the name
    return make_response(decision, 200, {"Content-Type": "text/plain"})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
