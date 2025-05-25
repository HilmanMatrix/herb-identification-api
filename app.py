import io, os, requests
from flask import Flask, request, make_response
from PIL import Image
from ultralytics import YOLO
from ultralytics.engine.results import Probs

app = Flask(__name__)

# ───── CONFIG ─────
MODEL_PATH    = "best.pt"
GOOGLE_ID     = "107Egyp0zJih7XTlNq2pJMFsb1JSiSPK2"
YOLO_CONF     = 0.25
CONF_CUTOFF   = 0.8
HERB_CLASSES = [
    "Variegated Mexican Mint",
    "Mexican Mint",
    "Java Tea",
    "Java Pennywort",
    "Green Chiretta",
    "Chinese Gynura"
]
# ──────────────────

def download_model():
    if not os.path.exists(MODEL_PATH):
        url = f"https://drive.google.com/uc?id={GOOGLE_ID}"
        resp = requests.get(url, stream=True)
        resp.raise_for_status()
        with open(MODEL_PATH, "wb") as f:
            for chunk in resp.iter_content(1024):
                f.write(chunk)
        print("Downloaded model.")

download_model()
model = YOLO(MODEL_PATH)
model.conf = YOLO_CONF  # no-op for classify but OK

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(force=True)
    if "image_url" not in data:
        return make_response("Error: No image URL provided", 400)

    url = data["image_url"]
    print("Fetching:", url)
    resp = requests.get(url, timeout=10)
    if resp.status_code != 200:
        return make_response(f"Error: download failed ({resp.status_code})", 400)

    img = Image.open(io.BytesIO(resp.content)).convert("RGB")
    img = img.resize((640, 640))

    results = model(img)
    p: Probs = results[0].probs

    # Log entire vector (optional):
    print("Raw confs:",
          ", ".join(f"{HERB_CLASSES[i]} {p[i]:.2f}"
                    for i in range(len(HERB_CLASSES))))

    top_idx  = int(p.top1)
    top_conf = float(p.top1conf)
    if top_conf < CONF_CUTOFF:
        decision = "Not a Herb"
    else:
        decision = HERB_CLASSES[top_idx]

    print("Final decision:", decision)
    return make_response(decision, 200, {"Content-Type":"text/plain"})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
