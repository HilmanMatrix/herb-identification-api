import io, os, requests
from flask     import Flask, request, make_response
from PIL       import Image
from ultralytics import YOLO

app = Flask(__name__)

# ── CONFIG ─────────────────────────────────────────────────────────────────────
MODEL_PATH    = "best.pt"         # your classification weights
DOWNLOAD_ID   = "107Egyp0zJih7XTlNq2pJMFsb1JSiSPK2"
MIN_CONF      = 0.8               # require ≥80% to call it a herb
HERB_CLASSES  = [
    "Variegated Mexican Mint",
    "Java Pennywort",
    "Mexican Mint",
    "Green Chiretta",
    "Java Tea",
    "Chinese Gynura"
]
# ────────────────────────────────────────────────────────────────────────────────

def download_model():
    if not os.path.exists(MODEL_PATH):
        print("Downloading best.pt from Google Drive…")
        url = f"https://drive.google.com/uc?id={DOWNLOAD_ID}"
        r = requests.get(url, stream=True); r.raise_for_status()
        with open(MODEL_PATH, "wb") as f:
            for chunk in r.iter_content(1024):
                f.write(chunk)
        print("Download complete.")

download_model()
model = YOLO(MODEL_PATH)

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True)
    if not data or "image_url" not in data:
        return make_response("Error: No image URL provided", 400)

    url = data["image_url"]
    print("Fetching image from:", url)
    resp = requests.get(url, timeout=10)
    if resp.status_code != 200:
        return make_response("Error: Failed to download image", 400)

    img = Image.open(io.BytesIO(resp.content)).convert("RGB")
    img = img.resize((640, 640))

    results = model(img)
    # results[0].probs is a 6‐element tensor of softmax scores
    if not hasattr(results[0], "probs"):
        print("⚠️  No .probs; falling back to Not a Herb")
        return make_response("Not a Herb", 200, {"Content-Type":"text/plain"})

    # pull out the numpy array [p0, p1, … p5]
    probs = results[0].probs.cpu().numpy()
    # log entire vector in Render
    print("Raw confidences:", 
          ", ".join(f"{HERB_CLASSES[i]} {probs[i]:.2f}" 
                    for i in range(len(probs))))

    top_idx  = int(probs.argmax())
    top_conf = float(probs[top_idx])
    if top_conf < MIN_CONF:
        decision = "Not a Herb"
    else:
        decision = HERB_CLASSES[top_idx]

    print("Final decision:", decision)
    return make_response(decision, 200, {"Content-Type":"text/plain"})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    # bind 0.0.0.0 so Render can route traffic
    app.run(host="0.0.0.0", port=port)
