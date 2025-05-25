import io, os, requests
from flask import Flask, request, make_response
from PIL import Image
from ultralytics import YOLO

app = Flask(__name__)

# ───── CONFIG ─────
MODEL_PATH      = "best.pt"
GOOGLE_DRIVE_ID = "107Egyp0zJih7XTlNq2pJMFsb1JSiSPK2"
YOLO_CONF       = 0.25
MIN_CONF        = 0.8

HERB_CLASSES = [
    "Variegated Mexican Mint",  
    "Mexican Mint",               
    "Java Tea",                  
    "Java Pennywort",             
    "Green Chiretta",             
    "Chinese Gynura"              
]
# ───────────────────

def download_model():
    if not os.path.exists(MODEL_PATH):
        url = f"https://drive.google.com/uc?id={GOOGLE_DRIVE_ID}"
        resp = requests.get(url, stream=True)
        resp.raise_for_status()
        with open(MODEL_PATH, "wb") as f:
            for chunk in resp.iter_content(1024):
                f.write(chunk)

download_model()
model = YOLO(MODEL_PATH)
model.conf = YOLO_CONF

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(force=True)
    if "image_url" not in data:
        return make_response("Error: No image URL provided", 400)

    url = data["image_url"]
    resp = requests.get(url, timeout=10)
    if resp.status_code != 200:
        return make_response("Error: Failed to download image", 400)

    img = Image.open(io.BytesIO(resp.content)).convert("RGB")
    img = img.resize((640, 640))

    results = model(img)
    # .probs is a Ultralytics Probs object, extract its .data tensor:
    probs = results[0].probs.data.cpu().numpy()

    print("Raw confidences:",
          ", ".join(f"{HERB_CLASSES[i]} {probs[i]:.2f}"
                    for i in range(len(probs))))

    top_idx  = int(probs.argmax())
    top_conf = float(probs[top_idx])

    decision = HERB_CLASSES[top_idx] if top_conf >= MIN_CONF else "Not a Herb"
    print("Final decision:", decision)
    return make_response(decision, 200, {"Content-Type": "text/plain"})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
