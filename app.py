import io
import os
import requests
from flask import Flask, request, make_response
from PIL import Image
from ultralytics import YOLO

app = Flask(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# 1) Download & cache your model from Drive if needed, exactly as before
# ──────────────────────────────────────────────────────────────────────────────
def download_model():
    model_path = "best.pt"
    if not os.path.exists(model_path):
        print("Downloading best.pt from Google Drive…")
        file_id = "107Egyp0zJih7XTlNq2pJMFsb1JSiSPK2"
        url = f"https://drive.google.com/uc?id={file_id}"
        resp = requests.get(url, stream=True)
        resp.raise_for_status()
        with open(model_path, "wb") as f:
            for chunk in resp.iter_content(1024):
                f.write(chunk)
        print("Download complete.")

download_model()

# ──────────────────────────────────────────────────────────────────────────────
# 2) Load your YOLO model and class list
# ──────────────────────────────────────────────────────────────────────────────
model = YOLO("best.pt")
model.conf = 0.25  # pre-filter threshold

HERB_CLASSES = [
    "Variegated Mexican Mint",
    "Java Pennywort",
    "Mexican Mint",
    "Green Chiretta",
    "Java Tea",
    "Chinese Gynura"
]

# ──────────────────────────────────────────────────────────────────────────────
# 3) New /predict route: prints & returns plain-text
# ──────────────────────────────────────────────────────────────────────────────
@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True)
    if not data or "image_url" not in data:
        msg = "Error: No image URL provided"
        print(msg)
        return make_response(msg, 400)

    # 3a) download & preprocess
    image_url = data["image_url"]
    print("Fetching image from:", image_url)
    resp = requests.get(image_url, stream=True)
    if resp.status_code != 200:
        msg = "Error: Failed to download image"
        print(msg)
        return make_response(msg, 400)

    img = Image.open(io.BytesIO(resp.content)).convert("RGB")
    img = img.resize((640, 640))

    # 3b) run detection
    results = model(img)

    # 3c) no boxes → immediate “Not a Herb”
    if not results[0].boxes:
        decision = "Not a Herb"
        print(decision)
        return make_response(decision, 200, {"Content-Type": "text/plain"})

    # 3d) sort by confidence desc, pick top
    boxes = sorted(results[0].boxes, key=lambda b: b.conf[0].item(), reverse=True)
    top = boxes[0]
    cls_id = int(top.cls[0].item())
    conf  = float(top.conf[0].item())

    # 3e) enforce 0.8 minimum
    if conf < 0.8:
        decision = "Not a Herb"
    else:
        decision = HERB_CLASSES[cls_id]

    # 3f) print to stdout *and* return plain-text
    print(f"Decision: {decision}  (confidence {conf:.2f})")
    return make_response(decision, 200, {"Content-Type": "text/plain"})

if __name__ == "__main__":
    # bind to 0.0.0.0 and port from env or 5000
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
