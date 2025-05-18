import io, os, requests
from flask import Flask, request, make_response
from PIL import Image
from ultralytics import YOLO

app = Flask(__name__)

# CONFIG
MODEL_PATH      = "best.pt"
GOOGLE_DRIVE_ID = "107Egyp0zJih7XTlNq2pJMFsb1JSiSPK2"
CONF_THRESHOLD  = 0.8  # minimum to call it “a herb”
YOLO_CONF       = 0.25 # pre‐filter threshold
HERB_CLASSES    = [
    "Variegated Mexican Mint",
    "Java Pennywort",
    "Mexican Mint",
    "Green Chiretta",
    "Java Tea",
    "Chinese Gynura"
]

def download_model():
    if not os.path.exists(MODEL_PATH):
        print("Downloading best.pt from Google Drive…")
        url = f"https://drive.google.com/uc?id={GOOGLE_DRIVE_ID}"
        r = requests.get(url, stream=True)
        r.raise_for_status()
        with open(MODEL_PATH, "wb") as f:
            for chunk in r.iter_content(1024):
                f.write(chunk)
        print("Download complete.")

# ensure model, then load it
download_model()
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

    # open & resize
    img = Image.open(io.BytesIO(resp.content)).convert("RGB")
    img = img.resize((640, 640))

    # run detection
    results = model(img)

    # no boxes → not a herb
    if not results[0].boxes:
        decision = "Not a Herb"
        print("Decision:", decision)
        return make_response(decision, 200, {"Content-Type": "text/plain"})

    # sort by confidence desc, pick top
    boxes = sorted(
        results[0].boxes,
        key=lambda b: b.conf[0].item(),
        reverse=True
    )
    top = boxes[0]
    cls_id = int(top.cls[0].item())
    conf   = float(top.conf[0].item())

    # apply threshold
    if conf < CONF_THRESHOLD:
        decision = "Not a Herb"
    else:
        decision = HERB_CLASSES[cls_id]

    print("Decision:", decision)
    return make_response(decision, 200, {"Content-Type": "text/plain"})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
