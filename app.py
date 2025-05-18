import io
import os
import requests
from flask import Flask, request, Response
from PIL import Image
from ultralytics import YOLO

app = Flask(__name__)

# ───────────────────────── CONFIG ─────────────────────────
MODEL_PATH     = "best.pt"
GOOGLE_FILE_ID = "107Egyp0zJih7XTlNq2pJMFsb1JSiSPK2"
YOLO_CONF      = 0.25   # internal filter for YOLOv8 detections
CONF_THRESHOLD = 0.8    # cutoff for “is it a herb?”
HERB_CLASSES   = [
    "Variegated Mexican Mint",
    "Java Pennywort",
    "Mexican Mint",
    "Green Chiretta",
    "Java Tea",
    "Chinese Gynura"
]

def download_model():
    """Download best.pt from GDrive if missing."""
    if not os.path.exists(MODEL_PATH):
        print("Downloading best.pt from Google Drive…")
        url = f"https://drive.google.com/uc?id={GOOGLE_FILE_ID}"
        r = requests.get(url, stream=True)
        r.raise_for_status()
        with open(MODEL_PATH, "wb") as f:
            for chunk in r.iter_content(1024):
                f.write(chunk)
        print("Download complete.")

# Ensure model file is present, then load it
download_model()
model = YOLO(MODEL_PATH)
model.conf = YOLO_CONF

@app.route("/predict", methods=["POST"])
def predict():
    # 1) parse JSON body
    data = request.get_json(force=True)
    if not data or "image_url" not in data:
        return Response("Error: No image_url provided", status=400)
    image_url = data["image_url"]
    print("Fetching image from:", image_url)

    # 2) download & open image
    try:
        r = requests.get(image_url, stream=True, timeout=10)
        r.raise_for_status()
        img = Image.open(io.BytesIO(r.content)).convert("RGB").resize((640,640))
    except Exception as e:
        print("❌ Failed to fetch/process image:", e)
        return Response("Error: Could not download or process image", status=400)

    # 3) run YOLO detection
    results = model(img)

    # 4) extract all confidences for Render logs
    confs = [b.conf[0].item() for b in results[0].boxes]
    names = [HERB_CLASSES[int(b.cls[0].item())] for b in results[0].boxes]
    if confs:
        # log “Mexican Mint 1.00, Java Tea 0.00, …”
        log_line = ", ".join(f"{n} {c:.2f}" for n,c in zip(names, confs))
        print("Raw detections:", log_line)
    else:
        print("Raw detections: (none)")

    # 5) decide top prediction
    if not confs:
        decision = "Not a Herb"
    else:
        # find highest-confidence box
        top_idx = max(range(len(confs)), key=lambda i: confs[i])
        top_conf = confs[top_idx]
        if top_conf < CONF_THRESHOLD:
            decision = "Not a Herb"
        else:
            decision = names[top_idx]

    # 6) log & return _only_ that decision
    print("Final decision:", decision)
    return Response(decision, mimetype="text/plain")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
