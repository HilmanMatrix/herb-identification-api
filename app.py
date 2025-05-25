import io, os, requests
from flask import Flask, request, make_response
from PIL       import Image
from ultralytics import YOLO

app = Flask(__name__)

MODEL_PATH   = "best.pt"
GOOGLE_ID    = "107Egyp0zJih7XTlNq2pJMFsb1JSiSPK2"
YOLO_CONF    = 0.25
CONF_CUTOFF  = 0.8
HERB_CLASSES = [
    "Chinese Gynura",
    "Green Chiretta",
    "Java Pennywort",
    "Java Tea",
    "Mexican Mint",
    "Variegated Mexican Mint"
]

def download_model():
    if not os.path.exists(MODEL_PATH):
        url = f"https://drive.google.com/uc?id={GOOGLE_ID}"
        resp = requests.get(url, stream=True); resp.raise_for_status()
        with open(MODEL_PATH, "wb") as f:
            for chunk in resp.iter_content(1024):
                f.write(chunk)
        print("Model downloaded.")

download_model()
model = YOLO(MODEL_PATH)
model.conf = YOLO_CONF

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(force=True)
    if not data or "image_url" not in data:
        return make_response("Error: No image URL provided", 400)

    # 1) fetch & open
    resp = requests.get(data["image_url"], timeout=10)
    if resp.status_code != 200:
        return make_response(f"Error: download failed ({resp.status_code})", 400)
    img = Image.open(io.BytesIO(resp.content)).convert("RGB").resize((640, 640))

    # 2) run
    results = model(img)
    raw = results[0].probs.data.cpu().numpy()  # type: np.ndarray

    # 3) log all six in the correct order
    print("Raw confs:",
          ", ".join(f"{HERB_CLASSES[i]} {raw[i]:.2f}"
                    for i in range(len(raw))))

    # 4) pick argmax
    top_idx  = int(raw.argmax())
    top_conf = float(raw[top_idx])
    decision = "Not a Herb" if top_conf < CONF_CUTOFF else HERB_CLASSES[top_idx]

    print("Final decision:", decision)
    return make_response(decision, 200, {"Content-Type":"text/plain"})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
