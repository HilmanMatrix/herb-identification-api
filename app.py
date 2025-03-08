import io
import requests
from flask import Flask, request, jsonify
from PIL import Image
from ultralytics import YOLO

app = Flask(__name__)
model = YOLO("best.pt")
HERB_CLASSES = ["Variegated Mexican Mint", "Java Pennywort", "Mexican Mint", "Green Chiretta", "Java Tea", "Chinese Gynura"]

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    image_url = data.get("image_url")
    response = requests.get(image_url)
    img = Image.open(io.BytesIO(response.content))
    results = model(img)
    
    if not results[0].boxes:
        return jsonify({"herb_name": "Not a Herb", "confidence": 0.0})
    
    class_id = int(results[0].boxes[0].cls[0].item())
    conf = float(results[0].boxes[0].conf[0].item())
    herb_name = HERB_CLASSES[class_id] if conf >= 0.5 else "Not a Herb"
    
    return jsonify({"herb_name": herb_name, "confidence": conf})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)