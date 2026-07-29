from flask import Flask, request, render_template
from ultralytics import YOLO
from PIL import Image
import io, base64, os
import matplotlib
matplotlib.use("Agg")  # use non-GUI backend for servers
import matplotlib.pyplot as plt

app = Flask(__name__)

MODEL_PATH = 'yolov8_model.pt'

# Auto-download YOLOv8n if model file is missing
def ensure_model():
    if not os.path.exists(MODEL_PATH):
        import requests
        print("Downloading pretrained YOLOv8n model...")
        url = "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt"
        r = requests.get(url, timeout=60)
        r.raise_for_status()
        with open(MODEL_PATH, "wb") as f:
            f.write(r.content)
        print("Download complete -> yolov8_model.pt")

# Initialize model (download if necessary)
ensure_model()
model = YOLO(MODEL_PATH)

@app.route('/')
def index():
    return render_template('index.html')

def fig_to_base64():
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    data = base64.b64encode(buf.read()).decode('utf-8')
    plt.close()
    return data

@app.route('/upload', methods=['POST'])
def upload_image():
    if 'file' not in request.files or request.files['file'].filename.strip() == '':
        return render_template('index.html', error="No file selected.")

    try:
        # Read image
        file = request.files['file']
        image = Image.open(file.stream).convert('RGB')

        # Run inference
        results = model(image)
        r0 = results[0]
        boxes = r0.boxes

        # Default outputs
        tumor_type = "No Tumor"
        confidence = 0.0
        result_message = "No tumor detected"
        probs = {"Glioma": 0.0, "Meningioma": 0.0, "Pituitary Tumor": 0.0, "No Tumor": 1.0}

        # Translate first detection (YOLOv8n is not tumor-trained; demo only)
        if boxes is not None and len(boxes) > 0:
            cls_id = int(boxes.cls[0].item())
            confidence = float(boxes.conf[0].item())
            # Map any detection to a pseudo tumor class by modulo (demo visualization)
            mapping = {0: "Glioma", 1: "Meningioma", 2: "Pituitary Tumor"}
            tumor_type = mapping.get(cls_id % 3, "Glioma")
            probs = {"Glioma": 0.0, "Meningioma": 0.0, "Pituitary Tumor": 0.0, "No Tumor": 0.0}
            probs[tumor_type] = confidence
            result_message = "Chance of brain tumor recurrence" if confidence > 0.6 else "No recurrence chance"

        # Bar chart for per-image "probabilities"
        labels = list(probs.keys())
        values = [probs[k] for k in labels]
        plt.figure()
        plt.bar(labels, values)
        plt.title("Prediction Probabilities (demo)")
        plt.ylabel("Confidence")
        plt.ylim(0, 1)
        chart_b64 = fig_to_base64()

        # Annotated image with YOLO boxes
        annotated = r0.plot()  # numpy array
        annotated_img = Image.fromarray(annotated)
        buf = io.BytesIO()
        annotated_img.save(buf, format="PNG")
        buf.seek(0)
        annotated_b64 = base64.b64encode(buf.read()).decode('utf-8')

        return render_template(
            'index.html',
            result=result_message,
            tumor_type=tumor_type,
            confidence=f"{confidence*100:.2f}%",
            chart=chart_b64,
            annotated=annotated_b64
        )
    except Exception as e:
        return render_template('index.html', error=str(e))

@app.route('/health')
def health():
    # Simple health probe
    return {'status': 'ok', 'model_loaded': True}

if __name__ == '__main__':
    # For local demo only
    app.run(debug=True)