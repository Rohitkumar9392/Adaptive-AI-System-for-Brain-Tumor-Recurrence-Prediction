import base64
import io
import logging
import os
from flask import Flask, request, render_template
from ultralytics import YOLO
from PIL import Image
import matplotlib
matplotlib.use("Agg")  # use non-GUI backend for servers
import matplotlib.pyplot as plt

app = Flask(__name__)

# Maximum upload size: 50 MB
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024

MODEL_PATH = os.path.join("models", "yolov8_model.pt")
MODEL_URL = "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt"

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}
CONFIDENCE_THRESHOLD = 0.6

logging.basicConfig(level=logging.INFO)

def ensure_model():
    """
    Ensure the YOLO model exists.
    If it's missing, download it automatically.
    """
    if os.path.exists(MODEL_PATH):
        logging.info("YOLO model found.")
        return

    logging.warning("YOLO model not found. Downloading...")
    try:
        response = requests.get(MODEL_URL, timeout=60)
        response.raise_for_status()

        with open(MODEL_PATH, "wb") as model_file:
            model_file.write(response.content)

        logging.info("YOLO model downloaded successfully.")

    except requests.RequestException as e:
        raise RuntimeError(f"Failed to download YOLO model: {e}")

# Initialize model
ensure_model()
model = YOLO(MODEL_PATH)

def allowed_file(filename):
     """Return True if the uploaded file has an allowed image extension."""
    return (
        "." in filename and
        filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS
    )

@app.route('/')
def index():
    return render_template('index.html')

def fig_to_base64():
    """Convert the Matplotlib figure into a Base64-encoded PNG image."""
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    data = base64.b64encode(buf.read()).decode('utf-8')
    plt.close()
    return data

@app.route('/upload', methods=['POST'])
def upload_image():
     """Handle image upload, run prediction, and display the results."""
    if 'file' not in request.files or request.files['file'].filename.strip() == '':
        return render_template('index.html', error="No file selected.")

    try:
        # Read image
        file = request.files['file']
        if not allowed_file(file.filename):
            return render_template('index.html', error="Only PNG, JPG, and JPEG files are allowed.")
        try:
            image = Image.open(file.stream)
            image.verify()
            file.stream.seek(0)
            image = Image.open(file.stream).convert("RGB")
        except Exception:
            return render_template('index.html', error="Invalid or corrupted image.")

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
            result_message = "Chance of brain tumor recurrence" if confidence > CONFIDENCE_THRESHOLD else "No recurrence chance"

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
    app.run(debug=False)
