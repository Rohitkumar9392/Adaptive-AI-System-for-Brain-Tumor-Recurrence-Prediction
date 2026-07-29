# Brain Tumor Detection (Flask + YOLOv8) — Visual Demo

> ⚠️ This is a **demo**: the auto-downloaded YOLOv8n model is **not trained on MRI**. 
> It is used only to demonstrate the end-to-end web app, annotated image, and per-image probability chart.

## Features
- Auto-downloads YOLOv8n if `yolov8_model.pt` is missing
- Upload MRI → shows:
  - Detection result + confidence
  - **Probability bar chart** (per image)
  - **Annotated MRI image** with bounding boxes

## Quick Start
```bash
pip install -r requirements.txt
python app.py
# then open http://127.0.0.1:5000
```

## Notes for Presentation
- Mention that to get **accurate medical predictions**, you would train YOLOv8 on a brain MRI dataset (e.g., Kaggle Brain MRI).
- Replace `yolov8_model.pt` with your trained weights for real results.