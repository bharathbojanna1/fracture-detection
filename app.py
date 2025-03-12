from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse
from transformers import pipeline
from PIL import Image, ImageDraw
import numpy as np
import io
import uvicorn
import base64

app = FastAPI()

# Load AI models for bone fracture detection
def load_models():
    return {
        "BoneEye": pipeline("object-detection", model="D3STRON/bone-fracture-detr"),
        "BoneGuardian": pipeline("image-classification", model="Heem2/bone-fracture-detection-using-xray"),
        "XRayMaster": pipeline("image-classification", 
            model="nandodeomkar/autotrain-fracture-detection-using-google-vit-base-patch-16-54382127388")
    }

models = load_models()

# Translate labels to English
def translate_label(label):
    translations = {
        "fracture": "Bone Fracture",
        "no fracture": "No Bone Fracture",
        "normal": "Normal",
        "abnormal": "Abnormal",
        "F1": "Bone Fracture",
        "NF": "No Bone Fracture"
    }
    return translations.get(label.lower(), label)

# Create a heatmap overlay for fracture detection
def create_heatmap_overlay(image, box, score):
    overlay = Image.new('RGBA', image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    x1, y1 = box['xmin'], box['ymin']
    x2, y2 = box['xmax'], box['ymax']

    if score > 0.8:
        fill_color = (255, 0, 0, 100)  # Red for high confidence
        border_color = (255, 0, 0, 255)
    elif score > 0.6:
        fill_color = (255, 165, 0, 100)  # Orange for medium confidence
        border_color = (255, 165, 0, 255)
    else:
        fill_color = (255, 255, 0, 100)  # Yellow for low confidence
        border_color = (255, 255, 0, 255)

    draw.rectangle([x1, y1, x2, y2], fill=fill_color)
    draw.rectangle([x1, y1, x2, y2], outline=border_color, width=2)

    return overlay

# Draw bounding boxes around detected fractures
def draw_boxes(image, predictions):
    result_image = image.copy().convert('RGBA')

    for pred in predictions:
        box = pred['box']
        score = pred['score']

        overlay = create_heatmap_overlay(image, box, score)
        result_image = Image.alpha_composite(result_image, overlay)

        draw = ImageDraw.Draw(result_image)
        temperature = 36.5 + (score * 2.5)
        label = f"{translate_label(pred['label'])} ({score:.1%} • {temperature:.1f}°C)"

        text_bbox = draw.textbbox((box['xmin'], box['ymin'] - 20), label)
        draw.rectangle(text_bbox, fill=(0, 0, 0, 180))

        draw.text(
            (box['xmin'], box['ymin'] - 20),
            label,
            fill=(255, 255, 255, 255)
        )

    return result_image

# Convert image to base64 for display in HTML
def image_to_base64(image):
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return f"data:image/png;base64,{img_str}"

# Home Page with File Upload
@app.get("/", response_class=HTMLResponse)
async def main():
    content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Bone Fracture Detection</title>
    </head>
    <body>
        <h2>Upload an X-ray Image for Fracture Detection</h2>
        <form action="/analyze" method="post" enctype="multipart/form-data">
            <input type="file" name="file" accept="image/*" required>
            <button type="submit">Analyze</button>
        </form>
    </body>
    </html>
    """
    return content

# Analyze Uploaded Image
@app.post("/analyze", response_class=HTMLResponse)
async def analyze_file(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))

        # Run AI models
        predictions_watcher = models["BoneGuardian"](image)
        predictions_master = models["XRayMaster"](image)
        predictions_locator = models["BoneEye"](image)

        # Filter predictions with confidence above 0.6
        filtered_preds = [p for p in predictions_locator if p['score'] >= 0.6]
        if filtered_preds:
            result_image = draw_boxes(image, filtered_preds)
        else:
            result_image = image

        result_image_b64 = image_to_base64(result_image)

        # Display Results
        results_html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Fracture Detection Results</title>
        </head>
        <body>
            <h2>Analysis Results</h2>
            <img src="{result_image_b64}" alt="Detection Results">
            <br><br>
            <a href="/">Upload Another Image</a>
        </body>
        </html>
        """
        return results_html

    except Exception as e:
        return f"<h2>Error: {str(e)}</h2>"

# Run API Server (for manual execution)
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
