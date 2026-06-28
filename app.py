import os
import sys
import uuid
import cv2
import spacy
import numpy as np
from flask import Flask, render_template, request, redirect, url_for

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.config import OUTPUT_DIR
from src.predict import get_predictions

app = Flask(__name__)

UPLOAD_FOLDER = os.path.join('static', 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

print("Loading NER model...")
model_ner = spacy.load(OUTPUT_DIR)
print("Model loaded!")


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if 'card_image' not in request.files:
        return redirect(url_for('index'))

    file = request.files['card_image']
    if file.filename == '':
        return redirect(url_for('index'))

    # Read image
    file_bytes = np.frombuffer(file.read(), np.uint8)
    image      = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if image is None:
        return render_template('index.html', error="Could not read image. Please upload a valid JPG or PNG.")

    # Run prediction
    annotated_image, entities = get_predictions(image, model_ner)

    # DEBUG
    print("DEBUG entities:", entities)
    print("DEBUG results:", {k: v for k, v in entities.items() if v})

    # Save annotated image
    filename    = f"{uuid.uuid4().hex}.jpg"
    output_path = os.path.join(UPLOAD_FOLDER, filename)
    cv2.imwrite(output_path, annotated_image)

    results = {k: v for k, v in entities.items() if v}
    return render_template('result.html', image_file=filename, entities=results)


if __name__ == '__main__':
    app.run(debug=True)