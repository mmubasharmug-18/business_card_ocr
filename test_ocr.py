from src.extract_text import load_image, extract_ocr_data
from src.config import IMAGES_DIR
import os

img_path = os.path.join(IMAGES_DIR, '000.jpeg')
image    = load_image(img_path)
df       = extract_ocr_data(image)

print("Columns:", df.columns.tolist())
print("Shape:", df.shape)
print("\nDetected words:")
print(df[['text', 'conf', 'left', 'top', 'width', 'height']].head(15).to_string())