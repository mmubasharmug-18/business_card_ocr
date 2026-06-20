import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import pytesseract
import pandas as pd
from glob import glob
from tqdm import tqdm
from src.config import IMAGES_DIR, EXTRACTED_CSV, OCR_CONF_THRESHOLD


def load_image(image_path):
    """
    Load an image from disk using OpenCV.
    """
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")
    return image


def extract_ocr_data(image):
    """
    Run Tesseract OCR on an image and return a DataFrame of words
    with their pixel coordinates and confidence scores.
    """
    raw_data = pytesseract.image_to_data(image)
    rows = list(map(lambda x: x.split('\t'), raw_data.split('\n')))
    df = pd.DataFrame(rows[1:], columns=rows[0])
    df.dropna(inplace=True)
    df['conf'] = df['conf'].astype(float).astype(int)
    df = df[df['conf'] >= OCR_CONF_THRESHOLD]
    df.reset_index(drop=True, inplace=True)
    return df


def extract_all_images(images_dir):
    """
    Extract OCR data from all business card images in a directory.
    """
    image_paths = glob(os.path.join(images_dir, '*.jpeg'))

    if len(image_paths) == 0:
        raise FileNotFoundError(f"No .jpeg files found in: {images_dir}")

    print(f"Found {len(image_paths)} images. Starting extraction...")

    all_data = []

    for img_path in tqdm(image_paths, desc='Extracting OCR'):
        _, filename = os.path.split(img_path)

        try:
            image  = load_image(img_path)
            df     = extract_ocr_data(image)

            card_df         = pd.DataFrame()
            card_df['id']   = [filename] * len(df)
            card_df['text'] = df['text'].values

            all_data.append(card_df)

        except Exception as e:
            print(f"\nSkipping {filename}: {e}")
            continue

    combined = pd.concat(all_data, ignore_index=True)
    return combined


def save_to_csv(dataframe, output_path):
    """
    Save DataFrame to a CSV file.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    dataframe.to_csv(output_path, index=False)
    print(f"Saved {len(dataframe):,} rows to: {output_path}")


if __name__ == '__main__':
    print("=" * 50)
    print("OCR EXTRACTION PIPELINE")
    print("=" * 50)

    all_text = extract_all_images(IMAGES_DIR)

    print(f"\nTotal words extracted: {len(all_text):,}")
    print("\nSample output (first 10 rows):")
    print(all_text.head(10).to_string())

    save_to_csv(all_text, EXTRACTED_CSV)
    print("\nDone! Open data/extracted_text.csv to see the result.")