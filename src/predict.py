import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import re
import string
import numpy as np
import pandas as pd
import cv2
import pytesseract
from src.config import OCR_CONF_THRESHOLD
from src.clean_text import clean_text


class GroupGenerator:
    def __init__(self):
        self.id   = 0
        self.text = ''

    def get_group(self, label):
        if self.text == label:
            return self.id
        else:
            self.id  += 1
            self.text = label
            return self.id


def parse_entity(text, label):
    if label == 'PHONE':
        text = re.sub(r'\D', '', text)

    elif label == 'EMAIL':
        text = text.lower()
        text = re.sub(r'[^A-Za-z0-9@_.\-]', '', text)

    elif label == 'WEB':
        text = text.lower()
        text = re.sub(r'[^A-Za-z0-9:/.%#\-]', '', text)

    elif label in ('NAME', 'DES'):
        text = text.lower()
        text = re.sub(r'[^a-z ]', '', text)
        text = text.title()

    elif label == 'ORG':
        text = text.lower()
        text = re.sub(r'[^a-z0-9 ]', '', text)
        text = text.title()

    return text.strip()


def get_predictions(image, model_ner):

    # ── PART A: OCR ──────────────────────────────────────────────────────────
    raw_data = pytesseract.image_to_data(image)
    rows     = list(map(lambda x: x.split('\t'), raw_data.split('\n')))
    df       = pd.DataFrame(rows[1:], columns=rows[0])
    df.dropna(inplace=True)

    df['conf'] = df['conf'].astype(float).astype(int)
    df = df[df['conf'] >= OCR_CONF_THRESHOLD]

    df['text'] = df['text'].apply(clean_text)
    df_clean   = df[df['text'] != ''].copy()
    df_clean.reset_index(drop=True, inplace=True)

    content = " ".join(df_clean['text'].tolist())
    
    # ── PART B: NER ──────────────────────────────────────────────────────────
    doc      = model_ner(content)
    doc_json = doc.to_json()
    doc_text = doc_json['text']

    
    tokens_df = pd.DataFrame(doc_json['tokens'])
    tokens_df['token'] = tokens_df.apply(
        lambda row: doc_text[row['start']:row['end']], axis=1
    )

    if len(doc_json['ents']) > 0:
        ents_df = pd.DataFrame(doc_json['ents'])[['start', 'label']]
    else:
        ents_df = pd.DataFrame(columns=['start', 'label'])

    tokens_df = pd.merge(tokens_df, ents_df, how='left', on='start')
    tokens_df['label'] = tokens_df['label'].fillna('O')

    # ── PART C: ALIGN ────────────────────────────────────────────────────────
    df_clean = df_clean.copy()
    df_clean['end']   = df_clean['text'].apply(lambda x: len(x) + 1).cumsum() - 1
    df_clean['start'] = df_clean.apply(lambda row: row['end'] - len(row['text']), axis=1)

    dataframe_info = pd.merge(
        df_clean,
        tokens_df[['start', 'token', 'label']],
        how='inner',
        on='start'
    )

    
    # ── PART D: BOUNDING BOXES ───────────────────────────────────────────────
    bb_df = dataframe_info[dataframe_info['label'] != 'O'].copy()
    bb_df.reset_index(drop=True, inplace=True)

    annotated_image = image.copy()

    if len(bb_df) > 0:
        bb_df['label'] = bb_df['label'].apply(lambda x: x[2:] if len(x) > 2 and x[1] == '-' else x)

        grp_gen = GroupGenerator()
        bb_df['group'] = bb_df['label'].apply(grp_gen.get_group)

        bb_df[['left', 'top', 'width', 'height']] = \
            bb_df[['left', 'top', 'width', 'height']].astype(int)
        bb_df['right']  = bb_df['left'] + bb_df['width']
        bb_df['bottom'] = bb_df['top']  + bb_df['height']

        img_tagging = bb_df[['left','top','right','bottom','label','token','group']] \
            .groupby('group') \
            .agg({
                'left'  : min,
                'right' : max,
                'top'   : min,
                'bottom': max,
                'label' : lambda x: list(x)[0],
                'token' : lambda x: " ".join(x)
            })

        
        for _, row in img_tagging.iterrows():
            l = int(row['left'])
            t = int(row['top'])
            r = int(row['right'])
            b = int(row['bottom'])

            cv2.rectangle(annotated_image, (l, t), (r, b), (0, 255, 0), 2)
            cv2.putText(
                annotated_image,
                str(row['label']),
                (l, max(t - 5, 10)),
                cv2.FONT_HERSHEY_PLAIN,
                1,
                (255, 0, 255),
                2
            )

        # ── PART E: ENTITY EXTRACTION ────────────────────────────────────────
        entities = {k: [] for k in ['NAME', 'ORG', 'DES', 'PHONE', 'EMAIL', 'WEB']}

        for _, row in img_tagging.iterrows():
            label = str(row['label']).strip()
            token = str(row['token']).strip()

            if label not in entities:
                continue

            parsed = parse_entity(token, label)
            if parsed:
                entities[label].append(parsed)

        return annotated_image, entities

    return annotated_image, {k: [] for k in ['NAME', 'ORG', 'DES', 'PHONE', 'EMAIL', 'WEB']}


if __name__ == '__main__':
    import spacy
    from src.config import OUTPUT_DIR, IMAGES_DIR

    print("Loading NER model...")
    model_ner = spacy.load(OUTPUT_DIR)

    test_path = os.path.join(IMAGES_DIR, '001.jpeg')
    image     = cv2.imread(test_path)

    if image is None:
        print(f"Could not load: {test_path}")
    else:
        print(f"Running prediction on: {test_path}\n")
        annotated_img, entities = get_predictions(image, model_ner)

        print("Extracted Entities:")
        for entity_type, values in entities.items():
            if values:
                print(f"  {entity_type:<8}: {values}")

        cv2.imwrite('test_prediction.jpg', annotated_img)
        print("\nAnnotated image saved as: test_prediction.jpg")