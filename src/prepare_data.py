import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import random
import pickle
import pandas as pd
from src.config import LABELED_CSV, TRAIN_PICKLE, TEST_PICKLE
from src.clean_text import clean_text


def load_labeled_csv(csv_path):
    """
    Load your labeled business card CSV file.
    Expected columns: id, text, tag
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"Labeled CSV not found: {csv_path}\n"
            f"Run src/label_data.py first to create your training data."
        )

    df = pd.read_csv(csv_path, encoding='utf-8', on_bad_lines='skip')
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)

    cards = df['id'].nunique()
    rows  = len(df)
    print(f"Loaded {rows:,} labeled rows from {cards} cards.")
    return df


def preprocess_dataframe(df):
    """
    Apply text cleaning to the labeled DataFrame.
    Removes rows that become empty after cleaning.
    """
    df = df.copy()
    df['text'] = df['text'].apply(clean_text)
    df = df[df['text'] != '']
    df.dropna(subset=['tag'], inplace=True)
    df.reset_index(drop=True, inplace=True)
    print(f"After cleaning: {len(df):,} rows remain.")
    return df


def convert_to_spacy_format(df):
    """
    Convert labeled DataFrame into spaCy NER training format.
    Groups by card id, builds content string, calculates character
    positions for each entity word.
    """
    all_cards  = []
    null_count = 0

    for card_id, group in df.groupby(by='id'):
        word_tag_pairs = group[['text', 'tag']].values

        content  = ''
        entities = []
        char_pos = 0

        for text, tag in word_tag_pairs:
            text     = str(text)
            word_len   = len(text)
            word_start = char_pos
            word_end   = char_pos + word_len

            if tag != 'O':
                label = tag[2:]   # strip "B-" or "I-" -> "NAME", "PHONE", etc.
                entities.append((word_start, word_end, label))

            content  += text + ' '
            char_pos  = word_end + 1

        if len(entities) > 0:
            all_cards.append((content, {'entities': entities}))
        else:
            null_count += 1

    print(f"Converted {len(all_cards)} cards to spaCy format.")
    if null_count > 0:
        print(f"Skipped {null_count} cards with no entities (all tags were O).")

    return all_cards


def split_and_save(all_data, train_path, test_path, train_ratio=0.9):
    """
    Shuffle, split into training and test sets, save as pickle files.
    """
    random.shuffle(all_data)

    split_at   = int(len(all_data) * train_ratio)
    train_data = all_data[:split_at]
    test_data  = all_data[split_at:]

    print(f"Training samples: {len(train_data)}")
    print(f"Test samples:     {len(test_data)}")

    with open(train_path, 'wb') as f:
        pickle.dump(train_data, f)

    with open(test_path, 'wb') as f:
        pickle.dump(test_data, f)

    print(f"Saved: {train_path}")
    print(f"Saved: {test_path}")


if __name__ == '__main__':
    print("=" * 50)
    print("DATA PREPARATION PIPELINE")
    print("=" * 50)

    df_raw    = load_labeled_csv(LABELED_CSV)
    df_clean  = preprocess_dataframe(df_raw)
    spacy_data = convert_to_spacy_format(df_clean)
    split_and_save(spacy_data, TRAIN_PICKLE, TEST_PICKLE)

    print("\nData preparation complete!")
    print("You can now run Phase 6 to train the NER model.")