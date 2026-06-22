import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from src.config import EXTRACTED_CSV, LABELED_CSV
from src.clean_text import clean_text


VALID_TAGS = {
    'B-NAME', 'I-NAME',
    'B-ORG',  'I-ORG',
    'B-DES',  'I-DES',
    'B-PHONE','I-PHONE',
    'B-EMAIL','I-EMAIL',
    'B-WEB',  'I-WEB',
    'O'
}

TAG_SHORTCUTS = {
    'bn': 'B-NAME',  'in': 'I-NAME',
    'bo': 'B-ORG',   'io': 'I-ORG',
    'bd': 'B-DES',   'id': 'I-DES',
    'bp': 'B-PHONE', 'ip': 'I-PHONE',
    'be': 'B-EMAIL', 'ie': 'I-EMAIL',
    'bw': 'B-WEB',   'iw': 'I-WEB',
    'o' : 'O',
}


def load_extracted_data(csv_path):
    """
    Load the extracted text CSV produced by extract_text.py.
    """
    df = pd.read_csv(csv_path, encoding='utf-8')
    df.dropna(inplace=True)
    df['text'] = df['text'].apply(clean_text)
    df = df[df['text'] != '']
    df.reset_index(drop=True, inplace=True)
    print(f"Loaded {len(df):,} words from {len(df['id'].unique())} cards.")
    return df


def load_existing_labels(labeled_csv_path):
    """
    Load already-labeled rows so you can continue from where you stopped.
    Returns empty DataFrame if no labels file exists yet.
    """
    if os.path.exists(labeled_csv_path):
        df = pd.read_csv(labeled_csv_path, encoding='utf-8')
        already_labeled = df['id'].unique().tolist()
        print(f"Found existing labels for {len(already_labeled)} cards. Resuming...")
        return df
    else:
        print("No existing labels found. Starting fresh.")
        return pd.DataFrame(columns=['id', 'text', 'tag'])


def label_card_interactive(card_id, words):
    """
    Show words from one business card one at a time and collect labels.
    Returns None if user types 'quit', empty list if skipped.
    """
    labeled_rows = []

    print("\n" + "=" * 55)
    print(f"CARD: {card_id}  ({len(words)} words)")
    print("=" * 55)
    print("Shortcuts: bn=B-NAME in=I-NAME bo=B-ORG io=I-ORG")
    print("           bd=B-DES  id=I-DES  bp=B-PHONE ip=I-PHONE")
    print("           be=B-EMAIL ie=I-EMAIL bw=B-WEB iw=I-WEB  o=O")
    print("Type 'skip' to skip this card, 'quit' to stop and save.")
    print("-" * 55)

    for i, word in enumerate(words):
        while True:
            user_input = input(f"  [{i+1:3d}/{len(words)}] '{word}'  ->  ").strip().lower()

            if user_input == 'quit':
                return None

            if user_input == 'skip':
                print(f"  Skipping card {card_id}.")
                return []

            # Resolve shortcut to full tag name
            tag = TAG_SHORTCUTS.get(user_input, user_input.upper())

            if tag in VALID_TAGS:
                labeled_rows.append((card_id, word, tag))
                break
            else:
                print(f"  Invalid tag '{user_input}'. Valid shortcuts: bn, in, bo, io, bd, id, bp, ip, be, ie, bw, iw, o")

    return labeled_rows


def save_labels(labeled_df, output_path):
    """
    Save labeled DataFrame to CSV.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    labeled_df.to_csv(output_path, index=False)
    cards_count = len(labeled_df['id'].unique())
    print(f"\nSaved {len(labeled_df):,} labeled rows ({cards_count} cards) to:")
    print(f"  {output_path}")


def validate_labels(labeled_df):
    """
    Check labeled data for common mistakes (invalid tags, I- without B-).
    """
    errors = []

    for card_id, group in labeled_df.groupby('id'):
        tags  = group['tag'].tolist()
        words = group['text'].tolist()

        prev_tag = 'O'
        for i, (word, tag) in enumerate(zip(words, tags)):

            if tag not in VALID_TAGS:
                errors.append(f"Card {card_id}, word '{word}': invalid tag '{tag}'")

            elif tag.startswith('I-'):
                expected_same = tag[2:]
                prev_type     = prev_tag[2:] if '-' in prev_tag else ''
                if prev_type != expected_same:
                    errors.append(
                        f"Card {card_id}, word '{word}': '{tag}' follows '{prev_tag}' "
                        f"(I- tag without matching B- or I- before it)"
                    )

            prev_tag = tag

    if errors:
        print(f"\nFound {len(errors)} labeling error(s):")
        for err in errors[:10]:
            print(f"  - {err}")
        if len(errors) > 10:
            print(f"  ... and {len(errors) - 10} more.")
        return False
    else:
        print("Validation passed! No labeling errors found.")
        return True


if __name__ == '__main__':
    import sys

    print("=" * 55)
    print("BUSINESS CARD LABELING TOOL")
    print("=" * 55)

    extracted_df = load_extracted_data(EXTRACTED_CSV)
    labeled_df   = load_existing_labels(LABELED_CSV)
    already_done = set(labeled_df['id'].unique()) if len(labeled_df) > 0 else set()

    all_card_ids = extracted_df['id'].unique().tolist()
    remaining    = [cid for cid in all_card_ids if cid not in already_done]

    print(f"\nTotal cards: {len(all_card_ids)}")
    print(f"Already labeled: {len(already_done)}")
    print(f"Remaining: {len(remaining)}")
    print("\nPress Enter to start labeling. Type 'quit' at any word to stop and save.")
    input()

    all_rows = labeled_df.values.tolist() if len(labeled_df) > 0 else []

    for card_id in remaining:
        card_words = extracted_df[extracted_df['id'] == card_id]['text'].tolist()
        result     = label_card_interactive(card_id, card_words)

        if result is None:
            print("\nStopping. Saving progress...")
            break

        if result:
            all_rows.extend(result)

        # Save after every card
        current_df = pd.DataFrame(all_rows, columns=['id', 'text', 'tag'])
        save_labels(current_df, LABELED_CSV)

    final_df      = pd.DataFrame(all_rows, columns=['id', 'text', 'tag'])
    labeled_cards = len(final_df['id'].unique()) if len(final_df) > 0 else 0
    print(f"\nDone for now. Labeled {labeled_cards} cards total.")
    print("Run this script again to continue labeling more cards.")

    if len(final_df) > 0:
        print("\nRunning validation...")
        validate_labels(final_df)