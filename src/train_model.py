import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pickle
import random
import spacy
from spacy.training import Example
from spacy.util import minibatch, compounding
from src.config import TRAIN_PICKLE, TEST_PICKLE, OUTPUT_DIR, ENTITY_LABELS


def load_data(train_path, test_path):
    """
    Load training and test data from pickle files.
    """
    with open(train_path, 'rb') as f:
        train_data = pickle.load(f)

    with open(test_path, 'rb') as f:
        test_data = pickle.load(f)

    print(f"Training samples: {len(train_data)}")
    print(f"Test samples:     {len(test_data)}")
    return train_data, test_data


def create_blank_model(entity_labels):
    """
    Create a blank spaCy English model and configure NER with our labels.
    """
    nlp = spacy.blank('en')
    ner = nlp.add_pipe('ner')

    for label in entity_labels:
        ner.add_label(label)

    print(f"Created blank model with {len(entity_labels)} entity labels: {entity_labels}")
    return nlp


def train_ner_model(nlp, train_data, output_dir, n_iter=30):
    """
    Train the NER model and save the best version to disk.
    """
    os.makedirs(output_dir, exist_ok=True)

    optimizer = nlp.begin_training()

    print(f"\nTraining for {n_iter} iterations on {len(train_data)} samples...")
    print("-" * 45)

    best_loss = float('inf')

    for iteration in range(1, n_iter + 1):
        random.shuffle(train_data)
        losses  = {}
        batches = minibatch(train_data, size=compounding(4.0, 32.0, 1.001))

        for batch in batches:
            examples = []
            for text, annotations in batch:
                doc     = nlp.make_doc(text)
                example = Example.from_dict(doc, annotations)
                examples.append(example)

            nlp.update(
                examples,
                sgd=optimizer,
                drop=0.3,
                losses=losses
            )

        current_loss = losses.get('ner', 0)
        status = " <- best" if current_loss < best_loss else ""
        print(f"  Iteration {iteration:2d}/{n_iter}  |  Loss: {current_loss:8.4f}{status}")

        if current_loss < best_loss:
            best_loss = current_loss
            nlp.to_disk(output_dir)

    print("-" * 45)
    print(f"Training complete. Best loss: {best_loss:.4f}")
    print(f"Model saved to: {output_dir}")


def evaluate_model(model_dir, test_data):
    """
    Load saved model and compare predictions against test labels.
    """
    print("\nLoading model for evaluation...")
    nlp = spacy.load(model_dir)

    print(f"Evaluating on {len(test_data)} test samples:\n")

    for i, (text, annotations) in enumerate(test_data[:3]):
        doc = nlp(text)

        print(f"--- Sample {i + 1} ---")
        print(f"Text: {text[:90]}...")

        print("  Predicted:")
        if doc.ents:
            for ent in doc.ents:
                print(f"    {ent.label_:<8} -> '{ent.text}'")
        else:
            print("    (no entities predicted)")

        print("  True labels (first 6):")
        for start, end, label in annotations['entities'][:6]:
            print(f"    {label:<8} -> '{text[start:end]}'")
        print()


if __name__ == '__main__':
    print("=" * 50)
    print("NER MODEL TRAINING")
    print("=" * 50)

    train_data, test_data = load_data(TRAIN_PICKLE, TEST_PICKLE)
    nlp = create_blank_model(ENTITY_LABELS)
    train_ner_model(nlp, train_data, OUTPUT_DIR, n_iter=30)
    evaluate_model(OUTPUT_DIR, test_data)

    print("Model is ready. Proceed to Phase 7.")