import pickle
from src.config import TRAIN_PICKLE

with open(TRAIN_PICKLE, 'rb') as f:
    train = pickle.load(f)

print(f"Training samples: {len(train)}")

text, annotations = train[0]

print(f"\nFirst card text (first 80 chars):")
print(f"  '{text[:80]}'")

print(f"\nFirst 5 entities and the words they point to:")
for start, end, label in annotations['entities'][:5]:
    word = text[start:end]
    print(f"  [{start}:{end}] label='{label}' word='{word}'")