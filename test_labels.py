import pandas as pd
from src.label_data import validate_labels

df = pd.read_csv('data/labeled/businessCard.csv')
print(f'Total rows: {len(df)}')
print(f'Cards labeled: {df["id"].nunique()}')
print(df['tag'].value_counts())
validate_labels(df)