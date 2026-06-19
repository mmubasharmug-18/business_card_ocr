import numpy as np
import pandas as pd
import cv2
import pytesseract
import spacy
from PIL import Image
from tqdm import tqdm

print("All imports successful!")
print(f"OpenCV version:   {cv2.__version__}")
print(f"spaCy version:    {spacy.__version__}")
print(f"pandas version:   {pd.__version__}")