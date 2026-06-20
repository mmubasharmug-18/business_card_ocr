import os
import pytesseract

# ── Tesseract path ────────────────────────────────────────────────────────────
# Windows: update this if Tesseract is installed in a different location
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# ── Project root ──────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ── Data paths ────────────────────────────────────────────────────────────────
IMAGES_DIR    = os.path.join(BASE_DIR, 'data', 'images')
EXTRACTED_CSV = os.path.join(BASE_DIR, 'data', 'extracted_text.csv')
LABELED_CSV   = os.path.join(BASE_DIR, 'data', 'labeled', 'businessCard.csv')

# ── Pickle paths (generated in Phase 5) ──────────────────────────────────────
TRAIN_PICKLE  = os.path.join(BASE_DIR, 'data', 'TrainData.pickle')
TEST_PICKLE   = os.path.join(BASE_DIR, 'data', 'TestData.pickle')

# ── Model path (generated in Phase 6) ────────────────────────────────────────
OUTPUT_DIR    = os.path.join(BASE_DIR, 'output', 'model-best')

# ── OCR settings ──────────────────────────────────────────────────────────────
OCR_CONF_THRESHOLD = 30

# ── NER entity labels ─────────────────────────────────────────────────────────
ENTITY_LABELS = ['NAME', 'ORG', 'DES', 'PHONE', 'EMAIL', 'WEB']