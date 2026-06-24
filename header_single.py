from ultralytics import YOLO
import cv2
import pytesseract
import os
from pathlib import Path

# =====================================================
# CONFIGURATION
# =====================================================

MODEL_PATH = "Newend.pt"

# Put the path of the invoice image here
# Examples:
# image_path = "image.png"
# image_path = "test_invoices/image.png"
# image_path = r"C:\Users\Shreyansh\Downloads\invoice.png"

image_path = "image.png"

OCR_OUTPUT_FOLDER = "ocr_output"
HEADER_CROP_FOLDER = "header_crops"

os.makedirs(OCR_OUTPUT_FOLDER, exist_ok=True)
os.makedirs(HEADER_CROP_FOLDER, exist_ok=True)

# =====================================================
# LOAD MODEL
# =====================================================

print("Loading model...")
model = YOLO(MODEL_PATH)

# =====================================================
# LOAD IMAGE
# =====================================================

image_name = Path(image_path).stem

print("\n" + "=" * 70)
print(f"Processing: {image_path}")

img = cv2.imread(image_path)

if img is None:
    raise ValueError(
        f"Could not read image: {image_path}"
    )

height, width = img.shape[:2]

print(f"Image Size : {width} x {height}")

# =====================================================
# YOLO DETECTION
# =====================================================

results = model(
    image_path,
    verbose=False
)

# =====================================================
# FIND TOP-MOST TABLE
# =====================================================

table_y = None
table_count = 0

for r in results:

    for box in r.boxes:

        cls_id = int(box.cls[0])
        label = model.names[cls_id]

        if label == "Table":

            table_count += 1

            x1, y1, x2, y2 = map(
                int,
                box.xyxy[0]
            )

            if y1 < 0 or y1 > height:
                continue

            if table_y is None:
                table_y = y1
            else:
                table_y = min(table_y, y1)

print(f"Tables Detected : {table_count}")

# =====================================================
# FALLBACK IF NO TABLE FOUND
# =====================================================

if table_y is None:

    print(
        "No table detected -> using 50% page height"
    )

    table_y = int(height * 0.5)

table_y = max(
    50,
    min(table_y, height)
)

print(f"Header Ends At Y : {table_y}")

# =====================================================
# CROP HEADER
# =====================================================

header_crop = img[
    0:table_y,
    :
]

if header_crop.size == 0:
    raise ValueError(
        "Header crop is empty"
    )

crop_h, crop_w = header_crop.shape[:2]

if crop_h < 20 or crop_w < 20:
    raise ValueError(
        f"Crop too small ({crop_w}x{crop_h})"
    )

print(
    f"Header Crop Size : {crop_w} x {crop_h}"
)

# =====================================================
# SAVE HEADER IMAGE
# =====================================================

crop_path = os.path.join(
    HEADER_CROP_FOLDER,
    f"{image_name}_header.png"
)

cv2.imwrite(
    crop_path,
    header_crop
)

print(
    f"Header Crop Saved -> {crop_path}"
)

# =====================================================
# OCR
# =====================================================

print("Running OCR...")

raw_text = pytesseract.image_to_string(
    header_crop
)

# =====================================================
# SAVE OCR TEXT
# =====================================================

txt_path = os.path.join(
    OCR_OUTPUT_FOLDER,
    f"{image_name}.txt"
)

with open(
    txt_path,
    "w",
    encoding="utf-8"
) as f:

    f.write(raw_text)

print(
    f"OCR Saved -> {txt_path}"
)

# =====================================================
# PREVIEW OCR
# =====================================================

print("\n" + "=" * 70)
print("OCR OUTPUT PREVIEW")
print("=" * 70)

print(raw_text[:3000])

print("\nFinished.")