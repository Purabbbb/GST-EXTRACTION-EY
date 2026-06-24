from ultralytics import YOLO
import cv2
import pytesseract
import os
from pathlib import Path

# =====================================================
# CONFIGURATION
# =====================================================

MODEL_PATH = "Newend.pt"
INPUT_FOLDER = "dataset_new"

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
# GET ALL PNG FILES
# =====================================================

image_files = sorted(Path(INPUT_FOLDER).glob("*.png"))  

print(f"Found {len(image_files)} PNG files")

# =====================================================
# PROCESS EACH IMAGE
# =====================================================

for image_path in image_files:

    try:

        print("\n" + "=" * 70)
        print(f"Processing: {image_path.name}")

        # ---------------------------------------------
        # Read image
        # ---------------------------------------------

        img = cv2.imread(str(image_path))

        if img is None:
            print("Could not read image")
            continue

        height, width = img.shape[:2]

        # ---------------------------------------------
        # YOLO Detection
        # ---------------------------------------------

        results = model(str(image_path), verbose=False)

        # ---------------------------------------------
        # Find highest table
        # ---------------------------------------------

        table_y = None

        for r in results:

            for box in r.boxes:

                cls_id = int(box.cls[0])
                label = model.names[cls_id]

                if label == "Table":

                    x1, y1, x2, y2 = map(int, box.xyxy[0])

                    # Ignore invalid detections
                    if y1 < 0 or y1 > height:
                        continue

                    if table_y is None:
                        table_y = y1
                    else:
                        table_y = min(table_y, y1)

        # ---------------------------------------------
        # Fallback if no table found
        # ---------------------------------------------

        if table_y is None:

            print("No table detected -> using 50% page height")

            table_y = int(height * 0.5)

        # Clamp to image boundaries
        table_y = max(50, min(table_y, height))

        print(f"Image Height : {height}")
        print(f"Table Y      : {table_y}")

        # ---------------------------------------------
        # Crop Header Region
        # ---------------------------------------------

        header_crop = img[0:table_y, :]

        # Validate crop
        if header_crop.size == 0:

            print("Empty crop -> skipped")
            continue

        crop_h, crop_w = header_crop.shape[:2]

        if crop_h < 20 or crop_w < 20:

            print(
                f"Crop too small ({crop_w}x{crop_h}) -> skipped"
            )
            continue

        # ---------------------------------------------
        # Save Header Crop
        # ---------------------------------------------

        crop_path = os.path.join(
            HEADER_CROP_FOLDER,
            image_path.name
        )

        cv2.imwrite(crop_path, header_crop)

        # ---------------------------------------------
        # OCR
        # ---------------------------------------------

        raw_text = pytesseract.image_to_string(header_crop)

        # ---------------------------------------------
        # Save OCR Text
        # ---------------------------------------------

        txt_path = os.path.join(
            OCR_OUTPUT_FOLDER,
            image_path.stem + ".txt"
        )

        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(raw_text)

        print(f"OCR Saved -> {txt_path}")

    except Exception as e:

        print(f"ERROR in {image_path.name}")
        print(str(e))

        continue

print("\nFinished processing all invoices.")