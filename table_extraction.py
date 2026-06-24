from ultralytics import YOLO
from paddleocr import PaddleOCR
from pathlib import Path
import cv2
import os

# =====================================================
# CONFIG
# =====================================================

MODEL_PATH = "Newend.pt"
INPUT_FOLDER = "dataset_new"

TABLE_CROP_FOLDER = "table_crops"
TABLE_OUTPUT_FOLDER = "table_output"

os.makedirs(TABLE_CROP_FOLDER, exist_ok=True)
os.makedirs(TABLE_OUTPUT_FOLDER, exist_ok=True)

# =====================================================
# LOAD MODEL + OCR
# =====================================================

print("Loading YOLO...")
model = YOLO(MODEL_PATH)

print("Loading PaddleOCR...")
ocr = PaddleOCR(
    use_angle_cls=True,
    lang="en"
)

# =====================================================
# IOU FUNCTION (REMOVE DUPLICATES)
# =====================================================

def iou(box1, box2):

    xA = max(box1[0], box2[0])
    yA = max(box1[1], box2[1])
    xB = min(box1[2], box2[2])
    yB = min(box1[3], box2[3])

    inter = max(0, xB - xA) * max(0, yB - yA)

    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

    union = area1 + area2 - inter

    if union == 0:
        return 0

    return inter / union

# =====================================================
# OCR FUNCTIONS
# =====================================================

def extract_table_words(table_crop):

    result = ocr.ocr(table_crop)

    if not result or result[0] is None:
        return []

    words = []

    try:
        res = result[0]

        # PaddleOCR v3 format
        if isinstance(res, dict):

            texts = res["rec_texts"]
            scores = res["rec_scores"]
            boxes = res["dt_polys"]

            for text, score, bbox in zip(
                texts,
                scores,
                boxes
            ):

                if score < 0.80:
                    continue

                x_center = bbox[:, 0].mean()
                y_center = bbox[:, 1].mean()

                words.append({
                    "text": text,
                    "x": round(float(x_center), 2),
                    "y": round(float(y_center), 2)
                })

        else:
            # PaddleOCR older format
            for line in res:

                bbox = line[0]
                text = line[1][0]
                score = line[1][1]

                if score < 0.80:
                    continue

                x_center = sum(
                    p[0] for p in bbox
                ) / 4

                y_center = sum(
                    p[1] for p in bbox
                ) / 4

                words.append({
                    "text": text,
                    "x": round(x_center, 2),
                    "y": round(y_center, 2)
                })

    except Exception as e:
        print("OCR Error:", e)

    return words


def group_rows(words, threshold=15):

    words = sorted(
        words,
        key=lambda x: x["y"]
    )

    rows = []
    current_row = []

    for word in words:

        if not current_row:
            current_row.append(word)
            continue

        if abs(
            word["y"] -
            current_row[0]["y"]
        ) < threshold:

            current_row.append(word)

        else:

            rows.append(current_row)
            current_row = [word]

    if current_row:
        rows.append(current_row)

    return rows

# =====================================================
# PROCESS ALL IMAGES
# =====================================================

image_files = sorted(
    Path(INPUT_FOLDER).glob("*.png")
)

print(f"\nFound {len(image_files)} images")

for image_path in image_files:

    try:

        print("\n" + "=" * 70)
        print(f"Processing: {image_path.name}")

        image = cv2.imread(str(image_path))

        if image is None:
            print("Could not read image")
            continue

        # =================================================
        # YOLO DETECTION
        # =================================================

        results = model(
            str(image_path),
            conf=0.30,
            iou=0.45,
            verbose=False
        )

        result = results[0]

        table_boxes = []

        for box in result.boxes:

            cls_id = int(box.cls[0])
            class_name = model.names[cls_id]

            if class_name == "Table":

                x1, y1, x2, y2 = map(
                    int,
                    box.xyxy[0]
                )

                conf = float(box.conf[0])

                table_boxes.append([
                    x1,
                    y1,
                    x2,
                    y2,
                    conf
                ])

        # =================================================
        # REMOVE DUPLICATES
        # =================================================

        table_boxes = sorted(
            table_boxes,
            key=lambda x: x[4],
            reverse=True
        )

        filtered_boxes = []

        for box in table_boxes:

            keep = True

            for existing in filtered_boxes:

                if iou(box, existing) > 0.70:
                    keep = False
                    break

            if keep:
                filtered_boxes.append(box)

        if len(filtered_boxes) == 0:

            print("No tables found")
            continue

        print(
            f"Detected {len(filtered_boxes)} unique table(s)"
        )

        # =================================================
        # ONE FILE PER INVOICE
        # =================================================

        invoice_txt_path = os.path.join(
            TABLE_OUTPUT_FOLDER,
            f"{image_path.stem}.txt"
        )

        invoice_file = open(
            invoice_txt_path,
            "w",
            encoding="utf-8"
        )

        invoice_file.write(
            "=" * 80 + "\n"
        )

        invoice_file.write(
            f"INVOICE : {image_path.name}\n"
        )

        invoice_file.write(
            "=" * 80 + "\n\n"
        )

        # =================================================
        # PROCESS ALL TABLES
        # =================================================

        for idx, box in enumerate(
            filtered_boxes,
            start=1
        ):

            x1, y1, x2, y2 = box[:4]

            table_crop = image[
                y1:y2,
                x1:x2
            ]

            if table_crop.size == 0:
                continue

            # ---------------------------------------------
            # SAVE TABLE IMAGE
            # ---------------------------------------------

            crop_name = (
                f"{image_path.stem}"
                f"_table_{idx}.png"
            )

            crop_path = os.path.join(
                TABLE_CROP_FOLDER,
                crop_name
            )

            cv2.imwrite(
                crop_path,
                table_crop
            )

            # ---------------------------------------------
            # OCR
            # ---------------------------------------------

            words = extract_table_words(
                table_crop
            )

            rows = group_rows(words)

            # ---------------------------------------------
            # WRITE TABLE
            # ---------------------------------------------

            invoice_file.write(
                f"\nTABLE {idx}\n"
            )

            invoice_file.write(
                "-" * 60 + "\n"
            )

            for row in rows:

                row.sort(
                    key=lambda x: x["x"]
                )

                row_text = " | ".join(
                    item["text"]
                    for item in row
                )

                invoice_file.write(
                    row_text + "\n"
                )

            invoice_file.write(
                "\n"
            )

        invoice_file.close()

        print(
            f"Saved -> {invoice_txt_path}"
        )

    except Exception as e:

        print(
            f"ERROR in {image_path.name}"
        )

        print(str(e))

        continue

print("\nFinished processing all invoices.")