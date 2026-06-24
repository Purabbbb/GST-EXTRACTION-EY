import cv2
import numpy as np
import pandas as pd
from paddleocr import PaddleOCR
from ultralytics import YOLO

# ============================================================
# CONFIG
# ============================================================

IMAGE_PATH = "invoice_test2.png"
MODEL_PATH = "Newend.pt"
OUTPUT_EXCEL = "all_tables_output.xlsx"

# ============================================================
# LOAD MODELS
# ============================================================

model = YOLO(MODEL_PATH)

ocr = PaddleOCR(
    use_angle_cls=True,
    lang="en"
)

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def calculate_iou(box1, box2):
    """
    Calculate Intersection over Union (IoU)
    """

    x1a, y1a, x2a, y2a = box1
    x1b, y1b, x2b, y2b = box2

    inter_x1 = max(x1a, x1b)
    inter_y1 = max(y1a, y1b)
    inter_x2 = min(x2a, x2b)
    inter_y2 = min(y2a, y2b)

    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)

    intersection = inter_w * inter_h

    area1 = (x2a - x1a) * (y2a - y1a)
    area2 = (x2b - x1b) * (y2b - y1b)

    union = area1 + area2 - intersection

    if union == 0:
        return 0

    return intersection / union


def remove_overlapping_tables(table_boxes, iou_threshold=0.5):
    """
    Keep only the largest box among overlapping detections.
    Keep all non-overlapping tables.
    """

    table_boxes = sorted(
        table_boxes,
        key=lambda x: x["area"],
        reverse=True
    )

    final_tables = []

    for candidate in table_boxes:

        keep = True

        for selected in final_tables:

            iou = calculate_iou(
                candidate["bbox"],
                selected["bbox"]
            )

            if iou > iou_threshold:
                keep = False
                break

        if keep:
            final_tables.append(candidate)

    return final_tables


def extract_table(table_crop):
    """
    Extract OCR words along with coordinates
    """

    result = ocr.ocr(table_crop)

    if not result or result[0] is None:
        return []

    res = result[0]

    texts = res["rec_texts"]
    scores = res["rec_scores"]
    boxes = res["dt_polys"]

    words = []

    for text, score, bbox in zip(texts, scores, boxes):

        if score < 0.80:
            continue

        bbox = np.array(bbox)

        x_center = bbox[:, 0].mean()
        y_center = bbox[:, 1].mean()

        words.append(
            {
                "text": text,
                "x": round(float(x_center), 2),
                "y": round(float(y_center), 2),
                "confidence": round(float(score), 2),
            }
        )

    return words


def group_rows(words, threshold=15):
    """
    Group OCR words into rows using Y coordinate
    """

    words = sorted(words, key=lambda x: x["y"])

    rows = []
    current_row = []

    for word in words:

        if not current_row:
            current_row.append(word)
            continue

        if abs(word["y"] - current_row[0]["y"]) < threshold:
            current_row.append(word)
        else:
            rows.append(current_row)
            current_row = [word]

    if current_row:
        rows.append(current_row)

    return rows

def build_table_using_header(rows):
    """
    Use first row as header and assign all OCR words
    to nearest header column.
    """

    if len(rows) < 2:
        return pd.DataFrame()

    # -------------------------------
    # Header row
    # -------------------------------
    header_row = rows[0]
    header_row.sort(key=lambda x: x["x"])

    header_positions = []
    header_names = []

    for item in header_row:
        header_positions.append(item["x"])
        header_names.append(item["text"])

    # -------------------------------
    # Check if data exists before
    # first header
    # -------------------------------

    first_header_x = header_positions[0]

    add_left_column = False

    for row in rows[1:]:

        for item in row:

            if item["x"] < first_header_x - 50:
                add_left_column = True
                break

        if add_left_column:
            break

    if add_left_column:

        header_names = ["Column_0"] + header_names
        header_positions = [0] + header_positions

    # -------------------------------
    # Build dataframe rows
    # -------------------------------

    table_rows = []

    for row in rows[1:]:

        row_dict = {
            col: ""
            for col in header_names
        }

        for item in row:

            nearest_idx = min(
                range(len(header_positions)),
                key=lambda i: abs(
                    item["x"] - header_positions[i]
                )
            )

            column_name = header_names[nearest_idx]

            if row_dict[column_name]:
                row_dict[column_name] += " " + item["text"]
            else:
                row_dict[column_name] = item["text"]

        table_rows.append(row_dict)

    return pd.DataFrame(table_rows)
# ============================================================
# MAIN
# ============================================================

print("Loading image...")

image = cv2.imread(IMAGE_PATH)

if image is None:
    raise FileNotFoundError(f"Could not read image: {IMAGE_PATH}")

print("Running YOLO detection...")

results = model(IMAGE_PATH)

result = results[0]

# ============================================================
# FIND ALL TABLES
# ============================================================

table_boxes = []

for box in result.boxes:

    cls_id = int(box.cls[0])

    class_name = model.names[cls_id]

    if class_name == "Table":

        x1, y1, x2, y2 = map(int, box.xyxy[0])

        area = (x2 - x1) * (y2 - y1)

        table_boxes.append(
            {
                "bbox": (x1, y1, x2, y2),
                "area": area
            }
        )

print(f"\nTotal table detections: {len(table_boxes)}")

if len(table_boxes) == 0:
    print("No tables detected!")
    exit()

# ============================================================
# REMOVE OVERLAPPING TABLES
# ============================================================

final_tables = remove_overlapping_tables(
    table_boxes,
    iou_threshold=0.5
)

print(
    f"Tables after overlap filtering: "
    f"{len(final_tables)}"
)

# ============================================================
# SORT TABLES TOP TO BOTTOM
# ============================================================

final_tables = sorted(
    final_tables,
    key=lambda x: x["bbox"][1]
)

# ============================================================
# EXTRACT ALL TABLES
# ============================================================

all_tables_data = []

for table_idx, table in enumerate(final_tables, start=1):

    x1, y1, x2, y2 = table["bbox"]

    print(f"\n{'='*80}")
    print(f"PROCESSING TABLE {table_idx}")
    print(f"{'='*80}")

    print(
        f"Coordinates: "
        f"({x1}, {y1}, {x2}, {y2})"
    )

    table_crop = image[y1:y2, x1:x2]

    cv2.imwrite(
        f"table_{table_idx}.png",
        table_crop
    )

    words = extract_table(table_crop)

    rows = group_rows(words, threshold=25)

    print("\nROWS:")
    print("-" * 80)

    for row_num, row in enumerate(rows):

        row.sort(key=lambda x: x["x"])

        print(
            f"Row {row_num}:",
            [item["text"] for item in row]
        )

    # Build dataframe using header coordinates
    df = build_table_using_header(rows)

    all_tables_data.append(df)

    print("\nDataFrame:")
    print(df)

    print(
        f"\nRows extracted: "
        f"{len(df)}"
    )

# ============================================================
# SAVE ALL TABLES
# ============================================================

with pd.ExcelWriter(
    OUTPUT_EXCEL,
    engine="openpyxl"
) as writer:

    for idx, df in enumerate(all_tables_data):

        df.to_excel(
            writer,
            sheet_name=f"Table_{idx+1}",
            index=False
        )

print(
    f"\nAll tables saved to: "
    f"{OUTPUT_EXCEL}"
)

# ============================================================
# DISPLAY TABLES
# ============================================================

for idx, df in enumerate(all_tables_data):

    print("\n" + "=" * 60)
    print(f"TABLE {idx + 1}")
    print("=" * 60)

    print(df)
