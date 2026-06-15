from ultralytics import YOLO
import cv2
import os
import pandas as pd
import jwt
import json
import base64
from pyzbar.pyzbar import decode
from PIL import Image

# =====================================================
# CONFIGURATION
# =====================================================

MODEL_PATH = "Newend.pt"
INPUT_FOLDER = "DATASET_Invoice"

QR_FOLDER = "qr_crops"

OUTPUT_EXCEL = "decoded_qr_results.xlsx"
FAILED_EXCEL = "failed_invoices.xlsx"

os.makedirs(QR_FOLDER, exist_ok=True)

# =====================================================
# LOAD MODEL
# =====================================================

print("Loading YOLO model...")
model = YOLO(MODEL_PATH)

# =====================================================
# STORAGE
# =====================================================

all_records = []
failed_records = []

supported_extensions = (
    ".png",
    ".jpg",
    ".jpeg",
    ".bmp",
    ".tif",
    ".tiff"
)

files = [
    f for f in os.listdir(INPUT_FOLDER)
    if f.lower().endswith(supported_extensions)
]

print(f"\nTotal Files Found: {len(files)}")

# =====================================================
# PROCESS EACH FILE
# =====================================================

for idx, filename in enumerate(files, start=1):

    print(f"\n[{idx}/{len(files)}] Processing {filename}")

    image_path = os.path.join(INPUT_FOLDER, filename)

    try:

        image = cv2.imread(image_path)

        if image is None:

            failed_records.append({
                "Invoice_File": filename,
                "Reason": "Image could not be read"
            })

            continue

        results = model(image)

        qr_found = False
        qr_decoded = False

        for result in results:

            boxes = result.boxes

            for box in boxes:

                cls = int(box.cls[0])
                class_name = model.names[cls]

                if class_name.upper() != "QR":
                    continue

                qr_found = True

                x1, y1, x2, y2 = map(int, box.xyxy[0])

                # Add padding
                pad = 15

                h, w = image.shape[:2]

                x1 = max(0, x1 - pad)
                y1 = max(0, y1 - pad)

                x2 = min(w, x2 + pad)
                y2 = min(h, y2 + pad)

                qr_crop = image[y1:y2, x1:x2]

                base_name = os.path.splitext(filename)[0]

                crop_path = os.path.join(
                    QR_FOLDER,
                    f"{base_name}_qr.png"
                )

                cv2.imwrite(crop_path, qr_crop)

                qr_data = None

                # ==========================================
                # METHOD 1 : OpenCV Decoder
                # ==========================================

                try:

                    detector = cv2.QRCodeDetector()

                    data, bbox, _ = detector.detectAndDecode(
                        cv2.imread(crop_path)
                    )

                    if data and len(data.strip()) > 0:
                        qr_data = data.strip()

                except:
                    pass

                # ==========================================
                # METHOD 2 : PYZBAR FALLBACK
                # ==========================================

                if not qr_data:

                    try:

                        decoded = decode(
                            Image.open(crop_path)
                        )

                        if len(decoded) > 0:

                            qr_data = decoded[0].data.decode(
                                "utf-8",
                                errors="ignore"
                            ).strip()

                    except:
                        pass

                # ==========================================
                # QR DECODE FAILED
                # ==========================================

                if not qr_data:

                    failed_records.append({
                        "Invoice_File": filename,
                        "Reason": "QR detected but decode failed"
                    })

                    continue

                # ==========================================
                # PARSE DATA
                # ==========================================

                parsed_data = None

                # -----------------------------
                # TRY JWT
                # -----------------------------

                try:

                    jwt_data = jwt.decode(
                        qr_data,
                        options={"verify_signature": False}
                    )

                    if "data" in jwt_data:

                        if isinstance(jwt_data["data"], str):

                            parsed_data = json.loads(
                                jwt_data["data"]
                            )

                        else:

                            parsed_data = jwt_data["data"]

                    else:

                        parsed_data = jwt_data

                except:
                    pass

                # -----------------------------
                # TRY JSON
                # -----------------------------

                if parsed_data is None:

                    try:

                        parsed_data = json.loads(
                            qr_data
                        )

                    except:
                        pass

                # -----------------------------
                # TRY BASE64 JSON
                # -----------------------------

                if parsed_data is None:

                    try:

                        decoded_string = base64.b64decode(
                            qr_data
                        ).decode("utf-8")

                        parsed_data = json.loads(
                            decoded_string
                        )

                    except:
                        pass

                # ==========================================
                # FINAL FAILURE
                # ==========================================

                if parsed_data is None:

                    failed_records.append({
                        "Invoice_File": filename,
                        "Reason": "Unable to parse QR data",
                        "Raw_QR": qr_data[:200]
                    })

                    continue

                # ==========================================
                # STORE RECORD
                # ==========================================

                record = {
                    "Invoice_File": filename
                }

                for key, value in parsed_data.items():

                    if isinstance(value, (dict, list)):

                        record[key] = json.dumps(
                            value,
                            ensure_ascii=False
                        )

                    else:

                        record[key] = value

                record["QR_Raw_Data"] = qr_data

                all_records.append(record)

                qr_decoded = True

                print("✓ Success")

                break

            if qr_decoded:
                break

        # ==========================================
        # NO QR FOUND
        # ==========================================

        if not qr_found:

            failed_records.append({
                "Invoice_File": filename,
                "Reason": "No QR detected"
            })

    except Exception as e:

        failed_records.append({
            "Invoice_File": filename,
            "Reason": str(e)
        })

# =====================================================
# CREATE DATAFRAMES
# =====================================================

success_df = pd.DataFrame(all_records)

failed_df = pd.DataFrame(failed_records)

# =====================================================
# REORDER IMPORTANT GST FIELDS
# =====================================================

preferred_order = [
    "Invoice_File",
    "SellerGstin",
    "BuyerGstin",
    "DocNo",
    "DocTyp",
    "DocDt",
    "TotInvVal",
    "ItemCnt",
    "MainHsnCode",
    "Irn",
    "AckNo",
    "AckDt",
    "QR_Raw_Data"
]

existing_cols = [
    c for c in preferred_order
    if c in success_df.columns
]

remaining_cols = [
    c for c in success_df.columns
    if c not in existing_cols
]

success_df = success_df[
    existing_cols + remaining_cols
]

# =====================================================
# SAVE EXCEL FILES
# =====================================================

success_df.to_excel(
    OUTPUT_EXCEL,
    index=False
)

failed_df.to_excel(
    FAILED_EXCEL,
    index=False
)

# =====================================================
# SUMMARY
# =====================================================

print("\n=================================")
print("PROCESS COMPLETED")
print("=================================")

print(f"Total Images       : {len(files)}")
print(f"Successfully Parsed: {len(success_df)}")
print(f"Failed             : {len(failed_df)}")

print(f"\nSaved: {OUTPUT_EXCEL}")
print(f"Saved: {FAILED_EXCEL}")