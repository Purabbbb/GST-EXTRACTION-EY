from ultralytics import YOLO
import cv2
import numpy as np
import os
import pandas as pd
import jwt
import json
import base64
from pyzbar.pyzbar import decode
from PIL import Image, ImageFilter, ImageEnhance

# =====================================================
# CONFIGURATION
# =====================================================

MODEL_PATH    = "Newend.pt"
INPUT_FOLDER  = "DATASET_Invoice"
QR_FOLDER     = "qr_crops"
OUTPUT_EXCEL  = "decoded_qr_results.xlsx"
FAILED_EXCEL  = "failed_invoices.xlsx"
CONF_THRESH   = 0.25   # lower to catch more QRs the model is less sure about

os.makedirs(QR_FOLDER, exist_ok=True)

# =====================================================
# LOAD MODEL
# =====================================================

print("Loading YOLO model...")
model = YOLO(MODEL_PATH)

# =====================================================
# HELPERS: IMAGE PREPROCESSING FOR BETTER DECODE
# =====================================================

def preprocess_variants(crop_bgr: np.ndarray) -> list:
    """
    Returns a list of image variants to try decoding one by one.
    Ordered from least to most aggressive processing.
    Each variant is a PIL Image.
    """
    variants = []
    h, w = crop_bgr.shape[:2]

    # Variant 1: original crop as-is
    variants.append(Image.fromarray(cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)))

    # Variant 2: grayscale
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    variants.append(Image.fromarray(gray))

    # Variant 3: upscaled 2x (helps with small/low-res QRs)
    upscaled = cv2.resize(gray, (w * 2, h * 2), interpolation=cv2.INTER_CUBIC)
    variants.append(Image.fromarray(upscaled))

    # Variant 4: adaptive threshold (handles uneven lighting)
    thresh_adapt = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 11, 2
    )
    variants.append(Image.fromarray(thresh_adapt))

    # Variant 5: Otsu threshold + upscale
    _, thresh_otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    thresh_up = cv2.resize(thresh_otsu, (w * 2, h * 2), interpolation=cv2.INTER_NEAREST)
    variants.append(Image.fromarray(thresh_up))

    # Variant 6: denoised + sharpened
    denoised = cv2.fastNlMeansDenoising(gray, h=10)
    pil_img   = Image.fromarray(denoised)
    sharpened = pil_img.filter(ImageFilter.SHARPEN)
    enhanced  = ImageEnhance.Contrast(sharpened).enhance(2.0)
    variants.append(enhanced)

    # Variant 7: morphological closing (fills gaps in QR modules)
    kernel  = np.ones((3, 3), np.uint8)
    closed  = cv2.morphologyEx(thresh_otsu, cv2.MORPH_CLOSE, kernel)
    variants.append(Image.fromarray(closed))

    return variants


def try_decode_qr(crop_bgr: np.ndarray) -> str | None:
    """
    Attempts to decode a QR from a crop using multiple methods and preprocessing.
    Returns decoded string or None.
    """
    variants = preprocess_variants(crop_bgr)

    for i, pil_img in enumerate(variants):
        cv_img = cv2.cvtColor(np.array(pil_img.convert("RGB")), cv2.COLOR_RGB2GRAY)

        # Method 1: pyzbar (more reliable)
        try:
            decoded_list = decode(pil_img)
            if decoded_list:
                data = decoded_list[0].data.decode("utf-8", errors="ignore").strip()
                if data:
                    return data
        except Exception:
            pass

        # Method 2: OpenCV QRCodeDetector
        try:
            detector = cv2.QRCodeDetector()
            data, bbox, _ = detector.detectAndDecode(cv_img)
            if data and data.strip():
                return data.strip()
        except Exception:
            pass

        # Method 3: OpenCV WeChatQRCode (more powerful, if available)
        try:
            wechat = cv2.wechat_qrcode_WeChatQRCode()
            texts, _ = wechat.detectAndDecode(cv_img)
            if texts and texts[0].strip():
                return texts[0].strip()
        except Exception:
            pass

    return None

# =====================================================
# HELPERS: QR DATA PARSING
# =====================================================

def parse_qr_data(qr_data: str) -> dict | None:
    """
    Tries JWT → JSON → Base64-JSON → URL/UPI fallback.
    Returns a dict of parsed fields, or None if all methods fail.
    """

    # --- Try JWT ---
    try:
        jwt_payload = jwt.decode(qr_data, options={"verify_signature": False})
        inner = jwt_payload.get("data", jwt_payload)

        if isinstance(inner, str):
            inner = json.loads(inner)

        # BUG FIX: ensure it's a dict before iterating
        if isinstance(inner, dict):
            return inner
        elif isinstance(inner, (int, float, bool)):
            return {"value": inner}
        elif isinstance(inner, list):
            return {"items": json.dumps(inner)}
    except Exception:
        pass

    # --- Try JSON ---
    try:
        parsed = json.loads(qr_data)
        if isinstance(parsed, dict):
            return parsed
        elif isinstance(parsed, list):
            return {"items": json.dumps(parsed)}
    except Exception:
        pass

    # --- Try Base64 → JSON ---
    try:
        decoded_bytes = base64.b64decode(qr_data + "==")  # padding-safe
        decoded_str   = decoded_bytes.decode("utf-8")
        parsed        = json.loads(decoded_str)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass

    # --- Fallback: URL or UPI string (non-GST QRs) ---
    stripped = qr_data.strip()
    if stripped.startswith("http://") or stripped.startswith("https://"):
        return {"QR_Type": "URL", "URL": stripped}
    if stripped.startswith("upi://"):
        # Parse UPI query params into a dict
        from urllib.parse import urlparse, parse_qs
        parsed_url = urlparse(stripped)
        params     = {k: v[0] for k, v in parse_qs(parsed_url.query).items()}
        return {"QR_Type": "UPI", **params}

    return None

# =====================================================
# STORAGE
# =====================================================

all_records    = []
failed_records = []

supported_ext = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
files = [f for f in os.listdir(INPUT_FOLDER) if f.lower().endswith(supported_ext)]
print(f"\nTotal Files Found: {len(files)}")

# =====================================================
# TRACK ALREADY-DECODED BASE INVOICE NAMES
# This avoids logging page_2 as "No QR detected" when page_1 already succeeded
# =====================================================

decoded_bases = set()   # base invoice names (without _page_X.ext) already decoded

# =====================================================
# PROCESS EACH FILE
# =====================================================

for idx, filename in enumerate(files, start=1):
    print(f"\n[{idx}/{len(files)}] {filename}")

    base_name = os.path.splitext(filename)[0]
    # Strip _page_N suffix to get the root invoice name
    import re
    invoice_base = re.sub(r'_page_\d+$', '', base_name)

    # Skip if this invoice was already decoded from a different page
    if invoice_base in decoded_bases:
        print(f"  ⏭  Skipping — already decoded from another page")
        continue

    image_path = os.path.join(INPUT_FOLDER, filename)

    try:
        image = cv2.imread(image_path)
        if image is None:
            failed_records.append({"Invoice_File": filename, "Reason": "Image could not be read"})
            continue

        results  = model(image, conf=CONF_THRESH)
        h, w     = image.shape[:2]
        qr_found  = False
        qr_decoded = False

        for result in results:
            for box in result.boxes:
                cls        = int(box.cls[0])
                class_name = model.names[cls]

                if class_name.upper() != "QR":
                    continue

                qr_found = True
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                # Padding — clamped to image bounds
                pad = 15
                x1 = max(0, x1 - pad)
                y1 = max(0, y1 - pad)
                x2 = min(w, x2 + pad)
                y2 = min(h, y2 + pad)

                qr_crop = image[y1:y2, x1:x2]

                # Save crop for inspection
                crop_path = os.path.join(QR_FOLDER, f"{base_name}_qr.png")
                cv2.imwrite(crop_path, qr_crop)

                # --- Decode with preprocessing pipeline ---
                qr_data = try_decode_qr(qr_crop)

                if not qr_data:
                    failed_records.append({
                        "Invoice_File": filename,
                        "Reason": "QR detected but decode failed (all preprocessing tried)"
                    })
                    continue

                # --- Parse decoded data ---
                parsed_data = parse_qr_data(qr_data)

                if parsed_data is None:
                    failed_records.append({
                        "Invoice_File": filename,
                        "Reason": "Unable to parse QR data",
                        "Raw_QR": qr_data[:300]
                    })
                    continue

                # --- Build record ---
                record = {"Invoice_File": filename}
                for key, value in parsed_data.items():
                    if isinstance(value, (dict, list)):
                        record[key] = json.dumps(value, ensure_ascii=False)
                    else:
                        record[key] = value

                record["QR_Raw_Data"] = qr_data
                all_records.append(record)

                decoded_bases.add(invoice_base)
                qr_decoded = True
                print("  ✓ Success")
                break

            if qr_decoded:
                break

        if not qr_found:
            failed_records.append({"Invoice_File": filename, "Reason": "No QR detected"})

    except Exception as e:
        failed_records.append({"Invoice_File": filename, "Reason": str(e)})

# =====================================================
# BUILD & SAVE DATAFRAMES
# =====================================================

preferred_order = [
    "Invoice_File", "SellerGstin", "BuyerGstin", "DocNo", "DocTyp",
    "DocDt", "TotInvVal", "ItemCnt", "MainHsnCode", "Irn", "AckNo",
    "AckDt", "QR_Type", "URL", "QR_Raw_Data"
]

success_df = pd.DataFrame(all_records)
if not success_df.empty:
    existing  = [c for c in preferred_order if c in success_df.columns]
    remaining = [c for c in success_df.columns if c not in existing]
    success_df = success_df[existing + remaining]

failed_df = pd.DataFrame(failed_records)

success_df.to_excel(OUTPUT_EXCEL, index=False)
failed_df.to_excel(FAILED_EXCEL, index=False)

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