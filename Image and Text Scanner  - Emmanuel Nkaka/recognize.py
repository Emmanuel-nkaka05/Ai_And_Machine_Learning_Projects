#!/usr/bin/env python3
"""
Project 4: Image or Text Recognition (Basic) - "Building the Machine's Optic Nerve"

Two execution paths (the "Perception Matrix"):
  Path 1  OCR              pytesseract  -> formatted text strings (+ confidence)
  Path 2  Object detection cv2.dnn + MobileNet-SSD -> (X, Y, W, H) bounding boxes

Usage:
  python recognize.py --mode ocr    --image samples/text_sample.png --psm 6
  python recognize.py --mode detect --image samples/street.jpg
  python recognize.py --mode both   --image samples/street.jpg --show
"""
import argparse
import os
import sys

import cv2
import numpy as np
import pytesseract
import shutil

# Windows: find Tesseract even if it is not on PATH
if not shutil.which("tesseract"):
    for _p in (r"C:\Program Files\Tesseract-OCR\tesseract.exe",
               r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe"):
        if os.path.exists(_p):
            pytesseract.pytesseract.tesseract_cmd = _p
            break

CONF_THRESHOLD = 0.80  # "In Project 4, 80% is the absolute minimum standard."
HERE = os.path.dirname(os.path.abspath(__file__))
PROTO = os.path.join(HERE, "models", "MobileNetSSD_deploy.prototxt")
WEIGHTS = os.path.join(HERE, "models", "MobileNetSSD_deploy.caffemodel")
OUT_DIR = os.path.join(HERE, "output")

CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus",
           "car", "cat", "chair", "cow", "diningtable", "dog", "horse",
           "motorbike", "person", "pottedplant", "sheep", "sofa", "train",
           "tvmonitor"]

PSM_HELP = {
    3: "fully automatic (default, varied layouts)",
    6: "single uniform block of text (book pages)",
    7: "single text line (number plates / headers)",
    11: "sparse, scattered text (invoices)",
}


# ----------------------------------------------------------------- helpers
def load_image(path):
    img = cv2.imread(path)
    if img is None:
        sys.exit(f"[error] cannot read image: {path}")
    h, w, c = img.shape
    print(f"[input] {path}  |  H={h} W={w} channels={c}  |  {h * w * c:,} data points")
    return img


def show_or_save(img, name, show):
    os.makedirs(OUT_DIR, exist_ok=True)
    out = os.path.join(OUT_DIR, name)
    cv2.imwrite(out, img)
    print(f"[output] annotated image saved -> {out}")
    if show:
        cv2.imshow(name, img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


# ------------------------------------------------------- Path 1: OCR
def deskew(gray):
    """Step 3 of pre-processing: snap tilted text back to a horizontal baseline."""
    inv = cv2.bitwise_not(gray)
    _, bw = cv2.threshold(inv, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    coords = np.column_stack(np.where(bw > 0))
    if len(coords) < 50:
        return gray
    angle = cv2.minAreaRect(coords)[-1]
    angle = -(90 + angle) if angle < -45 else (-angle if angle < 45 else 90 - angle)
    if abs(angle) < 0.5 or abs(angle) > 30:  # ignore noise / implausible skew
        return gray
    h, w = gray.shape
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    return cv2.warpAffine(gray, M, (w, h), flags=cv2.INTER_CUBIC,
                          borderMode=cv2.BORDER_REPLICATE)


def preprocess(img):
    """Grayscale -> Gaussian blur -> deskew -> adaptive threshold."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)          # step 1
    blur = cv2.GaussianBlur(gray, (5, 5), 0)              # step 2
    straight = deskew(blur)                               # step 3
    binary = cv2.adaptiveThreshold(                       # force the binary decision
        straight, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 31, 15)
    return binary


def run_ocr(img, psm=3, conf_threshold=CONF_THRESHOLD, show=False):
    print(f"\n=== PATH 1: OCR (pytesseract) | psm {psm}: {PSM_HELP.get(psm, 'custom')} ===")
    binary = preprocess(img)
    cv2.imwrite(os.path.join(os.makedirs(OUT_DIR, exist_ok=True) or OUT_DIR,
                             "ocr_preprocessed.png"), binary)

    config = f"--psm {psm}"
    data = pytesseract.image_to_data(binary, config=config,
                                     output_type=pytesseract.Output.DICT)
    annotated = img.copy()
    kept, dropped = [], 0
    for i, word in enumerate(data["text"]):
        word = word.strip()
        conf = float(data["conf"][i])
        if not word or conf < 0:
            continue
        conf /= 100.0
        x, y, w, h = (data["left"][i], data["top"][i],
                      data["width"][i], data["height"][i])
        if conf >= conf_threshold:                        # the confidence gate
            kept.append((word, conf, (x, y, w, h), data["line_num"][i],
                         data["block_num"][i], data["par_num"][i]))
            cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 200, 0), 2)
            cv2.putText(annotated, f"{word} {conf:.0%}", (x, max(y - 6, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 120, 0), 1)
        else:
            dropped += 1

    # rebuild readable lines from accepted words
    lines = {}
    for word, conf, _, ln, blk, par in kept:
        lines.setdefault((blk, par, ln), []).append(word)
    print("\n--- Extracted text (confidence >= {:.0%}) ---".format(conf_threshold))
    for key in sorted(lines):
        print("  " + " ".join(lines[key]))
    print("\n--- Word table ---")
    print(f"  {'WORD':<20}{'CONF':>7}   BOX (x, y, w, h)")
    for word, conf, box, *_ in kept:
        print(f"  {word:<20}{conf:>7.0%}   {box}")
    print(f"\n[summary] accepted={len(kept)}  dropped(below {conf_threshold:.0%})={dropped}")
    show_or_save(annotated, "ocr_result.png", show)
    return [k[0] for k in kept]


# ------------------------------------------------- Path 2: Object detection
def run_detection(img, conf_threshold=CONF_THRESHOLD, show=False):
    print("\n=== PATH 2: Object Detection (cv2.dnn + MobileNet-SSD) ===")
    if not (os.path.exists(PROTO) and os.path.exists(WEIGHTS)):
        sys.exit("[error] model files missing in ./models (see README)")
    net = cv2.dnn.readNetFromCaffe(PROTO, WEIGHTS)        # pre-trained (transfer learning)

    (H, W) = img.shape[:2]
    # Step 1: blob construction - mean subtraction + resize to 300x300
    blob = cv2.dnn.blobFromImage(cv2.resize(img, (300, 300)), 0.007843,
                                 (300, 300), 127.5)
    print(f"[blob] shape={blob.shape}  (N, C, H, W)")
    net.setInput(blob)
    detections = net.forward()                            # (1,1,N,7)

    annotated = img.copy()
    results, dropped = [], 0
    for i in range(detections.shape[2]):
        conf = float(detections[0, 0, i, 2])              # softmax confidence
        idx = int(detections[0, 0, i, 1])
        if conf < 0.05:                                   # ignore pure noise in the count
            continue
        if conf >= conf_threshold:
            # normalized coords -> physical pixel coords
            box = detections[0, 0, i, 3:7] * np.array([W, H, W, H])
            x1, y1, x2, y2 = box.astype(int)
            x1, y1 = max(x1, 0), max(y1, 0)
            x2, y2 = min(x2, W - 1), min(y2, H - 1)
            label = f"{CLASSES[idx]}: {conf:.0%}"
            results.append((CLASSES[idx], conf, (x1, y1, x2 - x1, y2 - y1)))
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 2)
            ty = y1 - 8 if y1 - 8 > 12 else y1 + 18
            cv2.putText(annotated, label, (x1, ty), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 0, 255), 2)
        else:
            dropped += 1                                  # drop_detection()

    print(f"\n--- Detections (confidence >= {conf_threshold:.0%}) ---")
    if not results:
        print("  (none above threshold)")
    for name, conf, (x, y, w, h) in results:
        print(f"  {name:<12}{conf:>6.0%}   X={x} Y={y} W={w} H={h}")
    print(f"\n[summary] accepted={len(results)}  dropped(below {conf_threshold:.0%})={dropped}")
    show_or_save(annotated, "detection_result.png", show)
    return results


# --------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description="Project 4: image / text recognition")
    ap.add_argument("--mode", choices=["ocr", "detect", "both"], default="both")
    ap.add_argument("--image", default=os.path.join(HERE, "samples", "street.jpg"),
                    help="path to image (default: samples/street.jpg)")
    ap.add_argument("--psm", type=int, default=3, choices=[3, 6, 7, 11])
    ap.add_argument("--conf", type=float, default=CONF_THRESHOLD,
                    help="confidence threshold (default 0.80)")
    ap.add_argument("--show", action="store_true", help="open a display window")
    a = ap.parse_args()

    img = load_image(a.image)
    if a.mode in ("ocr", "both"):
        run_ocr(img, a.psm, a.conf, a.show)
    if a.mode in ("detect", "both"):
        run_detection(img, a.conf, a.show)


if __name__ == "__main__":
    main()
