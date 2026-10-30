"""Verifies the Project 4 requirements from the spec against the actual code + a live run."""
import re, subprocess, sys, os
src = open("recognize.py").read()
def run(args):
    return subprocess.run([sys.executable, "recognize.py", *args], capture_output=True, text=True)
ocr = run(["--mode","ocr","--image","samples/text_sample.png","--psm","6"]).stdout
det = run(["--mode","detect","--image","samples/street.jpg"]).stdout
checks = [
 ("Goal: image/text recognition with available libraries", "import pytesseract" in src and "import cv2" in src),
 ("Req 1: pre-trained model / library (MobileNet-SSD via cv2.dnn + pytesseract)", "readNetFromCaffe" in src and os.path.getsize("models/MobileNetSSD_deploy.caffemodel")>1e6),
 ("Req 2: recognition performed on sample input (OCR)", "INVOICE" in ocr and "1,250.00" in ocr),
 ("Req 2: recognition performed on sample input (objects)", bool(re.search(r"person\s+\d+%", det))),
 ("Req 3: output displayed clearly (tables + annotated images)", os.path.exists("output/ocr_result.png") and os.path.exists("output/detection_result.png") and "Word table" in ocr),
 ("Skill: AI libraries + understanding model outputs (confidence, boxes)", "X=" in det and "CONF" in ocr),
 ("Pre-processing: grayscale", "COLOR_BGR2GRAY" in src),
 ("Pre-processing: Gaussian blur", "GaussianBlur" in src),
 ("Pre-processing: deskew", "def deskew" in src and "warpAffine" in src),
 ("Pre-processing: adaptive threshold", "adaptiveThreshold" in src),
 ("PSM modes 3/6/7/11 supported", "choices=[3, 6, 7, 11]" in src),
 ("Blob construction via blobFromImage, 300x300, mean subtraction", "blobFromImage" in src and "(300, 300)" in src and "127.5" in src),
 ("Normalized coords scaled to pixel W/H", "np.array([W, H, W, H])" in src),
 ("Confidence gate 80% (draw vs drop)", "CONF_THRESHOLD = 0.80" in src and "drop_detection" in src),
 ("Detections below threshold are dropped", "dropped" in det),
]
bad = 0
for name, ok in checks:
    print(("PASS  " if ok else "FAIL  ") + name); bad += (not ok)
print(f"\n{len(checks)-bad}/{len(checks)} requirements met")
sys.exit(bad)
