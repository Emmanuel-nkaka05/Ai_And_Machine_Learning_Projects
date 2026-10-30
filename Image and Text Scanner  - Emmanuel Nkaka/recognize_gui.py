""" image & text recognition (MobileNet-SSD + Tesseract OCR)"""
import os, sys, csv, shutil, threading, queue, ctypes
import numpy as np
import cv2
import pytesseract
from pytesseract import Output
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

PROTO_NAME = "MobileNetSSD_deploy.prototxt"
WEIGHTS_NAME = "MobileNetSSD_deploy.caffemodel"
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair",
           "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa",
           "train", "tvmonitor"]
MAX_SIDE = 1600          
PSM_CHOICES = ["3  - Fully automatic (default)", "4  - Single column of variable sizes",
               "6  - Single uniform block", "7  - Single text line", "11 - Sparse text", "12 - Sparse text + OSD"]

def model_dirs():
    """Every place the models folder may live (works as .py, onefile exe and onedir exe)."""
    d = []
    if getattr(sys, "frozen", False):
        d.append(os.path.join(os.path.dirname(sys.executable), "models"))
        if hasattr(sys, "_MEIPASS"):
            d.append(os.path.join(sys._MEIPASS, "models"))
    d.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "models"))
    d.append(os.path.join(os.getcwd(), "models"))
    return d

def find_models():
    for d in model_dirs():
        p, w = os.path.join(d, PROTO_NAME), os.path.join(d, WEIGHTS_NAME)
        if os.path.isfile(p) and os.path.isfile(w):
            return p, w
    raise FileNotFoundError("MobileNet-SSD files not found. Put %s and %s in a 'models' folder.\nLooked in:\n  %s"
                            % (PROTO_NAME, WEIGHTS_NAME, "\n  ".join(model_dirs())))

def setup_tesseract():
    cmd = shutil.which("tesseract")
    if not cmd:
        for p in (r"C:\Program Files\Tesseract-OCR\tesseract.exe",
                  r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
                  os.path.expandvars(r"%LOCALAPPDATA%\Programs\Tesseract-OCR\tesseract.exe")):
            if os.path.isfile(p):
                cmd = p
                break
    if not cmd:
        raise RuntimeError("Tesseract OCR is not installed (or not found).\n"
                           "Install it from https://github.com/UB-Mannheim/tesseract/wiki and re-run.")
    pytesseract.pytesseract.tesseract_cmd = cmd


_net = None
def get_net():
    """Load the network once and reuse it (loading is slow)."""
    global _net
    if _net is None:
        p, w = find_models()
        with open(p, "rb") as f: proto = f.read()
        with open(w, "rb") as f: weights = f.read()
        _net = cv2.dnn.readNetFromCaffe(proto, weights) if False else cv2.dnn.readNetFromCaffe(p, w)
    return _net

def load_image(path):
    data = np.fromfile(path, dtype=np.uint8)          
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not read that image file.")
    return img

def shrink(img, max_side=MAX_SIDE):
    h, w = img.shape[:2]
    s = max_side / max(h, w)
    return cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else img

def preprocess(img):
    """Grayscale -> blur -> adaptive threshold (spec preprocessing for OCR)."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    return cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 10)

def detect_objects(img, gate):
    net = get_net()
    h, w = img.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(img, (300, 300)), 0.007843, (300, 300), 127.5)
    net.setInput(blob)
    out = net.forward()
    rows = []
    for i in range(out.shape[2]):
        conf = float(out[0, 0, i, 2])
        if conf < gate:
            continue
        idx = int(out[0, 0, i, 1])
        x1, y1, x2, y2 = (out[0, 0, i, 3:7] * np.array([w, h, w, h])).astype(int)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w - 1, x2), min(h - 1, y2)
        label = CLASSES[idx] if 0 <= idx < len(CLASSES) else "class %d" % idx
        rows.append(("Object", label, conf * 100, (int(x1), int(y1), int(x2 - x1), int(y2 - y1))))
    return rows

def read_text(binary, psm, gate_pct):
    """One Tesseract call gives words, confidences and boxes together."""
    d = pytesseract.image_to_data(binary, config="--oem 3 --psm %d" % psm, output_type=Output.DICT, timeout=90)
    rows, lines = [], {}
    for i, txt in enumerate(d["text"]):
        txt = txt.strip()
        try: conf = float(d["conf"][i])
        except ValueError: conf = -1
        if not txt or conf < gate_pct:
            continue
        rows.append(("Text", txt, conf, (d["left"][i], d["top"][i], d["width"][i], d["height"][i])))
        lines.setdefault((d["block_num"][i], d["par_num"][i], d["line_num"][i]), []).append(txt)
    return rows, "\n".join(" ".join(v) for v in lines.values())

def annotate(img, rows):
    out = img.copy()
    t = max(1, round(max(out.shape[:2]) / 700))
    scale = 0.45 * t
    for kind, label, conf, (x, y, w, h) in rows:
        col = (60, 200, 60) if kind == "Object" else (230, 120, 20)
        cv2.rectangle(out, (x, y), (x + w, y + h), col, t + 1)
        tag = "%s %d%%" % (label, conf)
        (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, scale, t)
        ty = y - 4 if y - th - 6 > 0 else y + th + 6
        cv2.rectangle(out, (x, ty - th - 4), (x + tw + 4, ty + 3), col, -1)
        cv2.putText(out, tag, (x + 2, ty), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), t, cv2.LINE_AA)
    return out


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title(" Image & Text Recognition")
        self.geometry("1200x760"); self.minsize(980, 620)
        self.q = queue.Queue()
        self.img = self.work = self.binary = self.annotated = None
        self.rows, self.text, self._photo, self._resize_job = [], "", None, None
        self.path_var = tk.StringVar(value="No image selected")
        self.mode = tk.StringVar(value="det"); self.view = tk.StringVar(value="ann")
        self.gate = tk.DoubleVar(value=80); self.psm = tk.StringVar(value=PSM_CHOICES[0])
        self.status = tk.StringVar(value="Ready. Open an image to begin.")
        self._build()

    def _build(self):
        self.columnconfigure(1, weight=1); self.rowconfigure(1, weight=1)
        top = ttk.Frame(self, padding=8); top.grid(row=0, column=0, columnspan=2, sticky="ew")
        ttk.Button(top, text="Open Image...", command=self.open_image).pack(side="left")
        ttk.Label(top, textvariable=self.path_var).pack(side="left", padx=12)
        self.run_btn = ttk.Button(top, text="Run Recognition", command=self.run); self.run_btn.pack(side="right")

        side = ttk.Frame(self, padding=12); side.grid(row=1, column=0, sticky="ns")
        ttk.Label(side, text="SETTINGS & PARAMETERS").pack(anchor="w", pady=(0, 8))
        ttk.Label(side, text="1. Execution Path").pack(anchor="w")
        for t, v in (("Object Detection (MobileNet-SSD)", "det"), ("Text Recognition (OCR)", "ocr"), ("Both", "both")):
            ttk.Radiobutton(side, text=t, value=v, variable=self.mode).pack(anchor="w")
        ttk.Label(side, text="OCR layout (PSM):").pack(anchor="w", pady=(8, 0))
        ttk.Combobox(side, textvariable=self.psm, values=PSM_CHOICES, state="readonly", width=32).pack(anchor="w")
        ttk.Label(side, text="2. Thresholds").pack(anchor="w", pady=(14, 0))
        gf = ttk.Frame(side); gf.pack(fill="x")
        ttk.Label(gf, text="Confidence gate:").pack(side="left")
        self.gate_lbl = ttk.Label(gf, text="80 %"); self.gate_lbl.pack(side="right")
        ttk.Scale(side, from_=0, to=100, variable=self.gate,
                  command=lambda _: self.gate_lbl.config(text="%d %%" % round(self.gate.get()))).pack(fill="x")
        ttk.Label(side, text="3. View Mode").pack(anchor="w", pady=(14, 0))
        for t, v in (("Annotated result", "ann"), ("Original", "orig"), ("Pre-processed (OCR)", "pre")):
            ttk.Radiobutton(side, text=t, value=v, variable=self.view, command=self.show).pack(anchor="w")

        main = ttk.Frame(self); main.grid(row=1, column=1, sticky="nsew")
        main.columnconfigure(0, weight=1); main.rowconfigure(1, weight=3); main.rowconfigure(3, weight=2)
        bar = ttk.Frame(main, padding=6); bar.grid(row=0, column=0)
        ttk.Button(bar, text="Save Annotated Image...", command=self.save_image).pack(side="left", padx=4)
        ttk.Button(bar, text="Export Data Table", command=self.export_csv).pack(side="left", padx=4)
        self.canvas = tk.Canvas(main, bg="black", highlightthickness=0); self.canvas.grid(row=1, column=0, sticky="nsew")
        self.canvas.bind("<Configure>", self._on_resize)
        bar2 = ttk.Frame(main, padding=6); bar2.grid(row=2, column=0, sticky="e")
        ttk.Button(bar2, text="Copy Recognised Text", command=self.copy_text).pack()
        tf = ttk.Frame(main); tf.grid(row=3, column=0, sticky="nsew"); tf.columnconfigure(0, weight=1); tf.rowconfigure(0, weight=1)
        self.tree = ttk.Treeview(tf, columns=("type", "label", "conf", "box"), show="headings")
        for c, t, w in (("type", "TYPE", 90), ("label", "LABEL / TEXT", 320), ("conf", "CONFIDENCE", 110), ("box", "BOX (X, Y, W, H)", 200)):
            self.tree.heading(c, text=t); self.tree.column(c, width=w, anchor="center")
        sb = ttk.Scrollbar(tf, command=self.tree.yview); self.tree.configure(yscrollcommand=sb.set)
        self.tree.grid(row=0, column=0, sticky="nsew"); sb.grid(row=0, column=1, sticky="ns")
        ttk.Label(self, textvariable=self.status, relief="sunken", anchor="w", padding=4).grid(row=2, column=0, columnspan=2, sticky="ew")

    
    def open_image(self):
        p = filedialog.askopenfilename(filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff *.webp"), ("All files", "*.*")])
        if not p: return
        try:
            self.img = load_image(p)
        except Exception as e:
            return messagebox.showerror("Problem", str(e))
        self.work = shrink(self.img); self.binary = preprocess(self.work)
        self.annotated, self.rows, self.text = None, [], ""
        self.path_var.set(os.path.basename(p)); self.fill_table([])
        self.view.set("orig"); self.show()
        self.status.set("Loaded %dx%d image (working size %dx%d)." % (self.img.shape[1], self.img.shape[0], self.work.shape[1], self.work.shape[0]))

    def run(self):
        if self.work is None:
            return messagebox.showinfo("Run Recognition", "Open an image first.")
        self.run_btn.state(["disabled"]); self.status.set("Working...")
        args = (self.mode.get(), int(self.psm.get().split()[0]), self.gate.get())
        threading.Thread(target=self._worker, args=args, daemon=True).start()
        self.after(60, self._poll)

    def _worker(self, mode, psm, gate):        
        rows, text, errs = [], "", []
        if mode in ("det", "both"):
            try: rows += detect_objects(self.work, gate / 100)
            except Exception as e: errs.append("Detection: " + str(e))
        if mode in ("ocr", "both"):
            try:
                setup_tesseract(); r, text = read_text(self.binary, psm, gate); rows += r
            except Exception as e: errs.append("OCR: " + str(e))
        self.q.put((rows, text, errs, annotate(self.work, rows)))

    def _poll(self):
        try: rows, text, errs, ann = self.q.get_nowait()
        except queue.Empty: return self.after(60, self._poll)
        self.rows, self.text, self.annotated = rows, text, ann
        self.run_btn.state(["!disabled"]); self.fill_table(rows); self.view.set("ann"); self.show()
        self.status.set(" | ".join(errs) if errs else "Done: %d result(s) above the %d%% gate." % (len(rows), round(self.gate.get())))
        if errs and not rows: messagebox.showerror("Problem", "\n\n".join(errs))

    def fill_table(self, rows):
        self.tree.delete(*self.tree.get_children())
        if not rows and self.annotated is not None:
            self.tree.insert("", "end", values=("", "Nothing found above the %d%% confidence gate." % round(self.gate.get()), "", ""))
        for k, l, c, b in rows:
            self.tree.insert("", "end", values=(k, l, "%.1f %%" % c, "%d, %d, %d, %d" % b))

    def show(self):
        arr = {"ann": self.annotated, "orig": self.work, "pre": self.binary}[self.view.get()]
        if arr is None: arr = self.work
        if arr is None: return
        if arr.ndim == 2: arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2RGB)
        else: arr = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
        cw, ch = max(self.canvas.winfo_width(), 50), max(self.canvas.winfo_height(), 50)
        h, w = arr.shape[:2]; s = min(cw / w, ch / h)
        im = Image.fromarray(arr).resize((max(1, int(w * s)), max(1, int(h * s))), Image.BILINEAR)
        self._photo = ImageTk.PhotoImage(im)
        self.canvas.delete("all"); self.canvas.create_image(cw // 2, ch // 2, image=self._photo)

    def _on_resize(self, _):                      
        if self._resize_job: self.after_cancel(self._resize_job)
        self._resize_job = self.after(120, self.show)

    def save_image(self):
        if self.annotated is None: return messagebox.showinfo("Save", "Run recognition first.")
        p = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg")])
        if p:
            ok, buf = cv2.imencode(os.path.splitext(p)[1] or ".png", self.annotated)
            if ok: buf.tofile(p); self.status.set("Saved " + p)

    def export_csv(self):
        if not self.rows: return messagebox.showinfo("Export", "Nothing to export yet.")
        p = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV", "*.csv")])
        if p:
            with open(p, "w", newline="", encoding="utf-8") as f:
                wr = csv.writer(f); wr.writerow(["Type", "Label/Text", "Confidence %", "X", "Y", "W", "H"])
                for k, l, c, b in self.rows: wr.writerow([k, l, "%.1f" % c, *b])
            self.status.set("Exported " + p)

    def copy_text(self):
        t = self.text or "\n".join(r[1] for r in self.rows if r[0] == "Object")
        if not t: return messagebox.showinfo("Copy", "No recognised text yet.")
        self.clipboard_clear(); self.clipboard_append(t); self.status.set("Copied to clipboard.")

if __name__ == "__main__":
    try: ctypes.windll.shcore.SetProcessDpiAwareness(1)      
    except Exception: pass
    App().mainloop()
