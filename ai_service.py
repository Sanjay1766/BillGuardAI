"""
Advanced ai_service.py
- Keeps your current YOLOv8 detection, size rules, angle, prohibited-word checks
- Adds: OCR fallback chain (Tesseract -> EasyOCR optional)
- Adds: HuggingFace text classification for content categories (toxic/abusive/political/misinformation heuristic)
- Adds: NSFW image classification for the cropped billboard region
- Returns extra fields used by test.py / gui_app.py and new ones that won’t break UI

Safe-by-default: if any optional model is missing, the code degrades gracefully and still returns results.
"""

import os
import traceback
from io import BytesIO

import numpy as np
import requests
from PIL import Image


try:
    import cv2
except Exception:
    cv2 = None

try:
    import pytesseract
except Exception:
    pytesseract = None


try:
    import easyocr
except Exception:
    easyocr = None


HF_AVAILABLE = True
try:
    from transformers import pipeline
except Exception:
    HF_AVAILABLE = False

from huggingface_hub import hf_hub_download
from ultralytics import YOLO


PROHIBITED = {
    "obscene": [
        "fuck","shit","bitch","bastard","cunt","dick","pussy","ass","whore","slut",
        "motherfucker","dickhead","asshole","jerkoff","prick","slutty","horny","blowjob","handjob","porn","xxx"
    ],
    "casteist_racial": [
        "chamar","bhangi","chinki","nigga","nigger","kafir","mleccha",
        "low caste","untouchable","dalit dog","negro","ape",
        "terrorist muslim","hindu pig","christian dog","jew rat"
    ],
    "religious_offensive": [
        "kill hindus","kill muslims","kill christians","burn quran","burn church","burn mosque","destroy temple",
        "down with hindus","down with muslims","down with christians",
        "all hindus must die","all muslims must die","all christians must die"
    ],
    "banned_products": [
        "cigarette","tobacco","gutka","bidi","hookah","alcohol","rum","whisky","vodka","beer","wine",
        "marijuana","weed","ganja","cocaine","heroin","opium","ecstasy","meth","lsd",
        "gambling","casino","betting","lottery scam","nicotine"
    ],
    "misleading_claims": [
        "100% cure","guaranteed returns","no risk investment","lose 10 kg in 7 days","instant fairness",
        "permanent solution for baldness","instant hair regrowth","miracle cure","risk-free trading",
        "cure cancer guaranteed","forever young formula","grow taller instantly"
    ],
    "political_antinational": [
        "down with india","khalistan zindabad","pakistan zindabad","maoist revolution",
        "break india","india murdabad","hail isis","support taliban","naxal revolution",
        "red army uprising","kill modi","kill amit shah","kill yogi"
    ],
    "sexual_exploitation": [
        "call girls","escort service","massage parlour","sex for sale","hot girls","adult entertainment",
        "girls available","cheap sex","underage girls","teen sex","pornography"
    ],
    "scam_fraud": [
        "work from home earn","lottery winner","send money to claim prize","advance fee required",
        "crypto guaranteed returns","fake job offers","quick loan no documents","black money exchange"
    ]
}


BILLBOARD_RULES = {
    "Tamil Nadu": {
        "Urban": {"max_width_m": 6, "max_height_m": 3},
        "Rural": {"max_width_m": 8, "max_height_m": 4},
        "Highway": {"max_width_m": 10, "max_height_m": 5}
    },
    "Delhi": {
        "Urban": {"max_width_m": 4, "max_height_m": 2},
        "Rural": {"max_width_m": 6, "max_height_m": 3},
        "Highway": {"max_width_m": 8, "max_height_m": 4}
    }
}


CAMERA_REAL_WORLD_WIDTH_M = 10


local_path = hf_hub_download(
    repo_id="maco018/billboard-detection-Yolo12",
    filename="yolo12m.pt"
)
_yolo = YOLO(local_path)
print(f"[INFO] Custom YOLOv8 model loaded from: {local_path}")



def _ensure_tesseract_path_windows():
    """Set a default Windows path if pytesseract is available and path not set."""
    if pytesseract is None:
        return
    try:
        cmd = getattr(pytesseract.pytesseract, "tesseract_cmd", None)
        if not cmd or not os.path.exists(cmd):
           
            default_path = r"C:\\Program Files\\Tesseract-OCR\\tesseract.exe"
            if os.path.exists(default_path):
                pytesseract.pytesseract.tesseract_cmd = default_path
    except Exception:
        pass

_ensure_tesseract_path_windows()



def get_billboard_angle(full_image: Image.Image, xmin, ymin, xmax, ymax):
    if cv2 is None:
        return None
    try:
        crop = np.array(full_image.crop((xmin, ymin, xmax, ymax)))
        if crop.size == 0:
            return None
        gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        largest_contour = max(contours, key=cv2.contourArea)
        rect = cv2.minAreaRect(largest_contour)
        angle = rect[2]
        if angle < -45:
            angle = 90 + angle
        return round(float(angle), 2)
    except Exception as e:
        print(f"[WARN] Angle detection failed: {e}")
        return None



def preprocess_for_ocr(image_path: str):
    if cv2 is None:
        return Image.open(image_path).convert("RGB")
    try:
        img = cv2.imread(image_path)
        if img is None:
            return Image.open(image_path).convert("RGB")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 9)
        denoised = cv2.medianBlur(thresh, 3)
        
        scale_percent = 200
        width = int(denoised.shape[1] * scale_percent / 100)
        height = int(denoised.shape[0] * scale_percent / 100)
        resized = cv2.resize(denoised, (width, height), interpolation=cv2.INTER_CUBIC)
        return Image.fromarray(resized)
    except Exception as e:
        print(f"[WARN] Preprocessing failed: {e}")
        return Image.open(image_path).convert("RGB")


def extract_billboard_text(cropped_image_path: str):
    text = None
    
    if pytesseract is not None:
        try:
            pil_img = preprocess_for_ocr(cropped_image_path)
            custom_config = r"--oem 3 --psm 6"
            text = pytesseract.image_to_string(pil_img, lang="eng", config=custom_config)
        except Exception as e:
            print(f"[WARN] Tesseract OCR failed: {e}")
    
    if (not text or not text.strip()) and easyocr is not None:
        try:
            reader = easyocr.Reader(["en"], gpu=False)
            result = reader.readtext(cropped_image_path, detail=0)
            text = " ".join(result)
        except Exception as e:
            print(f"[WARN] EasyOCR failed: {e}")
    if not text:
        return None
    text = text.strip()
    print(f"[OCR] Extracted: {text}")
    return text or None



_text_pipe = None
_text_models_try = [
    
    "unitary/unbiased-toxic-roberta",
    
    "cardiffnlp/twitter-roberta-base-offensive",
]

def _load_text_pipeline():
    global _text_pipe
    if _text_pipe is not None or not HF_AVAILABLE:
        return _text_pipe
    for mid in _text_models_try:
        try:
            _text_pipe = pipeline("text-classification", model=mid, truncation=True)
            print(f"[INFO] Text classifier loaded: {mid}")
            break
        except Exception as e:
            print(f"[WARN] Could not load {mid}: {e}")
    return _text_pipe


def classify_text_advanced(text: str):
    if not text:
        return {
            "content_category": "Unknown",
            "content_scores": {},
        }
    pipe = _load_text_pipeline()
    if pipe is None:
        
        lt = text.lower()
        if any(w in lt for w in ["vote", "party", "election", "bjp", "congress", "dmk", "aiadmk"]):
            return {"content_category": "Political", "content_scores": {}}
        if any(w in lt for w in ["offer", "sale", "discount"]):
            return {"content_category": "Commercial", "content_scores": {}}
        return {"content_category": "Unknown", "content_scores": {}}

    try:
        out = pipe(text)
        
        scores = {}
        label = "Neutral"
        for item in out:
            lbl = str(item.get("label", "")).lower()
            score = float(item.get("score", 0.0))
            scores[lbl] = score
            if "toxic" in lbl or "offensive" in lbl or "abusive" in lbl:
                if score > 0.5:
                    label = "Obscene/Abusive"
        return {
            "content_category": label,
            "content_scores": scores,
        }
    except Exception as e:
        print(f"[WARN] Text pipeline inference failed: {e}")
        return {"content_category": "Unknown", "content_scores": {}}



_img_pipe = None
_img_models_try = [
    
    "Falconsai/nsfw_image_detection",
]

def _load_image_pipeline():
    global _img_pipe
    if _img_pipe is not None or not HF_AVAILABLE:
        return _img_pipe
    for mid in _img_models_try:
        try:
            _img_pipe = pipeline("image-classification", model=mid)
            print(f"[INFO] NSFW image classifier loaded: {mid}")
            break
        except Exception as e:
            print(f"[WARN] Could not load {mid}: {e}")
    return _img_pipe


def classify_image_nsfw(image_path: str):
    pipe = _load_image_pipeline()
    if pipe is None:
        return {"visual_category": "Unknown", "visual_scores": {}}
    try:
        preds = pipe(Image.open(image_path).convert("RGB"))
        scores = {p["label"].lower(): float(p["score"]) for p in preds}
        nsfw_score = max([scores.get(k, 0.0) for k in ("nsfw", "porn", "explicit", "sexy")] or [0.0])
        if nsfw_score > 0.5:
            label = "NSFW"
        else:
            label = "Neutral"   
        return {"visual_category": label, "visual_scores": scores}
    except Exception as e:
        print(f"[WARN] NSFW pipeline inference failed: {e}")
        return {"visual_category": "Unknown", "visual_scores": {}}


def detect_prohibited_content(text: str):
    if not text:
        return None, []
    lower_text = text.lower()
    for category, words in PROHIBITED.items():
        flagged = [w for w in words if w in lower_text]
        if flagged:
            return category, flagged
    return None, []


    pipe = _load_image_pipeline()
    if pipe is None:
        return {"visual_category": "Unknown", "visual_scores": {}}
    try:
        preds = pipe(Image.open(image_path).convert("RGB"))
        scores = {p["label"].lower(): float(p["score"]) for p in preds}
        nsfw_score = max([scores.get(k, 0.0) for k in ("nsfw", "porn", "explicit", "sexy")] or [0.0])
        if nsfw_score > 0.5:
            label = "NSFW"
        else:
            label = "Neutral"   
        return {"visual_category": label, "visual_scores": scores}
    except Exception as e:
        print(f"[WARN] NSFW pipeline inference failed: {e}")
        return {"visual_category": "Unknown", "visual_scores": {}}


def analyze_billboard(image_input, state: str, area_type: str, visualize: bool = False):
    try:
        
        if isinstance(image_input, str):
            original_image = Image.open(BytesIO(requests.get(image_input).content)).convert("RGB")
        else:
            original_image = image_input

        img_w, img_h = original_image.size
        total_area = float(img_w * img_h)

        
        target_w = 640
        resized = original_image.resize((target_w, int(target_w * img_h / img_w)))
        results = _yolo(np.array(resized), conf=0.25)
        detections = results[0].boxes.xyxy.cpu().numpy()
        if detections.shape[0] == 0:
            raise ValueError("No billboards detected by YOLOv8 custom model.")

        
        areas = (detections[:, 2] - detections[:, 0]) * (detections[:, 3] - detections[:, 1])
        i = int(areas.argmax())
        xmin_r, ymin_r, xmax_r, ymax_r = detections[i]

        
        scale_x = img_w / float(resized.size[0])
        scale_y = img_h / float(resized.size[1])
        xmin = float(xmin_r * scale_x)
        xmax = float(xmax_r * scale_x)
        ymin = float(ymin_r * scale_y)
        ymax = float(ymax_r * scale_y)

        bb_w_px = max(1.0, xmax - xmin)
        bb_h_px = max(1.0, ymax - ymin)

        
        bb_w_m = (bb_w_px / img_w) * CAMERA_REAL_WORLD_WIDTH_M
        bb_h_m = (bb_h_px / img_w) * CAMERA_REAL_WORLD_WIDTH_M

        bb_area_px = bb_w_px * bb_h_px
        area_pct = float((bb_area_px / total_area) * 100.0)
        oversized = bool(area_pct >= 50.0)

        rules = BILLBOARD_RULES.get(state, {}).get(area_type)
        if rules:
            if bb_w_m > rules["max_width_m"] or bb_h_m > rules["max_height_m"]:
                legal_status = "illegal"
                reason = f"Billboard exceeds legal size limit for {state} ({area_type})"
            else:
                legal_status = "legal"
                reason = "Billboard within legal size limits"
        else:
            legal_status = "unknown"
            reason = f"No rules found for {state} - {area_type}"

        angle_deg = get_billboard_angle(original_image, xmin, ymin, xmax, ymax)

        
        crop = original_image.crop((xmin, ymin, xmax, ymax))
        cropped_path = "cropped_local.jpg"
        crop.save(cropped_path)

        
        extracted_text = extract_billboard_text(cropped_path)
        text_cls = classify_text_advanced(extracted_text)
        prohibited_category, flagged_terms = detect_prohibited_content(extracted_text)

        
        if prohibited_category:
            legal_status = "illegal"
            reason = f"Prohibited content detected ({prohibited_category}): {', '.join(flagged_terms)}"

        
        visual_cls = classify_image_nsfw(cropped_path)
        if visual_cls.get("visual_category") == "NSFW":
            legal_status = "illegal"
            reason = "Prohibited visual content detected (NSFW)"

        visualized_path = None
        if visualize:
            try:
                dbg = results[0].plot()
                vis_pil = Image.fromarray(dbg)
                visualized_path = "visualized_temp.jpg"
                vis_pil.save(visualized_path)
            except Exception:
                visualized_path = None

        return {
            "analysis": {
                "is_compliant": legal_status == "legal",
                "legal_status": legal_status,
                "reason": reason,
                "billboard_width_m": float(bb_w_m),
                "billboard_height_m": float(bb_h_m),
                "billboard_angle_deg": angle_deg,
                "prohibited_category": prohibited_category,
                "prohibited_terms": flagged_terms,
            },
            "croppedImageUrl": cropped_path,
            "oversized": oversized,
            "billboard_area_percentage": area_pct,
            "visualized_image": visualized_path,
            "content_text": extracted_text,
            "content_category": text_cls.get("content_category", "Unknown"),
            "content_scores": text_cls.get("content_scores", {}),
            "visual_category": visual_cls.get("visual_category", "Unknown"),
            "visual_scores": visual_cls.get("visual_scores", {}),
            "prohibited_detected": prohibited_category is not None or visual_cls.get("visual_category") == "NSFW",
        }

    except Exception as e:
        print(f"[ERROR] AI Service exception: {e}")
        traceback.print_exc()
        return {
            "analysis": {"error": str(e)},
            "croppedImageUrl": None,
            "oversized": False,
            "billboard_area_percentage": 0.0,
            "visualized_image": None,
            "content_text": None,
            "content_category": "Unknown",
            "content_scores": {},
            "visual_category": "Unknown",
            "visual_scores": {},
            "prohibited_detected": False,
        }
