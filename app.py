"""
PaddyGuard AI — Flask Web Application
Run: python app.py
Visit: http://localhost:5000
"""

import os
# ── Must be set BEFORE importing tensorflow ──
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import uuid
import json
import numpy as np
from flask import Flask, render_template, request, jsonify
from PIL import Image, ImageEnhance
import tensorflow as tf

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024
UPLOAD_FOLDER = os.path.join('static', 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

MODEL_PATH = r"C:\Users\Admin\OneDrive\Pictures\Desktop\paddygaurd\final_model.keras"
IMAGE_SIZE = (224, 224)

# ── Load class names from JSON ────────────────────────────────────────────────
with open('class_names.json', 'r') as f:
    CLASS_NAMES = json.load(f)
print(f"[INFO] Class names loaded: {CLASS_NAMES}")

# ── Load model + warmup ───────────────────────────────────────────────────────
print("[INFO] Loading model...")
model = tf.keras.models.load_model(MODEL_PATH)

# Warmup — makes first real prediction fast
dummy = np.zeros((1, 224, 224, 3), dtype=np.float32)
model.predict(dummy, verbose=0)
print("[INFO] Model ready and warmed up ✓")

# ── Better image preprocessing ────────────────────────────────────────────────
def preprocess_image(filepath):
    """
    Better preprocessing for higher accuracy:
    - Resize to 224x224
    - Convert to RGB
    - Normalize to [0, 1]
    - Slight contrast enhancement for better feature detection
    """
    img = Image.open(filepath).convert("RGB")

    # Enhance contrast slightly — helps with disease detection
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(1.2)

    # Enhance sharpness slightly
    enhancer = ImageEnhance.Sharpness(img)
    img = enhancer.enhance(1.1)

    # Resize using high quality resampling
    img = img.resize(IMAGE_SIZE, Image.LANCZOS)

    # Convert to array and normalize
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = np.expand_dims(arr, axis=0)
    return arr

# ── Test Time Augmentation (TTA) for better accuracy ─────────────────────────
def predict_with_tta(filepath):
    """
    Test Time Augmentation — runs 5 versions of the image
    and averages the predictions for higher accuracy.
    """
    img = Image.open(filepath).convert("RGB")

    # Enhance
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(1.2)
    enhancer = ImageEnhance.Sharpness(img)
    img = enhancer.enhance(1.1)

    augmented = []

    # Original
    orig = img.resize(IMAGE_SIZE, Image.LANCZOS)
    augmented.append(np.array(orig, dtype=np.float32) / 255.0)

    # Slight horizontal flip
    flipped = orig.transpose(Image.FLIP_LEFT_RIGHT)
    augmented.append(np.array(flipped, dtype=np.float32) / 255.0)

    # Slight brightness increase
    bright = ImageEnhance.Brightness(orig).enhance(1.1)
    augmented.append(np.array(bright, dtype=np.float32) / 255.0)

    # Slight brightness decrease
    dark = ImageEnhance.Brightness(orig).enhance(0.9)
    augmented.append(np.array(dark, dtype=np.float32) / 255.0)

    # Slight color enhancement
    color = ImageEnhance.Color(orig).enhance(1.1)
    augmented.append(np.array(color, dtype=np.float32) / 255.0)

    # Stack all augmented images
    batch = np.stack(augmented, axis=0)  # Shape: (5, 224, 224, 3)

    # Predict all at once
    preds = model.predict(batch, verbose=0)  # Shape: (5, 10)

    # Average predictions
    avg_preds = np.mean(preds, axis=0)  # Shape: (10,)
    return avg_preds


RECOMMENDATIONS = {
    "healthy": {
        "display_name": "Healthy", "severity_level": 0, "severity_label": "No Disease",
        "description": "No disease detected. Your crop looks healthy and strong.",
        "action": "Continue regular crop management practices.",
        "chemical": [], "organic": [],
        "tips": ["Maintain proper plant spacing for air circulation.", "Use balanced NPK fertilizers.", "Monitor fields regularly for early disease signs.", "Avoid overwatering; maintain proper drainage."],
    },
    "bacterial_leaf_blight": {
        "display_name": "Bacterial Leaf Blight", "severity_level": 3, "severity_label": "High",
        "description": "Caused by Xanthomonas oryzae. Yellowing and wilting of leaf margins. Can cause 20–30% yield loss.",
        "action": "Apply bactericides immediately. Avoid excessive nitrogen fertilization.",
        "chemical": [
            {"name": "Copper Oxychloride 50% WP", "dosage": "2.5 g/litre water", "frequency": "Every 10–12 days, 2–3 sprays"},
            {"name": "Streptomycin + Tetracycline (Plantomycin)", "dosage": "0.5 g/litre water", "frequency": "2 sprays at 10-day intervals"},
            {"name": "Kasugamycin 3% SL", "dosage": "2 ml/litre water", "frequency": "2 sprays at 7–10 day intervals"},
        ],
        "organic": [
            {"name": "Pseudomonas fluorescens", "dosage": "5 g/litre water", "frequency": "At onset, repeat after 15 days"},
            {"name": "Neem Oil 5000 ppm", "dosage": "3 ml/litre water + 1 ml soap", "frequency": "Every 7 days"},
            {"name": "Garlic Extract", "dosage": "100 g garlic in 1 L water, dilute 1:10", "frequency": "Once a week"},
        ],
        "tips": ["Use resistant varieties (IR64, Swarna Sub1).", "Avoid high nitrogen doses.", "Drain fields during severe outbreaks.", "Avoid working in wet fields to reduce spread."],
    },
    "brown_spot": {
        "display_name": "Brown Spot", "severity_level": 2, "severity_label": "Moderate",
        "description": "Caused by Bipolaris oryzae. Circular brown spots with yellow halo. Yield loss up to 45%.",
        "action": "Apply fungicides at boot and heading stages. Improve soil nutrition.",
        "chemical": [
            {"name": "Mancozeb 75% WP", "dosage": "2.5 g/litre water", "frequency": "2–3 sprays at 10–14 day intervals"},
            {"name": "Edifenphos (Hinosan) 50% EC", "dosage": "1 ml/litre water", "frequency": "2 sprays at 10-day intervals"},
            {"name": "Propiconazole 25% EC", "dosage": "1 ml/litre water", "frequency": "2 sprays at 14-day intervals"},
        ],
        "organic": [
            {"name": "Trichoderma viride", "dosage": "4 g/kg seed OR 2.5 kg/ha soil", "frequency": "Seed treatment + transplanting"},
            {"name": "Neem Leaf Extract", "dosage": "500 g leaves in 10 L water, dilute to 20 L", "frequency": "Every 10 days"},
            {"name": "Cow Urine (fermented)", "dosage": "Dilute 1:5 with water", "frequency": "Every 7–10 days"},
        ],
        "tips": ["Apply potassium fertilizers.", "Use silicon-based fertilizers to strengthen leaf tissue.", "Avoid water stress during tillering.", "Destroy infected crop debris after harvest."],
    },
    "leaf_blast": {
        "display_name": "Leaf Blast", "severity_level": 3, "severity_label": "High",
        "description": "Caused by Magnaporthe oryzae. Diamond-shaped grey lesions with brown borders. Can destroy entire crop.",
        "action": "Apply systemic fungicides immediately. Leaf blast can escalate to neck blast.",
        "chemical": [
            {"name": "Tricyclazole 75% WP", "dosage": "0.6 g/litre water", "frequency": "2 sprays at 10–14 day intervals"},
            {"name": "Carbendazim 50% WP", "dosage": "1 g/litre water", "frequency": "2–3 sprays at 10-day intervals"},
            {"name": "Isoprothiolane (Fuji-one) 40% EC", "dosage": "1.5 ml/litre water", "frequency": "2 sprays at 14-day intervals"},
        ],
        "organic": [
            {"name": "Pseudomonas fluorescens", "dosage": "5 g/litre water", "frequency": "3 sprays at 10-day intervals"},
            {"name": "Silicon (rice husk ash extract)", "dosage": "2 g/litre water", "frequency": "At tillering and panicle initiation"},
            {"name": "Neem Oil (cold pressed)", "dosage": "5 ml/litre water + 1 ml soap", "frequency": "Every 7 days"},
        ],
        "tips": ["Avoid excessive nitrogen fertilisation.", "Use blast-resistant varieties (Sahbhagi Dhan, IR-64).", "Avoid overhead irrigation at night.", "Spray preventively at 20–25°C with high humidity."],
    },
    "leaf_scald": {
        "display_name": "Leaf Scald", "severity_level": 2, "severity_label": "Moderate",
        "description": "Caused by Microdochium oryzae. Zonate lesions on leaf tips and margins. Yield loss up to 15%.",
        "action": "Apply foliar fungicides. Improve field drainage.",
        "chemical": [
            {"name": "Iprodione 50% WP", "dosage": "2 g/litre water", "frequency": "2 sprays at 14-day intervals"},
            {"name": "Propiconazole 25% EC", "dosage": "1 ml/litre water", "frequency": "2 sprays at 14-day intervals"},
            {"name": "Thiram 75% WP", "dosage": "2 g/litre water OR 3 g/kg seed", "frequency": "Seed treatment + 1–2 foliar sprays"},
        ],
        "organic": [
            {"name": "Trichoderma harzianum", "dosage": "4 g/kg seed OR 2.5 kg/ha soil", "frequency": "At transplanting"},
            {"name": "Neem Cake", "dosage": "250 kg/ha", "frequency": "One-time soil application"},
        ],
        "tips": ["Avoid dense planting.", "Improve drainage to lower leaf wetness.", "Avoid late-evening irrigation.", "Use tolerant varieties where available."],
    },
    "narrow_brown_spot": {
        "display_name": "Narrow Brown Spot", "severity_level": 1, "severity_label": "Low",
        "description": "Caused by Cercospora janseana. Narrow dark brown streaks parallel to veins. Mild yield loss (<10%).",
        "action": "Apply fungicides at moderate infection. Correct soil nutrition.",
        "chemical": [
            {"name": "Mancozeb 75% WP", "dosage": "2.5 g/litre water", "frequency": "2 sprays at 14-day intervals"},
            {"name": "Carbendazim 50% WP", "dosage": "1 g/litre water", "frequency": "1–2 sprays at disease onset"},
        ],
        "organic": [
            {"name": "Trichoderma viride", "dosage": "4 g/kg seed", "frequency": "Seed treatment before sowing"},
            {"name": "Neem Oil", "dosage": "3 ml/litre water", "frequency": "Once a week at early infection"},
        ],
        "tips": ["Apply balanced nitrogen and potassium fertilizers.", "Correct iron or zinc deficiencies in soil.", "Use certified, disease-free seeds.", "Maintain proper water management."],
    },
    "neck_blast": {
        "display_name": "Neck Blast", "severity_level": 4, "severity_label": "Critical",
        "description": "Caused by Magnaporthe oryzae on panicle neck. Can cause 70–80% yield loss; entire panicle may be empty.",
        "action": "URGENT — Apply fungicides immediately at panicle emergence.",
        "chemical": [
            {"name": "Tricyclazole 75% WP", "dosage": "0.6 g/litre water", "frequency": "At 50% panicle emergence + 10 days later"},
            {"name": "Hexaconazole 5% EC", "dosage": "2 ml/litre water", "frequency": "2 sprays at panicle emergence and grain filling"},
            {"name": "Azoxystrobin 23% SC", "dosage": "1 ml/litre water", "frequency": "At flag leaf and panicle emergence"},
        ],
        "organic": [
            {"name": "Pseudomonas fluorescens + Trichoderma viride", "dosage": "5 g each per litre water", "frequency": "At flag leaf; repeat at panicle emergence"},
            {"name": "Potassium Silicate Solution", "dosage": "2 g/litre water", "frequency": "At flag leaf stage"},
        ],
        "tips": ["Preventive spray at booting stage is most effective.", "Avoid water stress at panicle initiation.", "Do not apply high nitrogen at heading.", "Spray before rain-forecasted humid nights."],
    },
    "rice_hispa": {
        "display_name": "Rice Hispa", "severity_level": 2, "severity_label": "Moderate",
        "description": "Caused by Dicladispa armigera. Grubs mine leaves causing white streaks. Yield loss 10–30%.",
        "action": "Apply insecticides to control adults and larvae. Remove affected tillers.",
        "chemical": [
            {"name": "Chlorpyrifos 20% EC", "dosage": "2.5 ml/litre water", "frequency": "At first sign; repeat after 10 days"},
            {"name": "Monocrotophos 36% SL", "dosage": "1.5 ml/litre water", "frequency": "1–2 sprays at 10-day intervals"},
            {"name": "Imidacloprid 17.8% SL", "dosage": "0.5 ml/litre water", "frequency": "2 sprays at 10–14 day intervals"},
        ],
        "organic": [
            {"name": "Neem Oil 10,000 ppm", "dosage": "5 ml/litre water + 2 ml soap", "frequency": "Every 5–7 days during outbreak"},
            {"name": "NSKE 5%", "dosage": "50 g/litre water", "frequency": "Every 7 days"},
            {"name": "Beauveria bassiana", "dosage": "5 ml/litre water (10⁸ cfu/ml)", "frequency": "Spray in evening; repeat after 7 days"},
        ],
        "tips": ["Cut and destroy affected leaves.", "Avoid close planting density.", "Flood field to drown fallen grubs.", "Conserve natural predators."],
    },
    "sheath_blight": {
        "display_name": "Sheath Blight", "severity_level": 3, "severity_label": "High",
        "description": "Caused by Rhizoctonia solani. Oval greenish-grey lesions on leaf sheaths. Yield loss 25–50%.",
        "action": "Apply systemic fungicides. Reduce planting density and nitrogen.",
        "chemical": [
            {"name": "Hexaconazole 5% EC", "dosage": "2 ml/litre water", "frequency": "2–3 sprays at 10–14 day intervals"},
            {"name": "Propiconazole 25% EC", "dosage": "1 ml/litre water", "frequency": "2 sprays at 14-day intervals"},
            {"name": "Carbendazim + Mancozeb", "dosage": "2 g/litre water", "frequency": "2–3 sprays at 10-day intervals"},
        ],
        "organic": [
            {"name": "Pseudomonas fluorescens", "dosage": "5 g/litre water", "frequency": "3 sprays at 10-day intervals"},
            {"name": "Trichoderma viride / harzianum", "dosage": "2.5 kg/ha with 50 kg FYM", "frequency": "Soil application at transplanting"},
            {"name": "Bacillus subtilis", "dosage": "5 ml/litre water", "frequency": "Every 7 days in early infection"},
        ],
        "tips": ["Reduce hill density (1–2 seedlings per hill).", "Maintain 2.5 cm water level in field.", "Avoid excess nitrogen application.", "Remove infected debris after harvest."],
    },
    "tungro": {
        "display_name": "Tungro", "severity_level": 4, "severity_label": "Critical",
        "description": "Caused by Rice Tungro Virus via green leafhopper. Yellow-orange leaves, stunting. Up to 100% yield loss.",
        "action": "URGENT — Control green leafhopper immediately. Remove and destroy infected plants.",
        "chemical": [
            {"name": "Imidacloprid 70% WS", "dosage": "10 g/kg seed", "frequency": "Seed treatment before sowing"},
            {"name": "Buprofezin 25% SC", "dosage": "1 ml/litre water", "frequency": "When leafhopper count > 2 per hill"},
            {"name": "Thiamethoxam 25% WG", "dosage": "0.4 g/litre water", "frequency": "2 sprays at 10–14 day intervals"},
        ],
        "organic": [
            {"name": "Neem Oil 5%", "dosage": "5 ml/litre water", "frequency": "Every 5 days to repel leafhoppers"},
            {"name": "Yellow Sticky Traps", "dosage": "20 traps/ha (20×25 cm)", "frequency": "Replace every 2 weeks"},
            {"name": "Verticillium lecanii", "dosage": "5 ml/litre water (10⁸ cfu/ml)", "frequency": "Spray in evening every 7 days"},
        ],
        "tips": ["Use tungro-resistant varieties.", "Synchronise planting with neighbours.", "Remove and burn infected plants immediately.", "Avoid planting near previously infected fields."],
    },
}

SEVERITY_CONFIG = {
    0: {"color": "#22c55e", "bg": "#f0fdf4", "border": "#86efac", "icon": "✓",   "label": "No Disease"},
    1: {"color": "#eab308", "bg": "#fefce8", "border": "#fde047", "icon": "!",   "label": "Low"},
    2: {"color": "#f97316", "bg": "#fff7ed", "border": "#fdba74", "icon": "!!",  "label": "Moderate"},
    3: {"color": "#ef4444", "bg": "#fef2f2", "border": "#fca5a5", "icon": "!!!", "label": "High"},
    4: {"color": "#7c3aed", "bg": "#f5f3ff", "border": "#c4b5fd", "icon": "⚠",  "label": "Critical"},
}
URGENCY = {0: "None", 1: "Within 1 week", 2: "Within 2–3 days", 3: "Within 24 hours", 4: "Immediately"}


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400
    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400
    allowed = {"jpg", "jpeg", "png", "webp"}
    ext = file.filename.rsplit(".", 1)[-1].lower()
    if ext not in allowed:
        return jsonify({"error": "Invalid file type. Use JPG or PNG."}), 400

    filename = f"{uuid.uuid4().hex}.{ext}"
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)

    # Use TTA for better accuracy
    preds = predict_with_tta(filepath)

    idx = int(np.argmax(preds))
    label = CLASS_NAMES[idx]
    confidence = round(float(preds[idx]) * 100, 1)

    top3 = sorted(
        [{"label": CLASS_NAMES[i], "display": CLASS_NAMES[i].replace("_", " ").title(),
          "prob": round(float(preds[i]) * 100, 1)} for i in range(len(CLASS_NAMES))],
        key=lambda x: x["prob"], reverse=True
    )[:3]

    rec = RECOMMENDATIONS[label]
    sev = SEVERITY_CONFIG[rec["severity_level"]]

    return jsonify({
        "image_url": f"/static/uploads/{filename}",
        "label": label,
        "display_name": rec["display_name"],
        "confidence": confidence,
        "low_confidence": confidence < 60,
        "severity_level": rec["severity_level"],
        "severity_label": rec["severity_label"],
        "severity_color": sev["color"],
        "severity_bg": sev["bg"],
        "severity_border": sev["border"],
        "severity_icon": sev["icon"],
        "urgency": URGENCY[rec["severity_level"]],
        "description": rec["description"],
        "action": rec["action"],
        "chemical": rec["chemical"],
        "organic": rec["organic"],
        "tips": rec["tips"],
        "top3": top3,
    })


@app.route("/map")
def map_page():
    return render_template("map.html")


if __name__ == "__main__":
    app.run(debug=True, port=5000)