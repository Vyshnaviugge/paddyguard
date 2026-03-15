"""
PaddyGuard AI — Prediction + Recommendation Script
====================================================
Usage:
    python paddyguard_predict.py --image path/to/leaf.jpg

Requirements:
    pip install tensorflow pillow numpy
"""

import argparse
import numpy as np
from PIL import Image
import tensorflow as tf

# ─────────────────────────────────────────────
#  CONFIG — Edit these if needed
# ─────────────────────────────────────────────

MODEL_PATH = r"C:\Users\Admin\OneDrive\Pictures\Desktop\paddygaurd\final_model.keras"

IMAGE_SIZE = (224, 224)

# Must match the order your model was trained with
# (sorted alphabetically — this is the default Keras ImageDataGenerator order)
CLASS_NAMES = [
    "bacterial_leaf_blight",
    "brown_spot",
    "healthy",
    "leaf_blast",
    "leaf_scald",
    "narrow_brown_spot",
    "neck_blast",
    "rice_hispa",
    "sheath_blight",
    "tungro",
]

# ─────────────────────────────────────────────
#  PESTICIDE RECOMMENDATION DATABASE
# ─────────────────────────────────────────────

DISEASE_RECOMMENDATIONS = {
    "healthy": {
        "display_name": "Healthy",
        "severity": "None",
        "severity_level": 0,
        "description": "No disease detected. The crop appears healthy.",
        "action": "Continue regular crop management practices.",
        "chemical": [],
        "organic": [],
        "preventive_tips": [
            "Maintain proper spacing between plants for air circulation.",
            "Use balanced NPK fertilizers to keep plants strong.",
            "Monitor fields regularly for early signs of disease.",
            "Avoid overwatering; maintain proper drainage.",
        ],
    },

    "bacterial_leaf_blight": {
        "display_name": "Bacterial Leaf Blight",
        "severity": "High",
        "severity_level": 3,
        "description": (
            "Caused by Xanthomonas oryzae pv. oryzae. "
            "Yellowing and wilting of leaf margins. "
            "Can cause 20–30% yield loss if untreated."
        ),
        "action": "Apply bactericides immediately. Avoid excessive nitrogen.",
        "chemical": [
            {
                "name": "Copper Oxychloride 50% WP",
                "dosage": "2.5 g/litre of water",
                "frequency": "Spray every 10–12 days, 2–3 sprays",
                "note": "Highly effective against bacterial infections.",
            },
            {
                "name": "Streptomycin Sulphate + Tetracycline (Plantomycin)",
                "dosage": "0.5 g/litre of water",
                "frequency": "2 sprays at 10-day intervals",
                "note": "Antibiotic-based; use sparingly to avoid resistance.",
            },
            {
                "name": "Kasugamycin 3% SL",
                "dosage": "2 ml/litre of water",
                "frequency": "2 sprays at 7–10 day intervals",
                "note": "Also effective against leaf blast.",
            },
        ],
        "organic": [
            {
                "name": "Pseudomonas fluorescens",
                "dosage": "2.5 kg/ha mixed with 50 kg FYM or 5 g/litre water for foliar spray",
                "frequency": "Apply at disease onset, repeat after 15 days",
                "note": "Biological control agent; improves plant immunity.",
            },
            {
                "name": "Neem Oil (5000 ppm Azadirachtin)",
                "dosage": "3 ml/litre of water + 1 ml liquid soap as sticker",
                "frequency": "Spray every 7 days",
                "note": "Reduces bacterial spread on leaf surface.",
            },
            {
                "name": "Garlic extract",
                "dosage": "100 g crushed garlic in 1 litre water, dilute 1:10",
                "frequency": "Spray once a week",
                "note": "Natural antibacterial properties.",
            },
        ],
        "preventive_tips": [
            "Use resistant varieties (IR64, Swarna Sub1).",
            "Avoid high nitrogen doses.",
            "Drain fields during severe outbreaks.",
            "Avoid working in wet fields to reduce spread.",
        ],
    },

    "brown_spot": {
        "display_name": "Brown Spot",
        "severity": "Moderate",
        "severity_level": 2,
        "description": (
            "Caused by Bipolaris oryzae. Circular brown spots with yellow halo on leaves. "
            "Causes grain discolouration and yield loss up to 45%."
        ),
        "action": "Apply fungicides at boot and heading stages. Improve soil nutrition.",
        "chemical": [
            {
                "name": "Mancozeb 75% WP",
                "dosage": "2.5 g/litre of water",
                "frequency": "2–3 sprays at 10–14 day intervals",
                "note": "Broad-spectrum protectant fungicide.",
            },
            {
                "name": "Edifenphos (Hinosan) 50% EC",
                "dosage": "1 ml/litre of water",
                "frequency": "2 sprays at 10-day intervals",
                "note": "Systemic action; effective on brown spot and blast.",
            },
            {
                "name": "Propiconazole 25% EC",
                "dosage": "1 ml/litre of water",
                "frequency": "2 sprays at 14-day intervals",
                "note": "DMI fungicide; good systemic activity.",
            },
        ],
        "organic": [
            {
                "name": "Trichoderma viride",
                "dosage": "4 g/kg seed (seed treatment) OR 2.5 kg/ha soil application",
                "frequency": "Seed treatment before sowing; soil drench at transplanting",
                "note": "Biocontrol agent suppressing fungal pathogens.",
            },
            {
                "name": "Neem leaf extract",
                "dosage": "500 g neem leaves boiled in 10 litres water, dilute to 20 litres",
                "frequency": "Spray every 10 days",
                "note": "Natural antifungal properties.",
            },
            {
                "name": "Cow urine (fermented)",
                "dosage": "Dilute 1:5 with water",
                "frequency": "Spray every 7–10 days",
                "note": "Traditional remedy with antifungal properties.",
            },
        ],
        "preventive_tips": [
            "Apply potassium fertilizers (crop is more susceptible under K deficiency).",
            "Use silicon-based fertilizers to strengthen leaf tissue.",
            "Avoid water stress especially during tillering.",
            "Collect and destroy infected crop debris after harvest.",
        ],
    },

    "leaf_blast": {
        "display_name": "Leaf Blast",
        "severity": "High",
        "severity_level": 3,
        "description": (
            "Caused by Magnaporthe oryzae. Diamond-shaped grey lesions with brown borders. "
            "Can destroy entire crop in cool humid weather."
        ),
        "action": "Apply systemic fungicides immediately. Leaf blast can escalate to neck blast.",
        "chemical": [
            {
                "name": "Tricyclazole 75% WP",
                "dosage": "0.6 g/litre of water",
                "frequency": "2 sprays at 10–14 day intervals",
                "note": "Most effective blast-specific fungicide.",
            },
            {
                "name": "Carbendazim 50% WP",
                "dosage": "1 g/litre of water",
                "frequency": "2–3 sprays at 10-day intervals",
                "note": "Systemic; also controls sheath blight.",
            },
            {
                "name": "Isoprothiolane (Fuji-one) 40% EC",
                "dosage": "1.5 ml/litre of water",
                "frequency": "2 sprays at 14-day intervals",
                "note": "Systemic; also improves grain quality.",
            },
        ],
        "organic": [
            {
                "name": "Pseudomonas fluorescens",
                "dosage": "5 g/litre water for foliar spray",
                "frequency": "3 sprays at 10-day intervals from seedling stage",
                "note": "Induces systemic resistance against blast.",
            },
            {
                "name": "Silicon (rice husk ash extract)",
                "dosage": "2 g/litre water",
                "frequency": "Spray at tillering and panicle initiation",
                "note": "Strengthens cell wall; reduces blast infection.",
            },
            {
                "name": "Neem oil (cold pressed)",
                "dosage": "5 ml/litre water + 1 ml soap",
                "frequency": "Spray every 7 days in early infection",
                "note": "Reduces spore germination.",
            },
        ],
        "preventive_tips": [
            "Avoid excessive nitrogen fertilisation.",
            "Use blast-resistant varieties (Sahbhagi Dhan, IR-64).",
            "Avoid overhead irrigation during night.",
            "Apply preventive spray when temperature is 20–25°C with high humidity.",
        ],
    },

    "leaf_scald": {
        "display_name": "Leaf Scald",
        "severity": "Moderate",
        "severity_level": 2,
        "description": (
            "Caused by Microdochium oryzae. Zonate onion-like lesions on leaf tips and margins. "
            "More common in humid warm environments. Yield loss up to 15%."
        ),
        "action": "Apply foliar fungicides. Improve field drainage and reduce humidity.",
        "chemical": [
            {
                "name": "Iprodione 50% WP",
                "dosage": "2 g/litre of water",
                "frequency": "2 sprays at 14-day intervals",
                "note": "Dicarboximide fungicide; effective against leaf scald.",
            },
            {
                "name": "Propiconazole 25% EC",
                "dosage": "1 ml/litre of water",
                "frequency": "2 sprays at 14-day intervals",
                "note": "Systemic triazole fungicide.",
            },
            {
                "name": "Thiram 75% WP",
                "dosage": "2 g/litre water OR 3 g/kg seed",
                "frequency": "Seed treatment + 1–2 foliar sprays",
                "note": "Protective fungicide; useful for seed treatment too.",
            },
        ],
        "organic": [
            {
                "name": "Trichoderma harzianum",
                "dosage": "4 g/kg seed treatment OR 2.5 kg/ha in soil",
                "frequency": "Apply at transplanting",
                "note": "Soil application reduces primary inoculum.",
            },
            {
                "name": "Neem cake",
                "dosage": "250 kg/ha at transplanting",
                "frequency": "One-time application in soil",
                "note": "Improves soil health and suppresses fungal activity.",
            },
        ],
        "preventive_tips": [
            "Avoid dense planting; maintain proper row spacing.",
            "Improve field drainage to lower leaf wetness period.",
            "Avoid late-evening irrigation.",
            "Use tolerant varieties where available.",
        ],
    },

    "narrow_brown_spot": {
        "display_name": "Narrow Brown Spot",
        "severity": "Low",
        "severity_level": 1,
        "description": (
            "Caused by Cercospora janseana. Narrow dark brown streaks on leaves parallel to veins. "
            "Generally mild yield loss (<10%); indicates nutrient stress."
        ),
        "action": "Apply fungicides at moderate to severe infection. Correct soil nutrition deficiencies.",
        "chemical": [
            {
                "name": "Mancozeb 75% WP",
                "dosage": "2.5 g/litre of water",
                "frequency": "2 sprays at 14-day intervals",
                "note": "Broad-spectrum protective fungicide.",
            },
            {
                "name": "Carbendazim 50% WP",
                "dosage": "1 g/litre of water",
                "frequency": "1–2 sprays at disease onset",
                "note": "Systemic; controls multiple fungal diseases.",
            },
        ],
        "organic": [
            {
                "name": "Trichoderma viride",
                "dosage": "4 g/kg seed for seed treatment",
                "frequency": "Seed treatment before sowing",
                "note": "Reduces seedborne fungal pathogens.",
            },
            {
                "name": "Neem oil",
                "dosage": "3 ml/litre water",
                "frequency": "Spray once a week at early infection",
                "note": "Reduces fungal spread.",
            },
        ],
        "preventive_tips": [
            "Apply balanced nitrogen and potassium fertilizers.",
            "Correct iron or zinc deficiencies in soil.",
            "Use certified, disease-free seeds.",
            "Maintain proper water management.",
        ],
    },

    "neck_blast": {
        "display_name": "Neck Blast",
        "severity": "Very High",
        "severity_level": 4,
        "description": (
            "Caused by Magnaporthe oryzae on the panicle neck. "
            "White/grey discolouration of panicle neck. "
            "Can cause 70–80% yield loss; entire panicle may be empty."
        ),
        "action": "URGENT — Apply fungicides immediately at panicle emergence.",
        "chemical": [
            {
                "name": "Tricyclazole 75% WP",
                "dosage": "0.6 g/litre of water",
                "frequency": "Spray at 50% panicle emergence and again 10 days later",
                "note": "Most critical timing for neck blast control.",
            },
            {
                "name": "Hexaconazole 5% EC",
                "dosage": "2 ml/litre of water",
                "frequency": "2 sprays at panicle emergence and grain filling",
                "note": "Triazole fungicide; excellent systemic activity.",
            },
            {
                "name": "Azoxystrobin 23% SC",
                "dosage": "1 ml/litre of water",
                "frequency": "Spray at flag leaf and panicle emergence stage",
                "note": "Strobilurin; protection during grain filling.",
            },
        ],
        "organic": [
            {
                "name": "Pseudomonas fluorescens + Trichoderma viride (combined)",
                "dosage": "5 g each per litre of water",
                "frequency": "Spray at flag leaf stage; repeat at panicle emergence",
                "note": "Synergistic biocontrol effect.",
            },
            {
                "name": "Potassium silicate solution",
                "dosage": "2 g/litre water",
                "frequency": "Spray at flag leaf stage",
                "note": "Strengthens panicle neck tissue.",
            },
        ],
        "preventive_tips": [
            "Preventive fungicide spray at booting stage is most effective.",
            "Avoid water stress at panicle initiation stage.",
            "Do not apply high doses of nitrogen at heading.",
            "Monitor weather — spray before rain-forecasted humid nights.",
        ],
    },

    "rice_hispa": {
        "display_name": "Rice Hispa",
        "severity": "Moderate",
        "severity_level": 2,
        "description": (
            "Caused by Dicladispa armigera (insect pest). "
            "Grubs mine inside leaves causing white parallel streaks; adults scrape leaf surface. "
            "Yield loss of 10–30% in severe cases."
        ),
        "action": "Apply insecticides to control adult and larval stages. Remove affected tillers.",
        "chemical": [
            {
                "name": "Chlorpyrifos 20% EC",
                "dosage": "2.5 ml/litre of water",
                "frequency": "Spray at first sign; repeat after 10 days",
                "note": "Contact + systemic action against adults and larvae.",
            },
            {
                "name": "Monocrotophos 36% SL",
                "dosage": "1.5 ml/litre of water",
                "frequency": "1–2 sprays at 10-day intervals",
                "note": "Systemic organophosphate; handle with care (toxic).",
            },
            {
                "name": "Imidacloprid 17.8% SL",
                "dosage": "0.5 ml/litre of water",
                "frequency": "2 sprays at 10–14 day intervals",
                "note": "Neonicotinoid; highly effective systemic insecticide.",
            },
        ],
        "organic": [
            {
                "name": "Neem oil (10,000 ppm Azadirachtin)",
                "dosage": "5 ml/litre water + 2 ml soap",
                "frequency": "Spray every 5–7 days during outbreak",
                "note": "Repels adults and disrupts larval development.",
            },
            {
                "name": "NSKE (Neem Seed Kernel Extract) 5%",
                "dosage": "50 g/litre water",
                "frequency": "Spray at 7-day intervals",
                "note": "Reduces adult feeding and oviposition.",
            },
            {
                "name": "Beauveria bassiana",
                "dosage": "5 ml/litre water (10^8 cfu/ml)",
                "frequency": "Spray in evening; repeat after 7 days",
                "note": "Entomopathogenic fungus; infects and kills adult hispa beetles.",
            },
        ],
        "preventive_tips": [
            "Cut and destroy affected leaves with mines to remove larvae.",
            "Avoid close planting density.",
            "Flood field to drown fallen grubs when possible.",
            "Conserve natural predators — avoid broad-spectrum pesticides early.",
        ],
    },

    "sheath_blight": {
        "display_name": "Sheath Blight",
        "severity": "High",
        "severity_level": 3,
        "description": (
            "Caused by Rhizoctonia solani. Oval/irregular greenish-grey lesions on leaf sheaths. "
            "Common in high-density plantings. Yield loss of 25–50% possible."
        ),
        "action": "Apply systemic fungicides. Reduce planting density and nitrogen.",
        "chemical": [
            {
                "name": "Hexaconazole 5% EC",
                "dosage": "2 ml/litre of water",
                "frequency": "2–3 sprays at 10–14 day intervals from disease onset",
                "note": "Most effective against sheath blight.",
            },
            {
                "name": "Propiconazole 25% EC",
                "dosage": "1 ml/litre of water",
                "frequency": "2 sprays at 14-day intervals",
                "note": "Systemic; also controls other fungal diseases.",
            },
            {
                "name": "Carbendazim 50% WP + Mancozeb 25% WP",
                "dosage": "2 g/litre of water",
                "frequency": "2–3 sprays at 10-day intervals",
                "note": "Combination improves efficacy spectrum.",
            },
        ],
        "organic": [
            {
                "name": "Pseudomonas fluorescens",
                "dosage": "5 g/litre water for foliar spray",
                "frequency": "3 sprays at 10-day intervals",
                "note": "Produces antifungal compounds against Rhizoctonia.",
            },
            {
                "name": "Trichoderma viride / harzianum",
                "dosage": "2.5 kg/ha mixed with 50 kg FYM; apply at transplanting",
                "frequency": "Soil application; repeat if disease persists",
                "note": "Colonises soil and suppresses Rhizoctonia solani.",
            },
            {
                "name": "Bacillus subtilis",
                "dosage": "5 ml/litre water",
                "frequency": "Spray at 7-day intervals in early infection",
                "note": "Produces lipopeptides toxic to Rhizoctonia.",
            },
        ],
        "preventive_tips": [
            "Reduce hill density (transplant 1–2 seedlings per hill).",
            "Maintain 2.5 cm water level in field.",
            "Avoid excess nitrogen application.",
            "Remove infected plant debris after harvest.",
        ],
    },

    "tungro": {
        "display_name": "Tungro",
        "severity": "Very High",
        "severity_level": 4,
        "description": (
            "Caused by Rice Tungro Bacilliform Virus + Rice Tungro Spherical Virus. "
            "Transmitted by green leafhopper. Symptoms: yellow-orange discolouration, stunting. "
            "Can cause 100% yield loss in severe outbreaks."
        ),
        "action": "URGENT — Control vector (green leafhopper) immediately. Remove and destroy infected plants.",
        "chemical": [
            {
                "name": "Imidacloprid 70% WS",
                "dosage": "10 g/kg seed (seed treatment)",
                "frequency": "Seed treatment before sowing",
                "note": "Prevents early vector infestation — most effective preventive.",
            },
            {
                "name": "Buprofezin 25% SC",
                "dosage": "1 ml/litre of water",
                "frequency": "Spray when leafhopper count > 2 per hill",
                "note": "Insect growth regulator; controls nymph stages.",
            },
            {
                "name": "Thiamethoxam 25% WG",
                "dosage": "0.4 g/litre of water",
                "frequency": "2 sprays at 10–14 day intervals",
                "note": "Fast knockdown of leafhoppers.",
            },
            {
                "name": "Carbofuran 3% G (granules)",
                "dosage": "33 kg/ha applied in field water",
                "frequency": "Apply at 2–3 weeks after transplanting",
                "note": "Systemic granule insecticide; do not use near water bodies.",
            },
        ],
        "organic": [
            {
                "name": "Neem oil 5%",
                "dosage": "5 ml/litre water",
                "frequency": "Spray every 5 days to repel leafhoppers",
                "note": "Reduces leafhopper feeding and virus transmission.",
            },
            {
                "name": "Yellow sticky traps",
                "dosage": "20 traps/ha (20×25 cm yellow cards coated with grease)",
                "frequency": "Install at crop canopy height; replace every 2 weeks",
                "note": "Monitors and mass-traps green leafhoppers.",
            },
            {
                "name": "Verticillium lecanii",
                "dosage": "5 ml/litre water (10^8 cfu/ml)",
                "frequency": "Spray in evening every 7 days",
                "note": "Entomopathogenic fungus infecting leafhoppers.",
            },
        ],
        "preventive_tips": [
            "Use tungro-resistant varieties (check local ICAR recommendations).",
            "Synchronise planting with neighbours — avoid staggered planting.",
            "Remove and burn infected plants immediately.",
            "Avoid planting near previously infected fields.",
        ],
    },
}

SEVERITY_GUIDE = {
    0: {"label": "No Disease",         "urgency": "None",             "icon": "✅"},
    1: {"label": "Low",                "urgency": "Within 1 week",    "icon": "🟡"},
    2: {"label": "Moderate",           "urgency": "Within 2–3 days",  "icon": "🟠"},
    3: {"label": "High",               "urgency": "Within 24 hours",  "icon": "🔴"},
    4: {"label": "Very High / Critical","urgency": "Immediately",     "icon": "🚨"},
}


# ─────────────────────────────────────────────
#  IMAGE PREPROCESSING
# ─────────────────────────────────────────────

def preprocess_image(image_path: str) -> np.ndarray:
    """Load and preprocess a leaf image for model prediction."""
    img = Image.open(image_path).convert("RGB")
    img = img.resize(IMAGE_SIZE)
    img_array = np.array(img, dtype=np.float32) / 255.0      # Normalize to [0, 1]
    img_array = np.expand_dims(img_array, axis=0)             # Shape: (1, 224, 224, 3)
    return img_array


# ─────────────────────────────────────────────
#  PREDICTION
# ─────────────────────────────────────────────

def predict_disease(model, image_path: str) -> tuple:
    """
    Predict disease from a leaf image.

    Returns:
        (predicted_label, confidence_percent, all_probabilities_dict)
    """
    img_array = preprocess_image(image_path)
    predictions = model.predict(img_array, verbose=0)[0]       # Shape: (10,)

    predicted_index = int(np.argmax(predictions))
    predicted_label = CLASS_NAMES[predicted_index]
    confidence = float(predictions[predicted_index]) * 100

    all_probs = {CLASS_NAMES[i]: round(float(predictions[i]) * 100, 2)
                 for i in range(len(CLASS_NAMES))}

    return predicted_label, round(confidence, 2), all_probs


# ─────────────────────────────────────────────
#  RECOMMENDATION
# ─────────────────────────────────────────────

def get_recommendation(disease_label: str) -> dict:
    """Get full pesticide recommendation for a predicted disease."""
    return DISEASE_RECOMMENDATIONS.get(disease_label, {})


# ─────────────────────────────────────────────
#  REPORT PRINTER
# ─────────────────────────────────────────────

def print_report(image_path: str, predicted_label: str,
                 confidence: float, all_probs: dict):
    """Print a formatted recommendation report to the console."""

    rec = get_recommendation(predicted_label)
    sev_level = rec.get("severity_level", 0)
    sev_info  = SEVERITY_GUIDE[sev_level]

    print("\n" + "=" * 65)
    print("        🌾  PADDYGUARD AI — DISEASE REPORT")
    print("=" * 65)
    print(f"  Image           : {image_path}")
    print(f"  Disease Detected: {rec.get('display_name', predicted_label)}")
    print(f"  Confidence      : {confidence:.2f}%")
    print(f"  Severity        : {sev_info['icon']}  {sev_info['label']}  (Level {sev_level}/4)")
    print(f"  Urgency         : {sev_info['urgency']}")
    print(f"\n  Description:\n    {rec.get('description', '')}")
    print(f"\n  Recommended Action:\n    {rec.get('action', '')}")

    # ── Chemical Pesticides ──────────────────
    print("\n" + "-" * 65)
    print("  CHEMICAL PESTICIDES")
    print("-" * 65)
    chem = rec.get("chemical", [])
    if chem:
        for i, p in enumerate(chem, 1):
            print(f"\n  [{i}] {p['name']}")
            print(f"      Dosage    : {p['dosage']}")
            print(f"      Frequency : {p['frequency']}")
            print(f"      Note      : {p['note']}")
    else:
        print("  No chemical treatment required.")

    # ── Organic / Biological ─────────────────
    print("\n" + "-" * 65)
    print("  ORGANIC / BIOLOGICAL PESTICIDES")
    print("-" * 65)
    org = rec.get("organic", [])
    if org:
        for i, p in enumerate(org, 1):
            print(f"\n  [{i}] {p['name']}")
            print(f"      Dosage    : {p['dosage']}")
            print(f"      Frequency : {p['frequency']}")
            print(f"      Note      : {p['note']}")
    else:
        print("  No organic treatment required.")

    # ── Preventive Tips ──────────────────────
    print("\n" + "-" * 65)
    print("  PREVENTIVE TIPS")
    print("-" * 65)
    for tip in rec.get("preventive_tips", []):
        print(f"  •  {tip}")

    # ── Top 3 Probabilities ──────────────────
    print("\n" + "-" * 65)
    print("  TOP PREDICTION PROBABILITIES")
    print("-" * 65)
    top3 = sorted(all_probs.items(), key=lambda x: x[1], reverse=True)[:3]
    for label, prob in top3:
        bar = "█" * int(prob / 5)
        print(f"  {label:<30} {prob:>6.2f}%  {bar}")

    print("\n" + "=" * 65 + "\n")


# ─────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="PaddyGuard AI — Paddy Disease Prediction & Recommendation"
    )
    parser.add_argument(
        "--image", type=str, required=True,
        help="Path to the paddy leaf image (jpg/png)"
    )
    args = parser.parse_args()

    # Load model
    print(f"\n[INFO] Loading model from:\n       {MODEL_PATH}")
    model = tf.keras.models.load_model(MODEL_PATH)
    print("[INFO] Model loaded successfully.")

    # Predict
    print(f"[INFO] Analysing image: {args.image}")
    predicted_label, confidence, all_probs = predict_disease(model, args.image)

    # Print report
    print_report(args.image, predicted_label, confidence, all_probs)


if __name__ == "__main__":
    main()