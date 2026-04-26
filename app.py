"""
PaddyGuard AI — Flask Web Application with MongoDB
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# Removed CPU forcing to allow GPU usage if available

import uuid
import json
import math
import numpy as np
import requests as http_requests
from datetime import datetime, timedelta
from flask import Flask, render_template, request, jsonify, redirect, url_for, session
from flask_mail import Mail, Message
from PIL import Image, ImageOps
import tensorflow as tf
from functools import wraps
from pymongo import MongoClient
from dotenv import load_dotenv
from werkzeug.security import generate_password_hash, check_password_hash
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

# Load environment variables from .env
load_dotenv()

app = Flask(__name__)

# ── Secret key from environment (never hardcode!)
app.secret_key = os.getenv('SECRET_KEY', 'fallback_dev_key_change_in_prod')
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024

# ── Session security: expire after 2 hours of inactivity
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(hours=2)
app.config['SESSION_COOKIE_HTTPONLY'] = True   # prevent JS access to cookie
app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'  # CSRF protection

# ── Flask-Mail (free Gmail SMTP) ──────────────────────────────────────────────
app.config['MAIL_SERVER'] = 'smtp.gmail.com'
app.config['MAIL_PORT'] = 587
app.config['MAIL_USE_TLS'] = True
app.config['MAIL_USERNAME'] = os.getenv('MAIL_USERNAME')
app.config['MAIL_PASSWORD'] = os.getenv('MAIL_PASSWORD')
app.config['MAIL_DEFAULT_SENDER'] = os.getenv('MAIL_USERNAME')
mail = Mail(app)

# ── Rate limiter: brute-force protection on login
limiter = Limiter(
    get_remote_address,
    app=app,
    default_limits=[],
    storage_uri="memory://"
)
UPLOAD_FOLDER = os.path.join('static', 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ── MongoDB Connection (URI loaded from .env) ─────────────────────────────────
MONGO_URI = os.getenv('MONGO_URI')
if not MONGO_URI:
    raise RuntimeError("MONGO_URI not set in .env file!")
client = MongoClient(MONGO_URI)
db = client["paddyguard"]
users_col = db["users"]
history_col = db["predictions"]
bookings_col = db["bookings"]
print("[INFO] MongoDB connected")

# ── Seed default users if not exist ──────────────────────────────────────────
default_users = [
    {"email": "admin@paddyguard.com", "password": generate_password_hash("admin123"), "role": "admin", "name": "Admin"},
    {"email": "farmer@paddyguard.com", "password": generate_password_hash("farmer123"), "role": "farmer", "name": "Ravi Kumar"},
    {"email": "sprayer@paddyguard.com", "password": generate_password_hash("sprayer123"), "role": "sprayer", "name": "Suresh Sprayer"},
    {"email": "shop@paddyguard.com", "password": generate_password_hash("shop123"), "role": "shop", "name": "AgriShop Owner"},
]
for u in default_users:
    if not users_col.find_one({"email": u["email"]}):
        users_col.insert_one(u)
print("[INFO] Default users seeded")

# ── Seed demo sprayers with location data ─────────────────────────────────────
demo_sprayers = [
    {
        "email": "sprayer1@paddyguard.com", "password": "sprayer123",
        "role": "sprayer", "name": "Ravi Kumar",
        "phone": "+91 98765 43210", "lat": 17.3850, "lng": 78.4867,
        "rating": 4.9, "jobs_done": 147, "rate": 350,
        "rate_unit": "acre", "available": True,
        "available_from": "Today from 7 AM",
        "certifications": ["Certified Pesticide Applicator", "Organic Farming"],
        "diseases": ["Neck Blast", "Leaf Blast", "Sheath Blight", "Brown Spot"],
        "bio": "7 years experience in paddy disease management across Telangana."
    },
    {
        "email": "sprayer2@paddyguard.com", "password": "sprayer123",
        "role": "sprayer", "name": "Muthu Selvam",
        "phone": "+91 87654 32109", "lat": 17.3900, "lng": 78.4950,
        "rating": 4.7, "jobs_done": 89, "rate": 300,
        "rate_unit": "acre", "available": True,
        "available_from": "Tomorrow 6 AM",
        "certifications": ["Certified Pesticide Applicator"],
        "diseases": ["Blast", "Brown Spot", "Tungro", "Bacterial Leaf Blight"],
        "bio": "Specialises in fungal and bacterial disease treatment."
    },
    {
        "email": "sprayer3@paddyguard.com", "password": "sprayer123",
        "role": "sprayer", "name": "Anand Prakash",
        "phone": "+91 76543 21098", "lat": 17.3800, "lng": 78.5000,
        "rating": 4.5, "jobs_done": 62, "rate": 280,
        "rate_unit": "acre", "available": False,
        "available_from": "Available in 2 days",
        "certifications": ["Organic Farming Specialist"],
        "diseases": ["All Diseases", "Organic Treatments Only"],
        "bio": "Focuses exclusively on organic and bio-pesticide applications."
    },
]
for s in demo_sprayers:
    if not users_col.find_one({"email": s["email"]}):
        s["password"] = generate_password_hash(s["password"])  # hash sprayer passwords too
        users_col.insert_one(s)
print("[INFO] Demo sprayers seeded")

# ── Model Loading with Error Handling ─────────────────────────────────────────
MODEL_PATH = r"C:\Users\Admin\OneDrive\Pictures\Desktop\paddygaurd\rice_disease_final.keras"
IMAGE_SIZE = (224, 224)

# Load class names with validation
try:
    with open('class_names.json', 'r') as f:
        raw = json.load(f)
        CLASS_NAMES = raw if isinstance(raw, list) else raw['class_names']
    print(f"[INFO] Loaded {len(CLASS_NAMES)} classes: {CLASS_NAMES}")
except FileNotFoundError:
    print("[ERROR] class_names.json not found!")
    # Default fallback classes (adjust based on your model)
    CLASS_NAMES = [
        "healthy", "bacterial_leaf_blight", "brown_spot", "leaf_blast", 
        "leaf_scald", "narrow_brown_spot", "neck_blast", "rice_hispa", 
        "sheath_blight", "tungro"
    ]
    print("[INFO] Using default class names")

# Load model with error handling
print("[INFO] Loading model...")
try:
    # Clear any existing session
    tf.keras.backend.clear_session()
    
    # Load model without compilation to avoid optimizer issues
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    
    # Test with dummy input
    dummy = np.zeros((1, 224, 224, 3), dtype=np.float32)
    test_output = model.predict(dummy, verbose=0)
    print(f"[INFO] Model loaded successfully. Output shape: {test_output.shape}")
    print(f"[INFO] Number of output classes: {test_output.shape[1]}")
    
    # Verify class count matches
    if test_output.shape[1] != len(CLASS_NAMES):
        print(f"[WARNING] Model output classes ({test_output.shape[1]}) don't match class_names ({len(CLASS_NAMES)})")
        
except Exception as e:
    print(f"[ERROR] Failed to load model: {e}")
    model = None

# ── Fast Single-Pass Prediction ──────────────────────────────────────────────
def preprocess_image(img):
    """Preprocess image: fix mobile EXIF rotation, crop to square, resize."""
    # 1. Fix EXIF orientation (mobile phones)
    img = ImageOps.exif_transpose(img)
    
    # 2. Crop to center square to avoid squashing tall mobile photos
    width, height = img.size
    min_dim = min(width, height)
    left = (width - min_dim) / 2
    top = (height - min_dim) / 2
    right = (width + min_dim) / 2
    bottom = (height + min_dim) / 2
    img = img.crop((left, top, right, bottom))
    
    # 3. Resize to model input size
    img = img.resize(IMAGE_SIZE, Image.LANCZOS)
    img_array = np.array(img, dtype=np.float32)
    return img_array

def predict_fast(filepath):
    """Fast single-pass prediction without TTA"""
    if model is None:
        return None
    
    img = Image.open(filepath).convert("RGB")
    
    try:
        # Preprocess single image
        orig = preprocess_image(img)
        # Expand dims to create batch of 1
        batch = np.expand_dims(orig, axis=0)
        
        # Fast 1-pass prediction
        preds = model.predict(batch, verbose=0)
        final_prediction = preds[0]
        
    except Exception as e:
        print(f"[ERROR] Prediction failed: {e}")
        raise Exception(f"Failed to process image: {e}")
    
    # Apply softmax if needed (in case model doesn't output probabilities)
    if final_prediction.max() > 1 or final_prediction.min() < 0:
        final_prediction = tf.nn.softmax(final_prediction).numpy()
    
    return final_prediction

# ── Decorators ────────────────────────────────────────────────────────────────
def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if 'user' not in session:
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated

# ── Haversine distance (metres) ───────────────────────────────────────────────
def haversine(lat1, lng1, lat2, lng2):
    R = 6371000
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lng2 - lng1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlam/2)**2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

def fmt_dist(m):
    return f"{m/1000:.1f} km" if m >= 1000 else f"{round(m)} m"

# ── Recommendations (same as before, kept for brevity) ────────────────────────
RECOMMENDATIONS = {
    "healthy": {
        "display_name": "Healthy", "severity_level": 0, "severity_label": "No Disease",
        "description": "No disease detected. Your crop looks healthy and strong.",
        "action": "Continue regular crop management practices.",
        "chemical": [], "organic": [],
        "tips": ["Maintain proper plant spacing.", "Use balanced NPK fertilizers.", "Monitor regularly.", "Avoid overwatering."],
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
        ],
        "tips": ["Use resistant varieties (IR64, Swarna Sub1).", "Avoid high nitrogen doses.", "Drain fields during severe outbreaks.", "Avoid working in wet fields."],
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
            {"name": "Neem Leaf Extract", "dosage": "500 g leaves in 10 L water", "frequency": "Every 10 days"},
        ],
        "tips": ["Apply potassium fertilizers.", "Use silicon-based fertilizers.", "Avoid water stress during tillering.", "Destroy infected crop debris."],
    },
    "leaf_blast": {
        "display_name": "Leaf Blast", "severity_level": 3, "severity_label": "High",
        "description": "Caused by Magnaporthe oryzae. Diamond-shaped grey lesions. Can destroy entire crop.",
        "action": "Apply systemic fungicides immediately. Leaf blast can escalate to neck blast.",
        "chemical": [
            {"name": "Tricyclazole 75% WP", "dosage": "0.6 g/litre water", "frequency": "2 sprays at 10–14 day intervals"},
            {"name": "Carbendazim 50% WP", "dosage": "1 g/litre water", "frequency": "2–3 sprays at 10-day intervals"},
            {"name": "Isoprothiolane 40% EC", "dosage": "1.5 ml/litre water", "frequency": "2 sprays at 14-day intervals"},
        ],
        "organic": [
            {"name": "Pseudomonas fluorescens", "dosage": "5 g/litre water", "frequency": "3 sprays at 10-day intervals"},
            {"name": "Neem Oil (cold pressed)", "dosage": "5 ml/litre water + 1 ml soap", "frequency": "Every 7 days"},
        ],
        "tips": ["Avoid excessive nitrogen.", "Use blast-resistant varieties.", "Avoid overhead irrigation at night.", "Spray preventively at high humidity."],
    },
    "leaf_scald": {
        "display_name": "Leaf Scald", "severity_level": 2, "severity_label": "Moderate",
        "description": "Caused by Microdochium oryzae. Zonate lesions on leaf tips. Yield loss up to 15%.",
        "action": "Apply foliar fungicides. Improve field drainage.",
        "chemical": [
            {"name": "Iprodione 50% WP", "dosage": "2 g/litre water", "frequency": "2 sprays at 14-day intervals"},
            {"name": "Propiconazole 25% EC", "dosage": "1 ml/litre water", "frequency": "2 sprays at 14-day intervals"},
        ],
        "organic": [
            {"name": "Trichoderma harzianum", "dosage": "4 g/kg seed", "frequency": "At transplanting"},
            {"name": "Neem Cake", "dosage": "250 kg/ha", "frequency": "One-time soil application"},
        ],
        "tips": ["Avoid dense planting.", "Improve drainage.", "Avoid late-evening irrigation.", "Use tolerant varieties."],
    },
    "narrow_brown_spot": {
        "display_name": "Narrow Brown Spot", "severity_level": 1, "severity_label": "Low",
        "description": "Caused by Cercospora janseana. Narrow dark brown streaks. Mild yield loss (<10%).",
        "action": "Apply fungicides at moderate infection. Correct soil nutrition.",
        "chemical": [
            {"name": "Mancozeb 75% WP", "dosage": "2.5 g/litre water", "frequency": "2 sprays at 14-day intervals"},
            {"name": "Carbendazim 50% WP", "dosage": "1 g/litre water", "frequency": "1–2 sprays at disease onset"},
        ],
        "organic": [
            {"name": "Trichoderma viride", "dosage": "4 g/kg seed", "frequency": "Seed treatment before sowing"},
            {"name": "Neem Oil", "dosage": "3 ml/litre water", "frequency": "Once a week at early infection"},
        ],
        "tips": ["Apply balanced fertilizers.", "Correct iron or zinc deficiencies.", "Use certified seeds.", "Maintain proper water management."],
    },
    "neck_blast": {
        "display_name": "Neck Blast", "severity_level": 4, "severity_label": "Critical",
        "description": "Caused by Magnaporthe oryzae on panicle neck. Can cause 70–80% yield loss.",
        "action": "URGENT — Apply fungicides immediately at panicle emergence.",
        "chemical": [
            {"name": "Tricyclazole 75% WP", "dosage": "0.6 g/litre water", "frequency": "At 50% panicle emergence + 10 days later"},
            {"name": "Hexaconazole 5% EC", "dosage": "2 ml/litre water", "frequency": "2 sprays at panicle emergence"},
            {"name": "Azoxystrobin 23% SC", "dosage": "1 ml/litre water", "frequency": "At flag leaf and panicle emergence"},
        ],
        "organic": [
            {"name": "Pseudomonas fluorescens + Trichoderma viride", "dosage": "5 g each per litre", "frequency": "At flag leaf; repeat at panicle emergence"},
            {"name": "Potassium Silicate Solution", "dosage": "2 g/litre water", "frequency": "At flag leaf stage"},
        ],
        "tips": ["Preventive spray at booting is most effective.", "Avoid water stress at panicle initiation.", "No high nitrogen at heading.", "Spray before humid nights."],
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
            {"name": "Beauveria bassiana", "dosage": "5 ml/litre water", "frequency": "Spray in evening; repeat after 7 days"},
        ],
        "tips": ["Cut and destroy affected leaves.", "Avoid close planting density.", "Flood field to drown grubs.", "Conserve natural predators."],
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
        ],
        "tips": ["Reduce hill density.", "Maintain 2.5 cm water level.", "Avoid excess nitrogen.", "Remove infected debris after harvest."],
    },
    "tungro": {
        "display_name": "Tungro", "severity_level": 4, "severity_label": "Critical",
        "description": "Caused by Rice Tungro Virus via green leafhopper. Up to 100% yield loss.",
        "action": "URGENT — Control green leafhopper immediately. Remove and destroy infected plants.",
        "chemical": [
            {"name": "Imidacloprid 70% WS", "dosage": "10 g/kg seed", "frequency": "Seed treatment before sowing"},
            {"name": "Buprofezin 25% SC", "dosage": "1 ml/litre water", "frequency": "When leafhopper count > 2 per hill"},
            {"name": "Thiamethoxam 25% WG", "dosage": "0.4 g/litre water", "frequency": "2 sprays at 10–14 day intervals"},
        ],
        "organic": [
            {"name": "Neem Oil 5%", "dosage": "5 ml/litre water", "frequency": "Every 5 days to repel leafhoppers"},
            {"name": "Yellow Sticky Traps", "dosage": "20 traps/ha", "frequency": "Replace every 2 weeks"},
        ],
        "tips": ["Use tungro-resistant varieties.", "Synchronise planting with neighbours.", "Remove and burn infected plants.", "Avoid planting near infected fields."],
    },
}

SEVERITY_CONFIG = {
    0: {"color": "#22c55e", "bg": "#f0fdf4", "border": "#86efac", "icon": "✓", "label": "No Disease"},
    1: {"color": "#eab308", "bg": "#fefce8", "border": "#fde047", "icon": "!", "label": "Low"},
    2: {"color": "#f97316", "bg": "#fff7ed", "border": "#fdba74", "icon": "!!", "label": "Moderate"},
    3: {"color": "#ef4444", "bg": "#fef2f2", "border": "#fca5a5", "icon": "!!!", "label": "High"},
    4: {"color": "#7c3aed", "bg": "#f5f3ff", "border": "#c4b5fd", "icon": "⚠", "label": "Critical"},
}
URGENCY = {0: "None", 1: "Within 1 week", 2: "Within 2–3 days", 3: "Within 24 hours", 4: "Immediately"}

# ── Auth Routes ───────────────────────────────────────────────────────────────
@app.route("/login", methods=["GET", "POST"])
@limiter.limit("10 per minute")  # FIX #7: Brute-force rate limiting
def login():
    if request.method == "POST":
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "").strip()
        name = request.form.get("name", "").strip()
        phone = request.form.get("phone", "").strip()  # FIX #4: collect phone

        # Verify role is valid
        role = request.form.get("role", "farmer").strip()
        if role not in ("farmer", "sprayer", "shop", "admin"):
            role = "farmer"  # silently downgrade any invalid role

        if not email or not password:
            return render_template("login.html", error="Email and password are required.", success=None)

        if len(password) < 6:
            return render_template("login.html", error="Password must be at least 6 characters.", success=None)

        existing = users_col.find_one({"email": email})
        if existing:
            # FIX #1: Use secure password check (hashed comparison)
            if check_password_hash(existing["password"], password) or existing["password"] == password:
                if existing.get("is_approved") is False:
                    return render_template("login.html", error="Your account is pending administrator approval. Please wait for an admin to approve your account.", success=None)
                
                if existing["password"] == password:
                    # Fallback for plain-text passwords: upgrade to hash
                    users_col.update_one(
                        {"_id": existing["_id"]},
                        {"$set": {"password": generate_password_hash(password)}}
                    )
                session.permanent = True  # FIX #8: apply timeout to session
                session["user"] = {"email": email, "role": existing["role"], "name": existing["name"]}
                return redirect(url_for("dashboard"))
            else:
                return render_template("login.html", error="Wrong password. Please try again.", success=None)
        else:
            # New user registration
            if role == "admin":
                return render_template("login.html", error="Admin registration is not allowed. Please contact support.", success=None)
            
            requires_approval = role in ["sprayer", "shop"]
            
            user_name = name if name else email.split("@")[0].replace(".", " ").title()
            new_user = {
                "email": email,
                "password": generate_password_hash(password),  # FIX #1: hash password
                "role": role,
                "name": user_name,
                "phone": phone,  # FIX #4: save phone number
                "is_approved": not requires_approval,
                "created_at": datetime.now()
            }
            users_col.insert_one(new_user)
            
            if requires_approval:
                return render_template("login.html", success=f"Registration successful! Your {role.title()} account is pending administrator approval.", error=None)
            
            session.permanent = True  # FIX #8: apply timeout
            session["user"] = {"email": email, "role": role, "name": user_name}
            return redirect(url_for("dashboard"))

    return render_template("login.html", error=None, success=None)

@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))

@app.route("/dashboard")
@login_required
def dashboard():
    role = session["user"]["role"]
    if role == "admin":
        all_users = list(users_col.find({}, {"_id": 0}))
        active_users = [u for u in all_users if u.get("is_approved") is not False]
        pending_users = [u for u in all_users if u.get("is_approved") is False]
        
        all_history = list(history_col.find({}, {"_id": 0}).sort("timestamp", -1).limit(100))
        all_bookings = list(bookings_col.find({}, {"_id": 0}).sort("created_at", -1).limit(100))
        return render_template("admin_dashboard.html", user=session["user"], history=all_history, users=active_users, pending_users=pending_users, bookings=all_bookings)
    elif role == "farmer":
        return render_template("farmer_dashboard.html", user=session["user"])
    elif role == "sprayer":
        bookings = list(bookings_col.find(
            {"sprayer_email": session["user"]["email"]},
            {"_id": 0}
        ).sort("created_at", -1).limit(50))
        return render_template("sprayer_dashboard.html", user=session["user"], bookings=bookings)
    elif role == "shop":
        return render_template("shop_dashboard.html", user=session["user"])
    return redirect(url_for("login"))

# ── Main Routes ───────────────────────────────────────────────────────────────
@app.route("/")
def index():
    if "user" not in session:
        return redirect(url_for("login"))
    return redirect(url_for("dashboard"))

@app.route("/detect")
@login_required
def detect():
    if model is None:
        return render_template("error.html", error="Model not loaded. Please contact administrator.")
    return render_template("index.html", user=session["user"])

@app.route("/predict", methods=["POST"])
@login_required
def predict():
    if model is None:
        return jsonify({"error": "Model not loaded. Please try again later."}), 503
    
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

    try:
        preds = predict_fast(filepath)
        if preds is None:
            return jsonify({"error": "Prediction failed"}), 500
        
        idx = int(np.argmax(preds))
        
        # Ensure index is within bounds
        if idx >= len(CLASS_NAMES):
            return jsonify({"error": f"Invalid prediction index {idx}"}), 500
            
        label = CLASS_NAMES[idx]
        confidence = round(float(preds[idx]) * 100, 1)

        # Get top 3 predictions
        top3_indices = np.argsort(preds)[::-1][:3]
        top3 = []
        for i in top3_indices:
            if i < len(CLASS_NAMES):
                top3.append({
                    "label": CLASS_NAMES[i],
                    "display": CLASS_NAMES[i].replace("_", " ").title(),
                    "prob": round(float(preds[i]) * 100, 1)
                })

        # Get recommendation (with fallback)
        rec = RECOMMENDATIONS.get(label, RECOMMENDATIONS["healthy"])
        sev = SEVERITY_CONFIG.get(rec["severity_level"], SEVERITY_CONFIG[0])

        # Get farmer's phone number from database if available
        farmer = users_col.find_one({"email": session["user"]["email"]})
        farmer_phone = farmer.get("phone", "") if farmer else ""

        # Save to history
        history_col.insert_one({
            "user": session["user"]["name"],
            "email": session["user"]["email"],
            "role": session["user"]["role"],
            "disease": rec["display_name"],
            "confidence": confidence,
            "severity": rec["severity_label"],
            "image": filename,
            "timestamp": datetime.now()
        })

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
            "urgency": URGENCY.get(rec["severity_level"], "Unknown"),
            "description": rec["description"],
            "action": rec["action"],
            "chemical": rec["chemical"],
            "organic": rec["organic"],
            "tips": rec["tips"],
            "top3": top3,
            "farmer_phone": farmer_phone,
            "farmer_name": session["user"]["name"],
            "farmer_email": session["user"]["email"],
        })
        
    except Exception as e:
        print(f"[ERROR] Prediction failed: {e}")
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500

# ── Debug endpoint to test model ──────────────────────────────────────────────
@app.route("/debug/model-test", methods=["POST"])
@login_required
def debug_model_test():
    """Debug endpoint to check model predictions"""
    if model is None:
        return jsonify({"error": "Model not loaded"}), 503
    
    if "image" not in request.files:
        return jsonify({"error": "No image"}), 400
    
    file = request.files["image"]
    filename = f"debug_{uuid.uuid4().hex}.jpg"
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)
    
    try:
        preds = predict_fast(filepath)
        
        # Get all predictions
        top5_idx = np.argsort(preds)[::-1][:5]
        results = []
        for idx in top5_idx:
            if idx < len(CLASS_NAMES):
                results.append({
                    "class": CLASS_NAMES[idx],
                    "confidence": round(float(preds[idx]) * 100, 2)
                })
        
        return jsonify({
            "top_predictions": results,
            "all_probabilities": [round(float(p), 4) for p in preds],
            "class_names": CLASS_NAMES,
            "model_output_shape": preds.shape[0]
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── Sprayer API ───────────────────────────────────────────────────────────────
@app.route("/api/sprayers")
@login_required
def get_sprayers():
    """Return nearest sprayers sorted by distance. Pass ?lat=&lng= for geo-sorting."""
    try:
        farmer_lat = float(request.args.get("lat", 0))
        farmer_lng = float(request.args.get("lng", 0))
    except (TypeError, ValueError):
        farmer_lat = farmer_lng = 0

    raw = list(users_col.find({"role": "sprayer", "is_approved": {"$ne": False}}, {"_id": 0, "password": 0}))

    results = []
    for s in raw:
        s_lat = s.get("lat", 0)
        s_lng = s.get("lng", 0)
        dist_m = haversine(farmer_lat, farmer_lng, s_lat, s_lng) if (farmer_lat and farmer_lng) else 0
        s["distance_m"] = round(dist_m)
        s["distance_fmt"] = fmt_dist(dist_m)
        # initials for avatar
        parts = s["name"].split()
        s["initials"] = (parts[0][0] + parts[-1][0]).upper() if len(parts) >= 2 else parts[0][:2].upper()
        results.append(s)

    results.sort(key=lambda x: x["distance_m"])
    return jsonify({"sprayers": results[:10]})

@app.route("/api/book-sprayer", methods=["POST"])
@login_required
def book_sprayer():
    """Create a new booking from farmer to sprayer."""
    data = request.get_json()
    if not data:
        return jsonify({"error": "No data provided"}), 400

    required = ["sprayer_email", "sprayer_name", "date", "time_slot", "acres"]
    for field in required:
        if not data.get(field):
            return jsonify({"error": f"Missing field: {field}"}), 400

    ref_number = f"PG-{uuid.uuid4().hex[:6].upper()}"
    
    # Get farmer's location if provided
    farmer_lat = data.get("farmer_lat", None)
    farmer_lng = data.get("farmer_lng", None)
    
    booking = {
        "ref": ref_number,
        "farmer_name": session["user"]["name"],
        "farmer_email": session["user"]["email"],
        "farmer_phone": data.get("farmer_phone", ""),
        "farmer_lat": farmer_lat,
        "farmer_lng": farmer_lng,
        "sprayer_email": data["sprayer_email"],
        "sprayer_name": data["sprayer_name"],
        "disease": data.get("disease", ""),
        "pesticides": data.get("pesticides", ""),
        "date": data["date"],
        "time_slot": data["time_slot"],
        "acres": float(data["acres"]),
        "notes": data.get("notes", ""),
        "status": "pending",
        "created_at": datetime.now()
    }
    bookings_col.insert_one(booking)
    print(f"[INFO] Booking {ref_number} created → sprayer: {data['sprayer_name']}")
    return jsonify({"success": True, "ref": ref_number, "sprayer_name": data["sprayer_name"]})

@app.route("/api/sprayer-bookings")
@login_required
def sprayer_bookings():
    """Return bookings for the logged-in sprayer."""
    if session["user"]["role"] != "sprayer":
        return jsonify({"error": "Unauthorized"}), 403
    
    bookings = list(bookings_col.find(
        {"sprayer_email": session["user"]["email"]},
        {"_id": 0}
    ).sort("created_at", -1).limit(50))
    
    # Convert datetime to string for JSON
    for b in bookings:
        if "created_at" in b:
            b["created_at"] = b["created_at"].isoformat()
    
    return jsonify({"bookings": bookings})

@app.route("/api/booking-action", methods=["POST"])
@login_required
def booking_action():
    """Sprayer accepts or declines a booking."""
    data = request.get_json()
    ref = data.get("ref")
    action = data.get("action")  # 'accept' or 'decline'
    if not ref or action not in ("accept", "decline"):
        return jsonify({"error": "Invalid request"}), 400

    # Only the assigned sprayer can act on it
    booking = bookings_col.find_one({"ref": ref})
    if not booking:
        return jsonify({"error": "Booking not found"}), 404
    if booking["sprayer_email"] != session["user"]["email"]:
        return jsonify({"error": "Unauthorised"}), 403

    new_status = "accepted" if action == "accept" else "declined"
    bookings_col.update_one({"ref": ref}, {"$set": {"status": new_status, "updated_at": datetime.now()}})
    return jsonify({"success": True, "status": new_status})

@app.route("/api/my-bookings")
@login_required
def my_bookings():
    """Return bookings for the logged-in farmer."""
    bookings = list(bookings_col.find(
        {"farmer_email": session["user"]["email"]},
        {"_id": 0}
    ).sort("created_at", -1).limit(20))
    
    for b in bookings:
        if "created_at" in b:
            b["created_at"] = b["created_at"].isoformat()
    
    return jsonify({"bookings": bookings})

@app.route("/api/farmer-location", methods=["POST"])
@login_required
def save_farmer_location():
    """Save farmer's GPS location to their profile."""
    data = request.get_json()
    lat = data.get("lat")
    lng = data.get("lng")
    
    if lat and lng:
        users_col.update_one(
            {"email": session["user"]["email"]},
            {"$set": {"lat": lat, "lng": lng}}
        )
        return jsonify({"success": True})
    return jsonify({"error": "Invalid coordinates"}), 400

@app.route("/map")
@login_required
def map_page():
    return render_template("map.html")

@app.route("/api/admin/approve-user", methods=["POST"])
@login_required
def approve_user():
    """Admin endpoint to approve pending users and send email notification."""
    if session["user"]["role"] != "admin":
        return jsonify({"error": "Unauthorized"}), 403
    
    data = request.get_json()
    email = data.get("email")
    if not email:
        return jsonify({"error": "Email required"}), 400

    user = users_col.find_one({"email": email})
    if not user:
        return jsonify({"error": "User not found"}), 404

    users_col.update_one({"email": email}, {"$set": {"is_approved": True, "approval_notified": False}})

    # ── Send free Gmail notification email ────────────────────────────────────
    try:
        user_name = user.get("name", "User")
        user_role = user.get("role", "user").title()
        msg = Message(
            subject="✅ Your PaddyGuard Account Has Been Approved!",
            recipients=[email]
        )
        msg.html = f"""
        <div style="font-family:sans-serif;max-width:480px;margin:auto;padding:24px;background:#f0fdf4;border-radius:12px;border:1px solid #bbf7d0">
          <h2 style="color:#15803d">🌾 PaddyGuard AI</h2>
          <h3 style="color:#111">Your account has been approved!</h3>
          <p style="color:#444">Hello <strong>{user_name}</strong>,</p>
          <p style="color:#444">Great news! Your <strong>{user_role}</strong> account on PaddyGuard AI has been <strong>approved</strong> by the administrator.</p>
          <p style="color:#444">You can now log in and start using the platform:</p>
          <a href="http://127.0.0.1:5000/login" style="display:inline-block;margin:16px 0;padding:12px 24px;background:#16a34a;color:#fff;text-decoration:none;border-radius:8px;font-weight:700">Login to PaddyGuard →</a>
          <p style="color:#888;font-size:12px;margin-top:16px">This is an automated message from PaddyGuard AI. Please do not reply.</p>
        </div>
        """
        mail.send(msg)
        print(f"[INFO] Approval email sent to {email}")
        email_sent = True
    except Exception as e:
        print(f"[WARN] Could not send approval email to {email}: {e}")
        email_sent = False

    return jsonify({"success": True, "email_sent": email_sent})


@app.route("/api/check-status", methods=["POST"])
def check_approval_status():
    """Public endpoint — lets pending sprayer/shop users check if admin approved them."""
    data = request.get_json()
    email = (data.get("email") or "").strip().lower()
    if not email:
        return jsonify({"error": "Email required"}), 400

    user = users_col.find_one({"email": email}, {"_id": 0, "password": 0})
    if not user:
        return jsonify({"found": False})

    role = user.get("role", "")

    # This checker is ONLY for sprayer/shop accounts that need admin approval
    if role not in ["sprayer", "shop"]:
        return jsonify({
            "found": True,
            "not_applicable": True,
            "role": role,
            "name": user.get("name", ""),
        })

    # Only return approved=True if admin has EXPLICITLY set is_approved=True in DB
    is_approved = user.get("is_approved") is True  # strict check — must be exactly True

    return jsonify({
        "found": True,
        "not_applicable": False,
        "approved": is_approved,
        "name": user.get("name", ""),
        "role": role,
    })

# ... (all your existing code above)

# ========== ADD THE NEW ROUTES HERE ==========
@app.route("/dashboard-data")
@login_required
def dashboard_data():
    """Get current user data for dashboard"""
    return jsonify({"user": session["user"]})

@app.route("/api/user-history")
@login_required
def user_history():
    """Get prediction history for logged-in user"""
    history = list(history_col.find(
        {"email": session["user"]["email"]},
        {"_id": 0}
    ).sort("timestamp", -1).limit(50))
    
    # Convert datetime to string
    for h in history:
        if "timestamp" in h:
            h["timestamp"] = h["timestamp"].isoformat()
    
    return jsonify(history)
# ========== END OF NEW ROUTES ==========

# ── AI Chatbot API (Context-Aware) ───────────────────────────────────────────
@app.route("/api/chat", methods=["POST"])
@login_required
def chat():
    """AI chatbot powered by OpenRouter — fully context-aware."""
    data = request.get_json()
    messages = data.get("messages", [])
    ctx = data.get("context", {})

    if not messages:
        return jsonify({"error": "No messages provided"}), 400

    OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
    if not OPENROUTER_API_KEY:
        return jsonify({"error": "Chat API key not configured"}), 500

    # ── Build dynamic context string ──────────────────
    user_name  = ctx.get("name", session["user"].get("name", "User"))
    user_role  = ctx.get("role", session["user"].get("role", "farmer"))
    user_email = ctx.get("email", session["user"].get("email", ""))
    current_page   = ctx.get("page", "PaddyGuard")
    last_diagnosis = ctx.get("lastDiagnosis", None)

    PAGE_DESCRIPTIONS = {
        "🔬 Disease Detection": (
            "The user is on the Disease Detection page where they upload paddy leaf photos "
            "to get AI-powered disease diagnosis. Help them understand their result, explain "
            "the detected disease, recommend treatments, and guide them through the upload process."
        ),
        "📊 Dashboard": (
            "The user is on their main Dashboard. Help them navigate their scan history, "
            "understand their disease reports, book a sprayer, or interpret their results."
        ),
        "🗺️ Sprayer Map": (
            "The user is on the Sprayer Map page looking to find and book a local pesticide sprayer. "
            "Help them choose the right sprayer, understand the booking process, and suggest which "
            "pesticides to ask the sprayer to apply based on their detected disease."
        ),
        "🧑‍🌾 Sprayer Portal": (
            "The user is a sprayer viewing their booking portal. Help them with pesticide dosages, "
            "application schedules, safety precautions, and managing their bookings."
        ),
        "⚙️ Admin Panel": (
            "The user is an admin viewing the system dashboard. Help them understand disease trends, "
            "user activity, booking status, and platform analytics."
        ),
        "🛒 Agri Shop": (
            "The user is an agri-shop owner. Help them with which pesticides to stock based on "
            "common diseases, product details, dosages, and organic alternatives to recommend to farmers."
        ),
    }

    ROLE_CONTEXT = {
        "farmer":  "They are a farmer trying to protect their paddy crop.",
        "sprayer": "They are a pesticide sprayer who applies treatments for farmers.",
        "admin":   "They are a platform administrator managing the PaddyGuard system.",
        "shop":    "They are an agri-shop owner supplying pesticides to farmers.",
    }

    user_language = ctx.get("languageName", "English")
    import re
    msg_content = messages[-1]['content'] if messages else ''
    has_telugu = bool(re.search(r'[\u0C00-\u0C7F]', msg_content))
    lang_instruction = (
        "LANGUAGE RULE: The user's preferred language is Telugu. "
        "You MUST respond ONLY in Telugu (తెలుగు) script. Do not use English in your reply. "
        "Use simple, everyday Telugu that a farmer can easily understand. "
        "For chemical/pesticide names and dosages, you may write them in English within your Telugu response."
    ) if (user_language == "Telugu" or has_telugu) else (
        "LANGUAGE RULE: Respond in English, unless the user writes in Telugu, then detect it and respond in Telugu."
    )
    page_desc  = PAGE_DESCRIPTIONS.get(current_page, "The user is on the PaddyGuard platform.")
    role_desc  = ROLE_CONTEXT.get(user_role, "They are a PaddyGuard user.")
    diag_note  = f" Their most recent AI diagnosis result was: **{last_diagnosis}**." if last_diagnosis else ""

    system_content = (
        f"You are PaddyGuard AI Assistant — a smart, friendly agricultural chatbot embedded in the PaddyGuard platform.\n\n"
        f"{lang_instruction}\n\n"
        f"CURRENT USER CONTEXT:\n"
        f"- Name: {user_name}\n"
        f"- Role: {user_role}\n"
        f"- Currently on: {current_page}\n"
        f"- Page context: {page_desc}\n"
        f"- User description: {role_desc}{diag_note}\n\n"
        f"INSTRUCTIONS:\n"
        f"- Always tailor your response to the current page and user role above.\n"
        f"- If the user is on the Disease Detection page, focus on helping them understand and act on their results.\n"
        f"- If the user is on the Map page, focus on sprayer booking and treatment logistics.\n"
        f"- If the user is a sprayer, focus on dosages, schedules, and safety.\n"
        f"- If the user is a shop owner, focus on product recommendations and stock.\n"
        f"- If the user asks about a specific disease, provide: symptoms, severity, chemical treatments with dosages, organic alternatives, and prevention tips.\n"
        f"- Always be concise, use simple farmer-friendly language, and use bullet points for clarity.\n"
        f"- If asked about non-agricultural topics, politely redirect back to paddy farming.\n"
        f"- Address the user by their name ({user_name}) occasionally to make it personal.\n"
        f"\nYou know these diseases: Leaf Blast, Neck Blast, Bacterial Leaf Blight, Brown Spot, "
        f"Sheath Blight, Tungro, Rice Hispa, Leaf Scald, Narrow Brown Spot, and Healthy crops."
    )

    system_prompt = {"role": "system", "content": system_content}

    try:
        resp = http_requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://paddyguard.ai",
                "X-Title": "PaddyGuard AI"
            },
            json={
                "model": "openai/gpt-4o-mini",
                "messages": [system_prompt] + messages,
                "max_tokens": 700,
                "temperature": 0.7
            },
            timeout=30
        )
        resp.raise_for_status()
        reply = resp.json()["choices"][0]["message"]["content"]
        
        # Safely print to terminal to avoid UnicodeEncodeError on Windows
        log_msg = f"[CHAT] {user_name} ({user_role}) on '{current_page}': {messages[-1]['content'][:60]}..."
        print(log_msg.encode('ascii', 'ignore').decode('ascii'))
        
        return jsonify({"reply": reply})
    except Exception as e:
        import traceback
        # Safely print the error
        print(f"[ERROR] Chat failed: {str(e)}".encode('ascii', 'ignore').decode('ascii'))
        traceback.print_exc()
        return jsonify({"error": "Chat service unavailable. Please try again."}), 500

@app.route("/api/tts")
def proxy_tts():
    """Proxy Google Translate TTS to avoid browser CORS/referrer restrictions."""
    text = request.args.get("text", "").strip()
    lang = request.args.get("lang", "te") # default telugu
    if not text:
        return "No text", 400
    try:
        import urllib.parse
        from flask import Response
        limited_text = text[:190]
        url = f"https://translate.googleapis.com/translate_tts?ie=UTF-8&tl={lang}&client=tw-ob&q={urllib.parse.quote(limited_text)}"
        resp = http_requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
        return Response(resp.content, mimetype="audio/mpeg")
    except Exception as e:
        print(f"[TTS Error] {e}")
        return "Audio failed", 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True, use_reloader=False, port=5000)  # Disabled reloader to prevent keras crashes