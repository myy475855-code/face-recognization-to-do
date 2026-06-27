"""
Face-recognition login for a To-Do app — built on OpenCV's LBPH recognizer.

Why LBPH instead of `face_recognition`/dlib
--------------------------------------------
`face_recognition` needs `dlib`, which has to compile from source (cmake + a
C++ toolchain) and pulls in a large separate model package. That trips up a
lot of setups. OpenCV's LBPHFaceRecognizer ships as part of the
`opencv-contrib-python` wheel — no compiling, no extra downloads — at the
cost of being a bit less accurate, especially across lighting/angle changes.
We compensate by capturing several frames per user at registration instead
of just one.

Flow
----
Register : username + password (fallback factor) + a short burst of webcam
            frames -> each frame's face is detected (Haar cascade), cropped,
            resized, and saved as a training sample -> the LBPH model is
            retrained on all samples across all users.
Login     : webcam captures a frame every couple of seconds -> face detected
            and cropped -> LBPHFaceRecognizer.predict() returns the closest
            known label + a confidence (distance: lower = closer match).
Dashboard : plain CRUD to-do list scoped to session["user_id"].
"""

import base64
import io
import json
import os
from datetime import datetime
from functools import wraps

import cv2
import numpy as np
from flask import Flask, flash, jsonify, redirect, render_template, request, session, url_for
from flask_sqlalchemy import SQLAlchemy
from PIL import Image
from werkzeug.security import check_password_hash, generate_password_hash

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
FACES_DIR = os.path.join(BASE_DIR, "faces")
SAMPLES_DIR = os.path.join(FACES_DIR, "samples")
MODEL_PATH = os.path.join(FACES_DIR, "lbph_model.xml")
LABELS_PATH = os.path.join(FACES_DIR, "labels.json")
os.makedirs(SAMPLES_DIR, exist_ok=True)

app = Flask(__name__)
app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY", "dev-secret-change-me")
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///" + os.path.join(BASE_DIR, "database.db")
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

db = SQLAlchemy(app)

IMG_SIZE = (200, 200)
MIN_USABLE_SAMPLES = 3  # how many of the captured burst frames must have a usable face

# LBPH "confidence" is really a distance: lower = more similar.
# Tune by testing with your own face under your actual lighting.
LBPH_CONFIDENCE_THRESHOLD = 70

_face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
FACE_DETECTOR_AVAILABLE = not _face_cascade.empty()

try:
    cv2.face.LBPHFaceRecognizer_create  # noqa: B018 - attribute-existence probe
    LBPH_AVAILABLE = True
except AttributeError:
    LBPH_AVAILABLE = False  # opencv-contrib-python isn't installed (plain opencv-python lacks cv2.face)


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password_hash = db.Column(db.String(255), nullable=False)
    face_sample_dir = db.Column(db.String(255), nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    tasks = db.relationship("Task", backref="owner", cascade="all, delete-orphan")

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)


class Task(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=False)
    task = db.Column(db.String(255), nullable=False)
    completed = db.Column(db.Boolean, default=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)


# ---------------------------------------------------------------------------
# Face helpers
# ---------------------------------------------------------------------------
def decode_base64_image(data_url):
    """Turn a `data:image/jpeg;base64,...` string into an RGB numpy array."""
    _, encoded = data_url.split(",", 1)
    binary = base64.b64decode(encoded)
    image = Image.open(io.BytesIO(binary)).convert("RGB")
    return np.array(image)


def detect_and_crop_face(image_array):
    """Detect a single face, return it as a normalized grayscale crop."""
    if not FACE_DETECTOR_AVAILABLE:
        return None, "Face detector failed to load on the server."

    gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
    faces = _face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80))

    if len(faces) == 0:
        return None, "No face detected. Center your face in the frame and try again."
    if len(faces) > 1:
        return None, "Multiple faces detected. Only one person should be in frame."

    x, y, w, h = faces[0]
    face_img = gray[y:y + h, x:x + w]
    face_img = cv2.resize(face_img, IMG_SIZE)
    face_img = cv2.equalizeHist(face_img)  # softens lighting differences between frames
    return face_img, None


def retrain_model():
    """Rebuild the LBPH model from every saved sample across all users."""
    images, labels, label_map = [], [], {}
    next_label = 0

    for user_dir in sorted(os.listdir(SAMPLES_DIR)):
        full_path = os.path.join(SAMPLES_DIR, user_dir)
        if not user_dir.startswith("user_") or not os.path.isdir(full_path):
            continue
        try:
            user_id = int(user_dir.split("_")[1])
        except (IndexError, ValueError):
            continue

        sample_files = [f for f in os.listdir(full_path) if f.endswith(".png")]
        if not sample_files:
            continue

        label = next_label
        next_label += 1
        label_map[str(label)] = user_id

        for fname in sample_files:
            img = cv2.imread(os.path.join(full_path, fname), cv2.IMREAD_GRAYSCALE)
            if img is not None:
                images.append(img)
                labels.append(label)

    if not images:
        return False

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.train(images, np.array(labels))
    recognizer.write(MODEL_PATH)
    with open(LABELS_PATH, "w") as f:
        json.dump(label_map, f)
    return True


def recognize_face(face_img):
    """Return (user_id, confidence) for the closest match, or (None, None)."""
    if not os.path.exists(MODEL_PATH) or not os.path.exists(LABELS_PATH):
        return None, None

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(MODEL_PATH)
    with open(LABELS_PATH) as f:
        label_map = json.load(f)

    label, confidence = recognizer.predict(face_img)
    user_id = label_map.get(str(label))
    return user_id, confidence


# ---------------------------------------------------------------------------
# Auth guard
# ---------------------------------------------------------------------------
def login_required(view):
    @wraps(view)
    def wrapped(*args, **kwargs):
        if "user_id" not in session:
            return redirect(url_for("login"))
        return view(*args, **kwargs)

    return wrapped


# ---------------------------------------------------------------------------
# Routes - pages
# ---------------------------------------------------------------------------
@app.route("/")
def index():
    return redirect(url_for("dashboard") if "user_id" in session else url_for("login"))


@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "GET":
        return render_template("register.html")

    username = request.form.get("username", "").strip()
    password = request.form.get("password", "")
    image_data_raw = request.form.get("image_data")

    if not username or not password:
        flash("Username and password are both required.", "error")
        return redirect(url_for("register"))

    if User.query.filter_by(username=username).first():
        flash("That username is already taken.", "error")
        return redirect(url_for("register"))

    if not image_data_raw:
        flash("Capture your face before submitting.", "error")
        return redirect(url_for("register"))

    if not LBPH_AVAILABLE:
        flash("Face recognition isn't available on the server yet (opencv-contrib-python missing).", "error")
        return redirect(url_for("register"))

    try:
        frames = json.loads(image_data_raw)
        if not isinstance(frames, list) or not frames:
            raise ValueError
    except (ValueError, TypeError):
        flash("Could not read the captured images. Try again.", "error")
        return redirect(url_for("register"))

    crops = []
    for frame in frames:
        try:
            arr = decode_base64_image(frame)
        except Exception:
            continue
        face_img, _error = detect_and_crop_face(arr)
        if face_img is not None:
            crops.append(face_img)

    if len(crops) < MIN_USABLE_SAMPLES:
        flash(
            f"Only got a clear face in {len(crops)} of {len(frames)} captures "
            f"(need {MIN_USABLE_SAMPLES}+). Try again with better, even lighting.",
            "error",
        )
        return redirect(url_for("register"))

    user = User(username=username)
    user.set_password(password)
    db.session.add(user)
    db.session.commit()

    user_sample_dir = os.path.join(SAMPLES_DIR, f"user_{user.id}")
    os.makedirs(user_sample_dir, exist_ok=True)
    for i, crop in enumerate(crops):
        cv2.imwrite(os.path.join(user_sample_dir, f"sample_{i}.png"), crop)

    user.face_sample_dir = user_sample_dir
    db.session.commit()

    if not retrain_model():
        flash("Saved your samples, but training the recognizer failed.", "error")
        return redirect(url_for("register"))

    flash("Account created. Scan your face to sign in.", "success")
    return redirect(url_for("login"))


@app.route("/login", methods=["GET"])
def login():
    return render_template("login.html")


@app.route("/api/face-login", methods=["POST"])
def api_face_login():
    image_data = (request.get_json(silent=True) or {}).get("image_data")
    if not image_data:
        return jsonify({"success": False, "message": "No frame received."}), 400

    if not LBPH_AVAILABLE:
        return jsonify({"success": False, "message": "Face recognition isn't available on the server.", "retry": False}), 200

    try:
        image_array = decode_base64_image(image_data)
    except Exception:
        return jsonify({"success": False, "message": "Invalid image data."}), 400

    face_img, error = detect_and_crop_face(image_array)
    if error:
        return jsonify({"success": False, "message": error, "retry": True}), 200

    user_id, confidence = recognize_face(face_img)
    if user_id is None:
        return jsonify({"success": False, "message": "No registered faces yet.", "retry": False}), 200

    if confidence <= LBPH_CONFIDENCE_THRESHOLD:
        user = User.query.get(user_id)
        if user:
            session["user_id"] = user.id
            session["username"] = user.username
            return jsonify({
                "success": True,
                "message": f"Welcome back, {user.username}.",
                "redirect": url_for("dashboard"),
            })

    return jsonify({"success": False, "message": "Face not recognized.", "retry": True}), 200


@app.route("/login-password", methods=["POST"])
def login_password():
    """Fallback path for when the camera isn't available."""
    username = request.form.get("username", "").strip()
    password = request.form.get("password", "")
    user = User.query.filter_by(username=username).first()
    if user and user.check_password(password):
        session["user_id"] = user.id
        session["username"] = user.username
        return redirect(url_for("dashboard"))
    flash("Invalid username or password.", "error")
    return redirect(url_for("login"))


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))


@app.route("/dashboard")
@login_required
def dashboard():
    tasks = Task.query.filter_by(user_id=session["user_id"]).order_by(Task.created_at.desc()).all()
    return render_template("dashboard.html", tasks=tasks, username=session.get("username"))


@app.route("/tasks", methods=["POST"])
@login_required
def add_task():
    task_text = request.form.get("task", "").strip()
    if task_text:
        db.session.add(Task(user_id=session["user_id"], task=task_text))
        db.session.commit()
    return redirect(url_for("dashboard"))


@app.route("/tasks/<int:task_id>/toggle", methods=["POST"])
@login_required
def toggle_task(task_id):
    task = Task.query.filter_by(id=task_id, user_id=session["user_id"]).first_or_404()
    task.completed = not task.completed
    db.session.commit()
    return redirect(url_for("dashboard"))


@app.route("/tasks/<int:task_id>/delete", methods=["POST"])
@login_required
def delete_task(task_id):
    task = Task.query.filter_by(id=task_id, user_id=session["user_id"]).first_or_404()
    db.session.delete(task)
    db.session.commit()
    return redirect(url_for("dashboard"))


if __name__ == "__main__":
    with app.app_context():
        db.create_all()
    app.run(debug=True)
