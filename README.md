# Scan To-Do — face-recognition login for a to-do app

A working Flask app where signing in happens by looking at your webcam instead
of (or in addition to) typing a password.

**Face matching uses OpenCV's built-in LBPH recognizer**, not
`face_recognition`/`dlib`. `dlib` needs to compile from source (cmake, a C++
toolchain) and pulls in a separate large model package — that trips up a lot
of setups. `opencv-contrib-python` ships as a normal prebuilt wheel with
everything needed (Haar cascade face detector + LBPHFaceRecognizer), so
there's nothing to compile.

**Trade-off:** LBPH is less accurate than dlib's deep-learning face
embeddings, especially across lighting/angle changes. This project
compensates by capturing a short burst of 5 frames at registration instead
of one still image, which meaningfully helps.

## How it works

1. **Register** — pick a username + backup password, then the camera grabs a
   burst of 5 frames. Each frame's face is detected with a Haar cascade,
   cropped, resized to 200×200, and histogram-equalized. The crops are saved
   as that user's training samples, and the LBPH model is retrained on every
   sample across all users.
2. **Sign in** — the login page opens your camera and sends one frame to
   `/api/face-login` roughly every 1.6 seconds. The server detects + crops
   the face the same way, then calls `LBPHFaceRecognizer.predict()`, which
   returns the closest known user and a confidence score (this is really a
   *distance* — lower means more similar). A confidence at or below
   `LBPH_CONFIDENCE_THRESHOLD` logs you in.
3. **Dashboard** — a normal to-do list, scoped to your `user_id` via the
   Flask session. Add / complete / delete tasks.
4. A **password fallback** (`/login-password`) exists for when a camera
   isn't available — same password set at registration.

## Project layout

```
face_todo_app/
├── app.py                  Flask app: routes, models, LBPH face-matching logic
├── requirements.txt
├── faces/
│   ├── samples/user_<id>/  cropped training images per user (gitignore in real use)
│   ├── lbph_model.xml      trained recognizer (rebuilt on every registration)
│   └── labels.json         maps internal LBPH label -> user_id
├── database.db              created automatically on first run (SQLite)
├── templates/
│   ├── base.html
│   ├── register.html        webcam burst-capture + signup form
│   ├── login.html           live face-scan + password fallback
│   └── dashboard.html       task list
└── static/
    ├── css/style.css
    └── js/
        ├── camera.js         shared getUserMedia / frame-capture helpers
        ├── register.js       5-frame burst capture flow
        └── login.js          continuous scan-and-POST loop
```

## Setup

```bash
cd face_todo_app
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

That's it — `opencv-contrib-python` is a prebuilt wheel on Linux, macOS, and
Windows, so there's no compiler/cmake step like `dlib` requires.

> **Note:** `app.py` checks for `cv2.face` and the Haar cascade file at
> startup and sets `LBPH_AVAILABLE` / `FACE_DETECTOR_AVAILABLE` flags. If
> either is missing, the rest of the app (pages, task CRUD, password login)
> still works — registering/logging in with your face will just show a clear
> "not available" message instead of crashing.

## Run

```bash
python app.py
```

Visit `http://127.0.0.1:5000`. The webcam APIs (`getUserMedia`) require
either `localhost` or HTTPS — they won't work over plain `http://` on a
non-localhost address.

## Tuning the match

In `app.py`:

```python
LBPH_CONFIDENCE_THRESHOLD = 70  # lower = stricter (fewer false accepts, more false rejects)
MIN_USABLE_SAMPLES = 3          # of the 5 burst frames, how many must contain a clear face
```

LBPH confidence is a distance, not a percentage — typical "good match" values
are roughly 0–60, with values above ~80–100 usually being a different
person. Test with your own face under your actual lighting before settling
on a number; cameras, room lighting, and glasses on/off all shift this.

## Security notes (read before using this for anything real)

This is a learning/portfolio project, not a hardened auth system:

- **No liveness detection.** A printed photo or a phone screen showing the
  registered user's face can pass the current check. Add blink/head-turn
  detection before relying on this for anything sensitive.
- **LBPH is a classical (non-deep-learning) method.** It's noticeably less
  robust than embeddings from a model like dlib's or FaceNet's, especially
  with multiple registered users who look similar, varied lighting, or
  camera changes. Fine for a personal project/demo; not for production auth.
- **Keep the password fallback.** It's also the recovery path if the face
  match stops working (lighting, haircut, camera swap).
- **Use HTTPS in production** — camera access and session cookies both need
  it; browsers block `getUserMedia` on non-localhost HTTP anyway.
- **Set a real `SECRET_KEY`** via the `SECRET_KEY` environment variable
  instead of the placeholder in `app.py`.
- **Face crops are biometric data.** `faces/samples/` should never be
  committed to version control or left world-readable; encrypt at rest if
  you deploy this anywhere multi-user.
- **Rate-limit `/api/face-login`** in production so the scan loop can't be
  hammered.

## Possible next steps

- Swap SQLite for PostgreSQL (`SQLALCHEMY_DATABASE_URI`) for multi-user deployments.
- Add a basic liveness check (e.g., ask the user to blink or turn their head, detected frame-to-frame).
- If accuracy becomes a real problem, migrate to a deep-learning embedding model (e.g., `face_recognition`/dlib, or an ONNX-based face embedding model) — the detect/crop/compare structure in `app.py` carries over directly, only `detect_and_crop_face`/`retrain_model`/`recognize_face` would change.
- Add a "re-register face" flow on the dashboard for when matching degrades over time.
