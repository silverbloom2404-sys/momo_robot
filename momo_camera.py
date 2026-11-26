"""
momo_camera.py - Momo's eyes (the webcam).

  1. EYE TRACKING : finds the nearest face and sends "x,y" (-100..100) to the face
                    over MQTT (topic .../gaze) so the eyes follow you.
  2. FACE MEMORY  : recognizes people it has met and learns new ones
                    (name + one note), saved on THIS computer in momo_data/people.json.
  3. SEEING       : snapshot_jpeg() gives the current picture so Gemini can say what
                    an object is.

You don't run this file for chatting; momo_chat.py uses it.
Quick camera test on its own:   python momo_camera.py     (Ctrl+C to stop)

Install:  pip install "opencv-python<5" numpy
The first run downloads two small face models (about 37 MB) into momo_data/models.
Only numbers (face "fingerprints"), names and notes are saved. No face pictures.
"""

import collections
import json
import os
import shutil
import sys
import threading
import time
import urllib.request

import cv2
import numpy as np

from momo_brain import link

# ---------- settings ----------
CAMERA_INDEX = 0          # 0 = default webcam. Try 1 if you have more than one
SHOW_PREVIEW = True       # window with boxes. Press q in it to hide it. False = no window at all
GAZE_MIRROR = True        # True = eyes look toward you. If they go the WRONG way left/right, set False
MIN_FACE_PX = 80          # faces smaller than this (far away) are followed but not recognized
MEET_DELAY_S = 3.0        # an unknown face must stay this long before Momo introduces himself
GREET_COOLDOWN_S = 1800   # greet the same person at most once per 30 minutes
LOST_AFTER_S = 5          # nobody seen for this long = the person left
MATCH_THRESHOLD = 0.363   # OpenCV's recommended cosine similarity for "same person"
MAX_SAMPLES = 20          # face fingerprints kept per person

TOPIC_GAZE = "momo-robot-x7k2q9/gaze"       # payload "x,y"  (-100..100, y positive = down)

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "momo_data")
MODEL_DIR = os.path.join(DATA_DIR, "models")
PEOPLE_FILE = os.path.join(DATA_DIR, "people.json")
os.makedirs(MODEL_DIR, exist_ok=True)

WINDOW = "Momo's eyes"
YUNET = "face_detection_yunet_2023mar.onnx"
SFACE = "face_recognition_sface_2021dec.onnx"
MODELS = {YUNET: ("face_detection_yunet", 100_000),
          SFACE: ("face_recognition_sface", 10_000_000)}


def ensure_models():
    """Download the two face models once. Returns {filename: path}."""
    paths = {}
    for fname, (folder, min_size) in MODELS.items():
        path = os.path.join(MODEL_DIR, fname)
        if os.path.exists(path) and os.path.getsize(path) > min_size:
            paths[fname] = path
            continue
        urls = [f"https://github.com/opencv/opencv_zoo/raw/{b}/models/{folder}/{fname}" for b in ("main", "master")]
        urls.append(f"https://huggingface.co/opencv/{folder}/resolve/main/{fname}")
        done = False
        for url in urls:
            tmp = path + ".part"
            try:
                print(f"  (downloading {fname} ...)")
                req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
                with urllib.request.urlopen(req, timeout=60) as r, open(tmp, "wb") as out:
                    shutil.copyfileobj(r, out)
                if os.path.getsize(tmp) > min_size:
                    os.replace(tmp, path)
                    done = True
                    break
            except Exception as e:
                print(f"  (download failed from {url.split('/')[2]}: {e})")
            if os.path.exists(tmp):
                os.remove(tmp)
        if not done:
            raise RuntimeError(
                f"Could not download {fname}.\n"
                f"  Download it by hand from https://github.com/opencv/opencv_zoo/tree/main/models/{folder}\n"
                f"  and put it in: {MODEL_DIR}")
        paths[fname] = path
    return paths


def clamp(v, lo=-100.0, hi=100.0):
    return max(lo, min(hi, v))


# =====================================================================
# People memory (momo_data/people.json)
# =====================================================================
class PeopleStore:
    def __init__(self):
        self.lock = threading.Lock()
        self.people = {}      # key (lowercase name) -> {"name","about","summary","embeddings","first_seen","last_seen"}
        self.cache = {}       # key -> list of numpy arrays shaped (1, 128)
        self._load()

    def _load(self):
        if not os.path.exists(PEOPLE_FILE):
            return
        try:
            with open(PEOPLE_FILE, encoding="utf-8") as f:
                self.people = json.load(f)
        except (OSError, ValueError) as e:
            backup = PEOPLE_FILE + ".broken"
            print(f"  (could not read people.json: {e}; keeping a copy as {os.path.basename(backup)})")
            try:
                os.replace(PEOPLE_FILE, backup)
            except OSError:
                pass
            self.people = {}
        for key, p in self.people.items():
            self.cache[key] = [np.array(e, dtype=np.float32).reshape(1, -1) for e in p["embeddings"]]

    def _save(self):
        tmp = PEOPLE_FILE + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self.people, f, indent=1)
        os.replace(tmp, PEOPLE_FILE)

    def names(self):
        with self.lock:
            return [p["name"] for p in self.people.values()]

    def get(self, name):
        with self.lock:
            return dict(self.people.get((name or "").lower(), {}))

    def identify(self, recognizer, feat):
        """Returns the best matching person's name, or None if nobody matches."""
        best_key, best = None, -1.0
        with self.lock:
            for key, embs in self.cache.items():
                for e in embs:
                    score = recognizer.match(feat, e, cv2.FaceRecognizerSF_FR_COSINE)
                    if score > best:
                        best, best_key = score, key
            if best_key is not None and best >= MATCH_THRESHOLD:
                return self.people[best_key]["name"]
        return None

    def add(self, name, about, feats):
        """Save a person (new or existing). feats = list of face fingerprints (may be empty)."""
        name = name.strip()
        key = name.lower()
        now = time.time()
        with self.lock:
            p = self.people.get(key) or {"name": name, "about": "", "summary": "", "embeddings": [], "first_seen": now}
            if about:
                p["about"] = about.strip()
            p["embeddings"] = (p["embeddings"] + [f.flatten().tolist() for f in feats])[-MAX_SAMPLES:]
            p["last_seen"] = now
            self.people[key] = p
            self.cache[key] = [np.array(e, dtype=np.float32).reshape(1, -1) for e in p["embeddings"]]
            self._save()

    def set_summary(self, name, text):
        """Save Momo's short memory of past chats with this person."""
        with self.lock:
            p = self.people.get((name or "").lower())
            if p:
                p["summary"] = text.strip()
                self._save()

    def touch(self, name):
        with self.lock:
            p = self.people.get((name or "").lower())
            if p:
                p["last_seen"] = time.time()
                self._save()

    def forget(self, name):
        key = (name or "").lower()
        with self.lock:
            if key not in self.people:
                return False
            del self.people[key]
            self.cache.pop(key, None)
            self._save()
        return True


# =====================================================================
# The camera
# =====================================================================
class Camera:
    def __init__(self):
        paths = ensure_models()
        self.detector = cv2.FaceDetectorYN.create(paths[YUNET], "", (320, 320), 0.8, 0.3, 5000)
        self.recognizer = cv2.FaceRecognizerSF.create(paths[SFACE], "")
        self.store = PeopleStore()

        self.lock = threading.Lock()        # protects self.frame
        self.frame = None                   # newest camera picture
        self.running = False
        self.thread = None
        self.cap = None

        self.events = collections.deque()                   # ("greet", name) or ("meet", None)
        self.current_name = None                            # who Momo is looking at (None = unknown/nobody)
        self.recent_names = collections.deque(maxlen=4)
        self.recent_feats = collections.deque(maxlen=12)    # (time, fingerprint) of the nearest face
        self.unknown_since = None
        self.last_greeted = {}
        self.decline_until = 0.0                            # don't introduce Momo to strangers before this time

    # ---------------- start / stop ----------------
    def start(self):
        backend = cv2.CAP_DSHOW if sys.platform.startswith("win") else cv2.CAP_ANY
        self.cap = cv2.VideoCapture(CAMERA_INDEX, backend)
        if not self.cap.isOpened():
            raise RuntimeError(f"Can't open the camera (CAMERA_INDEX = {CAMERA_INDEX}). "
                               "Is another app using it?")
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.running = True
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=2)
        if self.cap:
            self.cap.release()
        try:
            cv2.destroyAllWindows()
        except cv2.error:
            pass
        link.publish(TOPIC_GAZE, "0,0")      # eyes back to the middle

    # ---------------- seeing ----------------
    def snapshot_jpeg(self):
        """The current camera picture as JPEG bytes (for Gemini). Nothing is saved to disk."""
        with self.lock:
            frame = None if self.frame is None else self.frame.copy()
        if frame is None:
            return None
        h, w = frame.shape[:2]
        if w > 960:
            frame = cv2.resize(frame, (960, int(h * 960 / w)))
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        return buf.tobytes() if ok else None

    # ---------------- people ----------------
    def next_event(self):
        try:
            return self.events.popleft()
        except IndexError:
            return None

    def context(self):
        """A sentence for Gemini about who Momo is talking to ('' if nobody known)."""
        p = self.store.get(self.current_name)
        if not p:
            return ""
        text = f"You are talking to {p['name']}, a friend you recognize with your camera."
        if p.get("about"):
            text += f" They once told you about themselves: {p['about'].rstrip('.!? ')}."
        if p.get("summary"):
            text += (f" What you remember from your earlier chats with them: {p['summary']} "
                     "Bring these memories up naturally, only when they fit.")
        return text + " Use their name now and then, but not in every sentence."

    def enroll(self, name, about=""):
        """Save the face Momo is looking at under this name. False if no clear face was seen lately."""
        now = time.time()
        samples = [s for s in list(self.recent_feats) if now - s[0] < 10]
        if len(samples) < 2:
            return False
        self.store.add(name, about, [s[1] for s in samples[-6:]])
        self.current_name = name.strip()
        self.last_greeted[self.current_name] = now
        self.unknown_since = None
        self.recent_names.clear()
        return True

    def forget(self, name):
        ok = self.store.forget(name)
        if self.current_name and self.current_name.lower() == (name or "").lower():
            self.current_name = None
            self.recent_names.clear()
        return ok

    def _lose_person(self):
        if self.current_name:
            self.events.append(("left", self.current_name))     # momo_chat saves a memory of the chat
        self.current_name = None
        self.unknown_since = None
        self.recent_names.clear()
        self.recent_feats.clear()

    def _update_people(self, now):
        names = list(self.recent_names)[-3:]
        if len(names) < 3 or len(set(names)) != 1:
            return                                       # not sure yet: wait for 3 matching looks
        who = names[0]
        if who is not None:                              # a face Momo knows
            self.unknown_since = None
            self.current_name = who
            if now - self.last_greeted.get(who, 0) > GREET_COOLDOWN_S:
                self.last_greeted[who] = now
                self.store.touch(who)
                self.events.append(("greet", who))
        else:                                            # a face Momo doesn't know
            self.current_name = None
            if self.unknown_since is None:
                self.unknown_since = now
            if now - self.unknown_since >= MEET_DELAY_S and now >= self.decline_until:
                self.decline_until = now + 60            # momo_chat sets the real wait afterwards
                self.events.append(("meet", None))

    # ---------------- the main loop ----------------
    def _loop(self):
        sm_x = sm_y = 0.0
        sent_x = sent_y = 0
        sent_t = 0.0
        idle_sent = True
        last_feat_t = 0.0
        face_seen_t = 0.0
        window_ok = SHOW_PREVIEW

        while self.running:
            ok, frame = self.cap.read()
            if not ok:
                time.sleep(0.05)
                continue
            now = time.time()
            h, w = frame.shape[:2]
            with self.lock:
                self.frame = frame

            # ----- find faces; the biggest one is the nearest person -----
            self.detector.setInputSize((w, h))
            faces = self.detector.detect(frame)[1]
            big = None
            if faces is not None and len(faces):
                big = max(faces, key=lambda f: f[2] * f[3])
                face_seen_t = now

            # ----- recognize the nearest face -----
            if big is not None and big[2] >= MIN_FACE_PX and now - last_feat_t > 0.4:
                last_feat_t = now
                try:
                    feat = self.recognizer.feature(self.recognizer.alignCrop(frame, big))
                except cv2.error:
                    feat = None
                if feat is not None:
                    self.recent_feats.append((now, feat.copy()))
                    self.recent_names.append(self.store.identify(self.recognizer, feat))
                    self._update_people(now)

            if now - face_seen_t > LOST_AFTER_S and (self.recent_names or self.current_name):
                self._lose_person()

            # ----- move the eyes -----
            if big is not None:
                nx = ((big[0] + big[2] / 2) / w - 0.5) * 2       # -1 (left) .. 1 (right)
                ny = ((big[1] + big[3] / 2) / h - 0.5) * 2       # -1 (top)  .. 1 (bottom)
                if GAZE_MIRROR:
                    nx = -nx
                sm_x += (clamp(nx * 100) - sm_x) * 0.5
                sm_y += (clamp(ny * 100) - sm_y) * 0.5
                idle_sent = False
                if now - sent_t > 0.1 and (abs(sm_x - sent_x) >= 4 or abs(sm_y - sent_y) >= 4):
                    sent_x, sent_y, sent_t = int(sm_x), int(sm_y), now
                    link.publish(TOPIC_GAZE, f"{sent_x},{sent_y}")
            elif now - face_seen_t > 2.0 and not idle_sent:
                sm_x = sm_y = 0.0
                sent_x = sent_y = 0
                idle_sent = True
                link.publish(TOPIC_GAZE, "0,0")                  # nobody there: eyes to the middle

            # ----- preview window -----
            if window_ok:
                try:
                    disp = cv2.flip(frame, 1)                    # mirrored, like a selfie camera
                    if faces is not None:
                        for f in faces:
                            x, y, fw, fh = [int(v) for v in f[:4]]
                            x = w - x - fw                       # box position in the mirrored picture
                            is_big = big is not None and f[0] == big[0] and f[1] == big[1]
                            color = (0, 200, 0) if (is_big and self.current_name) else (0, 165, 255)
                            cv2.rectangle(disp, (x, y), (x + fw, y + fh), color, 2)
                            if is_big:
                                cv2.putText(disp, self.current_name or "new face", (x, max(y - 8, 15)),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                    cv2.imshow(WINDOW, disp)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        cv2.destroyWindow(WINDOW)
                        window_ok = False
                except cv2.error:
                    print("  (the preview window isn't available; install opencv-python, not -headless)")
                    window_ok = False


if __name__ == "__main__":
    cam = Camera()
    cam.start()
    print("Camera test running. Move around: the green/orange box follows your face. Ctrl+C to stop.")
    try:
        while cam.running:
            ev = cam.next_event()
            if ev:
                print("  event:", ev)
            time.sleep(0.2)
    except KeyboardInterrupt:
        pass
    finally:
        cam.stop()
