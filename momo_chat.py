"""
momo_chat.py  -  Chat with Momo by TYPING. Face, camera and memory all work together.

  Face    : mood changes on the Wokwi face while you chat
  Camera  : eyes follow you, Momo recognizes people and looks at objects you hold up
  Memory  : new faces are introduced (name + a note); after each chat Momo saves a short
            summary of it, all in momo_data/people.json on this computer
  Smart   : hard questions (maths, logic, code) automatically use a stronger thinking model
  Music   : "play <song>" (your own files in momo_songs/, or YouTube), pause, resume, stop, volume

Needs momo_brain.py and momo_camera.py in the SAME folder.

Run:
    python momo_chat.py                 # everything on
    python momo_chat.py --no-voice      # text only, no speaking (fastest for testing)
    python momo_chat.py --no-camera     # without the webcam

Things you can type:
    solve x squared plus 5x plus 6 equals 0     (any hard question)
    play shape of you / pause / resume / stop the music / volume down
    what is this?   (hold something up to the camera)  |  solve this  (hold up a problem)
    forget me       (Momo deletes your saved face)
    /people   /face   /mood sad   stop   quit
"""

import random
import re
import sys
import threading
import time

import momo_brain
import momo_music
from momo_brain import (ask_and_speak, stop_speaking, speak, warm_up, shutdown, check_face,
                        set_mood, MOODS, API_KEY)

if "--no-voice" in sys.argv:
    momo_brain.VOICE_ENABLED = False
USE_CAMERA = "--no-camera" not in sys.argv

cam = None                 # the Camera (None if off)
stage = None               # None, "name" (waiting for a name) or "about" (waiting for a note)
pending_name = None
lock = threading.RLock()   # so the camera watcher and your typing never talk over each other
session_log = {}           # name -> [(what they said, what Momo said)], turned into a memory when they leave
mem_lock = threading.Lock()

GREETINGS = [
    "Hi {name}! So good to see you again!",
    "Oh, it's {name}! Hello hello!",
    "{name}! You're back! I missed you!",
]
SKIP_WORDS = {"skip", "no", "nope", "later", "never mind", "nevermind", "nah", "no thanks", "cancel"}
NAME_STOP = {"and", "but", "i", "im", "from", "the", "who", "so", "because"}
NOT_NAMES = {"hi", "hello", "hey", "yes", "no", "ok", "okay", "hmm", "what", "why", "how"}

SEE_RE = re.compile(r"\b(what is this|what's this|what is that|what's that|what do you see|what am i holding|"
                    r"what is in my hand|look at this|identify this|what object|can you see this|solve this|solve that|"
                    r"answer this|read this|what does this say|what is written|what's written|can you solve)\b", re.I)
FORGET_RE = re.compile(r"\bforget\s+([a-z]+)\b", re.I)


PLAY_RE = re.compile(
    r"^\s*(?:(?:please|pls|plz|can|could|would|will|you|u|ya|kindly|just|now|momo|hey|go ahead and)\s+)*"
    r"(?:play|put on|start playing|i want to hear|i wanna hear|i want to listen to|i wanna listen to|let'?s listen to)\s+"
    r"(.+?)\s*[.!?]*\s*$", re.I)
PAUSE_RE = re.compile(r"^\s*(?:please\s+)?pause(?:\s+the)?(?:\s+(?:music|song|it))?\s*[.!?]*$", re.I)
RESUME_RE = re.compile(r"^\s*(?:please\s+)?(?:resume|continue|unpause)(?:\s+the)?(?:\s+(?:music|song|it))?\s*[.!?]*$", re.I)
STOP_MUSIC_RE = re.compile(r"\b(?:stop|turn off|end|switch off)\b.*\b(?:music|song|playing)\b|\b(?:music|song)\b.*\b(?:off|stop)\b", re.I)
VOLUME_RE = re.compile(r"\b(volume|louder|quieter|softer|turn it (?:up|down))\b", re.I)
GAME_WORDS = re.compile(r"\b(game|games|riddle|riddles|quiz|questions|chess|hide and seek|truth or dare|trivia|with me|story|tag)\b", re.I)


WAKE_RE = re.compile(r"^\s*(?:(?:hey|hi|hello|ok|okay)\s+)?(?:momo|mo mo|mumu)\b[\s,.!?:\-]*", re.I)


def strip_wake(text):
    """'hi momo play shape of you' -> 'play shape of you' (so commands work after his name)."""
    out = WAKE_RE.sub("", text, count=1).strip()
    return out or text


def say(text, mood="happy", wait=False):
    """Momo says a ready-made line with a face."""
    with lock:
        set_mood(mood)
        print(f"Momo [{mood}]: {text}")
        speak(text, wait=wait)


# ---------------------------------------------------------------------
# music
# ---------------------------------------------------------------------
def play_job(query):
    """Background: find the song (your folder first, then YouTube) and start it."""
    try:
        path, title = momo_music.find_song(query)
    except Exception as e:
        print("  music problem:", e)
        say("Sorry, I couldn't get that song right now.", "sad")
        return
    momo_music.player.play(path, title)
    print(f"  now playing: {title}")
    say("Here it comes!", "love")


def try_music_command(text):
    """Returns True if the text was a music command (and has been handled)."""
    p = momo_music.player
    if STOP_MUSIC_RE.search(text):
        if p.is_active():
            p.stop()
            say("Okay, music off.", "happy")
        else:
            say("Nothing is playing right now.", "surprised")
        return True
    if PAUSE_RE.match(text) and p.is_active():
        p.pause()
        say("Paused.", "happy")
        return True
    if RESUME_RE.match(text) and p.is_active():
        p.resume()
        say("Back to the music!", "happy")
        return True
    if VOLUME_RE.search(text) and p.is_active():
        down = re.search(r"\b(down|quieter|softer|lower|decrease|less)\b", text, re.I)
        v = p.change_volume(-20 if down else 20)
        print(f"  (music volume {v}%)")
        say("Quieter." if down else "Louder!", "happy")
        return True
    m = PLAY_RE.match(text)
    if m and not GAME_WORDS.search(text):
        q = m.group(1).strip()
        while True:
            q2 = re.sub(r"^(?:me|a|an|some|the|song|songs|music)\b\s*", "", q, flags=re.I)
            if q2 == q:
                break
            q = q2
        q = re.sub(r"\s*\b(?:please|for me|now|again)\s*$", "", q, flags=re.I).strip()
        if not q or q.lower() in {"something", "anything", "random", "more"}:
            q = "popular hit songs"
        say("Okay, let me find that!", "happy")
        threading.Thread(target=play_job, args=(q,), daemon=True).start()
        return True
    return False


# ---------------------------------------------------------------------
# remembering chats
# ---------------------------------------------------------------------
def log_exchange(name, user, reply):
    if name and reply:
        with mem_lock:
            session_log.setdefault(name, []).append((user, reply))


def save_memory(name):
    """Turn the chat with this person into a short memory and save it."""
    with mem_lock:
        log = session_log.pop(name, [])
    if len(log) < 2 or cam is None:
        return
    old = cam.store.get(name).get("summary", "")
    convo = "\n".join(f"{name}: {u}\nMomo: {r}" for u, r in log[-30:])
    prompt = (f"You are Momo, a robot friend. Here is what you remember about {name} from before: "
              f"{old or 'nothing yet'}\n\nHere is your latest conversation with {name}:\n{convo}\n\n"
              f"Write your updated memory of {name} in at most 4 short plain sentences. Keep lasting facts "
              "(interests, plans, family, preferences, things they asked about or struggled with). "
              "Drop small talk and anything private like passwords or phone numbers. "
              "Write only the memory text, nothing else.")
    new = momo_brain.ask_once(prompt, max_tokens=300)
    if new:
        cam.store.set_summary(name, new[:700])
        print(f"  (Momo saved a memory of {name})")
    else:
        with mem_lock:
            session_log.setdefault(name, [])[:0] = log      # try again later


def save_all_memories():
    if any(len(v) >= 2 for v in session_log.values()):
        print("  (saving memories...)")
    for name in list(session_log):
        save_memory(name)


# ---------------------------------------------------------------------
# camera helpers
# ---------------------------------------------------------------------
def snapshot_jpeg():
    return cam.snapshot_jpeg()


def start_camera():
    global cam
    try:
        from momo_camera import Camera
        print("  [camera] starting (the first run downloads two small face models, please wait)...")
        cam = Camera()
        cam.start()
        threading.Thread(target=watch_camera, daemon=True).start()
        print("  [camera] Camera is on. In the preview window, press q to hide it.")
    except Exception as e:
        print(f"  [camera] could not start: {e}")
        print("           Momo will chat without the camera.")
        cam = None


def watch_camera():
    """Background: reacts when the camera sees someone he knows, or a new face."""
    while cam is not None and cam.running:
        ev = cam.next_event()
        if ev is None:
            time.sleep(0.3)
            continue
        kind, name = ev
        if kind == "greet":
            say(random.choice(GREETINGS).format(name=name), "happy")
        elif kind == "meet" and stage is None:
            start_intro()
        elif kind == "left":
            threading.Thread(target=save_memory, args=(name,), daemon=True).start()


# ---------------------------------------------------------------------
# meeting a new person
# ---------------------------------------------------------------------
def extract_name(text):
    t = text.strip().strip(".!?,")
    m = re.search(r"\b(?:my name is|my name's|name is|i am|i'm|im|this is|call me|it's|its)\s+"
                  r"([A-Za-z][A-Za-z'\-]*(?:\s+[A-Za-z][A-Za-z'\-]*){0,2})", t, re.I)
    raw = m.group(1) if m else (t if len(t.split()) <= 2 else "")
    words = []
    for w in raw.split():
        if w.lower() in NAME_STOP:
            break
        words.append(w)
    words = words[:2]
    if not words or words[0].lower() in NOT_NAMES:
        return None
    if not all(re.fullmatch(r"[A-Za-z][A-Za-z'\-]*", w) for w in words):
        return None
    return " ".join(w.capitalize() for w in words)


def start_intro():
    global stage
    with lock:
        stage = "name"
        cam.decline_until = time.time() + 600       # don't ask again for 10 minutes
    say("Oh, a new face! Hi, I'm Momo. What's your name?", "surprised")
    print("  (Momo asked your name. Just type it, or type 'skip'.)")


def handle_name(text):
    global stage, pending_name
    low = text.lower().strip(" .!?")
    if low in SKIP_WORDS:
        stage = None
        cam.decline_until = time.time() + 1800
        say("Okay, no problem! Talk to me anytime.", "happy")
        return
    name = extract_name(text)
    if not name:
        say("Sorry, I didn't catch your name. Try typing: my name is ... (or type skip)", "surprised")
        return
    if cam.enroll(name):
        pending_name = name
        stage = "about"
        say(f"Nice to meet you, {name}! I'll remember your face. "
            "Tell me one thing about yourself, or type skip.", "love")
    else:
        say("I can't see your face clearly yet. Look at the camera, then type your name again.", "surprised")


def handle_about(text):
    global stage
    low = text.lower().strip(" .!?")
    stage = None
    if low in SKIP_WORDS:
        say("Okay! Let's chat.", "happy")
    else:
        cam.store.add(pending_name, text.strip(), [])
        say("Got it, I'll remember that!", "happy")


# ---------------------------------------------------------------------
# typed commands for the camera
# ---------------------------------------------------------------------
def try_camera_command(text):
    """Returns True if the text was a camera command (and has been handled)."""
    if SEE_RE.search(text):
        jpg = snapshot_jpeg()
        if jpg is None:
            say("I can't see anything right now.", "sad")
        else:
            ask_and_speak(text, wait=False, context=cam.context(), image_jpeg=jpg)
        return True

    m = FORGET_RE.search(text)
    if m and m.group(1).lower() not in {"it", "that", "about", "this", "what", "everything", "to"}:
        who = m.group(1)
        if who.lower() == "me":
            who = cam.current_name
            if not who:
                say("I don't know who you are yet, so there's nothing to forget.", "surprised")
                return True
        if cam.forget(who):
            say(f"Okay, I forgot {who}.", "sad")
        else:
            say(f"I don't know anyone called {who}.", "surprised")
        return True

    return False


# ---------------------------------------------------------------------
# main loop
# ---------------------------------------------------------------------
def main():
    if not API_KEY:
        sys.exit("No GEMINI_API_KEY found. Run:  setx GEMINI_API_KEY your-key  and open a NEW Command Prompt.")

    warm_up()
    print("Checking the face connection...")
    check_face()
    if USE_CAMERA:
        start_camera()
    say("Hi hi! I'm Momo. Type to me!", "happy")
    print("\nMomo is ready. Type and press Enter. Try: 'play <song>', 'solve 3x + 5 = 20', 'what is this?'")
    print("Other: '/people', '/face', '/mood think', 'stop', 'quit'\n")

    while True:
        try:
            text = input("You: ").strip()
        except EOFError:
            break
        if not text:
            continue

        low = text.lower()
        if low in ("quit", "exit", "bye"):
            break
        if low in ("stop", "shh", "quiet", "enough"):
            stop_speaking()
            continue
        if low == "/face":
            check_face(3)
            continue
        if low == "/people":
            print("  people I know:", ", ".join(cam.store.names()) if cam and cam.store.names() else "nobody yet")
            continue
        if low.startswith("/mood"):
            parts = low.split()
            if len(parts) == 2 and parts[1] in MOODS:
                set_mood(parts[1])
                print(f"  (sent mood '{parts[1]}' to the face)")
            else:
                print("  Use: /mood " + " | ".join(MOODS))
            continue

        with lock:
            if stage == "name":
                handle_name(text)
                continue
            if stage == "about":
                handle_about(text)
                continue
            if try_music_command(strip_wake(text)):
                continue
            if cam is not None and try_camera_command(text):
                continue
            ctx = cam.context() if cam is not None else "Your camera is turned off, so you cannot see anything."
            ctx = (ctx + " You cannot take photos or record videos.").strip()
            who = cam.current_name if cam is not None else None
            try:
                _mood, reply = ask_and_speak(text, wait=False, context=ctx)
                log_exchange(who, text, reply)
            except Exception as e:
                print("Brain error:", e)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
    finally:
        try:
            if cam is not None:
                save_all_memories()
                cam.stop()
        except Exception:
            pass
        momo_music.player.stop()
        shutdown()
        print("\nBye!")
