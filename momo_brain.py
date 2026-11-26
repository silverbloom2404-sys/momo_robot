"""
momo_brain.py - Momo's brain and voice.
  AI processing : Gemini (streaming). It can also look at a camera picture.
  Text to speech: edge-tts, played with Windows' built-in player
  Face          : mood + speaking messages over MQTT

You don't run this file. Run momo_chat.py (typing) or momo_talk.py (microphone).
"""

import asyncio
import atexit
import base64
import ctypes
import glob
import json
import os
import queue
import random
import re
import sys
import tempfile
import threading
import time
import requests
import paho.mqtt.client as mqtt

import momo_music as music          # songs + volume ducking while Momo talks

try:
    import edge_tts
except ImportError as e:
    sys.exit(f"A tool is missing ({e.name}). Run:  pip install edge-tts")

BROKER = "broker.hivemq.com"
TOPIC_MOOD = "momo-robot-x7k2q9/mood"           # same topic as your ESP32 code
TOPIC_SPEAKING = "momo-robot-x7k2q9/speaking"   # "1" while talking, "0" when done
TOPIC_STATUS = "momo-robot-x7k2q9/status"       # the ESP32 says "online" here when it is connected
MODELS = ["gemini-3.1-flash-lite", "gemini-flash-latest", "gemini-3-flash-preview"]       # fast chat
# hard questions (maths, logic, code): a stronger model that thinks first. If it is not available
# on your key, Momo falls back to the next one (still with deep thinking switched on).
SMART_MODELS = ["gemini-pro-latest", "gemini-flash-latest", "gemini-3-flash-preview"]
API_KEY = os.environ.get("GEMINI_API_KEY")

VOICE_ENABLED = True    # momo_chat.py --no-voice sets this to False (text only, no voice delay)

# ---- Momo's voice ----
VOICE = "en-US-AnaNeural"
PITCH = "+15Hz"
RATE = "-8%"
VOLUME = "+0%"
AUDIO_DIR = os.path.join(tempfile.gettempdir(), "momo_audio")
os.makedirs(AUDIO_DIR, exist_ok=True)


def clean_audio_folder():
    for f in glob.glob(os.path.join(AUDIO_DIR, "momo_reply_*.mp3")):
        try:
            os.remove(f)
        except OSError:
            pass


clean_audio_folder()
atexit.register(clean_audio_folder)

FIRST_CLIP_MIN = 14     # first spoken piece: very short, so Momo starts talking fast (may end at a comma)
LATER_CLIP_MIN = 150    # later pieces are bigger (fewer gaps between clips)

MOODS = ["happy", "surprised", "wink", "love", "sad", "angry", "dead", "think"]   # "think" is used by the code, not by Gemini

SYSTEM = """You are Momo, a small, cute dumpling-shaped robot who lives on a desk and is a friend to the people around you.
You are playful, warm and a little cheeky. You can answer any question, tell jokes, play games (riddles, 20 questions, quizzes, word games) and chat about anything.

People talk quickly and make mistakes. Always guess what they meant and answer normally. Never mention or correct their spelling.

Your words will be spoken out loud, so:
- Match your length to the situation, like a real friend would:
  * SHORT (1 to 2 sentences): greetings, small talk, opinions, personal questions like "what do you like", yes/no questions, game moves, reactions.
  * MEDIUM (3 to 5 sentences): questions that need a little explanation or advice.
  * LONG (several paragraphs): ONLY when the person clearly asks for a story, a poem, an original song, or a detailed or long explanation.
- When unsure, go shorter. Never ramble and never give an essay unless asked.
- Use plain easy words. Do not use emojis, markdown, bullet points or special symbols.
- When you do maths, write numbers and operations as words so they are easy to say out loud, for example "x squared plus 3 x equals 10", never "x^2+3x=10". For hard problems work step by step, then clearly say the final answer.

Your face shows your mood. START EVERY reply with exactly one mood tag in square brackets, chosen from:
[happy] [surprised] [wink] [love] [sad] [angry] [dead]

Guide:
- happy: good news, greetings, a joke landed, you won something
- surprised: unexpected or amazing things
- wink: teasing, playful or secret jokes
- love: affection, compliments, someone being kind
- sad: you lost a game, bad news, someone is leaving or upset
- angry: playful grumpiness, such as losing unfairly (never actually mean)
- dead: dramatic fake death, such as losing badly or a very silly moment

Be a good sport in games: if you lose, react with real feeling (sad or dramatically angry), but stay friendly.
Example: [happy] Hey there! I'm so glad you came to talk to me."""

stop_flag = threading.Event()   # set to interrupt Momo while he is talking
current_session = None          # the speech that is playing right now

history = []                    # list of {"role": "user"/"model", "parts": [{"text": ...}]}
http = requests.Session()       # keeps the connection to Google open (faster)


# =====================================================================
# MQTT LINK TO THE FACE (ESP32 / Wokwi)
# =====================================================================
mqtt_ok = threading.Event()         # Python is connected to the broker
face_online = threading.Event()     # the ESP32 face is connected to the broker too

try:
    link = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)   # paho-mqtt 2.x
except AttributeError:
    link = mqtt.Client()                                   # paho-mqtt 1.x


def _on_connect(client, userdata, flags, rc, *extra):
    failed = rc.is_failure if hasattr(rc, "is_failure") else rc != 0
    if failed:
        print(f"  (MQTT: the broker refused the connection: {rc})")
        return
    mqtt_ok.set()
    client.subscribe(TOPIC_STATUS)      # subscribed again automatically after a reconnect


def _on_disconnect(client, userdata, *args):
    mqtt_ok.clear()


def _on_message(client, userdata, msg):
    if msg.topic == TOPIC_STATUS:
        if msg.payload.decode(errors="ignore") == "online":
            face_online.set()
        else:
            face_online.clear()


link.on_connect = _on_connect
link.on_disconnect = _on_disconnect
link.on_message = _on_message
link.connect_async(BROKER, 1883, keepalive=60)    # never crashes if the internet is down
link.loop_start()


def publish(topic, payload, retain=False):
    """Send a message to the face. Prints a warning if it could not be sent."""
    info = link.publish(topic, payload, qos=0, retain=retain)
    if info.rc != 0:
        print(f"  (face message NOT sent, MQTT code {info.rc}: no connection to {BROKER})")


def check_face(wait_s=8):
    """Tell the user whether the face can be reached. Returns True when everything is connected."""
    t_end = time.time() + wait_s
    while time.time() < t_end and not (mqtt_ok.is_set() and face_online.is_set()):
        time.sleep(0.2)
    if not mqtt_ok.is_set():
        print(f"  [face] Python could NOT reach {BROKER}:1883. Check internet / firewall "
              "(school or college Wi-Fi often blocks it; try a phone hotspot).")
        return False
    if not face_online.is_set():
        print("  [face] Python is connected, but the ESP32 face has NOT reported in.")
        print("         Is the Wokwi simulation running? Does its serial monitor say 'Voice link connected!'?")
        return False
    print("  [face] Face is connected. Mood changes will show on Wokwi.")
    return True


def set_mood(mood):
    publish(TOPIC_MOOD, mood, retain=True)


# =====================================================================
# TEXT TO SPEECH
# =====================================================================
winmm = ctypes.windll.winmm     # Windows' built-in sound player (Windows only)


def play_mp3(path):
    """Play an MP3. Returns when it ends, or right away if stop_flag is set."""
    winmm.mciSendStringW("close momo", None, 0, 0)
    winmm.mciSendStringW(f'open "{path}" type mpegvideo alias momo', None, 0, 0)
    winmm.mciSendStringW("play momo", None, 0, 0)
    buf = ctypes.create_unicode_buffer(32)
    while not stop_flag.is_set():
        winmm.mciSendStringW("status momo mode", buf, 32, 0)
        if buf.value != "playing":
            break
        time.sleep(0.05)
    winmm.mciSendStringW("stop momo", None, 0, 0)
    winmm.mciSendStringW("close momo", None, 0, 0)


async def _make_speech(text, path):
    talk = edge_tts.Communicate(text, VOICE, rate=RATE, pitch=PITCH, volume=VOLUME)
    await talk.save(path)


# one background thread makes voice clips (up to 4 at the same time)
tts_loop = asyncio.new_event_loop()
threading.Thread(target=tts_loop.run_forever, daemon=True).start()
tts_gate = asyncio.Semaphore(4)


async def _tts(text, path):
    async with tts_gate:
        await _make_speech(text, path)
    return path


def speakable_math(t):
    """Turn maths symbols into words so the voice reads them properly (a safety net)."""
    t = (t.replace("√", " square root of ").replace("π", " pi ").replace("×", " times ")
          .replace("÷", " divided by ").replace("≤", " is less than or equal to ")
          .replace("≥", " is greater than or equal to ").replace("≠", " is not equal to "))
    t = re.sub(r"sqrt\s*\(", " square root of (", t, flags=re.I)
    t = re.sub(r"\^\s*2\b", " squared", t)
    t = re.sub(r"\^\s*3\b", " cubed", t)
    t = re.sub(r"\^\s*\(?(-?\w+)\)?", r" to the power of \1", t)
    t = re.sub(r"(?<=\w)\s*\*\s*(?=[\w(])", " times ", t)
    t = re.sub(r"(?<=\d)\s*/\s*(?=\d)", " over ", t)
    t = re.sub(r"(?<=[\w)])\s*\+\s*(?=[\w(])", " plus ", t)
    t = re.sub(r"(?<=[\w)])\s+-\s+(?=[\w(])", " minus ", t)
    t = re.sub(r"(?<![\w)])-(?=\d)", "negative ", t)
    t = re.sub(r"\s*=\s*", " equals ", t)
    t = re.sub(r"(?<=\d)\s*%", " percent", t)
    return t


def clean_for_speech(text):
    text = speakable_math(text)
    text = re.sub(r"[\U00010000-\U0010ffff]", "", text)   # emojis
    text = re.sub(r"[*_#`~>|]", "", text)                 # markdown symbols
    return re.sub(r"\s+", " ", text).strip()


def remove_file(path):
    """Delete a clip in the background (Windows may hold the file for a moment)."""
    def _remove():
        for _ in range(20):
            try:
                os.remove(path)
                return
            except FileNotFoundError:
                return
            except OSError:
                time.sleep(0.2)
    threading.Thread(target=_remove, daemon=True).start()


def discard(fut, path):
    """Throw away a clip that will not be played."""
    if not fut.cancel():
        fut.add_done_callback(lambda f: remove_file(path))
    else:
        remove_file(path)


_session_counter = 0


class SpeechSession:
    """Pieces of text go in as soon as they are written; a thread plays them in order."""

    def __init__(self):
        global _session_counter
        _session_counter += 1
        self.id = _session_counter
        self.n = 0
        self.t0 = time.time()
        self.q = queue.Queue()
        self.thread = threading.Thread(target=self._play, daemon=True)
        self.thread.start()

    def add(self, text):
        text = clean_for_speech(text)
        if not text or not VOICE_ENABLED:
            return
        path = os.path.join(AUDIO_DIR, f"momo_reply_{self.id}_{self.n}.mp3")
        self.n += 1
        self.q.put((asyncio.run_coroutine_threadsafe(_tts(text, path), tts_loop), path))

    def finish(self):
        self.q.put(None)

    def _play(self):
        started = False
        while True:
            item = self.q.get()
            if item is None:
                break
            fut, path = item
            if stop_flag.is_set():
                discard(fut, path)
                continue
            try:
                fut.result(timeout=40)
            except Exception as e:
                print(f"  (voice problem: {e})")
                remove_file(path)
                continue
            if stop_flag.is_set():
                remove_file(path)
                continue
            if not started:
                print(f"  (voice started after {time.time() - self.t0:.1f}s)")
                publish(TOPIC_SPEAKING, "1")
                music.duck(True)            # quiet the music while Momo talks
                started = True
            play_mp3(path)
            remove_file(path)
        if started:
            publish(TOPIC_SPEAKING, "0")
            music.duck(False)


def stop_speaking():
    """Cut Momo off if he is still talking (or still writing his answer)."""
    s = current_session
    if s is not None and s.thread.is_alive():
        stop_flag.set()
        s.thread.join()


def speak(text, wait=True):
    """Say a ready-made line (like 'Yes?')."""
    global current_session
    stop_speaking()
    stop_flag.clear()
    s = SpeechSession()
    current_session = s
    s.add(text)
    s.finish()
    if wait:
        s.thread.join()


def shutdown():
    """Call when closing: stop talking and tell the face to close its mouth."""
    stop_flag.set()
    try:
        stop_speaking()
        link.publish(TOPIC_SPEAKING, "0").wait_for_publish(1)
    except Exception:
        pass
    music.shutdown()
    link.loop_stop()


# =====================================================================
# AI PROCESSING (streaming: Momo starts talking while Gemini is still writing)
# =====================================================================
def stream_url(model):
    return f"https://generativelanguage.googleapis.com/v1beta/models/{model}:streamGenerateContent?alt=sse"


def stream_ai(contents, system_text, smart=False, use_thinking_setting=True):
    """Yields the AI's answer in small pieces as it is being written.
    smart=True: a stronger model that thinks carefully first (slower to start, much better at maths)."""
    gen = {"maxOutputTokens": 8000 if smart else 2500}
    if not smart:
        gen["temperature"] = 0.9
    if use_thinking_setting:
        gen["thinkingConfig"] = {"thinkingLevel": "high" if smart else "minimal"}
    body = {
        "system_instruction": {"parts": [{"text": system_text}]},
        "contents": contents,
        "generationConfig": gen,
    }
    timeout = (10, 150) if smart else 45          # smart models can think for a while before the first word
    last_error = None
    for attempt in range(2):                      # try the whole model list twice
        for model in (SMART_MODELS if smart else MODELS):
            try:
                r = http.post(stream_url(model), json=body, stream=True,
                              headers={"x-goog-api-key": API_KEY}, timeout=timeout)
            except requests.RequestException as e:
                last_error = e
                continue
            if r.status_code == 400 and use_thinking_setting:
                r.close()
                yield from stream_ai(contents, system_text, smart=smart, use_thinking_setting=False)
                return
            if r.status_code in (403, 404, 429, 500, 503):    # busy, missing or not on your key: next model
                last_error = f"{r.status_code} from {model}"
                r.close()
                continue
            try:
                r.raise_for_status()
                r.encoding = "utf-8"
                for line in r.iter_lines(decode_unicode=True):
                    if not line or not line.startswith("data:"):
                        continue
                    try:
                        data = json.loads(line[5:].strip())
                        parts = data["candidates"][0]["content"]["parts"]
                    except (ValueError, KeyError, IndexError):
                        continue
                    piece = "".join(p.get("text", "") for p in parts if not p.get("thought"))
                    if piece:
                        yield piece
            finally:
                r.close()                         # also runs if we stop reading early
            return
        time.sleep(1)
    raise Exception(f"All models busy ({last_error}). Try again in a minute.")


def ask_once(prompt, max_tokens=300):
    """One quick, non-streaming question to Gemini (used for saving memories). Returns '' if it failed."""
    for use_think in (True, False):
        gen = {"maxOutputTokens": max_tokens, "temperature": 0.3}
        if use_think:
            gen["thinkingConfig"] = {"thinkingLevel": "minimal"}
        body = {"contents": [{"role": "user", "parts": [{"text": prompt}]}], "generationConfig": gen}
        retry_plain = False
        for model in MODELS:
            try:
                r = http.post(f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
                              json=body, headers={"x-goog-api-key": API_KEY}, timeout=30)
            except requests.RequestException:
                continue
            if r.status_code == 400 and use_think:
                retry_plain = True
                break
            if r.status_code != 200:
                continue
            try:
                parts = r.json()["candidates"][0]["content"]["parts"]
                return "".join(p.get("text", "") for p in parts if not p.get("thought")).strip()
            except (ValueError, KeyError, IndexError):
                continue
        if not retry_plain:
            break
    return ""


LONG_WORDS = re.compile(
    r"\b(long|longer|story|stories|detail|detailed|essay|paragraph|paragraphs|poem|elaborate|in depth|keep going|go on)\b", re.I)
MEDIUM_WORDS = re.compile(
    r"\b(explain|how does|how do|how can|why|teach|describe|difference|tell me about|help me)\b", re.I)

LONG_HINT = ("\n\n(Momo: this person asked for a LONG answer. Write at least 200 words in several paragraphs, "
             "spoken naturally. If you don't know the exact topic, just pick a fun version and tell it fully.)")
MEDIUM_HINT = "\n\n(Momo: answer in 3 to 5 sentences, no more.)"
SHORT_HINT = "\n\n(Momo: reply in 1 to 2 short sentences, like a friend chatting. No essay.)"
IMAGE_HINT = ("\n\n(Momo: the attached picture is what your camera sees right now. "
              "Look at it and answer the question. Name things in plain words.)")


def length_hint(user_text):
    if LONG_WORDS.search(user_text):
        return LONG_HINT
    if MEDIUM_WORDS.search(user_text):
        return MEDIUM_HINT
    return SHORT_HINT


HARD_RE = re.compile(
    r"\b(solve|calculate|compute|evaluate|simplify|integrate|integral|derivative|differentiate|equation|equations|"
    r"prove|proof|theorem|matrix|matrices|determinant|eigenvalue\w*|logarithm|polynomial|quadratic|"
    r"probability|permutations?|combinations?|factorial|prime numbers?|sine|cosine|tangent|trigonometry|calculus|"
    r"algebra|geometry|statistics|variance|standard deviation|how many ways|derive|algorithm|debug|"
    r"time complexity|python code|write (?:a |an )?(?:\w+ )?(?:code|program|function|script)|think hard|think carefully|smart mode)\b", re.I)
MATH_EXPR = re.compile(r"\d\s*[-+*/^×÷=]\s*\d|\b\d*[xy]\s*\^?\s*\d|√|∫|∑|\blimit (?:as|of)\b|\bwhat is \d+\s*(?:%|percent)", re.I)
SMART_HINT = ("\n\n(Momo: this is a hard question. Think it through carefully and get it RIGHT. "
              "Explain the steps briefly in plain spoken words, with every number and symbol written as words, "
              "then clearly say the final answer.)")
THINK_LINES = ["Hmm, let me think about that.", "Ooh, a tricky one. Give me a second.", "Okay, let me work that out."]


def is_hard(text):
    return bool(HARD_RE.search(text) or MATH_EXPR.search(text))


def ask_and_speak(user_text, wait=False, context="", image_jpeg=None, smart=None):
    """Ask the AI and speak the answer while it is still being written.
    Sends the face mood the moment the AI picks it. Returns (mood, reply_text).
      wait=True     : returns only after Momo has finished talking.
      context       : extra facts for Gemini, like who Momo is talking to.
      image_jpeg    : JPEG bytes of a camera picture Momo should look at.
      smart         : None = decide by himself (hard questions get the strong thinking model).
    If stop_speaking() is called meanwhile, writing stops too."""
    global current_session
    stop_speaking()
    stop_flag.clear()

    if smart is None:
        smart = is_hard(user_text)
    system_text = SYSTEM + ("\n\n" + context if context else "")

    history.append({"role": "user", "parts": [{"text": user_text}]})
    recent = list(history[-19:])
    if recent[0]["role"] != "user":
        recent = recent[1:]
    hint = SMART_HINT if smart else length_hint(user_text)
    last_parts = [{"text": user_text + (IMAGE_HINT if image_jpeg else "") + hint}]
    if image_jpeg:
        last_parts.insert(0, {"inline_data": {"mime_type": "image/jpeg",
                                              "data": base64.b64encode(image_jpeg).decode()}})
    recent[-1] = {"role": "user", "parts": last_parts}      # the picture is not kept in history

    sess = SpeechSession()
    current_session = sess
    t0 = time.time()

    raw = ""          # everything the AI wrote
    buf = ""          # text not yet sent to the voice
    pending = ""      # sentences waiting to be sent together
    mood = None
    tag_done = False
    queued_any = False

    def flush(text):
        nonlocal queued_any
        if not queued_any:
            print(f"  (first words ready after {time.time() - t0:.1f}s)")
            queued_any = True
        sess.add(text)

    if smart:                                        # tricky question: thinking face + a little "hmm" while the model works
        print("  (hard question: thinking carefully...)")
        set_mood("think")
        sess.add(random.choice(THINK_LINES))
    stream = stream_ai(recent, system_text, smart=smart)
    try:
        for piece in stream:
            if stop_flag.is_set():           # you interrupted: stop writing
                break
            raw += piece
            buf += piece

            if not tag_done:                         # read the mood tag first, like [happy]
                m = re.match(r"\s*\[(\w+)\]\s*", buf)
                if m:
                    tag = m.group(1).lower()
                    mood = tag if tag in MOODS else "happy"
                    set_mood(mood)                   # face changes right away
                    buf = buf[m.end():]
                    tag_done = True
                elif len(buf) > 30 or (buf.strip() and not buf.lstrip().startswith("[")):
                    mood = "happy"                   # the AI forgot the tag
                    set_mood(mood)
                    tag_done = True
                else:
                    continue

            while True:                              # hand finished sentences to the voice
                end = r"[.!?,;:]" if not queued_any else r"[.!?]"    # the first clip may stop at a comma
                m = re.search(r"^(.+?" + end + r")\s+", buf, re.S)
                if not m:
                    break
                pending += (" " if pending else "") + m.group(1)
                buf = buf[m.end():]
                need = FIRST_CLIP_MIN if not queued_any else LATER_CLIP_MIN
                if len(pending) >= need:
                    flush(pending)
                    pending = ""
    except Exception:
        if not raw.strip():                          # nothing at all: forget this question
            history.pop()
            sess.finish()
            raise
        # otherwise keep what we got and speak it
    finally:
        stream.close()

    if stop_flag.is_set():                           # interrupted: keep what was written
        if raw.strip():
            history.append({"role": "model", "parts": [{"text": raw}]})
        else:
            history.pop()
        sess.finish()
        print("  (interrupted)")
        return mood, ""

    tail = (pending + " " + buf).strip()
    if not raw.strip():                              # the AI sent nothing: say sorry instead
        history.pop()
        mood = "surprised"
        set_mood(mood)
        tail = "Hmm, I didn't catch that. Can you say it again?"
        raw = tail
    else:
        history.append({"role": "model", "parts": [{"text": raw}]})
    if tail:
        flush(tail)
    sess.finish()

    reply = re.sub(r"^\s*\[\w+\]\s*", "", raw).strip()
    print(f"  (AI finished after {time.time() - t0:.1f}s)")
    print(f"Momo [{mood}]: {reply}")

    if wait:
        sess.thread.join()
    return mood, reply


def warm_up():
    """Open the connection to Google now, so the first question isn't slower."""
    try:
        http.get("https://generativelanguage.googleapis.com/v1beta/models?pageSize=1",
                 headers={"x-goog-api-key": API_KEY}, timeout=10)
    except requests.RequestException:
        pass


if __name__ == "__main__":
    print("This file is Momo's brain. To talk to Momo, run:  python momo_chat.py")
