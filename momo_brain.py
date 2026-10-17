import asyncio
import ctypes
import os
import re
import sys
import time
import requests
import paho.mqtt.client as mqtt

try:
    import edge_tts
except ImportError as e:
    sys.exit(f"A tool is missing ({e.name}). Run:  pip install edge-tts")

BROKER = "broker.hivemq.com"
TOPIC_MOOD = "momo-robot-x7k2q9/mood"           # same topic as your ESP32 code
TOPIC_SPEAKING = "momo-robot-x7k2q9/speaking"   # "1" while talking, "0" when done
MODELS = ["gemini-3.1-flash-lite", "gemini-flash-latest", "gemini-3-flash-preview"]
API_KEY = os.environ.get("GEMINI_API_KEY")

# ---- Momo's voice (your friend's settings) ----
VOICE = "en-US-AnaNeural"
PITCH = "+15Hz"
RATE = "-8%"
VOLUME = "+0%"
AUDIO_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "momo_reply.mp3")


def url_for(model):
    return f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"


MOODS = ["happy", "surprised", "wink", "love", "sad", "angry", "dead"]

SYSTEM = """You are Momo, a small, cute dumpling-shaped robot who lives on a desk and is a friend to the people around you.
You are playful, warm and a little cheeky. You can answer any question, tell jokes, play games (riddles, 20 questions, quizzes, word games) and chat about anything.

People type quickly and make spelling mistakes. Always guess what they meant and answer normally. Never mention or correct their spelling.

Your words will be spoken out loud, so:
- Keep replies SHORT: 1 to 2 sentences, under 30 words, unless the person asks for a story.
- Use plain easy words. Do not use emojis, markdown, bullet points or special symbols.

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

history = []   # list of {"role": "user"/"model", "parts": [{"text": ...}]}
session = requests.Session()

# one MQTT connection that stays open
try:
    link = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
except AttributeError:
    link = mqtt.Client()
link.connect(BROKER, 1883, keepalive=60)
link.loop_start()


# ---------------- Momo's voice ----------------
winmm = ctypes.windll.winmm     # Windows' built-in sound player


def play_mp3(path):
    """Play an MP3 with Windows and wait until it finishes."""
    winmm.mciSendStringW("close momo", None, 0, 0)
    winmm.mciSendStringW(f'open "{path}" type mpegvideo alias momo', None, 0, 0)
    winmm.mciSendStringW("play momo wait", None, 0, 0)
    winmm.mciSendStringW("close momo", None, 0, 0)


async def _make_speech(text, path):
    talk = edge_tts.Communicate(text, VOICE, rate=RATE, pitch=PITCH, volume=VOLUME)
    await talk.save(path)


def clean_for_speech(text):
    text = re.sub(r"[\U00010000-\U0010ffff]", "", text)   # emojis
    text = re.sub(r"[*_#`~>|]", "", text)                 # markdown symbols
    return re.sub(r"\s+", " ", text).strip()


def speak(text):
    text = clean_for_speech(text)
    if not text:
        return
    try:
        asyncio.run(_make_speech(text, AUDIO_FILE))
    except Exception as e:
        print(f"  (voice problem: {e})")
        return
    link.publish(TOPIC_SPEAKING, "1")
    play_mp3(AUDIO_FILE)
    link.publish(TOPIC_SPEAKING, "0")


# ---------------- Momo's brain ----------------
def call_ai(contents, use_thinking_setting=True):
    gen = {"maxOutputTokens": 600, "temperature": 0.9}
    if use_thinking_setting:
        gen["thinkingConfig"] = {"thinkingLevel": "minimal"}
    body = {
        "system_instruction": {"parts": [{"text": SYSTEM}]},
        "contents": contents,
        "generationConfig": gen,
    }
    last_error = None
    for attempt in range(2):                 # try the whole list twice
        for model in MODELS:
            try:
                r = session.post(url_for(model), json=body,
                                 headers={"x-goog-api-key": API_KEY}, timeout=20)
            except requests.RequestException as e:
                last_error = e
                continue
            if r.status_code == 400 and use_thinking_setting:
                return call_ai(contents, use_thinking_setting=False)
            if r.status_code in (429, 500, 503, 404):    # busy or missing: try the next model
                last_error = f"{r.status_code} from {model}"
                continue
            r.raise_for_status()
            try:
                parts = r.json()["candidates"][0]["content"]["parts"]
                return "".join(p.get("text", "") for p in parts).strip()
            except (KeyError, IndexError):
                return ""
        time.sleep(1)
    raise Exception(f"All models busy ({last_error}). Try again in a minute.")


def ask_momo(user_text):
    """Send the person's words to the AI. Returns (mood, reply_text)."""
    history.append({"role": "user", "parts": [{"text": user_text}]})
    recent = history[-19:]
    if recent[0]["role"] != "user":
        recent = recent[1:]

    raw = call_ai(recent)
    if not raw:
        raw = call_ai(recent)    # one retry if the reply came back empty
    if not raw:
        history.pop()
        return "surprised", "Hmm, I didn't catch that. Can you say it again?"

    history.append({"role": "model", "parts": [{"text": raw}]})

    mood = None
    reply = raw
    match = re.match(r"\[(\w+)\]\s*(.*)", raw, re.S)
    if match:
        tag = match.group(1).lower()
        reply = match.group(2).strip()
        if tag in MOODS:
            mood = tag
    return mood, reply


if __name__ == "__main__":
    if not API_KEY:
        print("No GEMINI_API_KEY found. Do the key step, then open a NEW Command Prompt.")
        raise SystemExit
    print("Momo brain ready! Type a message (Ctrl+C to quit).")
    try:
        while True:
            text = input("\nYou: ").strip()
            if not text:
                continue
            try:
                mood, reply = ask_momo(text)
            except Exception as e:
                print("Brain error:", e)
                history.pop()
                continue
            if mood:
                link.publish(TOPIC_MOOD, mood)      # face changes first...
            print(f"Momo [{mood}]: {reply}")
            speak(reply)                            # ...then Momo talks
    except KeyboardInterrupt:
        link.loop_stop()
        print("\nBye!")