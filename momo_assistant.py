"""
Momo, the talking robot  (listen -> understand -> think -> speak)
-----------------------------------------------------------------
Works on any laptop (Windows / Mac / Linux). Free tools only. Needs internet.

SETUP (one time):
    pip install edge-tts SpeechRecognition sounddevice numpy pygame requests

    Then get a free Gemini API key at https://aistudio.google.com  ("Get API key"),
    and save it in a text file called  gemini_key.txt  next to this program
    (just the key, nothing else).

RUN:
    python momo_assistant.py            # the real thing: press Enter, then talk
    python momo_assistant.py --type     # type instead of talking (tests the AI + voice, no mic needed)
    python momo_assistant.py --devices  # list microphones (if Momo can't hear you)

HOW IT WORKS:
    1. You press Enter, wait for "Listening...", then talk.
    2. Momo stops listening when you stop talking.
    3. Speech -> text  (Google's free speech recognizer)
    4. Momo says "Hmm, let me think!" while the AI writes a short answer
    5. Answer -> Momo's voice (Ana_B_cute) -> speaker
"""

import argparse
import asyncio
import collections
import os
import random
import re
import sys
import tempfile
import threading
import time
import wave

try:
    import edge_tts
    import numpy as np
    import requests
    import sounddevice as sd
    import speech_recognition as sr
    os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
    import pygame
except ImportError as e:
    sys.exit(
        f"A tool is missing ({e.name}). Run this in the terminal, then try again:\n"
        "  pip install edge-tts SpeechRecognition sounddevice numpy pygame requests"
    )


# ======================================================================
# 1) MOMO'S VOICE  (Ana_B_cute - same as momo_voice.py)
# ======================================================================
VOICE = "en-US-AnaNeural"
PITCH = "+15Hz"
RATE = "-8%"
VOLUME = "+0%"


# ======================================================================
# 2) MOMO'S PERSONALITY  (what the AI is told. Edit this to change how Momo talks)
# ======================================================================
PERSONALITY = (
    "You are Momo, a cute, friendly little robot friend at a school exhibition. "
    "You look like a soft dumpling with a smiling screen face. "
    "Speak in one or two SHORT, simple sentences (under 25 words total). "
    "Be warm, curious and playful, like a kind little child. "
    "Use plain words that are easy to hear on a small speaker. "
    "Do not use emojis, lists, or special symbols. "
    "Only talk about kid-friendly topics. If someone asks something rude, scary or "
    "inappropriate, gently change the subject to something fun. "
    "If you don't know something, say so cheerfully."
)


# ======================================================================
# 3) HOW MOMO THINKS
# ======================================================================
BRAIN_MODE = "scripted"     # "scripted", "ollama" or "gemini"

# --- only for BRAIN_MODE = "ollama" ---
OLLAMA_MODEL = "llama3.2:1b"            # small model; works on modest laptops
OLLAMA_URL = "http://localhost:11434/api/chat"

# --- only for BRAIN_MODE = "gemini" ---
# Model names change over time. The program tries these in order and uses the first
# one that works. If all fail, copy a current model name from aistudio.google.com
# (look for a "flash" model) and put it first in this list.
MODEL_CANDIDATES = [
    "gemini-flash-latest",
    "gemini-3.8-flash",
    "gemini-3.6-flash",
    "gemini-2.5-flash",
]
MAX_REPLY_TOKENS = 300      # leave this generous; it is not the length of the spoken answer
REMEMBER_TURNS = 4          # how many back-and-forths Momo remembers


# ======================================================================
# 4) LISTENING SETTINGS
# ======================================================================
LANGUAGE = "en-US"          # for speech recognition. Indian English: "en-IN"
MIC_DEVICE = None           # None = default mic. Use a number from --devices to pick another
SAMPLE_RATE = 16000
MIN_VOLUME = 300            # raise if Momo "hears" noise as speech; lower if he misses quiet voices
WAIT_FOR_SPEECH_SECONDS = 6 # give up if nobody talks within this time
MAX_RECORD_SECONDS = 10     # longest question Momo will listen to
SILENCE_SECONDS = 1.2       # how long a pause means "I'm done talking"


# ======================================================================
# 5) BACKUP VOICE LINES  (made once, saved, played instantly - even when internet fails)
# ======================================================================
AUDIO_DIR = "momo_audio"
BACKUP_DIR = os.path.join(AUDIO_DIR, "backup")

BACKUP_LINES = {
    "hello": ["Hi hi! I'm Momo. Press the button and talk to me!"],
    "thinking": ["Hmm, let me think!", "Ooh, good question!", "Let me see, hehe!"],
    "again": ["Hm? Can you say that one more time?"],
    "offline": ["Oh no, Momo can't connect right now. Please try again!"],
    "busy": ["Whew, Momo needs a little rest. Please try again soon!"],
    "bye": ["Bye bye! Come visit Momo again, okay?"],
}


# ----------------------------------------------------------------------
# Helpers: audio output
# ----------------------------------------------------------------------
def play_file(path):
    """Play an MP3 and wait until it finishes."""
    pygame.mixer.music.load(path)
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():
        time.sleep(0.05)
    pygame.mixer.music.unload()


async def _make_speech(text, path):
    communicate = edge_tts.Communicate(text, VOICE, rate=RATE, pitch=PITCH, volume=VOLUME)
    await communicate.save(path)


def clean_for_speech(text):
    text = re.sub(r"[\U00010000-\U0010ffff]", "", text)  # emojis
    text = re.sub(r"[*_#`~>|]", "", text)                # markdown symbols
    return re.sub(r"\s+", " ", text).strip()


def speak(text):
    """Turn text into Momo's voice and play it. Falls back to a backup line if offline."""
    text = clean_for_speech(text)
    if not text:
        return
    path = os.path.join(AUDIO_DIR, "reply.mp3")
    try:
        asyncio.run(_make_speech(text, path))
    except Exception as e:
        print(f"  (voice problem: {e})")
        play_backup("offline")
        return
    play_file(path)


def ensure_backups():
    """Create the backup clips the first time (needs internet once)."""
    os.makedirs(BACKUP_DIR, exist_ok=True)
    for name, lines in BACKUP_LINES.items():
        for i, text in enumerate(lines, start=1):
            path = os.path.join(BACKUP_DIR, f"{name}_{i}.mp3")
            if os.path.exists(path):
                continue
            try:
                asyncio.run(_make_speech(text, path))
                print(f"  made backup clip: {name}_{i}.mp3")
            except Exception as e:
                print(f"  could not make {name}_{i}.mp3 ({e}). Connect to the internet and run again.")


def play_backup(name):
    options = [f for f in os.listdir(BACKUP_DIR) if f.startswith(name + "_")] if os.path.isdir(BACKUP_DIR) else []
    if options:
        play_file(os.path.join(BACKUP_DIR, random.choice(options)))


# ----------------------------------------------------------------------
# Helpers: listening (microphone -> text)
# ----------------------------------------------------------------------
def _level(block):
    return float(np.sqrt(np.mean(block.astype(np.float32) ** 2)))


def record_speech():
    """Record until the person stops talking. Returns a WAV path, or None if nobody spoke."""
    block = int(SAMPLE_RATE * 0.1)                       # 0.1 second chunks
    silence_blocks = int(SILENCE_SECONDS / 0.1)
    preroll = collections.deque(maxlen=3)                # keeps the first syllable
    frames = []
    started = False
    silent = 0

    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="int16", device=MIC_DEVICE) as stream:
        ambient = [_level(stream.read(block)[0]) for _ in range(3)]
        threshold = max(MIN_VOLUME, 2.5 * sum(ambient) / len(ambient))
        print("  Listening... (talk now!)")
        t0 = time.time()
        while time.time() - t0 < MAX_RECORD_SECONDS:
            data = stream.read(block)[0]
            loud = _level(data) > threshold
            if not started:
                preroll.append(data)
                if loud:
                    started = True
                    frames.extend(preroll)
                elif time.time() - t0 > WAIT_FOR_SPEECH_SECONDS:
                    return None
            else:
                frames.append(data)
                silent = 0 if loud else silent + 1
                if silent >= silence_blocks:
                    break

    if not started:
        return None
    path = os.path.join(tempfile.gettempdir(), "momo_heard.wav")
    with wave.open(path, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(SAMPLE_RATE)
        wav.writeframes(np.concatenate(frames).tobytes())
    return path


def speech_to_text(wav_path):
    """Returns the words heard, or None. Raises ConnectionError if the internet fails."""
    recognizer = sr.Recognizer()
    with sr.AudioFile(wav_path) as source:
        audio = recognizer.record(source)
    try:
        return recognizer.recognize_google(audio, language=LANGUAGE)
    except sr.UnknownValueError:
        return None
    except sr.RequestError as e:
        raise ConnectionError(str(e))


# ----------------------------------------------------------------------
# Helpers: the AI brain
# ----------------------------------------------------------------------
def load_api_key():
    key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not key:
        key_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gemini_key.txt")
        if os.path.exists(key_file):
            with open(key_file, encoding="utf-8") as f:
                key = f.read().strip()
    if not key:
        sys.exit(
            "No API key found.\n"
            "1) Get a free key at https://aistudio.google.com  ('Get API key')\n"
            "2) Save it in a file named gemini_key.txt next to this program."
        )
    return key


class RateLimited(Exception):
    pass


class Brain:
    def __init__(self, api_key):
        self.api_key = api_key
        self.history = []          # list of {"role": ..., "parts": [{"text": ...}]}
        self.working_model = None

    def ask(self, user_text):
        self.history.append({"role": "user", "parts": [{"text": user_text}]})
        self.history = self.history[-(REMEMBER_TURNS * 2):]

        models = [self.working_model] if self.working_model else MODEL_CANDIDATES
        last_error = "unknown error"
        for model in models:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
            body = {
                "system_instruction": {"parts": [{"text": PERSONALITY}]},
                "contents": self.history,
                "generationConfig": {"maxOutputTokens": MAX_REPLY_TOKENS, "temperature": 0.9},
            }
            try:
                resp = requests.post(url, headers={"x-goog-api-key": self.api_key}, json=body, timeout=20)
            except requests.RequestException as e:
                raise ConnectionError(str(e))

            if resp.status_code == 200:
                text = self._extract_text(resp.json())
                if text:
                    self.working_model = model
                    self.history.append({"role": "model", "parts": [{"text": text}]})
                    return text
                last_error = "empty answer"
                continue
            if resp.status_code == 404:          # that model name doesn't exist (any more)
                last_error = f"model {model} not found"
                continue
            if resp.status_code == 429:
                raise RateLimited()
            raise RuntimeError(f"AI error {resp.status_code}: {resp.text[:300]}")

        self.history.pop()                        # forget the question that failed
        raise RuntimeError(f"No working AI model found ({last_error}). See MODEL_CANDIDATES.")

    @staticmethod
    def _extract_text(data):
        try:
            parts = data["candidates"][0]["content"]["parts"]
            return " ".join(p.get("text", "") for p in parts).strip()
        except (KeyError, IndexError, TypeError):
            return ""


SCRIPTED_RULES = [
    # (words to listen for, [possible replies]) - Momo picks one reply at random.
    # These are SAMPLE lines. Edit or add your own!
    (r"\b(hi|hello|hey)\b", ["Hi hi! I'm Momo. Nice to meet you!", "Hello hello! I'm so happy you're here!"]),
    (r"your name|who are you", ["I'm Momo, your little robot friend!"]),
    (r"how are you|how do you feel", ["I'm super happy today! How about you?"]),
    (r"how old|your age", ["I'm brand new! Just born for this exhibition!"]),
    (r"who (made|built|created)|your (creator|maker)", ["The students at this school built me with lots of love!"]),
    (r"what can you do|what do you do", ["I can listen, talk, and be your friend. Try asking me something!"]),
    (r"favou?rite (food|thing to eat)|what do you eat|\bfood\b|hungry", ["I look like a dumpling, so I love dumplings! Hehe!"]),
    (r"favou?rite colou?r|\bcolou?r\b", ["I like soft cream and a little bit of pink!"]),
    (r"joke|funny", ["What do you call a robot who loves dumplings? A momo-bot! Hehe!"]),
    (r"\bsing\b|song", ["La la la! I only know a tiny song, but I'm learning!"]),
    (r"\bdance\b", ["I'd love to dance! My little arms go wiggle wiggle!"]),
    (r"cute|adorable|love you|like you", ["Hehe, you're making Momo blush!"]),
    (r"thank", ["You're welcome! That makes Momo happy!"]),
    (r"school|exhibition|project", ["This is our school exhibition! I'm so excited to be here!"]),
    (r"where are you from|where do you live", ["I live right here at the exhibition, with all my friends!"]),
]

SCRIPTED_FALLBACKS = [
    "Hehe, Momo doesn't know that one yet! Ask me something else?",
    "Ooh, that's a tricky one! Can you ask me another way?",
    "Hm, Momo is still learning that! Try asking my name!",
]


class ScriptedBrain:
    """No AI at all: Momo answers from SCRIPTED_RULES. Free, instant, and always safe."""
    needs_thinking_sound = False

    def ask(self, user_text):
        text = user_text.lower()
        for pattern, replies in SCRIPTED_RULES:
            if re.search(pattern, text):
                return random.choice(replies)
        return random.choice(SCRIPTED_FALLBACKS)


class OllamaBrain:
    """A small AI running on this laptop through Ollama (https://ollama.com). No account needed."""
    needs_thinking_sound = True

    def __init__(self):
        self.history = []

    def ask(self, user_text):
        self.history.append({"role": "user", "content": user_text})
        self.history = self.history[-(REMEMBER_TURNS * 2):]
        body = {
            "model": OLLAMA_MODEL,
            "messages": [{"role": "system", "content": PERSONALITY}] + self.history,
            "stream": False,
            "options": {"num_predict": 80, "temperature": 0.8},
        }
        try:
            resp = requests.post(OLLAMA_URL, json=body, timeout=120)
        except requests.RequestException:
            self.history.pop()
            raise ConnectionError("Ollama is not running. Open the Ollama app and try again.")
        if resp.status_code != 200:
            self.history.pop()
            raise RuntimeError(f"Ollama error {resp.status_code}: {resp.text[:200]}. "
                               f"Did you run:  ollama pull {OLLAMA_MODEL} ?")
        text = resp.json().get("message", {}).get("content", "").strip()
        if not text:
            self.history.pop()
            raise RuntimeError("Ollama gave an empty answer.")
        self.history.append({"role": "assistant", "content": text})
        return text


def make_brain():
    if BRAIN_MODE == "scripted":
        return ScriptedBrain()
    if BRAIN_MODE == "ollama":
        return OllamaBrain()
    if BRAIN_MODE == "gemini":
        return Brain(load_api_key())
    sys.exit('BRAIN_MODE must be "scripted", "ollama" or "gemini".')


def think_and_answer(brain, heard):
    """Play a 'thinking' line while the AI works, then return the AI's answer."""
    filler = None
    if getattr(brain, "needs_thinking_sound", True) and os.path.isdir(BACKUP_DIR):
        options = [f for f in os.listdir(BACKUP_DIR) if f.startswith("thinking_")]
        if options:
            filler = threading.Thread(target=play_file, args=(os.path.join(BACKUP_DIR, random.choice(options)),))
            filler.start()
    try:
        return brain.ask(heard)
    finally:
        if filler:
            filler.join()


# ----------------------------------------------------------------------
# One full conversation turn
# ----------------------------------------------------------------------
def handle_text(brain, heard):
    print(f"  You said: {heard}")
    if re.search(r"\b(goodbye|bye bye|see you)\b", heard.lower()):
        play_backup("bye")
        return
    try:
        reply = think_and_answer(brain, heard)
    except RateLimited:
        print("  (AI is busy / daily free limit reached)")
        play_backup("busy")
        return
    except ConnectionError:
        print("  (no internet)")
        play_backup("offline")
        return
    except RuntimeError as e:
        print(f"  ({e})")
        play_backup("offline")
        return
    print(f"  Momo says: {reply}")
    speak(reply)


def voice_turn(brain):
    wav = record_speech()
    if wav is None:
        print("  (nobody spoke)")
        return
    try:
        heard = speech_to_text(wav)
    except ConnectionError:
        print("  (no internet for speech recognition)")
        play_backup("offline")
        return
    if not heard:
        print("  (could not understand)")
        play_backup("again")
        return
    handle_text(brain, heard)


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Momo the talking robot")
    parser.add_argument("--type", action="store_true", help="type instead of talking (no microphone)")
    parser.add_argument("--devices", action="store_true", help="list microphones and exit")
    args = parser.parse_args()

    if args.devices:
        print(sd.query_devices())
        print("\nPut the number of your microphone into MIC_DEVICE at the top of this file.")
        return

    os.makedirs(AUDIO_DIR, exist_ok=True)
    pygame.mixer.init()
    brain = make_brain()
    print(f"Momo's brain: {BRAIN_MODE}")

    print("Getting Momo ready...")
    ensure_backups()
    play_backup("hello")
    print("\nMomo is ready!  (press Ctrl+C to stop)\n")

    while True:
        if args.type:
            heard = input("Type something for Momo: ").strip()
            if heard:
                handle_text(brain, heard)
        else:
            input("Press Enter, then talk to Momo...")
            voice_turn(brain)
        print()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nMomo is going to sleep. Bye bye!")
    except SystemExit as e:
        if e.code not in (None, 0):
            print(e)
    except Exception:
        import traceback
        print("\nSomething went wrong. Copy the message below and send it to your helper:\n")
        traceback.print_exc()
    input("\nPress Enter to close this window...")
