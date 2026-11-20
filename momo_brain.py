import asyncio
import atexit
import ctypes
import glob
import json
import os
import queue
import re
import sys
import tempfile
import threading
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
# voice clips are temporary: they live in Windows' temp folder and are deleted after playing
AUDIO_DIR = os.path.join(tempfile.gettempdir(), "momo_audio")
os.makedirs(AUDIO_DIR, exist_ok=True)


def clean_audio_folder():
    for f in glob.glob(os.path.join(AUDIO_DIR, "momo_reply_*.mp3")):
        try:
            os.remove(f)
        except OSError:
            pass


clean_audio_folder()                 # remove leftovers from earlier runs
atexit.register(clean_audio_folder)  # and clean up when Momo closes

FIRST_CLIP_MIN = 25     # first spoken piece: as short as possible so Momo starts talking fast
LATER_CLIP_MIN = 150    # later pieces are bigger (fewer gaps between clips)

MOODS = ["happy", "surprised", "wink", "love", "sad", "angry", "dead"]

SYSTEM = """You are Momo, a small, cute dumpling-shaped robot who lives on a desk and is a friend to the people around you.
You are playful, warm and a little cheeky. You can answer any question, tell jokes, play games (riddles, 20 questions, quizzes, word games) and chat about anything.

People type quickly and make spelling mistakes. Always guess what they meant and answer normally. Never mention or correct their spelling.

Your words will be spoken out loud, so:
- Match your length to the situation, like a real friend would:
  * SHORT (1 to 2 sentences): greetings, small talk, opinions, personal questions like "what do you like", yes/no questions, game moves, reactions.
  * MEDIUM (3 to 5 sentences): questions that need a little explanation or advice.
  * LONG (several paragraphs): ONLY when the person clearly asks for a story, a poem, an original song, or a detailed or long explanation.
- When unsure, go shorter. Never ramble and never give an essay unless asked.
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

stop_flag = threading.Event()   # set to interrupt Momo while he is talking
current_session = None          # the speech that is playing right now

history = []   # list of {"role": "user"/"model", "parts": [{"text": ...}]}
http = requests.Session()       # keeps the connection to Google open (faster)

# one MQTT connection that stays open
try:
    link = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
except AttributeError:
    link = mqtt.Client()
link.connect(BROKER, 1883, keepalive=60)
link.loop_start()


# =====================================================================
# Momo's voice
# =====================================================================
winmm = ctypes.windll.winmm     # Windows' built-in sound player


def play_mp3(path):
    """Play an MP3 with Windows. Returns when it ends, or right away if stop_flag is set."""
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


def clean_for_speech(text):
    text = re.sub(r"[\U00010000-\U0010ffff]", "", text)   # emojis
    text = re.sub(r"[*_#`~>|]", "", text)                 # markdown symbols
    return re.sub(r"\s+", " ", text).strip()


def remove_file(path):
    """Delete a clip in the background; retries because Windows may still hold the file for a moment."""
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
        if not text:
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
                link.publish(TOPIC_SPEAKING, "1")
                started = True
            play_mp3(path)
            remove_file(path)
        if started:
            link.publish(TOPIC_SPEAKING, "0")


def stop_speaking():
    """Cut Momo off if he is still talking."""
    s = current_session
    if s is not None and s.thread.is_alive():
        stop_flag.set()
        s.thread.join()


def speak(text, wait=True):
    """Say a ready-made text (used for short lines like 'Yes?')."""
    global current_session
    stop_speaking()
    stop_flag.clear()
    s = SpeechSession()
    current_session = s
    s.add(text)
    s.finish()
    if wait:
        s.thread.join()


# =====================================================================
# Momo's brain (streaming: Momo starts talking while the AI is still writing)
# =====================================================================
def stream_url(model):
    return f"https://generativelanguage.googleapis.com/v1beta/models/{model}:streamGenerateContent?alt=sse"


def stream_ai(contents, use_thinking_setting=True):
    """Yields the AI's answer in small pieces as it is being written."""
    gen = {"maxOutputTokens": 2500, "temperature": 0.9}
    if use_thinking_setting:
        gen["thinkingConfig"] = {"thinkingLevel": "minimal"}
    body = {
        "system_instruction": {"parts": [{"text": SYSTEM}]},
        "contents": contents,
        "generationConfig": gen,
    }
    last_error = None
    for attempt in range(2):                 # try the whole model list twice
        for model in MODELS:
            try:
                r = http.post(stream_url(model), json=body, stream=True,
                              headers={"x-goog-api-key": API_KEY}, timeout=45)
            except requests.RequestException as e:
                last_error = e
                continue
            if r.status_code == 400 and use_thinking_setting:
                r.close()
                yield from stream_ai(contents, use_thinking_setting=False)
                return
            if r.status_code in (429, 500, 503, 404):    # busy or missing: try the next model
                last_error = f"{r.status_code} from {model}"
                r.close()
                continue
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
                piece = "".join(p.get("text", "") for p in parts)
                if piece:
                    yield piece
            return
        time.sleep(1)
    raise Exception(f"All models busy ({last_error}). Try again in a minute.")


LONG_WORDS = re.compile(
    r"\b(long|longer|story|stories|detail|detailed|essay|paragraph|paragraphs|poem|elaborate|in depth|keep going|go on)\b", re.I)
MEDIUM_WORDS = re.compile(
    r"\b(explain|how does|how do|how can|why|teach|describe|difference|tell me about|help me)\b", re.I)

LONG_HINT = ("\n\n(Momo: this person asked for a LONG answer. Write at least 200 words in several paragraphs, "
             "spoken naturally. If you don't know the exact topic, just pick a fun version and tell it fully.)")
MEDIUM_HINT = "\n\n(Momo: answer in 3 to 5 sentences, no more.)"
SHORT_HINT = "\n\n(Momo: reply in 1 to 2 short sentences, like a friend chatting. No essay.)"


def length_hint(user_text):
    if LONG_WORDS.search(user_text):
        return LONG_HINT
    if MEDIUM_WORDS.search(user_text):
        return MEDIUM_HINT
    return SHORT_HINT


def ask_and_speak(user_text, wait=False):
    """Ask the AI and speak the answer while it is still being written.
    Sends the face mood the moment the AI picks it. Returns (mood, reply_text).
    wait=True: returns only after Momo has finished talking."""
    global current_session
    stop_speaking()
    stop_flag.clear()

    history.append({"role": "user", "parts": [{"text": user_text}]})
    recent = list(history[-19:])
    if recent[0]["role"] != "user":
        recent = recent[1:]
    recent[-1] = {"role": "user", "parts": [{"text": user_text + length_hint(user_text)}]}

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

    try:
        for piece in stream_ai(recent):
            raw += piece
            buf += piece

            if not tag_done:                         # read the mood tag first, like [happy]
                m = re.match(r"\s*\[(\w+)\]\s*", buf)
                if m:
                    tag = m.group(1).lower()
                    if tag in MOODS:
                        mood = tag
                        link.publish(TOPIC_MOOD, mood)   # face changes right away
                    buf = buf[m.end():]
                    tag_done = True
                elif len(buf) > 30 or (buf.strip() and not buf.lstrip().startswith("[")):
                    tag_done = True                  # the AI forgot the tag
                else:
                    continue

            while True:                              # hand finished sentences to the voice
                m = re.search(r"^(.+?[.!?])\s+", buf, re.S)
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

    tail = (pending + " " + buf).strip()
    if not raw.strip():                              # the AI sent nothing: say sorry instead
        history.pop()
        mood = "surprised"
        link.publish(TOPIC_MOOD, mood)
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
    print("This file is Momo's brain. To talk to Momo, run:  python momo_talk.py")
