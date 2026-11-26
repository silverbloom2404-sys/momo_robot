"""
momo_talk.py  -  Talk to Momo with your VOICE. Same Momo as momo_chat.py: face, camera,
memory and music all work, you just speak instead of typing.

Needs momo_brain.py, momo_camera.py, momo_music.py and momo_chat.py in the SAME folder.
Install once:   pip install sounddevice numpy

Run:
    python momo_talk.py                  # everything on, always listening
    python momo_talk.py --wake           # only answers when you say "Momo" (follow-ups stay open for 12 s)
    python momo_talk.py --no-camera      # without the webcam
    python momo_talk.py --no-voice       # Momo answers in text only (useful for testing)
    python momo_talk.py --list-mics      # show microphones, then:  --mic 2
    python momo_talk.py --threshold 400  # fixed loudness for "this is speech" (if auto is wrong)
    python momo_talk.py --no-barge       # turn off interrupting (he finishes every sentence)
    python momo_talk.py --barge-mult 2.2 # harder to interrupt (if he cuts himself off); 1.2 = easier

How it works:
  * Your mic is always open. When you start talking it records, and when you go quiet for
    about half a second it sends your voice to Gemini to turn it into text.
  * The text goes through exactly the same brain as momo_chat.py (music, camera, memory...).
  * INTERRUPTING: just talk over Momo (or say "stop", "wait", "never mind"). He learns how loud his
    own voice is in your room and stops when you are clearly louder than that. With headphones it
    works at normal volume; with speakers speak up a little.
  * Otherwise, while Momo talks (and for a moment after) the mic ignores echo, so he never answers himself.
  * While MUSIC is playing, Momo only reacts when you say "Momo ..." (so lyrics don't confuse him).
  * You can still TYPE any time and press Enter: stop, quit, /people, /face, /mood sad, /mic

Say things like:
    "hey Momo, play shape of you"   "pause"   "volume down"   "stop the music"
    "what is this?" (hold something up)   "solve x squared plus 5x plus 6 equals 0"   "forget me"
"""

import argparse
import base64
import collections
import io
import queue
import re
import sys
import threading
import time
import wave

try:
    import numpy as np
    import requests
    import sounddevice as sd
except ImportError as e:
    sys.exit(f"A tool is missing ({e.name}). Run:  pip install sounddevice numpy")

parser = argparse.ArgumentParser(description="Talk to Momo with your voice")
parser.add_argument("--wake", action="store_true", help='only answer when you say "Momo"')
parser.add_argument("--mic", type=int, default=None, help="microphone number (see --list-mics)")
parser.add_argument("--list-mics", action="store_true", help="show the microphones and exit")
parser.add_argument("--threshold", type=float, default=None, help="fixed loudness that counts as speech")
parser.add_argument("--no-barge", action="store_true", help="don't let me interrupt Momo by talking")
parser.add_argument("--barge-mult", type=float, default=1.6,
                    help="how much louder than Momo's own echo you must be to interrupt (default 1.6)")
parser.add_argument("--no-voice", action="store_true", help="Momo answers in text only")
parser.add_argument("--no-camera", action="store_true", help="run without the webcam")
ARGS, _ = parser.parse_known_args()

if ARGS.list_mics:
    print(sd.query_devices())
    sys.exit(0)

import momo_brain                     # noqa: E402  (imported after the quick --list-mics exit)
import momo_chat as chat              # noqa: E402  (reuses its music / camera / memory logic)
import momo_music                     # noqa: E402

# ---------------------------------------------------------------------
# settings (all the numbers you may want to tune live here)
# ---------------------------------------------------------------------
RATE = 16000              # samples per second sent to Gemini
FRAME_MS = 30             # the mic is checked in 30 ms pieces
FRAME = RATE * FRAME_MS // 1000
PREROLL_FRAMES = 12       # keep 0.36 s of audio from BEFORE you started, so the first word isn't cut
START_FRAMES = 3          # this many loud pieces in a row (90 ms) = you started talking
SPEC_SILENCE_S = 0.30     # after this much quiet, start turning speech into text EARLY (saves time)
END_SILENCE_S = 0.55      # this much quiet = you finished. Lower = faster reply, but may cut you off
MIN_SPEECH_S = 0.35       # shorter sounds (clicks, coughs) are ignored
MAX_UTTERANCE_S = 14      # safety limit for one sentence
TAIL_S = 0.6              # keep ignoring the mic this long after Momo stops (room echo)
MIN_THRESHOLD = 150       # quietest "speech" level allowed (16-bit loudness)
NOISE_MULT = 3.0          # speech must be this many times louder than the room noise
BARGE_FRAMES = 8          # you must stay louder than his echo for this many pieces (0.24 s) to interrupt
BARGE_GRACE_S = 0.5       # when Momo starts a new burst of speech, learn his echo level first (no interrupting)
BARGE_MIN_MULT = 1.5      # even with no echo (headphones), you must beat the normal speech level by this much
AWAKE_WINDOW_S = 12       # with --wake: how long Momo keeps listening after you say his name

WAKE_ANY = re.compile(r"\b(?:momo|mo mo|mumu|mommo)\b", re.I)
STOP_RE = re.compile(
    r"^(?:(?:ok(?:ay)?|hey|please|momo|mo mo)[\s,]+)*"
    r"(?:stop|stop it|stop talking|shh+|shush|quiet|be quiet|enough|that'?s enough|wait|hold on|never ?mind)"
    r"(?:[\s,]+(?:please|momo|now))*\W*$", re.I)
TAIL_NAME_RE = re.compile(r"[\s,.!?\-]*\b(?:momo|mo mo|mumu|mommo)\b[\s.!?]*$", re.I)

listener = None
awake_until = 0.0


# ---------------------------------------------------------------------
# voice -> text (Gemini hears the audio directly)
# ---------------------------------------------------------------------
STT_PROMPT = ("Transcribe this audio exactly as spoken. The speaker is chatting with a small robot named Momo "
              "(an Indian-English accent is possible). Write plain text with normal punctuation, in English "
              "letters. A robot voice may be faintly audible in the background: ignore it and write only what the "
              "main, closest person says. If there is no clear human speech (silence, music, background noise), reply with "
              "exactly: NONE. Output only the transcript, nothing else.")


def to_wav(pcm):
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(RATE)
        w.writeframes(pcm)
    return buf.getvalue()


def transcribe(pcm):
    """Audio bytes -> text. Returns '' when there was no real speech or every model failed."""
    data = base64.b64encode(to_wav(pcm)).decode()
    parts = [{"inline_data": {"mime_type": "audio/wav", "data": data}}, {"text": STT_PROMPT}]
    for use_think in (True, False):
        gen = {"maxOutputTokens": 150, "temperature": 0.0}
        if use_think:
            gen["thinkingConfig"] = {"thinkingLevel": "minimal"}
        body = {"contents": [{"role": "user", "parts": parts}], "generationConfig": gen}
        retry_plain = False
        for model in momo_brain.MODELS:
            try:
                r = momo_brain.http.post(
                    f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
                    json=body, headers={"x-goog-api-key": momo_brain.API_KEY}, timeout=15)
            except requests.RequestException:
                continue
            if r.status_code == 400 and use_think:
                retry_plain = True
                break
            if r.status_code != 200:
                continue
            try:
                out = r.json()["candidates"][0]["content"]["parts"]
                text = "".join(p.get("text", "") for p in out if not p.get("thought")).strip()
            except (ValueError, KeyError, IndexError):
                continue
            text = re.sub(r"^(?:transcript|transcription)\s*:\s*", "", text, flags=re.I).strip().strip('"')
            if text.upper().strip(" .") == "NONE" or len(text) < 2:
                return ""
            return text
        if not retry_plain:
            break
    return ""


class Transcription:
    """Turns one chunk of audio into text on a background thread.
    It can start EARLY (while you might still add more words) and is only delivered once the
    listener confirms you really finished (commit). If you keep talking it is cancelled."""

    def __init__(self, pcm, on_text):
        self.on_text = on_text
        self.lock = threading.Lock()
        self.text = None
        self.final = False
        self.cancelled = False
        self.delivered = False
        self.t_voice_end = 0.0
        threading.Thread(target=self._run, args=(pcm,), daemon=True).start()

    def _run(self, pcm):
        try:
            text = transcribe(pcm)
        except Exception as e:
            print(f"  (could not understand the audio: {e})")
            text = ""
        with self.lock:
            self.text = text
        self._maybe_deliver()

    def commit(self, t_voice_end):
        with self.lock:
            self.final = True
            self.t_voice_end = t_voice_end
        self._maybe_deliver()

    def cancel(self):
        with self.lock:
            self.cancelled = True

    def _maybe_deliver(self):
        with self.lock:
            if self.cancelled or self.delivered or not self.final or self.text is None:
                return
            self.delivered = True
            text, t_end = self.text, self.t_voice_end
        if text:
            print(f"  (voice to text: {time.time() - t_end:.1f}s after you stopped talking)")
            self.on_text(text)


# ---------------------------------------------------------------------
# microphone listener
# ---------------------------------------------------------------------
def busy_session():
    """The speech session Momo is in right now (writing or speaking an answer), or None."""
    s = momo_brain.current_session
    return s if (s is not None and s.thread.is_alive()) else None


# Momo's voice clips are played by momo_brain.play_mp3. We wrap it (without touching momo_brain.py)
# so the listener knows exactly when his voice is coming out of the speakers.
audio_state = {"playing": False, "clip_start": 0.0, "burst_start": 0.0, "last_end": 0.0}
_real_play_mp3 = momo_brain.play_mp3


def _tracked_play_mp3(path):
    now = time.time()
    if now - audio_state["last_end"] > 1.0:          # a new burst of speech (not the next clip of the same one)
        audio_state["burst_start"] = now
    audio_state["clip_start"] = now
    audio_state["playing"] = True
    try:
        _real_play_mp3(path)
    finally:
        audio_state["playing"] = False
        audio_state["last_end"] = time.time()


momo_brain.play_mp3 = _tracked_play_mp3


def rms_of(frame):
    a = np.frombuffer(frame, dtype=np.int16).astype(np.float32)
    return float(np.sqrt(np.mean(a * a))) if a.size else 0.0


class Listener:
    def __init__(self, inbox):
        self.inbox = inbox
        self.frames_q = queue.Queue()
        self.stream = None
        self.running = False
        self.noise = 60.0
        self.threshold = ARGS.threshold or 300.0
        self.level = 0.0
        self.quiet_until = 0.0
        self.echo_peak = 0.0            # how loud Momo's own voice is at the mic (learned while he talks)
        self.barge_run = 0
        self.barged_session = None      # the speech session you interrupted (the mic listens normally during it)

    def start(self):
        self.stream = sd.RawInputStream(samplerate=RATE, channels=1, dtype="int16", blocksize=FRAME,
                                        device=ARGS.mic, callback=self._callback)
        self.stream.start()
        self._calibrate()
        self.running = True
        threading.Thread(target=self._loop, daemon=True).start()

    def stop(self):
        self.running = False
        try:
            if self.stream:
                self.stream.stop()
                self.stream.close()
        except Exception:
            pass

    def describe(self):
        return (f"room noise {self.noise:.0f}, speech threshold {self.threshold:.0f}, "
                f"mic level right now {self.level:.0f}, "
                f"Momo's echo {self.echo_peak:.0f} (to interrupt him you must be louder than {self.barge_level():.0f})")

    def barge_level(self):
        return max(self.threshold * BARGE_MIN_MULT, self.echo_peak * ARGS.barge_mult)

    def _check_barge(self, level, now):
        """Called for every mic piece while Momo is talking/thinking. True = you are interrupting."""
        self.echo_peak *= 0.992                                      # forgets slowly (about 15% per second)
        st = audio_state
        if st["playing"] and now - st["burst_start"] < BARGE_GRACE_S:
            self.echo_peak = max(self.echo_peak, level)              # his voice just started: learn its loudness
            self.barge_run = 0
            return False
        if level > self.barge_level():                               # clearly louder than his own echo
            self.barge_run += 1                                      # (not learned as echo, or it would hide you)
        else:
            self.barge_run = 0
            self.echo_peak = max(self.echo_peak, level)
        return self.barge_run >= BARGE_FRAMES

    def _callback(self, indata, frames, t, status):
        self.frames_q.put(bytes(indata))

    def _calibrate(self):
        print("  [mic] listening to the room for a second, please stay quiet...")
        levels = []
        t_end = time.time() + 1.0
        while time.time() < t_end:
            try:
                levels.append(rms_of(self.frames_q.get(timeout=2)))
            except queue.Empty:
                raise RuntimeError("the microphone is not sending any sound")
        if not levels or max(levels) == 0:
            print("  [mic] WARNING: the microphone is completely silent. Check Windows Settings > "
                  "Privacy > Microphone, and that the right mic is selected (--list-mics).")
        self.noise = float(np.median(levels)) if levels else 60.0
        if ARGS.threshold is None:
            self.threshold = max(MIN_THRESHOLD, self.noise * NOISE_MULT)
        print(f"  [mic] ready. {self.describe()}")

    def _learn_noise(self, level):
        self.noise = 0.97 * self.noise + 0.03 * min(level, self.noise * 2 + 5)
        self.threshold = max(MIN_THRESHOLD, self.noise * NOISE_MULT)

    def _absorb_noise(self, levels):
        """Something loud and steady (fan, music) kept the mic busy: treat it as background."""
        self.noise = max(self.noise, float(np.mean(levels)) * 0.8)
        self.threshold = max(MIN_THRESHOLD, self.noise * NOISE_MULT)

    @staticmethod
    def _steady(levels):
        arr = np.array(levels, dtype=np.float32)
        return arr.mean() > 0 and float(arr.std() / arr.mean()) < 0.4      # speech wobbles, hum doesn't

    @staticmethod
    def _pcm(audio, silence):
        keep = len(audio) - max(0, silence - 5)        # drop most of the trailing quiet
        return b"".join(audio[:keep])

    def _on_text(self, text):
        self.inbox.put(("voice", text))

    def _loop(self):
        preroll = collections.deque(maxlen=PREROLL_FRAMES)
        audio, levels = [], []
        in_speech, voiced_run, silence = False, 0, 0
        spec = None
        fs = FRAME_MS / 1000.0

        while self.running:
            try:
                fr = self.frames_q.get(timeout=0.5)
            except queue.Empty:
                continue
            now = time.time()

            sess = busy_session()
            if sess is not None and sess is self.barged_session:
                pass                                   # you interrupted this one: listen to you normally
            elif sess is not None:
                # Momo is writing/talking: don't treat sound as a new sentence (it's mostly his own echo)
                self.quiet_until = now + TAIL_S
                if spec is not None:
                    spec.cancel()
                spec = None
                audio, levels = [], []
                in_speech, voiced_run, silence = False, 0, 0
                if ARGS.no_barge:
                    preroll.clear()
                    continue
                level = rms_of(fr)
                self.level = level
                preroll.append(fr)
                if self._check_barge(level, now):
                    # ---- you are talking over him: stop him and record what you say ----
                    self.barged_session = sess
                    self.quiet_until = 0.0
                    self.barge_run = 0
                    threading.Thread(target=momo_brain.stop_speaking, daemon=True).start()
                    print("  (interrupted)")
                    in_speech, silence = True, 0
                    audio, levels = list(preroll), []
                    preroll.clear()
                continue
            if now < self.quiet_until:                 # he just finished: let the room echo die away
                if spec is not None:
                    spec.cancel()
                spec = None
                preroll.clear()
                audio, levels = [], []
                in_speech, voiced_run, silence = False, 0, 0
                continue

            level = rms_of(fr)
            self.level = level
            voiced = level > self.threshold

            if not in_speech:
                preroll.append(fr)
                if not voiced and ARGS.threshold is None:
                    self._learn_noise(level)
                voiced_run = voiced_run + 1 if voiced else 0
                if voiced_run >= START_FRAMES:
                    in_speech = True
                    audio, levels, silence = list(preroll), [], 0
                continue

            audio.append(fr)
            levels.append(level)
            if voiced:
                silence = 0
                if spec is not None:                   # you kept talking: throw the early attempt away
                    spec.cancel()
                    spec = None
            else:
                silence += 1

            quiet_s = silence * fs
            if silence and spec is None and quiet_s >= SPEC_SILENCE_S:
                spec = Transcription(self._pcm(audio, silence), self._on_text)    # early start

            too_long = len(audio) * fs >= MAX_UTTERANCE_S
            if quiet_s < END_SILENCE_S and not too_long:
                continue

            # ---- the sentence is over ----
            speech_s = (len(audio) - silence) * fs
            steady = too_long and self._steady(levels)
            if steady:
                self._absorb_noise(levels)
            if speech_s < MIN_SPEECH_S or steady:
                if spec is not None:
                    spec.cancel()
            else:
                if spec is None:
                    spec = Transcription(self._pcm(audio, silence), self._on_text)
                spec.commit(now - silence * fs)
            spec = None
            preroll.clear()
            audio, levels = [], []
            in_speech, voiced_run, silence = False, 0, 0


# ---------------------------------------------------------------------
# handling what you said (same order of checks as momo_chat.py)
# ---------------------------------------------------------------------
def music_playing():
    p = momo_music.player
    return p.is_active() and not p.paused


def strip_name(text):
    """'hey momo play shape of you' / 'play shape of you momo' -> 'play shape of you'"""
    out = chat.strip_wake(text)
    out2 = TAIL_NAME_RE.sub("", out).strip()
    return out2 or out


def process(text, typed):
    """Handle one thing you said or typed. Returns False when it's time to quit."""
    global awake_until
    text = text.strip()
    if not text:
        return True
    low = text.lower().strip(" .!?,")

    if typed:
        if low in ("quit", "exit", "bye"):
            return False
        if low == "/face":
            momo_brain.check_face(3)
            return True
        if low == "/people":
            cam = chat.cam
            print("  people I know:", ", ".join(cam.store.names()) if cam and cam.store.names() else "nobody yet")
            return True
        if low == "/mic":
            print("  " + listener.describe())
            return True
        if low.startswith("/mood"):
            parts = low.split()
            if len(parts) == 2 and parts[1] in momo_brain.MOODS:
                momo_brain.set_mood(parts[1])
                print(f"  (sent mood '{parts[1]}' to the face)")
            else:
                print("  Use: /mood " + " | ".join(momo_brain.MOODS))
            return True
    else:
        print(f"You: {text}")

    if STOP_RE.match(strip_name(text)):          # "stop", "wait", "never mind"...: just go quiet
        momo_brain.stop_speaking()
        return True

    # ----- wake word rules -----
    now = time.time()
    in_intro = chat.stage is not None            # Momo just asked a name / a note: no wake word needed
    woke = bool(WAKE_ANY.search(text))
    if not typed and not in_intro:
        if music_playing():                      # music is on: only react when called
            need, window_open = True, False
        elif ARGS.wake:
            need, window_open = True, now < awake_until
        else:
            need, window_open = False, True
        if need and not woke and not window_open:
            print("  (ignored: say 'Momo' first)")
            return True
    if woke and not WAKE_ANY.sub("", text, count=1).strip(" ,.!?-:"):
        awake_until = now + AWAKE_WINDOW_S       # he was only called by name
        chat.say("Yes?", "happy")
        return True
    awake_until = now + AWAKE_WINDOW_S

    clean = strip_name(text)

    with chat.lock:
        if chat.stage == "name":
            chat.handle_name(text)
            return True
        if chat.stage == "about":
            chat.handle_about(text)
            return True
        if chat.try_music_command(clean):
            return True
        if chat.cam is not None and chat.try_camera_command(clean):
            return True
        ctx = chat.cam.context() if chat.cam is not None else "Your camera is turned off, so you cannot see anything."
        ctx = (ctx + " You cannot take photos or record videos.").strip()
        who = chat.cam.current_name if chat.cam is not None else None
        try:
            _mood, reply = momo_brain.ask_and_speak(clean, wait=False, context=ctx)
            chat.log_exchange(who, clean, reply)
        except Exception as e:
            print("Brain error:", e)
    return True


def typing_thread(inbox):
    """You can still type any time (stop, quit, /people ...)."""
    while True:
        try:
            line = input()
        except EOFError:
            return
        if line.strip():
            inbox.put(("typed", line))


def keep_connection_warm():
    """Keeps the connection to Google open, so the first words of each answer arrive faster."""
    while True:
        time.sleep(40)
        momo_brain.warm_up()


# ---------------------------------------------------------------------
# main
# ---------------------------------------------------------------------
def main():
    global listener
    if not momo_brain.API_KEY:
        sys.exit("No GEMINI_API_KEY found. Run:  setx GEMINI_API_KEY your-key  and open a NEW Command Prompt.")

    momo_brain.warm_up()
    print("Checking the face connection...")
    momo_brain.check_face()
    if chat.USE_CAMERA:
        chat.start_camera()

    inbox = queue.Queue()
    listener = Listener(inbox)
    try:
        listener.start()
    except Exception as e:
        sys.exit(f"Could not open the microphone: {e}\n"
                 "Try:  python momo_talk.py --list-mics   then   python momo_talk.py --mic <number>")

    threading.Thread(target=keep_connection_warm, daemon=True).start()
    threading.Thread(target=typing_thread, args=(inbox,), daemon=True).start()

    chat.say("Hi hi! I'm Momo. Talk to me!", "happy")
    mode = 'say "Momo" first, then your question' if ARGS.wake else "just talk, I'm always listening"
    print(f"\nMomo is listening ({mode}).")
    print("Try: 'play <song>', 'solve 3x + 5 = 20', 'what is this?'.")
    if not ARGS.no_barge:
        print("You can interrupt Momo any time by talking over him (or say 'stop'). With speakers, speak up a little.")
    print("You can also type: stop, /people, /mic, quit\n")

    while True:
        try:
            kind, text = inbox.get(timeout=0.2)
        except queue.Empty:
            continue
        if not process(text, kind == "typed"):
            break


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
    finally:
        try:
            if listener is not None:
                listener.stop()
            if chat.cam is not None:
                chat.save_all_memories()
                chat.cam.stop()
        except Exception:
            pass
        momo_music.player.stop()
        momo_brain.shutdown()
        print("\nBye!")
