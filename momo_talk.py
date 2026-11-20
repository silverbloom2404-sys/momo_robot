"""
momo_talk.py  -  THE file you run. Talk to Momo with your voice.

You speak, Momo listens, thinks, changes face and answers out loud.
Needs momo_brain.py (his brain) in the SAME folder.

Run:
    python momo_talk.py             # talk to Momo
    python momo_talk.py --devices   # list microphones
    python momo_talk.py --level     # test: shows how loud your mic hears you
"""

import re
import sys
import time
import numpy as np
import sounddevice as sd
import speech_recognition as sr

from momo_brain import ask_and_speak, stop_speaking, speak, warm_up, API_KEY, stop_flag

# ---------- settings ----------
HEADPHONES = False       # True = you wear headphones, so you can INTERRUPT Momo by talking.
                         # False = Momo stops listening while he talks (no echo problems on speakers).
ALWAYS_LISTEN = True     # True = answers everything you say. False = only when you say "Momo"
MIC_DEVICE = None        # None = default mic. Or a number from --devices
RATE = 16000
LANGUAGE = "en-IN"       # "en-US" for American English
MIN_VOLUME = 300         # raise if Momo reacts to noise, lower if he misses your voice
SILENCE_S = 0.7          # a pause this long means "I'm done talking" (lower = faster reply)
MAX_RECORD_S = 15        # longest sentence

STOP_WORDS = {"stop", "shh", "quiet", "enough", "wait", "okay stop", "ok stop"}
WAKE = r"\b(hey momo|hi momo|ok momo|okay momo|momo|mo mo|mumu|moomoo|mommo|hey mom|hi mom)\b[,.!?]*\s*"

recognizer = sr.Recognizer()


def level(block):
    return float(np.sqrt(np.mean(block.astype(np.float32) ** 2)))


def stop_talking():
    """Cut Momo off if he is still speaking."""
    stop_speaking()


def record_phrase(on_speech=None):
    """Wait for you to speak, record until you stop. Calls on_speech() the moment you start."""
    block = int(RATE * 0.1)
    silence_blocks = max(1, int(SILENCE_S / 0.1))
    before = []
    frames = []
    started = False
    silent = 0

    with sd.InputStream(samplerate=RATE, channels=1, dtype="int16", device=MIC_DEVICE) as stream:
        ambient = [level(stream.read(block)[0]) for _ in range(3)]
        threshold = max(MIN_VOLUME, 2.5 * sum(ambient) / len(ambient))
        t0 = time.time()
        while True:
            data = stream.read(block)[0]
            loud = level(data) > threshold
            if not started:
                before = (before + [data])[-3:]
                if loud:
                    started = True
                    frames.extend(before)
                    t0 = time.time()
                    if on_speech:
                        on_speech()
            else:
                frames.append(data)
                silent = 0 if loud else silent + 1
                if silent >= silence_blocks or time.time() - t0 > MAX_RECORD_S:
                    break
    return sr.AudioData(np.concatenate(frames).tobytes(), RATE, 2)


def to_text(audio):
    try:
        return recognizer.recognize_google(audio, language=LANGUAGE)
    except sr.UnknownValueError:
        return None
    except sr.RequestError:
        print("  (no internet for speech recognition)")
        return None


def answer(text):
    print(f"You: {text}")
    try:
        # Momo starts talking while the AI is still writing.
        # wait=True (speakers): comes back only after Momo finishes, so the mic never hears him.
        # wait=False (headphones): comes back at once, so you can interrupt him by talking.
        ask_and_speak(text, wait=not HEADPHONES)
    except Exception as e:
        print("Brain error:", e)
        return
    if not HEADPHONES:
        time.sleep(0.4)                     # let the room go quiet


def main():
    if "--devices" in sys.argv:
        print(sd.query_devices())
        print("\nPut the number of your microphone into MIC_DEVICE at the top of this file.")
        return
    if "--level" in sys.argv:
        print("Talk for 4 seconds...")
        rec = sd.rec(int(4 * RATE), samplerate=RATE, channels=1, dtype="int16", device=MIC_DEVICE)
        sd.wait()
        print("Loudest level:", int(np.abs(rec).max()), "(should be in the hundreds or thousands)")
        return
    if not API_KEY:
        sys.exit('No GEMINI_API_KEY found. Run setx GEMINI_API_KEY your-key and open a NEW Command Prompt.')

    warm_up()                                # open the connection to Google
    speak("Hi hi! I'm Momo. Talk to me!")    # hello, so you know the speaker works
    mode = "headphones: you can interrupt Momo by talking" if HEADPHONES else "speakers: Momo listens after he finishes"
    print(f"Momo is listening ({mode}). Just talk! Ctrl+C to stop.\n")

    while True:
        audio = record_phrase(on_speech=stop_talking if HEADPHONES else None)
        text = to_text(audio)
        if not text:
            continue

        low = re.sub(r"[^\w\s]", "", text.lower()).strip()
        if low in STOP_WORDS:
            print("You: (stop)")
            stop_talking()
            continue

        if not ALWAYS_LISTEN and not re.search(WAKE, text.lower()):
            continue                         # not meant for Momo
        clean = re.sub(WAKE, "", text.lower(), count=1).strip()
        answer(clean if clean else text)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        stop_flag.set()
        print("\nBye!")
