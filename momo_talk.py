"""
momo_talk.py  -  talk to Momo with your voice.

Say "Hey Momo" (or just "Momo"), then ask something.
Momo answers out loud and its face changes. After Momo answers it keeps
listening for ~20 seconds, so you can chat without saying the wake word again.

Needs momo_brain.py in the SAME folder.

Run:
    python momo_talk.py             # talk to Momo
    python momo_talk.py --devices   # list microphones (if Momo can't hear you)
"""

import re
import sys
import time
import numpy as np
import sounddevice as sd
import speech_recognition as sr

from momo_brain import ask_momo, speak, link, TOPIC_MOOD, API_KEY, history

# ---------- settings ----------
ALWAYS_LISTEN = True     # True = Momo answers everything you say. False = needs "Hey Momo" first
MIC_DEVICE = None        # None = default mic. Or a number from --devices
RATE = 16000
LANGUAGE = "en-IN"       # "en-US" for American English
MIN_VOLUME = 300         # raise if Momo reacts to noise, lower if it misses your voice
SILENCE_S = 0.9          # a pause this long means "I'm done talking"
MAX_RECORD_S = 10        # longest sentence
AWAKE_S = 20             # how long Momo keeps listening after an answer

# Google often mishears "Momo", so these all count as the wake word
WAKE = r"\b(hey momo|hi momo|ok momo|okay momo|momo|mo mo|mumu|moomoo|mommo|hey mom|hi mom)\b[,.!?]*\s*"

recognizer = sr.Recognizer()


def level(block):
    return float(np.sqrt(np.mean(block.astype(np.float32) ** 2)))


def record_phrase(wait_seconds=None):
    """Wait for speech, record until the person stops. Returns audio, or None if nobody spoke in time."""
    block = int(RATE * 0.1)
    silence_blocks = int(SILENCE_S / 0.1)
    before = []
    frames = []
    started = False
    silent = 0

    # the microphone is only open while we listen, so Momo never hears its own voice
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
                elif wait_seconds and time.time() - t0 > wait_seconds:
                    return None
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
        mood, reply = ask_momo(text)
    except Exception as e:
        print("Brain error:", e)
        if history:
            history.pop()
        return
    if mood:
        link.publish(TOPIC_MOOD, mood)
    print(f"Momo [{mood}]: {reply}")
    speak(reply)
    time.sleep(0.4)          # let the room go quiet before listening again


def main():
    if "--devices" in sys.argv:
        print(sd.query_devices())
        print("\nPut the number of your microphone into MIC_DEVICE at the top of this file.")
        return
    if not API_KEY:
        sys.exit("No GEMINI_API_KEY found. Run setx GEMINI_API_KEY \"your-key\" and open a NEW Command Prompt.")

    print("Momo is listening. Say 'Hey Momo' and then ask something. (Ctrl+C to stop)\n")
    awake_until = 0.0

    while True:
        if ALWAYS_LISTEN:                          # answer everything, no wake word needed
            audio = record_phrase()
            text = to_text(audio)
            if text:
                clean = re.sub(WAKE, "", text.lower(), count=1).strip()
                answer(clean if clean else text)
            continue

        awake = time.time() < awake_until

        if awake:
            print("  [listening...]")
            audio = record_phrase(wait_seconds=8)
            if audio is None:                      # nobody spoke: go back to waiting for the wake word
                awake_until = 0.0
                print("  (back to sleep, say 'Hey Momo')")
                continue
            text = to_text(audio)
            if not text:
                continue
            clean = re.sub(WAKE, "", text.lower(), count=1).strip()
            answer(clean if clean else text)
            awake_until = time.time() + AWAKE_S
        else:
            audio = record_phrase()                # waits as long as needed
            text = to_text(audio)
            if not text:
                continue
            low = text.lower()
            if not re.search(WAKE, low):
                continue                           # not for Momo, ignore
            question = re.sub(WAKE, "", low, count=1).strip()
            if question:
                answer(question)
            else:
                link.publish(TOPIC_MOOD, "happy")
                print("Momo: Yes?")
                speak("Yes? I'm listening!")
                time.sleep(0.4)
            awake_until = time.time() + AWAKE_S


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nBye!")
