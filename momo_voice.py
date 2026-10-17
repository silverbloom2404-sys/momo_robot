import time
import numpy as np
import sounddevice as sd
import speech_recognition as sr
import paho.mqtt.client as mqtt

BROKER = "broker.hivemq.com"
TOPIC = "momo-robot-x7k2q9/mood"   # must match the ESP32 code

# words that trigger each mood (add your own!)
MOODS = {
    "happy":     ["happy", "smile", "glad", "joy", "cheerful"],
    "surprised": ["surprised", "surprise", "shocked", "wow"],
    "wink":      ["wink", "winked", "winking", "wing", "ink", "win", "wick", "link","wynk"
                  "naughty", "flirt", "tease","flirty"],
    "love":      ["love", "heart", "hearts"],
    "sad":       ["sad", "cry", "crying", "upset"],
    "angry":     ["angry", "anger", "mad", "furious"],
    "dead":      ["dead", "die", "died", "dying", "dies"],
}

RATE = 16000
CHUNK = int(RATE * 0.05)      # 0.05 second pieces (faster reaction)
END_SILENCE = 8               # 8 x 0.05 = 0.4 s of silence ends the phrase

def find_mood(text):
    words = text.lower().replace(",", " ").replace(".", " ").split()
    for mood, triggers in MOODS.items():
        for t in triggers:
            if t in words:
                return mood
    return None

def record_phrase(stream, threshold):
    """Waits for you to speak, records until you stop, returns raw audio bytes."""
    frames = []
    before = []
    started = False
    silent = 0
    while True:
        data, _ = stream.read(CHUNK)
        level = np.abs(data).mean()
        if not started:
            before = (before + [data])[-3:]        # keep the last 0.15 s before speech
            if level > threshold:
                started = True
                frames.extend(before)
        else:
            frames.append(data)
            silent = silent + 1 if level < threshold else 0
            if silent >= END_SILENCE or len(frames) > 100:   # 5 s max
                break
    return np.concatenate(frames).tobytes()

# one connection that stays open (much faster than reconnecting every time)
try:
    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
except AttributeError:
    client = mqtt.Client()
client.connect(BROKER, 1883, keepalive=60)
client.loop_start()

r = sr.Recognizer()

try:
    with sd.InputStream(samplerate=RATE, channels=1, dtype="int16") as stream:
        print("Calibrating mic, stay quiet for a second...")
        ambient, _ = stream.read(RATE)
        threshold = np.abs(ambient).mean() * 3 + 150
        print("Listening! Say: 'angry', 'happy', 'love', 'wink', 'dead'. Ctrl+C to stop.")

        while True:
            raw = record_phrase(stream, threshold)
            t0 = time.time()
            try:
                audio = sr.AudioData(raw, RATE, 2)
                text = r.recognize_google(audio, language="en-IN")
                took = time.time() - t0
                print(f"Heard: {text}   (recognising took {took:.1f}s)")
                mood = find_mood(text)
                if mood:
                    print("  -> sending mood:", mood)
                    client.publish(TOPIC, mood)
                else:
                    print("  (no mood word found)")
            except sr.UnknownValueError:
                print("  (couldn't understand)")
            except sr.RequestError:
                print("No internet for speech recognition")
except KeyboardInterrupt:
    client.loop_stop()
    print("Bye!")