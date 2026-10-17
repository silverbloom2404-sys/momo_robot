# momo_robot
# Momo

AI desk-pet robot. Animated face (Wokwi ESP32), AI brain (Gemini), voice (edge-tts).

## Install (Command Prompt)

```
pip install requests paho-mqtt edge-tts SpeechRecognition sounddevice numpy
```

Only typing to Momo (no microphone):

```
pip install requests paho-mqtt edge-tts
```

## Gemini API key (free)

1. Create a key at https://aistudio.google.com/apikey
2. Save it, then open a NEW Command Prompt:

```
setx GEMINI_API_KEY "your-key-here"
```

Never upload the key to GitHub.

## Wokwi libraries (Library Manager tab)

- Adafruit ILI9341
- Adafruit GFX Library
- Adafruit BusIO
- PubSubClient

## Run

1. Press Play in Wokwi and wait for `Voice link connected!`
2. In Command Prompt:

```
python momo_brain.py
```

3. Type a message, for example `hi momo`.

Test the microphone:

```
python mic_test.py
```

## MQTT topics

Prefix: `momo-robot-x7k2q9/` (use the same letters in every file)

| Topic | Message |
|---|---|
| `mood` | happy, surprised, wink, love, sad, angry, dead |
| `speaking` | 1 while Momo talks, 0 when done |

## Notes

- Do not use `pyaudio` or `pygame` (they fail to install on very new Python).
- If Gemini gives a 404, update the `MODELS` list in `momo_brain.py`.
