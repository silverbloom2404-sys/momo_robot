# momo_robot
# Momo

AI desk-pet robot. Animated face (Wokwi ESP32), AI brain (Gemini), voice (edge-tts).

## Install (Command Prompt)

```pip install further packages -
C:\Users\HP>pip list
Package            Version
------------------ ----------
aiohappyeyeballs   2.7.1
aiohttp            3.14.5
aiosignal          1.4.0
attrs              26.1.0
audioop-lts        0.2.2
certifi            2026.7.22
cffi               2.1.1
charset-normalizer 3.5.2
edge-tts           7.3.1
frozenlist         1.8.0
idna               3.20
imageio-ffmpeg     0.8.1
multidict          7.1.0
numpy              2.5.4
opencv-python      4.14.0.94
paho-mqtt          2.1.0
pip                26.2.1
propcache          0.5.4
pycparser          3.11
requests           2.34.2
sounddevice        0.5.6
SpeechRecognition  3.17.0
standard-aifc      3.13.0
standard-chunk     3.13.0
tabulate           0.10.0
typing_extensions  4.16.0
urllib3            2.8.0
yarl               1.25.1
yt-dlp             2026.11.23
```

Only typing to Momo (no microphone):

```
pip install requests paho-mqtt edge-tts

```
pip install opencv-python numpy


## Gemini API key (free)

1. Create a key at https://aistudio.google.com/apikey
2. Save it, then open a NEW Command Prompt:

```
setx GEMINI_API_KEY AQ.Ab8RN6I5EUIgkBHPMQzgCbwYLaQ54PQOBInGe6uNPjaj61_fWg
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
