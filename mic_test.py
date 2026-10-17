import numpy as np
import sounddevice as sd

for d in [1, 2, 8, 14, 23]:
    try:
        print(f"Device {d}: talk now...")
        rec = sd.rec(int(3 * 16000), samplerate=16000, channels=1, dtype="int16", device=d)
        sd.wait()
        print("   level:", int(np.abs(rec).max()))
    except Exception as e:
        print("   failed:", str(e)[:60])