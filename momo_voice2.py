"""
Momo's Voice Generator
----------------------
Free, works on any laptop (Windows / Mac / Linux). No Raspberry Pi needed.

SETUP (one time):
    pip install edge-tts

USAGE:
    python momo_voice.py --audition        # make test clips in 3 pitch styles + a few voices
    python momo_voice.py                   # generate every line in LINES below
    python momo_voice.py --say "Hello!"    # make one quick clip from any text

Internet is needed ONLY while generating. Once the MP3 files exist,
you can play them offline at the exhibition.

If ffmpeg is installed, clips are also polished for a small speaker
(bass cut, clarity boost, even volume). If not, that step is skipped.
"""

import argparse
import asyncio
import os
import platform
import shutil
import subprocess
import sys

try:
    import edge_tts
except ImportError:
    sys.exit("edge-tts is not installed. Run:  pip install edge-tts")


# ----------------------------------------------------------------------
# 1) MOMO'S VOICE SETTINGS  (change these to tune the voice by ear)
# ----------------------------------------------------------------------
VOICE = "en-US-AnaNeural"   # child-like voice. Alternatives: en-US-JennyNeural, en-US-AriaNeural
PITCH = "+15Hz"             # higher = cuter, but stop when words get blurry (try +0Hz to +30Hz)
RATE = "-8%"                # negative = slower and clearer (try -5% to -12%)
VOLUME = "+0%"              # raise if too quiet, e.g. "+10%"

OUTPUT_DIR = "momo_audio"


# ----------------------------------------------------------------------
# 2) MOMO'S LINES  (placeholders for now - we'll write the real dialogue later)
#    Format:  "file_name": "what Momo says",
# ----------------------------------------------------------------------
LINES = {
    "greeting": "Hi hi! I'm Momo. Nice to meet you!",
    "listening": "Ooh, I'm listening!",
    "not_understood": "Hm? Can you say that one more time?",
    "happy": "Yay! That makes me so happy!",
    "goodbye": "Bye bye! Come visit me again, okay?",
}


# ----------------------------------------------------------------------
# 3) AUDITION SETTINGS
# ----------------------------------------------------------------------
TEST_SENTENCE = "Hi hi! I'm Momo. Nice to meet you! Do you want to see something cool?"

AUDITION_STYLES = {
    "A_natural": ("+0Hz", "-5%"),
    "B_cute": ("+15Hz", "-8%"),
    "C_extra_cute": ("+30Hz", "-10%"),
}

AUDITION_VOICES = [
    "en-US-AnaNeural",
    "en-US-JennyNeural",
    "en-US-AriaNeural",
]


# ----------------------------------------------------------------------
# Core functions
# ----------------------------------------------------------------------
async def make_clip(text, path, voice=VOICE, pitch=PITCH, rate=RATE, volume=VOLUME):
    """Generate one MP3 file from text."""
    communicate = edge_tts.Communicate(text, voice, rate=rate, pitch=pitch, volume=volume)
    await communicate.save(path)


def polish(src, dst):
    """Make the clip clearer on a small speaker using ffmpeg (if available)."""
    if not shutil.which("ffmpeg"):
        shutil.copyfile(src, dst)
        return False
    filters = "highpass=f=150,equalizer=f=2500:t=q:w=1:g=3,loudnorm=I=-16"
    result = subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", src, "-af", filters, dst],
        capture_output=True,
    )
    if result.returncode != 0:
        shutil.copyfile(src, dst)
        return False
    return True


def play(path):
    """Open an audio file with the computer's default player."""
    system = platform.system()
    try:
        if system == "Windows":
            os.startfile(path)  # type: ignore[attr-defined]
        elif system == "Darwin":
            subprocess.run(["afplay", path])
        else:
            subprocess.run(["xdg-open", path])
    except Exception as e:
        print(f"  (could not auto-play: {e})")


async def generate_all():
    raw_dir = os.path.join(OUTPUT_DIR, "raw")
    final_dir = os.path.join(OUTPUT_DIR, "final")
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(final_dir, exist_ok=True)

    print(f"Voice: {VOICE} | pitch {PITCH} | rate {RATE}\n")
    polished_any = False
    for name, text in LINES.items():
        raw_path = os.path.join(raw_dir, f"{name}.mp3")
        final_path = os.path.join(final_dir, f"{name}.mp3")
        try:
            await make_clip(text, raw_path)
            polished_any = polish(raw_path, final_path) or polished_any
            print(f"  done: {name}.mp3  ->  \"{text}\"")
        except Exception as e:
            print(f"  FAILED: {name} ({e}). Check your internet connection.")

    print(f"\nFinished! Use the files in:  {final_dir}")
    if not polished_any:
        print("(ffmpeg not found, so the small-speaker polish was skipped. That's okay!)")


async def audition():
    out = os.path.join(OUTPUT_DIR, "audition")
    os.makedirs(out, exist_ok=True)
    print("Making audition clips. Listen to all of them and pick your favorite.\n")
    for voice in AUDITION_VOICES:
        short = voice.replace("en-US-", "").replace("Neural", "")
        for style, (pitch, rate) in AUDITION_STYLES.items():
            path = os.path.join(out, f"{short}_{style}.mp3")
            try:
                await make_clip(TEST_SENTENCE, path, voice=voice, pitch=pitch, rate=rate)
                print(f"  done: {short}_{style}.mp3  (pitch {pitch}, rate {rate})")
            except Exception as e:
                print(f"  FAILED: {short}_{style} ({e})")
    print(f"\nOpen the folder '{out}' and compare the clips.")
    print("Then copy your favorite VOICE / PITCH / RATE into the settings at the top of this file.")


async def say(text, do_play):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    raw = os.path.join(OUTPUT_DIR, "say_raw.mp3")
    final = os.path.join(OUTPUT_DIR, "say.mp3")
    await make_clip(text, raw)
    polish(raw, final)
    print(f"Saved: {final}")
    if do_play:
        play(final)


def main():
    parser = argparse.ArgumentParser(description="Generate Momo's voice (free, laptop-friendly).")
    parser.add_argument("--audition", action="store_true", help="make test clips in different voices and styles")
    parser.add_argument("--say", metavar="TEXT", help="make a single clip from this text")
    parser.add_argument("--play", action="store_true", help="play the clip after making it (with --say)")
    args = parser.parse_args()

    if args.audition:
        asyncio.run(audition())
    elif args.say:
        asyncio.run(say(args.say, args.play))
    else:
        asyncio.run(generate_all())


if __name__ == "__main__":
    try:
        main()
    except SystemExit as e:
        if e.code not in (None, 0):
            print(e)
    except Exception:
        import traceback
        print("\nSomething went wrong. Copy the message below and send it to your helper:\n")
        traceback.print_exc()
    # Keep the window open so you can read everything
    input("\nPress Enter to close this window...")
