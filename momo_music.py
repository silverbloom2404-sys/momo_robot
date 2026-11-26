"""
momo_music.py - Momo plays music, like a smart speaker.  (Windows only for now.)

  * First looks for a matching file in the  momo_songs  folder next to the code
    (drop your own .mp3 / .wav files in there; it always works, even offline).
  * Otherwise finds the song on YouTube, downloads just the audio, and plays it.
  * Volume goes down by itself while Momo talks, and back up afterwards.

Install for online songs:   pip install yt-dlp imageio-ffmpeg
(If online songs stop working one day, run:  pip install -U yt-dlp )
"""

import atexit
import ctypes
import glob
import os
import queue
import re
import sys
import tempfile
import threading

HERE = os.path.dirname(os.path.abspath(__file__))
LOCAL_DIR = os.path.join(HERE, "momo_songs")                        # put your own songs here
MUSIC_DIR = os.path.join(tempfile.gettempdir(), "momo_music")       # temporary downloads
os.makedirs(LOCAL_DIR, exist_ok=True)
os.makedirs(MUSIC_DIR, exist_ok=True)

DUCK_LEVEL = 0.25          # music volume (as a fraction) while Momo is talking
MAX_FILE_MB = 40           # skip huge downloads (hour-long mixes)
ALIAS = "momo_music"

winmm = ctypes.windll.winmm if sys.platform.startswith("win") else None


def _mci(cmd):
    if winmm is None:
        return -1
    return winmm.mciSendStringW(cmd, None, 0, 0)


def _mode():
    if winmm is None:
        return ""
    buf = ctypes.create_unicode_buffer(32)
    winmm.mciSendStringW(f"status {ALIAS} mode", buf, 32, 0)
    return buf.value


class MusicPlayer:
    """All sound commands run on one worker thread, so they never clash."""

    def __init__(self):
        self.q = queue.Queue()
        self.base = 70            # volume you chose, 0..100
        self.ducked = False
        self.playing = False
        self.paused = False
        self.title = ""
        threading.Thread(target=self._worker, daemon=True).start()

    # ---- what the rest of Momo calls ----
    def play(self, path, title=""):
        self.title = title
        self.q.put(("play", path))

    def pause(self):
        self.q.put(("pause",))

    def resume(self):
        self.q.put(("resume",))

    def stop(self):
        self.q.put(("stop",))

    def change_volume(self, delta):
        self.base = max(10, min(100, self.base + delta))
        self.q.put(("vol",))
        return self.base

    def duck(self, on):
        self.ducked = on
        self.q.put(("vol",))

    def is_active(self):
        return self.playing

    # ---- worker ----
    def _apply_volume(self):
        level = self.base * (DUCK_LEVEL if self.ducked else 1.0)
        _mci(f"setaudio {ALIAS} volume to {int(level * 10)}")

    def _worker(self):
        while True:
            try:
                cmd = self.q.get(timeout=1)
            except queue.Empty:                       # check whether the song ended by itself
                if self.playing and not self.paused and _mode() != "playing":
                    _mci(f"close {ALIAS}")
                    self.playing = False
                continue
            op = cmd[0]
            if op == "play":
                _mci(f"close {ALIAS}")
                if _mci(f'open "{cmd[1]}" type mpegvideo alias {ALIAS}') != 0:
                    print("  (music: Windows could not open that file)")
                    self.playing = False
                    continue
                self._apply_volume()
                _mci(f"play {ALIAS}")
                self.playing, self.paused = True, False
            elif op == "pause":
                _mci(f"pause {ALIAS}")
                self.paused = True
            elif op == "resume":
                _mci(f"resume {ALIAS}")
                self.paused = False
            elif op == "stop":
                _mci(f"stop {ALIAS}")
                _mci(f"close {ALIAS}")
                self.playing = self.paused = False
            elif op == "vol":
                self._apply_volume()
            elif op == "quit":
                _mci(f"close {ALIAS}")
                return


player = MusicPlayer()


def duck(on):
    """Called by Momo's voice: quiet the music while he talks."""
    player.duck(on)


def shutdown():
    player.q.put(("quit",))


def clean_downloads():
    for f in glob.glob(os.path.join(MUSIC_DIR, "*")):
        try:
            os.remove(f)
        except OSError:
            pass


clean_downloads()
atexit.register(clean_downloads)


# ---------------------------------------------------------------------
# finding songs
# ---------------------------------------------------------------------
def _words(s):
    return [w for w in re.findall(r"[a-z0-9]+", s.lower()) if w not in {"the", "a", "an", "by", "song", "music"}]


def find_local(query):
    """Best matching file in momo_songs, or None."""
    want = _words(query)
    if not want:
        return None
    best, best_score = None, 0
    for f in glob.glob(os.path.join(LOCAL_DIR, "*")):
        if not f.lower().endswith((".mp3", ".wav", ".wma", ".m4a")):
            continue
        name = _words(os.path.splitext(os.path.basename(f))[0])
        score = sum(1 for w in want if w in name)
        if score > best_score and score >= max(1, len(want) // 2):
            best, best_score = f, score
    return best


def find_online(query):
    """Download the audio of the best YouTube match. Returns (path, title)."""
    try:
        import yt_dlp
    except ImportError:
        raise RuntimeError("online songs need:  pip install yt-dlp imageio-ffmpeg")
    ffmpeg = None
    try:
        import imageio_ffmpeg
        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        pass

    clean_downloads()
    opts = {
        "outtmpl": os.path.join(MUSIC_DIR, "%(id)s.%(ext)s"),
        "noplaylist": True, "quiet": True, "no_warnings": True,
        "socket_timeout": 20, "retries": 2,
        "max_filesize": MAX_FILE_MB * 1024 * 1024,
    }
    if ffmpeg:                                           # convert to mp3 so Windows can play it
        opts["format"] = "bestaudio/best"
        opts["ffmpeg_location"] = ffmpeg
        opts["postprocessors"] = [{"key": "FFmpegExtractAudio", "preferredcodec": "mp3", "preferredquality": "128"}]
    else:
        opts["format"] = "bestaudio[ext=m4a]/bestaudio"
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(f"ytsearch1:{query}", download=True)
    entries = (info or {}).get("entries") or ([info] if info else [])
    if not entries or not entries[0]:
        raise RuntimeError("no result found")
    e = entries[0]
    files = glob.glob(os.path.join(MUSIC_DIR, e["id"] + "*"))
    files = [f for f in files if not f.endswith((".part", ".ytdl"))]
    if not files:
        raise RuntimeError("the download didn't finish (maybe the file was too big)")
    return files[0], e.get("title", query)


def find_song(query):
    """Local file first, then YouTube. Returns (path, title)."""
    local = find_local(query)
    if local:
        return local, os.path.splitext(os.path.basename(local))[0]
    return find_online(query)
