"""Paths, token and persistent settings.

Everything lives in one data directory (override with OVS_HOME):
  Windows  %USERPROFILE%\\.openvoice-studio
  macOS    ~/Library/Application Support/OpenVoiceStudio
  Linux    $XDG_DATA_HOME/openvoice-studio (default ~/.local/share/openvoice-studio)
"""

from __future__ import annotations

import json
import os
import secrets
import shutil
import sys
from pathlib import Path

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8765
DEFAULT_ENGINE = "chatterbox-mtl"
DEFAULT_LANGUAGE = "pt"


def data_dir() -> Path:
    if env := os.environ.get("OVS_HOME"):
        base = Path(env).expanduser()
    elif sys.platform == "win32":
        # Not %LOCALAPPDATA%: packaged (MSIX) apps such as Claude Desktop get a private, virtualised
        # AppData, so a server started from inside the app and one started from a terminal would
        # otherwise see different tokens and voices.
        base = Path.home() / ".openvoice-studio"
    elif sys.platform == "darwin":
        base = Path.home() / "Library" / "Application Support" / "OpenVoiceStudio"
    else:
        base = Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share")) / "openvoice-studio"
    base.mkdir(parents=True, exist_ok=True)
    return base


def voices_dir() -> Path:
    p = data_dir() / "voices"
    p.mkdir(parents=True, exist_ok=True)
    return p


def outputs_dir() -> Path:
    p = data_dir() / "outputs"
    p.mkdir(parents=True, exist_ok=True)
    return p


def logs_dir() -> Path:
    p = data_dir() / "logs"
    p.mkdir(parents=True, exist_ok=True)
    return p


def get_token() -> str:
    """Return the local API token, creating it on first use."""
    path = data_dir() / "token"
    if path.exists():
        token = path.read_text(encoding="utf-8").strip()
        if token:
            return token
    token = secrets.token_urlsafe(24)
    path.write_text(token, encoding="utf-8")
    if sys.platform != "win32":
        path.chmod(0o600)
    return token


def load_settings() -> dict:
    path = data_dir() / "config.json"
    settings = {"engine": DEFAULT_ENGINE, "language": DEFAULT_LANGUAGE, "port": DEFAULT_PORT}
    if path.exists():
        try:
            settings.update(json.loads(path.read_text(encoding="utf-8")))
        except json.JSONDecodeError:
            pass
    if env := os.environ.get("OVS_ENGINE"):
        settings["engine"] = env
    if env := os.environ.get("OVS_PORT"):
        settings["port"] = int(env)
    return settings


def server_url(port: int | None = None) -> str:
    return f"http://{DEFAULT_HOST}:{port or load_settings()['port']}"


def ui_link(port: int | None = None) -> str:
    return f"{server_url(port)}/#token={get_token()}"


def desktop_dir() -> Path:
    """The user's real Desktop folder (handles OneDrive-redirected Desktops on Windows)."""
    if sys.platform == "win32":
        try:
            import ctypes
            from ctypes import wintypes

            buf = ctypes.create_unicode_buffer(wintypes.MAX_PATH)
            # CSIDL_DESKTOPDIRECTORY = 0x10
            if ctypes.windll.shell32.SHGetFolderPathW(None, 0x10, None, 0, buf) == 0 and buf.value:
                return Path(buf.value)
        except Exception:
            pass
    elif sys.platform.startswith("linux") and shutil.which("xdg-user-dir"):
        import subprocess

        try:
            out = subprocess.run(["xdg-user-dir", "DESKTOP"], capture_output=True, text=True, timeout=5).stdout.strip()
            if out:
                return Path(out)
        except Exception:
            pass
    return Path.home() / "Desktop"


def ffmpeg_path() -> str | None:
    return shutil.which("ffmpeg")
