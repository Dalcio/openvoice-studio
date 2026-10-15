"""`openvoice-studio doctor`: environment checks, printed as JSON with a top-level "ok"."""

from __future__ import annotations

import importlib.util
import platform
import shutil
import subprocess
import sys

from . import __version__
from .config import data_dir, ffmpeg_path, load_settings, server_url


def _check(results: dict, name: str, required: bool, fn) -> None:
    try:
        ok, detail = fn()
    except Exception as exc:  # noqa: BLE001
        ok, detail = False, f"{type(exc).__name__}: {exc}"
    results[name] = {"ok": bool(ok), "required": required, "detail": detail}


def run_doctor() -> dict:
    checks: dict = {}

    _check(checks, "python", True, lambda: ((3, 10) <= sys.version_info[:2] <= (3, 12), f"{platform.python_version()} ({sys.executable})"))

    def ffmpeg():
        path = ffmpeg_path()
        if not path:
            return False, "ffmpeg não encontrado no PATH (necessário para MP3/M4A e gravações do browser)"
        out = subprocess.run([path, "-version"], capture_output=True, text=True, timeout=15).stdout.splitlines()
        return True, (out[0] if out else path)

    _check(checks, "ffmpeg", True, ffmpeg)

    def torch_check():
        import torch

        dev = "cuda" if torch.cuda.is_available() else "mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else "cpu"
        return True, f"torch {torch.__version__}, dispositivo: {dev}"

    _check(checks, "torch", True, torch_check)

    def engine_check():
        from .engines import get_engine

        st = get_engine(load_settings()["engine"]).status()
        return st["available"] and st["models_installed"], f"{load_settings()['engine']}: {st['detail']}"

    _check(checks, "engine_models", True, engine_check)

    def evaluation_check():
        from .evaluation import whisper_status

        st = whisper_status()
        return st["installed"], st["detail"]

    _check(checks, "evaluation_models", False, evaluation_check)

    def mcp_sdk():
        return importlib.util.find_spec("mcp") is not None, "SDK MCP instalado"

    _check(checks, "mcp_sdk", True, mcp_sdk)

    def data():
        d = data_dir()
        probe = d / ".write-test"
        probe.write_text("ok")
        probe.unlink()
        free = shutil.disk_usage(d).free / 1e9
        return True, f"{d} (livre: {free:.1f} GB)"

    _check(checks, "data_dir", True, data)

    def disk():
        free = shutil.disk_usage(data_dir()).free / 1e9
        return free >= 2, f"{free:.1f} GB livres"

    _check(checks, "disk_space", False, disk)

    def server():
        import httpx

        try:
            r = httpx.get(f"{server_url()}/api/health", timeout=2).json()
            return True, f"a correr em {server_url()} (v{r.get('version')})"
        except Exception:  # noqa: BLE001
            return True, f"parado (arranca com `openvoice-studio ui`; porta {load_settings()['port']})"

    _check(checks, "server", False, server)

    def claude_code_mcp():
        cli = shutil.which("claude")
        if not cli:
            return False, "CLI `claude` não encontrada"
        r = subprocess.run([cli, "mcp", "get", "openvoice-studio"], capture_output=True, text=True, timeout=60,
                           encoding="utf-8", errors="replace")
        return r.returncode == 0, "registado" if r.returncode == 0 else "não registado: corre `openvoice-studio setup`"

    _check(checks, "claude_code_mcp", False, claude_code_mcp)

    def install_location():
        exe = sys.executable.lower()
        if sys.platform == "win32" and "\\packages\\" in exe and "\\localcache\\" in exe:
            return False, ("instalado a partir de uma app empacotada (MSIX): o Python ficou no AppData privado dessa app. "
                           "Funciona dentro dela (ex.: Claude Desktop), mas não num terminal normal; para isso, "
                           "reinstala a partir de um PowerShell normal.")
        return True, sys.prefix

    _check(checks, "install_location", False, install_location)

    errors = [f"{k}: {v['detail']}" for k, v in checks.items() if v["required"] and not v["ok"]]
    warnings = [f"{k}: {v['detail']}" for k, v in checks.items() if not v["required"] and not v["ok"]]
    return {
        "ok": not errors,
        "version": __version__,
        "platform": f"{platform.system()} {platform.release()} ({platform.machine()})",
        "checks": checks,
        "errors": errors,
        "warnings": warnings,
    }
