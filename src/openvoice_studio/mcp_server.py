"""MCP server (stdio). A thin client of the local HTTP API, which it starts on demand, so the web UI
and every Claude session share one model in memory and one job queue.

Never print to stdout here: stdout is the MCP transport.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import httpx
from mcp.server.fastmcp import FastMCP

from .config import desktop_dir, get_token, logs_dir, server_url

mcp = FastMCP(
    "openvoice-studio",
    instructions=(
        "OpenVoice Studio generates narration locally with voices the user created and consented to in the web UI. "
        "Typical flow: get_engine_capabilities -> list_voices -> generate_narration (returns when done or after "
        "wait_seconds; then poll get_job_status) -> get_audio_artifact to copy the WAV somewhere useful -> "
        "run_voice_evaluation for similarity/intelligibility metrics. On a laptop CPU, generation takes roughly 5-10x the audio "
        "duration and the first call also loads the model, so use wait_seconds of 300 for short texts."
    ),
)


def _client(timeout: float = 30) -> httpx.Client:
    return httpx.Client(base_url=server_url(), headers={"Authorization": f"Bearer {get_token()}"}, timeout=timeout)


def _server_up() -> bool:
    try:
        return httpx.get(f"{server_url()}/api/health", timeout=2).json().get("product") == "openvoice-studio"
    except Exception:  # noqa: BLE001
        return False


def _ensure_server() -> None:
    if _server_up():
        return
    log = open(logs_dir() / "server.log", "ab")
    kwargs: dict = {"stdout": log, "stderr": subprocess.STDOUT, "stdin": subprocess.DEVNULL, "close_fds": True}
    if sys.platform == "win32":
        kwargs["creationflags"] = 0x00000008 | 0x00000200 | 0x08000000  # DETACHED | NEW_GROUP | NO_WINDOW
    else:
        kwargs["start_new_session"] = True
    subprocess.Popen([sys.executable, "-m", "openvoice_studio", "serve"], **kwargs)
    for _ in range(120):
        if _server_up():
            return
        time.sleep(0.5)
    raise RuntimeError(f"O servidor local não arrancou; vê {logs_dir() / 'server.log'}")


def _call(method: str, path: str, timeout: float = 30, **kw) -> dict:
    _ensure_server()
    with _client(timeout) as c:
        r = c.request(method, path, **kw)
    if r.status_code >= 400:
        try:
            detail = r.json().get("detail")
        except Exception:  # noqa: BLE001
            detail = r.text
        raise RuntimeError(f"HTTP {r.status_code}: {detail}")
    return r.json()


def _wait(job_id: str, wait_seconds: float) -> dict:
    deadline = time.monotonic() + max(0.0, float(wait_seconds))
    while True:
        left = deadline - time.monotonic()
        job = _call("GET", f"/api/jobs/{job_id}", params={"wait": max(0.0, min(left, 30))} if left > 0 else {}, timeout=45)
        if job["status"] in {"done", "error"} or time.monotonic() >= deadline:
            return job


def _job_view(job: dict) -> dict:
    view = {k: job.get(k) for k in ["job_id", "kind", "status", "stage", "progress", "error"]}
    if job.get("result"):
        view["result"] = job["result"]
    if job["status"] in {"queued", "running"}:
        view["next_step"] = f"Still running. Call get_job_status(job_id='{job['job_id']}', wait_seconds=120)."
    return view


@mcp.tool()
def get_engine_capabilities() -> dict:
    """Engines, languages, narration styles, limits and whether models/evaluation are installed."""
    caps = _call("GET", "/api/capabilities")
    caps["web_ui"] = f"{server_url()}/#token=<token>  (run `openvoice-studio ui` to get the full link)"
    return caps


@mcp.tool()
def list_voices() -> dict:
    """Voices the user created (with consent) in the web UI. Use voice_id in generate_narration."""
    return _call("GET", "/api/voices")


@mcp.tool()
def generate_narration(text: str, voice_id: str, style: str = "documental", language: str = "pt",
                       wait_seconds: float = 120) -> dict:
    """Generate narration audio (WAV) with one of the user's voices.

    style: documental | natural | didactico | energetico.
    language: ISO code, e.g. "pt". wait_seconds: how long to block for the result (max 900); if the job is
    still running, poll get_job_status.
    """
    job = _call("POST", "/api/generate", json={"voice_id": voice_id, "text": text, "style": style, "language": language})
    return _job_view(_wait(job["job_id"], min(float(wait_seconds), 900)))


@mcp.tool()
def get_job_status(job_id: str, wait_seconds: float = 0) -> dict:
    """Status of a generation/evaluation job; optionally block up to wait_seconds (max 900) for completion."""
    return _job_view(_wait(job_id, min(float(wait_seconds), 900)))


@mcp.tool()
def get_audio_artifact(job_id: str, destination_dir: str | None = None, filename: str | None = None) -> dict:
    """Copy a finished narration WAV to a folder (default: the user's Desktop) and return the file path."""
    job = _call("GET", f"/api/jobs/{job_id}")
    if job.get("kind") != "generate" or job.get("status") != "done":
        raise RuntimeError(f"A tarefa {job_id} não é uma geração concluída (estado: {job.get('status')}).")
    src = Path(job["result"]["audio_path"])
    dest_dir = Path(destination_dir).expanduser() if destination_dir else desktop_dir()
    dest_dir.mkdir(parents=True, exist_ok=True)
    name = filename or f"openvoice-{job_id}.wav"
    if not name.lower().endswith(".wav"):
        name += ".wav"
    dest = dest_dir / Path(name).name
    shutil.copyfile(src, dest)
    return {"path": str(dest), "bytes": dest.stat().st_size, "duration_s": job["result"]["duration_s"],
            "source": str(src)}


@mcp.tool()
def run_voice_evaluation(job_id: str, wait_seconds: float = 300) -> dict:
    """Objective metrics for a finished narration: speaker similarity to the voice, WER/CER from a local
    transcription, loudness, pace and warnings. job_id is the generation job."""
    job = _call("POST", "/api/evaluate", json={"job_id": job_id})
    return _job_view(_wait(job["job_id"], min(float(wait_seconds), 900)))


def run() -> None:
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    mcp.run("stdio")
