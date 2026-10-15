"""Local HTTP API + web UI (FastAPI). Binds to 127.0.0.1 and requires the token for /api/*."""

from __future__ import annotations

import hmac
import tempfile
from importlib import resources
from pathlib import Path

from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from pydantic import BaseModel

from . import __version__
from .config import get_token
from .studio import Studio
from .text import RECORDING_SCRIPT
from .voices import CONSENT_STATEMENT, ConsentError

ALLOWED_HOSTS = {"127.0.0.1", "localhost", "[::1]"}
MAX_UPLOAD_MB = 60


class GenerateBody(BaseModel):
    voice_id: str
    text: str
    style: str | None = None
    language: str | None = None
    engine: str | None = None


class EvaluateBody(BaseModel):
    job_id: str


def create_app(studio: Studio | None = None, token: str | None = None) -> FastAPI:
    studio = studio or Studio()
    token = token or get_token()
    app = FastAPI(title="OpenVoice Studio", version=__version__, docs_url=None, redoc_url=None)
    app.state.studio = studio

    @app.middleware("http")
    async def host_guard(request: Request, call_next):
        # Block DNS-rebinding: only accept requests addressed to the loopback host.
        host = (request.headers.get("host") or "").rsplit(":", 1)[0]
        if host not in ALLOWED_HOSTS:
            return JSONResponse({"detail": "host não permitido"}, status_code=403)
        return await call_next(request)

    def auth(request: Request) -> None:
        supplied = request.headers.get("x-ovs-token") or ""
        bearer = request.headers.get("authorization") or ""
        if bearer.lower().startswith("bearer "):
            supplied = bearer[7:]
        supplied = supplied or request.query_params.get("token", "")
        if not hmac.compare_digest(supplied.encode(), token.encode()):
            raise HTTPException(status_code=401, detail="token inválido ou em falta")

    def err(exc: Exception, code: int = 400):
        return HTTPException(status_code=code, detail=str(exc))

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return resources.files("openvoice_studio.web").joinpath("index.html").read_text(encoding="utf-8")

    @app.get("/api/health")
    def health() -> dict:
        return {"ok": True, "product": "openvoice-studio", "version": __version__}

    @app.get("/api/capabilities", dependencies=[Depends(auth)])
    def capabilities() -> dict:
        return studio.capabilities()

    @app.get("/api/recording-script", dependencies=[Depends(auth)])
    def recording_script() -> dict:
        return {"paragraphs": RECORDING_SCRIPT, "consent_statement": CONSENT_STATEMENT}

    @app.get("/api/voices", dependencies=[Depends(auth)])
    def list_voices() -> dict:
        return {"voices": [studio.voices.summary(v) for v in studio.voices.list()]}

    async def _save_uploads(files: list[UploadFile], tmp: Path) -> list[Path]:
        paths = []
        for i, f in enumerate(files):
            suffix = Path(f.filename or "").suffix.lower()[:8] or ".bin"
            dst = tmp / f"in{i:02d}{suffix}"
            size = 0
            with dst.open("wb") as out:
                while chunk := await f.read(1 << 20):
                    size += len(chunk)
                    if size > MAX_UPLOAD_MB << 20:
                        raise HTTPException(413, f"Ficheiro demasiado grande (máx. {MAX_UPLOAD_MB} MB).")
                    out.write(chunk)
            paths.append(dst)
        return paths

    @app.post("/api/voices/check", dependencies=[Depends(auth)])
    async def check_voice(files: list[UploadFile] = File(...)) -> dict:
        with tempfile.TemporaryDirectory() as tmp:
            paths = await _save_uploads(files, Path(tmp))
            try:
                result = studio.voices.check(paths)
            except Exception as exc:  # noqa: BLE001
                raise err(exc) from exc
        for r, f in zip(result["files"], files):
            r["file"] = f.filename
        return result

    @app.post("/api/voices", dependencies=[Depends(auth)])
    async def create_voice(
        files: list[UploadFile] = File(...),
        name: str = Form("A minha voz"),
        language: str = Form("pt"),
        consent_confirmed: bool = Form(False),
        consent_owner: str = Form("self"),
        authorized_by: str = Form(""),
    ) -> dict:
        with tempfile.TemporaryDirectory() as tmp:
            paths = await _save_uploads(files, Path(tmp))
            try:
                voice = studio.voices.create(
                    name, paths,
                    {"confirmed": consent_confirmed, "owner": consent_owner, "authorized_by": authorized_by},
                    language=language, default_engine=studio.engine_name,
                    display_names=[f.filename or "" for f in files],
                )
            except ConsentError as exc:
                raise err(exc, 422) from exc
            except Exception as exc:  # noqa: BLE001
                raise err(exc) from exc
        job = studio.submit_prepare(voice["voice_id"])
        return {"voice": studio.voices.summary(voice), "assessment": voice["assessment"], "prepare_job": job}

    @app.delete("/api/voices/{voice_id}", dependencies=[Depends(auth)])
    def delete_voice(voice_id: str) -> dict:
        try:
            studio.voices.delete(voice_id)
        except KeyError as exc:
            raise err(f"voz '{voice_id}' não existe", 404) from exc
        return {"deleted": voice_id}

    @app.get("/api/voices/{voice_id}/reference.wav", dependencies=[Depends(auth)])
    def voice_reference(voice_id: str):
        try:
            return FileResponse(studio.voices.path(voice_id) / "reference.wav", media_type="audio/wav")
        except KeyError as exc:
            raise err(f"voz '{voice_id}' não existe", 404) from exc

    @app.post("/api/generate", dependencies=[Depends(auth)])
    def generate(body: GenerateBody) -> dict:
        try:
            return studio.submit_generate(body.voice_id, body.text, body.style, body.language, body.engine)
        except KeyError as exc:
            raise err(f"voz '{body.voice_id}' não existe", 404) from exc
        except ValueError as exc:
            raise err(exc) from exc

    @app.post("/api/evaluate", dependencies=[Depends(auth)])
    def evaluate(body: EvaluateBody) -> dict:
        try:
            return studio.submit_evaluate(body.job_id)
        except KeyError as exc:
            raise err(f"tarefa '{body.job_id}' não existe", 404) from exc
        except ValueError as exc:
            raise err(exc) from exc

    @app.get("/api/jobs", dependencies=[Depends(auth)])
    def jobs() -> dict:
        return {"jobs": studio.jobs.recent()}

    @app.get("/api/jobs/{job_id}", dependencies=[Depends(auth)])
    def job(job_id: str, wait: float = 0) -> dict:
        try:
            return studio.jobs.wait(job_id, min(wait, 60)) if wait else studio.jobs.snapshot(job_id)
        except KeyError as exc:
            raise err(f"tarefa '{job_id}' não existe", 404) from exc

    @app.get("/api/audio/{name}", dependencies=[Depends(auth)])
    def audio_file(name: str, download: bool = False):
        job_id = name.removesuffix(".wav")
        try:
            job = studio.jobs.snapshot(job_id)
        except KeyError as exc:
            raise err("áudio não encontrado", 404) from exc
        path = Path((job.get("result") or {}).get("audio_path") or "")
        if job.get("kind") != "generate" or not path.is_file():
            raise err("áudio não encontrado", 404)
        return FileResponse(path, media_type="audio/wav", filename=f"{job_id}.wav" if download else None)

    return app


def port_in_use(host: str, port: int) -> bool:
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.5)
        return s.connect_ex((host, port)) == 0


def run(host: str, port: int, quiet: bool = False, preload: bool = True) -> None:
    import uvicorn

    from .engines import get_engine

    studio = Studio()
    app = create_app(studio)
    eng = get_engine(studio.engine_name)
    if preload and eng.status().get("models_installed"):
        # Warm the model in the background so the first narration starts sooner.
        def load(_: dict) -> dict:
            eng.load()
            return {"loaded": eng.name, "device": getattr(eng, "device", None)}

        studio.jobs.submit("load", {"engine": eng.name}, load)
    uvicorn.run(app, host=host, port=port, log_level="warning" if quiet else "info", access_log=False)

