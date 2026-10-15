"""The service layer shared by the HTTP API, the CLI and the self-test.

All heavy work (model load, voice preparation, synthesis, evaluation) runs as jobs on ONE worker
thread, so a single copy of the model serves the web UI and every MCP client.
"""

from __future__ import annotations

import json
import queue
import re
import secrets
import threading
import time
import traceback
from pathlib import Path
from typing import Callable

import numpy as np

from . import __version__, audio, evaluation
from .config import load_settings, outputs_dir
from .engines import engine_names, get_engine
from .text import STYLES, expected_seconds, get_style, plan_chunks
from .voices import VoiceStore, now_iso

MAX_TEXT_CHARS = 5000
JOB_ID_RE = re.compile(r"[a-z]{3}_\d{8}_\d{6}_[0-9a-f]{6}")


class JobManager:
    def __init__(self, out_dir: Path) -> None:
        self.out_dir = out_dir
        self.jobs: dict[str, dict] = {}
        self.queue: queue.Queue = queue.Queue()
        self.cond = threading.Condition()
        self.worker: threading.Thread | None = None

    def submit(self, kind: str, params: dict, fn: Callable[[dict], dict]) -> dict:
        job = {
            "job_id": f"{kind[:3]}_{time.strftime('%Y%m%d_%H%M%S')}_{secrets.token_hex(3)}",
            "kind": kind,
            "status": "queued",
            "stage": "na fila",
            "progress": {"done": 0, "total": 0},
            "params": params,
            "created_at": now_iso(),
            "started_at": None,
            "finished_at": None,
            "result": None,
            "error": None,
        }
        with self.cond:
            self.jobs[job["job_id"]] = job
        self.queue.put((job["job_id"], fn))
        self._ensure_worker()
        return self.snapshot(job["job_id"])

    def _ensure_worker(self) -> None:
        if self.worker is None or not self.worker.is_alive():
            self.worker = threading.Thread(target=self._loop, name="ovs-worker", daemon=True)
            self.worker.start()

    def _loop(self) -> None:
        while True:
            job_id, fn = self.queue.get()
            self.update(job_id, status="running", stage="a iniciar", started_at=now_iso())
            try:
                result = fn(self.jobs[job_id])
                self.update(job_id, status="done", stage="concluído", result=result, finished_at=now_iso())
            except Exception as exc:  # noqa: BLE001
                self.update(job_id, status="error", stage="erro", error=f"{type(exc).__name__}: {exc}",
                            traceback=traceback.format_exc(limit=5), finished_at=now_iso())
            finally:
                self._persist(job_id)

    def update(self, job_id: str, **fields) -> None:
        with self.cond:
            self.jobs[job_id].update(fields)
            self.cond.notify_all()

    def _persist(self, job_id: str) -> None:
        job = self.snapshot(job_id)
        if job["kind"] in {"generate", "evaluate"}:
            (self.out_dir / f"{job_id}.json").write_text(json.dumps(job, ensure_ascii=False, indent=2), encoding="utf-8")

    def snapshot(self, job_id: str) -> dict:
        with self.cond:
            if job_id in self.jobs:
                return json.loads(json.dumps(self.jobs[job_id]))
        if not JOB_ID_RE.fullmatch(job_id or ""):
            raise KeyError(job_id)
        path = self.out_dir / f"{job_id}.json"
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
        raise KeyError(job_id)

    def wait(self, job_id: str, timeout: float) -> dict:
        deadline = time.monotonic() + max(0.0, timeout)
        with self.cond:
            while job_id in self.jobs and self.jobs[job_id]["status"] in {"queued", "running"}:
                left = deadline - time.monotonic()
                if left <= 0:
                    break
                self.cond.wait(timeout=min(left, 1.0))
        return self.snapshot(job_id)

    def recent(self, limit: int = 20) -> list[dict]:
        with self.cond:
            items = sorted(self.jobs.values(), key=lambda j: j["created_at"], reverse=True)[:limit]
            return json.loads(json.dumps(items))


class Studio:
    def __init__(self, engine: str | None = None, voices_root: Path | None = None, out_dir: Path | None = None) -> None:
        self.settings = load_settings()
        self.engine_name = engine or self.settings["engine"]
        get_engine(self.engine_name)  # validate name early
        self.voices = VoiceStore(voices_root)
        self.out_dir = Path(out_dir) if out_dir else outputs_dir()
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.jobs = JobManager(self.out_dir)

    # ---------------------------------------------------------------- info
    def capabilities(self) -> dict:
        engines = {name: get_engine(name).describe() for name in engine_names()}
        return {
            "product": "OpenVoice Studio",
            "version": __version__,
            "default_engine": self.engine_name,
            "engines": engines,
            "languages": engines[self.engine_name]["languages"],
            "styles": {k: {"label": s.label, "description": s.description} for k, s in STYLES.items()},
            "evaluation": evaluation.whisper_status(),
            "limits": {"max_text_chars": MAX_TEXT_CHARS, "recommended_voice_recording": "1–3 min, uma só pessoa, sem música"},
            "cost": "0 € — tudo corre localmente; nenhum áudio sai deste computador",
            "watermark": "Os áudios do Chatterbox levam a marca de água imperceptível PerTh (Resemble AI).",
        }

    # ---------------------------------------------------------------- jobs
    def _ensure_prepared(self, job_id: str, voice_id: str, engine) -> Path:
        vdir = self.voices.path(voice_id)
        voice = self.voices.get(voice_id)
        if engine.name not in voice.get("prepared_engines", []) or not any(vdir.glob(f"*{engine.name}*")):
            self.jobs.update(job_id, stage="a preparar a voz")
            engine.prepare_voice(vdir, vdir / "reference.wav", vdir / "samples.wav")
            self.voices.mark_prepared(voice_id, engine.name)
        return vdir

    def submit_prepare(self, voice_id: str, engine: str | None = None) -> dict:
        self.voices.get(voice_id)
        eng = get_engine(engine or self.engine_name)

        def run(job: dict) -> dict:
            self.jobs.update(job["job_id"], stage="a carregar o modelo (a 1.ª vez demora)")
            eng.load()
            self._ensure_prepared(job["job_id"], voice_id, eng)
            return {"voice": self.voices.summary(self.voices.get(voice_id))}

        return self.jobs.submit("prepare", {"voice_id": voice_id, "engine": eng.name}, run)

    def submit_generate(self, voice_id: str, text: str, style: str | None = None, language: str | None = None,
                        engine: str | None = None) -> dict:
        text = (text or "").strip()
        if not text:
            raise ValueError("O texto está vazio.")
        if len(text) > MAX_TEXT_CHARS:
            raise ValueError(f"Texto demasiado longo ({len(text)} caracteres; máximo {MAX_TEXT_CHARS}). Divide-o em partes.")
        voice = self.voices.get(voice_id)
        sty = get_style(style)
        lang = language or voice.get("language") or self.settings["language"]
        eng = get_engine(engine or self.engine_name)
        if lang.split("-")[0] not in eng.languages:
            raise ValueError(f"Idioma '{lang}' não suportado por {eng.name}.")
        params = {"voice_id": voice_id, "text": text, "style": sty.key, "language": lang, "engine": eng.name}

        def run(job: dict) -> dict:
            job_id = job["job_id"]
            self.jobs.update(job_id, stage="a carregar o modelo (a 1.ª vez demora)")
            eng.load()
            vdir = self._ensure_prepared(job_id, voice_id, eng)
            plan = plan_chunks(text, lang)
            self.jobs.update(job_id, stage="a gerar", progress={"done": 0, "total": len(plan)})
            sr = eng.sample_rate
            synth = {"exaggeration": sty.exaggeration, "cfg_weight": sty.cfg_weight, "temperature": sty.temperature}
            pieces: list[np.ndarray] = [audio.silence(0.15, sr)]
            warnings: list[str] = []
            t0 = time.monotonic()
            for i, chunk in enumerate(plan):
                wav = audio.trim_silence(eng.synthesize(chunk["text"], vdir, lang, synth), sr)
                ratio = (len(wav) / sr) / expected_seconds(chunk["text"])
                if eng.name != "selftest" and not 0.4 <= ratio <= 2.5:
                    # Likely a garbled or run-on take: retry once, more conservatively, keep the saner one.
                    retry = audio.trim_silence(eng.synthesize(chunk["text"], vdir, lang, {**synth, "temperature": 0.6}), sr)
                    r2 = (len(retry) / sr) / expected_seconds(chunk["text"])
                    if abs(np.log(r2)) < abs(np.log(ratio)):
                        wav, ratio = retry, r2
                    if not 0.4 <= ratio <= 2.5:
                        warnings.append(f"Frase {i + 1} tem duração invulgar ({ratio:.1f}× o esperado): ouve-a com atenção.")
                pieces.append(audio.fade(wav, sr, 8))
                if chunk["pause_after"]:
                    pieces.append(audio.silence(sty.pause_paragraph if chunk["pause_after"] == "paragraph" else sty.pause_sentence, sr))
                self.jobs.update(job_id, progress={"done": i + 1, "total": len(plan)})
            pieces.append(audio.silence(0.25, sr))
            mix = np.concatenate(pieces)
            if eng.name != "selftest":
                # Light clean-up of residual hiss the model may have copied from the reference.
                mix = audio.spectral_gate(mix, sr, strength=0.7, n_std=2.0)
            out = audio.normalize(mix, sr, target_lufs=-18.0)
            path = audio.write(self.out_dir / f"{job_id}.wav", out, sr)
            duration = len(out) / sr
            elapsed = time.monotonic() - t0
            return {
                "audio_path": str(path),
                "audio_url": f"/api/audio/{job_id}.wav",
                "duration_s": round(duration, 2),
                "sample_rate": sr,
                "chunks": len(plan),
                "generation_seconds": round(elapsed, 1),
                "real_time_factor": round(elapsed / duration, 2) if duration else None,
                "normalized_text": " ".join(c["text"] for c in plan),
                "warnings": warnings,
            }

        return self.jobs.submit("generate", params, run)

    def submit_evaluate(self, job_id: str) -> dict:
        gen = self.jobs.snapshot(job_id)
        if gen["kind"] != "generate" or gen["status"] != "done":
            raise ValueError("Só se pode avaliar uma geração concluída.")
        p = gen["params"]
        eng = get_engine(p["engine"])

        def run(job: dict) -> dict:
            self.jobs.update(job["job_id"], stage="a avaliar")
            eng.load()
            res = evaluation.evaluate(Path(gen["result"]["audio_path"]), p["text"], p["language"], eng,
                                      self.voices.path(p["voice_id"]))
            return {"generation_job_id": job_id, "audio_path": gen["result"]["audio_path"], **res,
                    "generation_warnings": gen["result"].get("warnings", [])}

        return self.jobs.submit("evaluate", {"generation_job_id": job_id}, run)
