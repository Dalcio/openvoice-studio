"""Voice profiles on disk.

voices/<voice_id>/
  voice.json            metadata, consent record, recording analysis
  samples/NN.wav        each uploaded/recorded take (24 kHz mono)
  samples.wav           all takes joined (used for the speaker embedding)
  reference.wav         cleanest ~10 s window (used as the cloning prompt)
  conds.<engine>.pt     engine cache, rebuilt on demand
"""

from __future__ import annotations

import json
import re
import secrets
import shutil
import tempfile
import unicodedata
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from . import audio
from .config import voices_dir

SR = 24000
CONSENT_STATEMENT = (
    "Confirmo que esta voz é minha, ou que tenho autorização expressa e informada da pessoa a quem pertence, "
    "e que não a vou usar para enganar, fazer-me passar por alguém ou prejudicar terceiros."
)


class ConsentError(ValueError):
    pass


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def slugify(name: str) -> str:
    s = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    s = re.sub(r"[^a-zA-Z0-9]+", "-", s).strip("-").lower()
    return s[:32] or "voz"


def validate_consent(consent: dict) -> dict:
    if not consent or consent.get("confirmed") is not True:
        raise ConsentError("É preciso confirmar o consentimento antes de criar a voz.")
    owner = consent.get("owner", "self")
    if owner not in {"self", "authorized"}:
        raise ConsentError("owner tem de ser 'self' (a voz é minha) ou 'authorized' (tenho autorização).")
    authorized_by = (consent.get("authorized_by") or "").strip()
    if owner == "authorized" and not authorized_by:
        raise ConsentError("Indica quem é a pessoa e como deu autorização (ex.: 'Maria Silva, autorização escrita de 2026-10-15').")
    return {
        "confirmed": True,
        "owner": owner,
        "authorized_by": authorized_by or None,
        "statement": CONSENT_STATEMENT,
        "timestamp": now_iso(),
    }


class VoiceStore:
    def __init__(self, root: Path | None = None) -> None:
        self.root = Path(root) if root else voices_dir()
        self.root.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------------- read
    def path(self, voice_id: str) -> Path:
        if not re.fullmatch(r"[a-z0-9-]{1,64}", voice_id or ""):
            raise KeyError(voice_id)
        p = self.root / voice_id
        if not (p / "voice.json").exists():
            raise KeyError(voice_id)
        return p

    def get(self, voice_id: str) -> dict:
        return json.loads((self.path(voice_id) / "voice.json").read_text(encoding="utf-8"))

    def list(self) -> list[dict]:
        out = []
        for meta in self.root.glob("*/voice.json"):
            try:
                out.append(json.loads(meta.read_text(encoding="utf-8")))
            except (OSError, json.JSONDecodeError):
                continue
        return sorted(out, key=lambda v: v.get("created_at", ""), reverse=True)

    def summary(self, voice: dict) -> dict:
        keys = ["voice_id", "name", "language", "created_at", "status", "prepared_engines", "default_engine"]
        out = {k: voice.get(k) for k in keys}
        out["speech_seconds"] = voice.get("assessment", {}).get("total_speech_s")
        out["quality"] = voice.get("assessment", {}).get("status")
        return out

    # ---------------------------------------------------------------- write
    def update(self, voice_id: str, **fields) -> dict:
        voice = self.get(voice_id)
        voice.update(fields)
        (self.path(voice_id) / "voice.json").write_text(json.dumps(voice, ensure_ascii=False, indent=2), encoding="utf-8")
        return voice

    def mark_prepared(self, voice_id: str, engine: str) -> dict:
        voice = self.get(voice_id)
        engines = sorted(set(voice.get("prepared_engines", [])) | {engine})
        return self.update(voice_id, prepared_engines=engines, status="pronta")

    def delete(self, voice_id: str) -> None:
        shutil.rmtree(self.path(voice_id))

    @staticmethod
    def check(files: list[Path]) -> dict:
        """Analyse recordings without saving anything."""
        results = []
        with tempfile.TemporaryDirectory() as tmp:
            for i, f in enumerate(files):
                wav_path = audio.to_wav(f, Path(tmp) / f"{i:02d}.wav", SR)
                wav, sr = audio.read(wav_path)
                results.append({"file": Path(f).name, **audio.analyze(wav, sr)})
        return {"files": results, "assessment": audio.assess(results)}

    def create(self, name: str, files: list[Path], consent: dict, language: str = "pt",
               default_engine: str = "chatterbox-mtl") -> dict:
        record = validate_consent(consent)
        name = (name or "").strip() or "A minha voz"
        if not files:
            raise ValueError("Envia pelo menos uma gravação.")
        voice_id = f"{slugify(name)}-{secrets.token_hex(2)}"
        vdir = self.root / voice_id
        (vdir / "samples").mkdir(parents=True)
        try:
            takes, analyses = [], []
            for i, f in enumerate(files):
                wav_path = audio.to_wav(f, vdir / "samples" / f"{i + 1:02d}.wav", SR)
                wav, _ = audio.read(wav_path)
                takes.append(wav)
                analyses.append({"file": Path(f).name, **audio.analyze(wav, SR)})
            assessment = audio.assess(analyses)
            if not assessment["ok"]:
                raise ValueError(" ".join(assessment["errors"]))
            gap = audio.silence(0.3, SR)
            joined = np.concatenate([np.concatenate([audio.trim_silence(t, SR), gap]) for t in takes])
            audio.write(vdir / "samples.wav", joined, SR)
            audio.write(vdir / "reference.wav", audio.best_segment(joined, SR, 10.0), SR)
            voice = {
                "voice_id": voice_id,
                "name": name,
                "language": language,
                "created_at": now_iso(),
                "status": "a preparar",
                "default_engine": default_engine,
                "prepared_engines": [],
                "consent": record,
                "samples": analyses,
                "assessment": assessment,
            }
            (vdir / "voice.json").write_text(json.dumps(voice, ensure_ascii=False, indent=2), encoding="utf-8")
            return voice
        except Exception:
            shutil.rmtree(vdir, ignore_errors=True)
            raise
