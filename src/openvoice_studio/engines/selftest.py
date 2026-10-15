"""A tiny formant synthesiser with no model weights.

It does not clone anything; it exists so `openvoice-studio selftest` and the test-suite can exercise
the whole pipeline (voices, jobs, HTTP API, MCP tools, artifacts, evaluation) in seconds.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .. import audio
from .base import Engine

_VOWELS = {"a": (800, 1200), "e": (500, 1900), "i": (300, 2300), "o": (500, 900), "u": (350, 800)}


class SelftestEngine(Engine):
    name = "selftest"
    label = "Selftest (sintetizador de formantes, sem clonagem)"
    license = "MIT"
    sample_rate = 24000
    languages = {"pt": "Portuguese", "en": "English"}

    def status(self) -> dict:
        return {"available": True, "models_installed": True, "detail": "sem modelos"}

    def load(self) -> None:
        return None

    def prepare_voice(self, voice_dir: Path, reference_wav: Path, samples_wav: Path) -> None:
        wav, sr = audio.read(samples_wav)
        (voice_dir / "selftest.json").write_text(json.dumps({"f0": _estimate_f0(wav, sr)}), encoding="utf-8")

    def synthesize(self, text: str, voice_dir: Path, language: str, params: dict) -> np.ndarray:
        meta = voice_dir / "selftest.json"
        f0 = json.loads(meta.read_text(encoding="utf-8"))["f0"] if meta.exists() else 140.0
        sr = self.sample_rate
        out: list[np.ndarray] = []
        for ch in text.lower():
            if ch in _VOWELS:
                out.append(_vowel(*_VOWELS[ch], f0, 0.11, sr))
            elif ch.isalpha():
                out.append(np.random.default_rng(ord(ch)).normal(0, 0.03, int(0.04 * sr)).astype(np.float32))
            elif ch in ",;:":
                out.append(audio.silence(0.15, sr))
            elif ch == " ":
                out.append(audio.silence(0.03, sr))
        return np.concatenate(out) if out else audio.silence(0.2, sr)

    def speaker_embedding(self, wav: np.ndarray, sr: int) -> np.ndarray:
        wav = audio.resample(wav, sr, 16000)
        n = 512
        frames = len(wav) // n
        if frames == 0:
            return np.ones(32, dtype=np.float32) / np.sqrt(32)
        spec = np.abs(np.fft.rfft(wav[: frames * n].reshape(frames, n) * np.hanning(n), axis=1))
        bands = np.array_split(np.log(spec + 1e-6).mean(axis=0), 32)
        emb = np.array([b.mean() for b in bands], dtype=np.float32)
        emb -= emb.mean()
        return emb / (np.linalg.norm(emb) + 1e-9)


def _vowel(f1: float, f2: float, f0: float, dur: float, sr: int) -> np.ndarray:
    t = np.arange(int(dur * sr)) / sr
    src = np.zeros_like(t)
    for k in range(1, 30):
        h = k * f0
        if h > sr / 2:
            break
        gain = np.exp(-((h - f1) ** 2) / (2 * 120**2)) + 0.6 * np.exp(-((h - f2) ** 2) / (2 * 180**2)) + 0.05
        src += gain / k * np.sin(2 * np.pi * h * t)
    env = np.sin(np.pi * np.linspace(0, 1, len(t))) ** 0.5
    return (0.25 * src * env / (np.max(np.abs(src)) + 1e-9)).astype(np.float32)


def _estimate_f0(wav: np.ndarray, sr: int) -> float:
    seg = wav[: sr * 5]
    if len(seg) < sr // 10:
        return 140.0
    seg = seg - seg.mean()
    corr = np.correlate(seg[: sr // 2], seg[: sr // 2], mode="full")[sr // 2 - 1 :]
    lo, hi = sr // 400, sr // 60
    lag = lo + int(np.argmax(corr[lo:hi]))
    return float(np.clip(sr / lag, 70, 350))
