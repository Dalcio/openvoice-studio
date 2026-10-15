"""Objective checks on a generated narration: signal, speaker similarity, intelligibility.

Similarity uses the engine's own speaker encoder (cosine between the voice's recordings and the
output). Intelligibility transcribes the output with faster-whisper and compares it to the input text
(WER/CER after normalisation). Thresholds are heuristics for a quick check, not a listening test.
"""

from __future__ import annotations

import re
import unicodedata
from pathlib import Path

import numpy as np

from . import audio
from .text import normalize_text

WHISPER_MODEL = "small"
SIMILARITY_HIGH = 0.85
SIMILARITY_OK = 0.75


def whisper_status() -> dict:
    try:
        import faster_whisper  # noqa: F401
    except ImportError:
        return {"installed": False, "detail": "faster-whisper não instalado"}
    try:
        from huggingface_hub import try_to_load_from_cache

        ok = isinstance(try_to_load_from_cache(f"Systran/faster-whisper-{WHISPER_MODEL}", "model.bin"), str)
    except Exception:  # noqa: BLE001
        ok = False
    return {"installed": ok, "model": WHISPER_MODEL,
            "detail": "pronto" if ok else "modelo por descarregar: openvoice-studio setup --evaluation"}


def download_whisper() -> str:
    from faster_whisper import download_model

    return download_model(WHISPER_MODEL)


_WHISPER = None


def transcribe(wav: np.ndarray, sr: int, language: str) -> str:
    global _WHISPER
    from faster_whisper import WhisperModel

    if _WHISPER is None:
        _WHISPER = WhisperModel(WHISPER_MODEL, device="cpu", compute_type="int8", local_files_only=True)
    segments, _ = _WHISPER.transcribe(audio.resample(wav, sr, 16000), language=language.split("-")[0], beam_size=5)
    return " ".join(s.text.strip() for s in segments).strip()


def _words(text: str, language: str) -> list[str]:
    text = normalize_text(text, language).lower()
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    return re.findall(r"[a-z0-9]+", text)


def _edit_distance(a: list, b: list) -> int:
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


def wer_cer(reference: str, hypothesis: str, language: str) -> tuple[float, float]:
    ref, hyp = _words(reference, language), _words(hypothesis, language)
    wer = _edit_distance(ref, hyp) / max(1, len(ref))
    rc, hc = list("".join(ref)), list("".join(hyp))
    cer = _edit_distance(rc, hc) / max(1, len(rc))
    return round(wer, 3), round(cer, 3)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def evaluate(wav_path: Path, text: str, language: str, engine, voice_dir: Path) -> dict:
    wav, sr = audio.read(wav_path)
    sig = audio.analyze(wav, sr)
    lufs = audio.loudness_lufs(wav, sr)
    n_words = len(_words(text, language))
    # Speaking rate over the narration including its pauses (lead/tail silence excluded), which is
    # what "palavras por minuto" means to a listener; articulation-only rates run ~25% higher.
    spoken_s = len(audio.trim_silence(wav, sr)) / sr
    metrics: dict = {
        "duration_s": sig["duration_s"],
        "loudness_lufs": round(lufs, 1) if lufs is not None and np.isfinite(lufs) else None,
        "peak_dbfs": sig["peak_dbfs"],
        "clipping_ratio": sig["clipping_ratio"],
        "speech_ratio": sig["speech_ratio"],
        "longest_pause_s": sig["longest_pause_s"],
        "words_per_minute": round(n_words / (spoken_s / 60), 1) if spoken_s > 1 else None,
    }
    warnings: list[str] = []
    if sig["clipping_ratio"] > 0.001:
        warnings.append("Há saturação (clipping) no áudio gerado.")
    if sig["longest_pause_s"] > 2.0:
        warnings.append(f"Pausa longa de {sig['longest_pause_s']} s: verifica se há um silêncio estranho.")
    wpm = metrics["words_per_minute"]
    if wpm and not 100 <= wpm <= 195:
        warnings.append(f"Ritmo invulgar ({wpm} palavras/min; narração típica: 130–175).")

    # Speaker similarity
    try:
        emb_file = voice_dir / f"embedding.{engine.name}.npy"
        if emb_file.exists():
            ref = np.load(emb_file)
        else:
            ref = engine.speaker_embedding(*audio.read(voice_dir / "samples.wav"))
        sim = cosine(ref, engine.speaker_embedding(wav, sr))
        metrics["speaker_similarity"] = round(sim, 3)
        metrics["speaker_similarity_label"] = "alta" if sim >= SIMILARITY_HIGH else "média" if sim >= SIMILARITY_OK else "baixa"
        if sim < SIMILARITY_OK:
            warnings.append("Semelhança de voz baixa. Experimenta gravar mais 30–60 s num local silencioso e recriar a voz.")
    except Exception as exc:  # noqa: BLE001
        metrics["speaker_similarity"] = None
        warnings.append(f"Não foi possível medir a semelhança: {exc}")

    # Intelligibility
    ws = whisper_status()
    if ws["installed"]:
        try:
            hyp = transcribe(wav, sr, language)
            wer, cer = wer_cer(text, hyp, language)
            metrics.update({"transcript": hyp, "wer": wer, "cer": cer})
            if wer > 0.15:
                warnings.append(f"Inteligibilidade fraca (WER {wer:.0%}): compara a transcrição com o texto.")
        except Exception as exc:  # noqa: BLE001
            warnings.append(f"Transcrição falhou: {exc}")
    else:
        metrics.update({"transcript": None, "wer": None, "cer": None})
        warnings.append("Métrica de inteligibilidade indisponível: corre `openvoice-studio setup --yes --evaluation`.")

    return {"metrics": metrics, "warnings": warnings,
            "thresholds": {"speaker_similarity_high": SIMILARITY_HIGH, "speaker_similarity_ok": SIMILARITY_OK, "wer_max": 0.15}}
