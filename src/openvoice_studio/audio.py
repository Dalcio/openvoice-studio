"""Audio I/O, analysis and small DSP helpers (numpy only, ffmpeg for decoding)."""

from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
import soundfile as sf

from .config import ffmpeg_path

FRAME_S = 0.03


class AudioError(RuntimeError):
    pass


def to_wav(src: Path, dst: Path, sr: int = 24000) -> Path:
    """Decode any input (wav/mp3/m4a/webm/ogg/flac) to mono PCM16 WAV at `sr`."""
    src, dst = Path(src), Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    ff = ffmpeg_path()
    if ff:
        cmd = [ff, "-nostdin", "-hide_banner", "-loglevel", "error", "-y", "-i", str(src),
               "-vn", "-ac", "1", "-ar", str(sr), "-c:a", "pcm_s16le", str(dst)]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0 or not dst.exists():
            raise AudioError(f"Não foi possível ler o áudio '{src.name}': {proc.stderr.strip()[:300]}")
        return dst
    # Fallback without ffmpeg: only formats libsndfile understands (wav/flac/ogg).
    try:
        wav, in_sr = read(src)
    except Exception as exc:  # noqa: BLE001
        raise AudioError(f"ffmpeg não encontrado e '{src.name}' não é WAV/FLAC/OGG legível: {exc}") from exc
    write(dst, resample(wav, in_sr, sr), sr)
    return dst


def read(path: Path) -> tuple[np.ndarray, int]:
    wav, sr = sf.read(str(path), dtype="float32", always_2d=True)
    return wav.mean(axis=1).astype(np.float32), int(sr)


def write(path: Path, wav: np.ndarray, sr: int) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), np.clip(wav, -1.0, 1.0), sr, subtype="PCM_16")
    return path


def resample(wav: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    if sr_in == sr_out or len(wav) == 0:
        return wav.astype(np.float32)
    try:
        import librosa

        return librosa.resample(wav, orig_sr=sr_in, target_sr=sr_out).astype(np.float32)
    except ImportError:
        n = int(round(len(wav) * sr_out / sr_in))
        return np.interp(np.linspace(0, len(wav) - 1, n), np.arange(len(wav)), wav).astype(np.float32)


def silence(seconds: float, sr: int) -> np.ndarray:
    return np.zeros(int(max(seconds, 0) * sr), dtype=np.float32)


def frame_db(wav: np.ndarray, sr: int, frame_s: float = FRAME_S) -> np.ndarray:
    n = max(1, int(sr * frame_s))
    count = len(wav) // n
    if count == 0:
        return np.array([-120.0])
    frames = wav[: count * n].reshape(count, n)
    rms = np.sqrt(np.mean(frames**2, axis=1) + 1e-12)
    return 20 * np.log10(rms + 1e-12)


def speech_mask(db: np.ndarray) -> np.ndarray:
    floor = np.percentile(db, 10)
    peak = np.percentile(db, 98)
    threshold = max(floor + 10.0, peak - 35.0, -55.0)
    return db > threshold


def analyze(wav: np.ndarray, sr: int) -> dict:
    """Objective measurements of a recording. All levels in dBFS."""
    duration = len(wav) / sr if sr else 0.0
    db = frame_db(wav, sr)
    mask = speech_mask(db)
    speech_s = float(mask.sum() * FRAME_S)
    speech_level = float(np.mean(db[mask])) if mask.any() else -120.0
    noise_floor = float(np.percentile(db, 10))
    peak = float(np.max(np.abs(wav))) if len(wav) else 0.0
    clipping = float(np.mean(np.abs(wav) >= 0.99)) if len(wav) else 0.0
    # Longest run of non-speech frames.
    longest, run = 0, 0
    for m in mask:
        run = 0 if m else run + 1
        longest = max(longest, run)
    return {
        "duration_s": round(duration, 2),
        "speech_s": round(speech_s, 2),
        "speech_ratio": round(speech_s / duration, 3) if duration else 0.0,
        "speech_level_dbfs": round(speech_level, 1),
        "noise_floor_dbfs": round(noise_floor, 1),
        "snr_db": round(speech_level - noise_floor, 1),
        "peak_dbfs": round(20 * np.log10(peak + 1e-12), 1),
        "clipping_ratio": round(clipping, 5),
        "longest_pause_s": round(longest * FRAME_S, 2),
    }


def assess(analyses: list[dict]) -> dict:
    """Turn per-file analyses into a verdict with Portuguese guidance for the user."""
    errors: list[str] = []
    warnings: list[str] = []
    total_speech = sum(a["speech_s"] for a in analyses)
    total = sum(a["duration_s"] for a in analyses)
    if not analyses:
        errors.append("Nenhuma gravação recebida.")
    elif total_speech < 10:
        errors.append(f"Muito pouca fala ({total_speech:.0f} s). Grava pelo menos 30 s; o ideal é perto de 1 minuto.")
    elif total_speech < 30:
        warnings.append(f"Só {total_speech:.0f} s de fala. Funciona, mas a semelhança melhora com 45–90 s.")
    for i, a in enumerate(analyses, 1):
        tag = f"Gravação {i}: " if len(analyses) > 1 else ""
        if a["clipping_ratio"] > 0.001:
            warnings.append(f"{tag}há saturação (clipping). Afasta um pouco o microfone ou baixa o ganho.")
        if a["speech_level_dbfs"] < -35:
            warnings.append(f"{tag}volume baixo ({a['speech_level_dbfs']} dBFS). Aproxima-te do microfone (15–20 cm).")
        if a["snr_db"] < 15:
            warnings.append(f"{tag}muito ruído de fundo (SNR estimado {a['snr_db']} dB). Procura um local mais silencioso.")
        elif a["snr_db"] < 22:
            warnings.append(f"{tag}algum ruído de fundo (SNR estimado {a['snr_db']} dB).")
    if errors:
        status = "insuficiente"
    elif warnings:
        status = "aceitável"
    else:
        status = "bom"
    return {
        "ok": not errors,
        "status": status,
        "total_duration_s": round(total, 1),
        "total_speech_s": round(total_speech, 1),
        "errors": errors,
        "warnings": warnings,
        "notes": ["Confirma que só se ouve uma pessoa e que não há música de fundo."],
    }


def trim_silence(wav: np.ndarray, sr: int, keep_s: float = 0.04) -> np.ndarray:
    if len(wav) == 0:
        return wav
    db = frame_db(wav, sr)
    idx = np.where(speech_mask(db))[0]
    if len(idx) == 0:
        return wav
    n = int(sr * FRAME_S)
    keep = int(sr * keep_s)
    start = max(0, idx[0] * n - keep)
    end = min(len(wav), (idx[-1] + 1) * n + keep)
    return wav[start:end]


def fade(wav: np.ndarray, sr: int, ms: float = 10) -> np.ndarray:
    n = min(len(wav) // 2, int(sr * ms / 1000))
    if n <= 0:
        return wav
    out = wav.copy()
    ramp = np.linspace(0, 1, n, dtype=np.float32)
    out[:n] *= ramp
    out[-n:] *= ramp[::-1]
    return out


def best_segment(wav: np.ndarray, sr: int, seconds: float = 10.0) -> np.ndarray:
    """Pick the window with the most clean speech (no clipping) to use as the cloning reference."""
    win = int(seconds * sr)
    if len(wav) <= win:
        return fade(trim_silence(wav, sr), sr)
    db = frame_db(wav, sr)
    mask = speech_mask(db).astype(np.float32)
    n = int(sr * FRAME_S)
    clip = np.abs(wav[: len(db) * n].reshape(len(db), n)).max(axis=1) >= 0.99
    score = mask - 3.0 * clip.astype(np.float32)
    frames = max(1, win // n)
    csum = np.concatenate([[0.0], np.cumsum(score)])
    windows = csum[frames:] - csum[:-frames]
    start = int(np.argmax(windows)) * n
    seg = wav[start : start + win]
    return fade(trim_silence(seg, sr), sr)


def split_speech(wav: np.ndarray, sr: int, max_s: float = 8.0, min_s: float = 1.5) -> list[np.ndarray]:
    """Split a long recording into speech chunks (for averaging speaker embeddings)."""
    db = frame_db(wav, sr)
    mask = speech_mask(db)
    n = int(sr * FRAME_S)
    chunks: list[np.ndarray] = []

    def emit(s: int, e: int) -> None:
        if (e - s + 1) * FRAME_S >= min_s:
            chunks.append(wav[s * n : (e + 1) * n])

    start = last = None
    for i, m in enumerate(mask):
        if m:
            if start is None:
                start = i
            last = i
            if (i - start + 1) * FRAME_S >= max_s:
                emit(start, i)
                start = None
        elif start is not None and i - last >= 10:  # pauses shorter than 0.3 s stay in the chunk
            emit(start, last)
            start = None
    if start is not None:
        emit(start, last)
    return chunks or [wav]


def loudness_lufs(wav: np.ndarray, sr: int) -> float | None:
    try:
        import pyloudnorm

        if len(wav) < sr * 0.5:
            return None
        return float(pyloudnorm.Meter(sr).integrated_loudness(wav.astype(np.float64)))
    except Exception:  # noqa: BLE001
        return None


def normalize(wav: np.ndarray, sr: int, target_lufs: float = -18.0, peak_dbfs: float = -1.0) -> np.ndarray:
    lufs = loudness_lufs(wav, sr)
    out = wav.astype(np.float32)
    if lufs is not None and np.isfinite(lufs):
        out = out * (10 ** ((target_lufs - lufs) / 20))
    ceiling = 10 ** (peak_dbfs / 20)
    peak = float(np.max(np.abs(out))) if len(out) else 0.0
    if peak > ceiling:
        out = out * (ceiling / peak)
    return out.astype(np.float32)
