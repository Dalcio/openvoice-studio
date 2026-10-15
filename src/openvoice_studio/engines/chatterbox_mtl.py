"""Adapter for Chatterbox Multilingual (Resemble AI, MIT licence for code and weights).

Zero-shot cloning from a ~10 s reference in 23 languages, Portuguese included. Outputs carry
Resemble's imperceptible PerTh watermark, which we keep on purpose (responsible use).
"""

from __future__ import annotations

import importlib.util
import os
import threading
from pathlib import Path

import numpy as np

from .. import audio
from .base import Engine, EngineUnavailable

REPO_ID = "ResembleAI/chatterbox"
MODEL_FILES = ["ve.pt", "t3_mtl23ls_v2.safetensors", "s3gen.pt", "grapheme_mtl_merged_expanded_v1.json",
               "conds.pt", "Cangjie5_TC.json"]
MODEL_SIZE_GB = 3.2
CONDS_FILE = "conds.chatterbox-mtl.pt"
SPEECH_TOKENS_PER_S = 25  # S3 tokenizer rate
# Share of one chunk's compute on a laptop CPU, measured: T3 token sampling ~35 %, S3Gen flow
# matching (10 Euler steps) ~60 %, HiFT vocoder ~5 %.
T3_SHARE = 0.35
CFM_SHARE = 0.60
EMBED_FILE = "embedding.chatterbox-mtl.npy"


def cached_snapshot() -> Path | None:
    """Directory holding all model files if they are already in the Hugging Face cache."""
    try:
        from huggingface_hub import try_to_load_from_cache
    except ImportError:
        return None
    folder = None
    for name in MODEL_FILES:
        path = try_to_load_from_cache(REPO_ID, name)
        if not isinstance(path, str):
            return None
        folder = Path(path).parent
    return folder


def download_models() -> Path:
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(repo_id=REPO_ID, repo_type="model", allow_patterns=MODEL_FILES,
                                  token=os.getenv("HF_TOKEN")))


def pick_device() -> str:
    import torch

    if forced := os.environ.get("OVS_DEVICE"):
        return forced
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class ChatterboxMultilingualEngine(Engine):
    name = "chatterbox-mtl"
    label = "Chatterbox Multilingual (Resemble AI)"
    license = "MIT (código e pesos)"
    sample_rate = 24000
    languages = {
        "ar": "Arabic", "da": "Danish", "de": "German", "el": "Greek", "en": "English", "es": "Spanish",
        "fi": "Finnish", "fr": "French", "he": "Hebrew", "hi": "Hindi", "it": "Italian", "ja": "Japanese",
        "ko": "Korean", "ms": "Malay", "nl": "Dutch", "no": "Norwegian", "pl": "Polish", "pt": "Portuguese",
        "ru": "Russian", "sv": "Swedish", "sw": "Swahili", "tr": "Turkish", "zh": "Chinese",
    }

    def __init__(self) -> None:
        self.model = None
        self.device: str | None = None
        self._lock = threading.Lock()
        self._conds: dict[str, tuple[float, object]] = {}
        self._on_token = None  # set during synthesize() to report sampling progress
        self._on_cfm_step = None

    def status(self) -> dict:
        if importlib.util.find_spec("chatterbox") is None:
            return {"available": False, "models_installed": False, "detail": "pacote chatterbox-tts não instalado"}
        installed = cached_snapshot() is not None
        return {
            "available": True,
            "models_installed": installed,
            "loaded": self.model is not None,
            "device": self.device,
            "detail": "pronto" if installed else f"modelos por descarregar (~{MODEL_SIZE_GB} GB): openvoice-studio setup",
        }

    def load(self) -> None:
        with self._lock:
            if self.model is not None:
                return
            try:
                import torch
                from chatterbox.mtl_tts import ChatterboxMultilingualTTS
            except ImportError as exc:
                raise EngineUnavailable(f"chatterbox-tts não está instalado: {exc}") from exc
            self.device = pick_device()
            if self.device == "cpu":
                torch.set_num_threads(max(1, (os.cpu_count() or 2)))
            folder = cached_snapshot() or download_models()
            self.model = ChatterboxMultilingualTTS.from_local(folder, self.device)
            self._hook_sampling_loop()

    def _hook_sampling_loop(self) -> None:
        """T3 samples speech tokens in `for i in tqdm(range(max_new_tokens))`. Swapping that module's
        `tqdm` for a counting generator gives real per-token progress (and no console bars)."""
        try:
            import chatterbox.models.t3.t3 as t3_module
        except ImportError:
            return
        engine = self

        def counting(iterable=None, *args, **kwargs):
            def gen():
                for i, item in enumerate(iterable):
                    if engine._on_token is not None:
                        engine._on_token(i)
                    yield item
            return gen()

        t3_module.tqdm = counting

        # S3Gen's Euler solver calls `self.estimator.forward(...)` once per step (bypassing module
        # hooks), so wrap that bound method on the instance.
        for module in self.model.s3gen.modules():
            est = getattr(module, "estimator", None)
            if est is None or not hasattr(module, "solve_euler"):
                continue
            original = est.forward
            steps = {"n": 0}

            def counted_forward(*args, _orig=original, _steps=steps, **kwargs):
                out = _orig(*args, **kwargs)
                _steps["n"] += 1
                if engine._on_cfm_step is not None:
                    engine._on_cfm_step(_steps["n"])
                return out

            est.forward = counted_forward
            self._cfm_steps = steps

    def _lang(self, language: str) -> str:
        code = (language or "pt").lower().split("-")[0].split("_")[0]
        if code not in self.languages:
            raise ValueError(f"Idioma '{language}' não suportado pelo {self.label}.")
        return code

    def prepare_voice(self, voice_dir: Path, reference_wav: Path, samples_wav: Path) -> None:
        import torch

        self.load()
        m = self.model
        m.prepare_conditionals(str(reference_wav), exaggeration=0.5)
        conds = m.conds
        # The stock recipe embeds only the 10 s reference; averaging over every speech chunk of the
        # full recording gives a steadier speaker identity.
        emb = self.speaker_embedding(*audio.read(samples_wav))
        conds.t3.speaker_emb = torch.from_numpy(emb).float().unsqueeze(0).to(m.device)
        conds.save(voice_dir / CONDS_FILE)
        np.save(voice_dir / EMBED_FILE, emb)
        self._conds.pop(str(voice_dir), None)

    def _voice_conds(self, voice_dir: Path):
        from chatterbox.mtl_tts import Conditionals

        path = voice_dir / CONDS_FILE
        if not path.exists():
            raise FileNotFoundError("voz ainda não preparada para este motor")
        key, mtime = str(voice_dir), path.stat().st_mtime
        cached = self._conds.get(key)
        if cached is None or cached[0] != mtime:
            cached = (mtime, Conditionals.load(path, map_location="cpu").to(self.model.device))
            self._conds[key] = cached
        return cached[1]

    def synthesize(self, text: str, voice_dir: Path, language: str, params: dict, progress=None) -> np.ndarray:
        from ..text import expected_seconds

        self.load()
        self.model.conds = self._voice_conds(voice_dir)
        if progress:
            expected_tokens = max(10.0, expected_seconds(text) * SPEECH_TOKENS_PER_S)
            n_steps = 2 if getattr(self.model.s3gen, "meanflow", False) else 10
            self._on_token = lambda i: progress(min(T3_SHARE * 0.98, T3_SHARE * i / expected_tokens))
            if getattr(self, "_cfm_steps", None) is not None:
                self._cfm_steps["n"] = 0
            self._on_cfm_step = lambda n: progress(T3_SHARE + CFM_SHARE * min(1.0, n / n_steps))
        try:
            wav = self.model.generate(
                text,
                language_id=self._lang(language),
                exaggeration=float(params.get("exaggeration", 0.5)),
                cfg_weight=float(params.get("cfg_weight", 0.5)),
                temperature=float(params.get("temperature", 0.8)),
            )
        finally:
            self._on_token = None
            self._on_cfm_step = None
        if progress:
            progress(1.0)
        return wav.squeeze(0).detach().cpu().numpy().astype(np.float32)

    def speaker_embedding(self, wav: np.ndarray, sr: int) -> np.ndarray:
        self.load()
        wav16 = audio.resample(wav, sr, 16000)
        chunks = audio.split_speech(wav16, 16000, max_s=8.0, min_s=2.0)[:60]
        emb = self.model.ve.embeds_from_wavs(chunks, sample_rate=16000, as_spk=True)
        return np.asarray(emb, dtype=np.float32)
