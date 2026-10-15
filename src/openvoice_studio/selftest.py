"""`openvoice-studio selftest`: end-to-end pipeline check in an isolated temp folder.

Uses the model-free `selftest` engine for the pipeline (voices, consent, jobs, HTTP API, auth, host
guard, artifacts, evaluation, MCP tool registry) and separately verifies the real engine imports.
"""

from __future__ import annotations

import asyncio
import socket
import tempfile
import threading
import time
import traceback
from pathlib import Path

import numpy as np


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def run_selftest() -> dict:
    results: dict = {}

    def step(name: str, fn) -> None:
        t0 = time.monotonic()
        try:
            detail = fn()
            results[name] = {"ok": True, "detail": detail, "seconds": round(time.monotonic() - t0, 2)}
        except Exception as exc:  # noqa: BLE001
            results[name] = {"ok": False, "detail": f"{type(exc).__name__}: {exc}",
                             "trace": traceback.format_exc(limit=3), "seconds": round(time.monotonic() - t0, 2)}

    def text_rules():
        from .text import normalize_text, number_to_words, plan_chunks

        assert number_to_words(1234) == "mil duzentos e trinta e quatro", number_to_words(1234)
        assert number_to_words(2_500_000) == "dois milhões e quinhentos mil", number_to_words(2_500_000)
        assert "zero euros" in normalize_text("custa 0 €."), normalize_text("custa 0 €.")
        assert "quinhentos kwanzas" in normalize_text("Paga 500 Kz"), normalize_text("Paga 500 Kz")
        assert len(plan_chunks("Uma frase curta. Outra frase que é um pouco maior do que a primeira.\n\nNovo parágrafo aqui.")) >= 2
        return "números, moeda e divisão em frases"

    step("text", text_rules)

    with tempfile.TemporaryDirectory(prefix="ovs-selftest-") as tmp:
        tmp = Path(tmp)
        from . import audio
        from .engines import get_engine
        from .studio import Studio
        from .text import RECORDING_SCRIPT
        from .voices import ConsentError

        eng = get_engine("selftest")
        sample = tmp / "amostra.wav"

        def make_sample():
            parts = []
            for p in RECORDING_SCRIPT:
                parts += [eng.synthesize(p, tmp, "pt", {}), audio.silence(0.6, eng.sample_rate)]
            wav = np.concatenate(parts)
            wav = wav + np.random.default_rng(0).normal(0, 0.0005, len(wav)).astype(np.float32)
            audio.write(sample, wav, eng.sample_rate)
            return f"{len(wav) / eng.sample_rate:.1f} s sintéticos"

        step("sample_audio", make_sample)

        studio = Studio(engine="selftest", voices_root=tmp / "voices", out_dir=tmp / "outputs")
        state: dict = {}

        def voices():
            chk = studio.voices.check([sample])
            assert chk["assessment"]["ok"], chk["assessment"]
            try:
                studio.voices.create("Sem consentimento", [sample], {"confirmed": False})
                raise AssertionError("criou voz sem consentimento")
            except ConsentError:
                pass
            v = studio.voices.create("Teste Selftest", [sample], {"confirmed": True, "owner": "self"}, default_engine="selftest")
            state["voice_id"] = v["voice_id"]
            return f"voz {v['voice_id']} ({chk['assessment']['status']}); consentimento obrigatório verificado"

        step("voices", voices)

        def http_and_jobs():
            import httpx
            import uvicorn

            from .server import create_app

            port = _free_port()
            token = "selftest-token"
            server = uvicorn.Server(uvicorn.Config(create_app(studio, token), host="127.0.0.1", port=port, log_level="error"))
            th = threading.Thread(target=server.run, daemon=True)
            th.start()
            for _ in range(100):
                if server.started:
                    break
                time.sleep(0.05)
            base = f"http://127.0.0.1:{port}"
            h = {"Authorization": f"Bearer {token}"}
            try:
                assert httpx.get(f"{base}/api/health").json()["ok"]
                assert httpx.get(f"{base}/api/voices").status_code == 401, "API sem token devia dar 401"
                assert httpx.get(f"{base}/api/health", headers={"Host": "evil.example"}).status_code == 403, "host guard"
                assert "OpenVoice Studio" in httpx.get(f"{base}/").text
                vs = httpx.get(f"{base}/api/voices", headers=h).json()["voices"]
                assert any(v["voice_id"] == state["voice_id"] for v in vs)
                job = httpx.post(f"{base}/api/generate", headers=h, json={
                    "voice_id": state["voice_id"], "style": "documental",
                    "text": "Bem-vindos. Hoje vamos ver, passo a passo, como funciona o OpenVoice Studio, que custa 0 €."}).json()
                for _ in range(60):
                    j = httpx.get(f"{base}/api/jobs/{job['job_id']}?wait=5", headers=h, timeout=10).json()
                    if j["status"] in {"done", "error"}:
                        break
                assert j["status"] == "done", j.get("error")
                assert j["progress"]["percent"] == 100 and j["progress"]["eta_seconds"] == 0, j["progress"]
                wav =httpx.get(f"{base}{j['result']['audio_url']}?token={token}")
                assert wav.status_code == 200 and wav.content[:4] == b"RIFF", "áudio"
                ev = httpx.post(f"{base}/api/evaluate", headers=h, json={"job_id": job["job_id"]}).json()
                for _ in range(60):
                    e = httpx.get(f"{base}/api/jobs/{ev['job_id']}?wait=5", headers=h, timeout=10).json()
                    if e["status"] in {"done", "error"}:
                        break
                assert e["status"] == "done", e.get("error")
                assert e["result"]["metrics"]["speaker_similarity"] is not None
                return f"API, token, host guard, geração ({j['result']['duration_s']} s) e avaliação"
            finally:
                server.should_exit = True
                th.join(timeout=10)

        step("http_api", http_and_jobs)

    def mcp_tools():
        from .mcp_server import mcp

        names = sorted(t.name for t in asyncio.run(mcp.list_tools()))
        expected = {"get_engine_capabilities", "list_voices", "generate_narration", "get_job_status",
                    "get_audio_artifact", "run_voice_evaluation"}
        missing = expected - set(names)
        assert not missing, f"faltam ferramentas: {missing}"
        return ", ".join(names)

    step("mcp_tools", mcp_tools)

    def real_engine_import():
        from .config import load_settings
        from .engines import get_engine

        name = load_settings()["engine"]
        if name == "chatterbox-mtl":
            import chatterbox.mtl_tts  # noqa: F401  (heavy import: torch + transformers)
            import perth

            assert perth.PerthImplicitWatermarker is not None, "marca de água PerTh indisponível (falta pkg_resources/setuptools<80?)"
        st = get_engine(name).status()
        return f"{name}: importa correctamente; modelos {'instalados' if st['models_installed'] else 'por instalar'}"

    step("engine_import", real_engine_import)

    return {"ok": all(r["ok"] for r in results.values()), "steps": results}
