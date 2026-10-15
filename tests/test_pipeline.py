import numpy as np
import pytest

from openvoice_studio import audio
from openvoice_studio.text import normalize_text, number_to_words, plan_chunks, split_sentences
from openvoice_studio.voices import ConsentError, validate_consent


@pytest.mark.parametrize("n,words", [
    (0, "zero"), (16, "dezasseis"), (21, "vinte e um"), (100, "cem"), (101, "cento e um"),
    (1001, "mil e um"), (1100, "mil e cem"), (1234, "mil duzentos e trinta e quatro"),
    (2000, "dois mil"), (1_000_000, "um milhão"), (2_500_000, "dois milhões e quinhentos mil"),
])
def test_number_to_words(n, words):
    assert number_to_words(n) == words


def test_normalize_currency_and_percent():
    assert normalize_text("custa 0 €.") == "custa zero euros."
    assert normalize_text("Paga 500 Kz hoje") == "Paga quinhentos kwanzas hoje"
    assert normalize_text("1 €") == "um euro"
    assert normalize_text("50%") == "cinquenta por cento"
    assert normalize_text("3,5") == "três vírgula cinco"
    assert normalize_text("price 5", language="en") == "price 5"


def test_chunking_keeps_text_and_marks_paragraphs():
    text = "Primeira frase bastante longa para não juntar. Segunda frase também com tamanho.\n\nOutro parágrafo com texto."
    plan = plan_chunks(text)
    assert [c["pause_after"] for c in plan][-1] is None
    assert "paragraph" in [c["pause_after"] for c in plan]
    long = "palavra " * 100
    assert all(len(s) <= 240 for s in split_sentences(long))


def test_consent_rules():
    with pytest.raises(ConsentError):
        validate_consent({"confirmed": False})
    with pytest.raises(ConsentError):
        validate_consent({"confirmed": True, "owner": "authorized"})
    rec = validate_consent({"confirmed": True, "owner": "authorized", "authorized_by": "Maria, autorização escrita"})
    assert rec["owner"] == "authorized" and rec["statement"]


def test_analysis_detects_speech_and_clipping():
    sr = 24000
    t = np.arange(sr * 3) / sr
    tone = 0.3 * np.sin(2 * np.pi * 200 * t).astype(np.float32)
    wav = np.concatenate([audio.silence(1, sr) + 1e-4, tone, audio.silence(1, sr) + 1e-4])
    a = audio.analyze(wav, sr)
    assert 2.5 <= a["speech_s"] <= 3.5
    assert a["clipping_ratio"] == 0
    assert audio.analyze(np.clip(tone * 10, -1, 1), sr)["clipping_ratio"] > 0.1


def test_compress_pauses_shortens_long_silences():
    sr = 24000
    t = np.arange(sr) / sr
    tone = 0.3 * np.sin(2 * np.pi * 200 * t).astype(np.float32)
    wav = np.concatenate([tone, audio.silence(2.0, sr) + 1e-4, tone])
    out = audio.compress_pauses(wav, sr, max_pause_s=0.35, keep_s=0.25)
    assert 2.1 < len(out) / sr < 2.5
    assert audio.analyze(out, sr)["speech_s"] >= 1.8


def test_selftest_end_to_end():
    from openvoice_studio.selftest import run_selftest

    report = run_selftest()
    failed = {k: v for k, v in report["steps"].items() if not v["ok"] and k != "engine_import"}
    assert not failed, failed
