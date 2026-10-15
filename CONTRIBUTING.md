# Contributing

Thanks for helping! Keep changes small, tested (`uv run pytest`, `openvoice-studio selftest`), and documented.

## Adding an engine adapter

Engines live in `src/openvoice_studio/engines/` and implement `engines/base.py::Engine`. Before any adapter is merged, it must pass both gates below.

### 1. Licence audit (before writing code)

Document the following in the pull request:

| Item | Requirement |
| --- | --- |
| Code licence | OSI-approved and permissive (MIT, Apache-2.0, BSD). No GPL/AGPL in-process. |
| Weights licence | Must allow commercial use. **Reject** CC-BY-NC, "research only", or custom non-commercial terms (for example, Coqui CPML, F5-TTS weights). |
| Training data | Note any known restrictions published by the authors. |
| Transitive deps | Check heavy deps (e.g. `espeak-ng` is GPL: only acceptable as an external process). |
| Watermark / safety | Keep any built-in watermark enabled. Note whether outputs are watermarked. |

Then add the engine to `THIRD_PARTY_NOTICES.md` with links to both licences.

### 2. Benchmark with the same protocol

Every engine is compared with the same protocol, so numbers are comparable:

1. **Voices**: the same 3 consented speakers (at least 1 Portuguese). Each is created from the same 60 s recording through `VoiceStore.create`, which applies the same reference selection.
2. **Texts**: the same 10 Portuguese sentences (numbers, currency, questions, long sentences) plus 2 paragraphs, generated with the `documental` style.
3. **Metrics** (`run_voice_evaluation`): speaker similarity (cosine on a **fixed** external encoder when comparing engines, not each engine's own), WER/CER with faster-whisper `small`, real-time factor on the same machine (state CPU/GPU), and peak RAM.
4. **Listening**: a blind MOS (1–5) for similarity and naturalness from at least 3 listeners.
5. **Report**: a table in the PR. A new default engine must not be worse on WER and must be better on similarity or MOS.

Pin the engine package version. Changing the version means running the benchmark again.

## Style

- Python 3.10+, type hints, no prints in `mcp_server.py` (stdout is the MCP transport).
- User-facing text in the UI is in European Portuguese; code and API docs are in English.
- Never commit audio, voices or tokens (`.gitignore` blocks `*.wav`, `voices/`, `outputs/`).
