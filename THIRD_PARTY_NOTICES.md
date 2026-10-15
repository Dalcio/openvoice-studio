# Third-party notices

OpenVoice Studio downloads or depends on the following components. Each keeps its own licence.

| Component | Use | Licence |
| --- | --- | --- |
| [Chatterbox / chatterbox-tts](https://github.com/resemble-ai/chatterbox) (Resemble AI), code | Speech engine | MIT |
| [ResembleAI/chatterbox](https://huggingface.co/ResembleAI/chatterbox) weights (`t3_mtl23ls_v2`, `s3gen`, `ve`, tokenizer) | Multilingual model | MIT |
| [resemble-perth](https://github.com/resemble-ai/Perth) | Imperceptible watermark on outputs | MIT |
| [faster-whisper](https://github.com/SYSTRAN/faster-whisper) | Transcription for the WER/CER metric | MIT |
| [Systran/faster-whisper-small](https://huggingface.co/Systran/faster-whisper-small) (converted from OpenAI Whisper) | Evaluation model (optional) | MIT |
| PyTorch, torchaudio | Inference runtime | BSD-3-Clause |
| FastAPI, Uvicorn, httpx, python-multipart | HTTP server and client | MIT / BSD-3-Clause / Apache-2.0 |
| [MCP Python SDK](https://github.com/modelcontextprotocol/python-sdk) | MCP server | MIT |
| NumPy, SoundFile, librosa, pyloudnorm | Audio processing | BSD-3-Clause / BSD-3-Clause / ISC / MIT |
| FFmpeg (external program, not bundled) | Audio decoding | LGPL-2.1+ / GPL depending on build |

The licences were checked on 2026-10-15 for the pinned versions in `pyproject.toml`.
