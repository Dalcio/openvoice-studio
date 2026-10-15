# Installing OpenVoice Studio with an AI coding agent

This file is a checklist for an agent (for example, Claude Code) that installs and verifies OpenVoice Studio for a user. Follow it in order and stop for the user only where it says **[user]**.

## Rules

- Never use `sudo` or admin rights without asking the user first.
- Do not edit configuration files other than those `openvoice-studio setup` edits. Setup backs them up as `*.bak-YYYYmmdd-HHMMSS`.
- Never create a voice from someone else's recordings unless the user confirms that person's explicit permission.
- Do not rewrite git history.

## Steps

1. **Detect the OS** and show `git log --oneline -5` and `git status`.
2. **Prerequisites**:
   - `uv`: if it is missing, use the official installer from astral.sh.
   - `ffmpeg`: if it is missing, propose the package-manager command and ask before using sudo.
   - A Chromium-based browser for the UI. Microphone recording needs `getUserMedia`, which works on `127.0.0.1`.
3. **Install from the checkout**:
   - Linux/macOS: `OVS_SOURCE=. sh install.sh`
   - Windows: `$env:OVS_SOURCE="."; powershell -ExecutionPolicy ByPass -File .\install.ps1`
   - Make sure `uv tool dir --bin` is on PATH (`uv tool update-shell`).
4. **Verify**: `openvoice-studio doctor` and `openvoice-studio selftest` must both print `"ok": true`. If not, read `checks`/`steps` and fix the cause. The usual causes are a missing ffmpeg, models not downloaded (`openvoice-studio setup --yes`), or the wrong Python version (requires 3.10–3.12; the installer uses 3.11).
5. **Open the UI**: start `openvoice-studio ui --no-browser` in the background, copy the `Web UI:` line (`http://127.0.0.1:8765/#token=...`), and open it in the user's browser.
6. **[user] Create the voice** in section "1. Voz": upload 1–3 min of audio, or record 2–3 takes reading the script. Then click "Verificar gravação", tick the consent box, and click "Criar voz". The browser asks for microphone permission.
7. Confirm the voice with `openvoice-studio voices`.
8. **[user] Test narration** in section "4. Narração" and ask how it sounded (similarity, clarity).
9. **MCP**: `claude mcp list` should show `openvoice-studio … Connected`. MCP tools load only in a new session, so the user must open a new Claude Code session or restart Claude Desktop.

## Optional

`openvoice-studio setup --yes --evaluation` downloads the faster-whisper `small` model (~0.5 GB, MIT). `run_voice_evaluation` then also reports WER/CER.

## Troubleshooting

- The server log is at `<data dir>/logs/server.log` when the MCP server started the server.
- The first generation loads the model (30–90 s on CPU), so use `wait_seconds` ≥ 300 for MCP calls.
- `HF_TOKEN` is not required: the default model is public.
