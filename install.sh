#!/bin/sh
# OpenVoice Studio installer for macOS / Linux.
#   Remote: curl -LsSf https://raw.githubusercontent.com/Dalcio/openvoice-studio/main/install.sh | sh
#   Local:  OVS_SOURCE=. sh install.sh
# Env: OVS_SOURCE (path or git URL), OVS_SKIP_SETUP=1, OVS_EVALUATION=1, OVS_PYTHON (default 3.11)
set -eu

SOURCE="${OVS_SOURCE:-git+https://github.com/Dalcio/openvoice-studio}"
if [ -d "$SOURCE" ]; then SOURCE="$(cd "$SOURCE" && pwd)"; fi
PYTHON="${OVS_PYTHON:-3.11}"

step() { printf '\033[36m==> %s\033[0m\n' "$1"; }

if ! command -v uv >/dev/null 2>&1; then
  step "A instalar o uv (astral.sh)"
  curl -LsSf https://astral.sh/uv/install.sh | sh
  PATH="$HOME/.local/bin:$PATH"; export PATH
fi

if ! command -v ffmpeg >/dev/null 2>&1; then
  if command -v brew >/dev/null 2>&1; then hint="brew install ffmpeg"
  elif command -v apt-get >/dev/null 2>&1; then hint="sudo apt-get install -y ffmpeg"
  elif command -v dnf >/dev/null 2>&1; then hint="sudo dnf install -y ffmpeg"
  elif command -v pacman >/dev/null 2>&1; then hint="sudo pacman -S ffmpeg"
  else hint="instala o ffmpeg pelo gestor de pacotes"; fi
  echo "AVISO: ffmpeg não encontrado. Instala com: $hint" >&2
fi

step "A instalar openvoice-studio a partir de $SOURCE (Python $PYTHON)"
uv tool install --force --python "$PYTHON" --reinstall-package openvoice-studio "$SOURCE"

step "A garantir que a pasta de executáveis do uv está no PATH"
uv tool update-shell >/dev/null 2>&1 || true
BIN="$(uv tool dir --bin)"
case ":$PATH:" in *":$BIN:"*) ;; *) PATH="$BIN:$PATH"; export PATH ;; esac

if [ "${OVS_SKIP_SETUP:-0}" != "1" ]; then
  step "Setup: modelos + registo MCP no Claude"
  if [ "${OVS_EVALUATION:-0}" = "1" ]; then openvoice-studio setup --yes --evaluation
  else openvoice-studio setup --yes; fi
fi

step "Diagnóstico"
openvoice-studio doctor || true
echo
echo "Pronto. Abre a interface com:  openvoice-studio ui"
echo "(Se o comando não for encontrado, abre um terminal novo para actualizar o PATH.)"
