"""`openvoice-studio setup`: download models and register the MCP server with Claude.

Every config file we touch is backed up first (<file>.bak-YYYYmmdd-HHMMSS next to it, or in
<data dir>/backups for ~/.claude.json).
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from .config import data_dir

MCP_NAME = "openvoice-studio"


def mcp_command() -> list[str]:
    exe = shutil.which("openvoice-studio")
    if exe:
        return [str(Path(exe).resolve()), "mcp"]
    return [sys.executable, "-m", "openvoice_studio", "mcp"]


def _stamp() -> str:
    return time.strftime("%Y%m%d-%H%M%S")


def backup(path: Path, into: Path | None = None) -> Path | None:
    if not path.exists():
        return None
    target_dir = into or path.parent
    target_dir.mkdir(parents=True, exist_ok=True)
    dst = target_dir / f"{path.name}.bak-{_stamp()}"
    shutil.copy2(path, dst)
    return dst


def _claude_cli() -> str | None:
    return shutil.which("claude")


def _run(cmd: list[str], timeout: int = 60) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, encoding="utf-8", errors="replace")


def register_claude_code(command: list[str]) -> dict:
    cli = _claude_cli()
    if not cli:
        return {"status": "skipped", "detail": "CLI `claude` não encontrada no PATH"}
    current = _run([cli, "mcp", "get", MCP_NAME])
    if current.returncode == 0 and all(part in current.stdout for part in command):
        return {"status": "unchanged", "detail": "já registado com o mesmo comando"}
    saved = backup(Path.home() / ".claude.json", into=data_dir() / "backups")
    if current.returncode == 0:
        _run([cli, "mcp", "remove", MCP_NAME, "-s", "user"])
    added = _run([cli, "mcp", "add", "-s", "user", MCP_NAME, "--", *command])
    if added.returncode != 0:
        return {"status": "error", "detail": (added.stderr or added.stdout).strip()[:400], "backup": str(saved) if saved else None}
    return {"status": "registered", "scope": "user", "command": command, "backup": str(saved) if saved else None}


def desktop_config_paths() -> list[Path]:
    if sys.platform == "win32":
        paths = [Path(os.environ.get("APPDATA", Path.home() / "AppData" / "Roaming")) / "Claude"]
        packages = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local")) / "Packages"
        if packages.exists():
            paths += [p / "LocalCache" / "Roaming" / "Claude" for p in packages.glob("Claude_*")]
    elif sys.platform == "darwin":
        paths = [Path.home() / "Library" / "Application Support" / "Claude"]
    else:
        paths = [Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config")) / "Claude"]
    return [p / "claude_desktop_config.json" for p in paths if p.is_dir()]


def register_claude_desktop(command: list[str]) -> list[dict]:
    results = []
    paths = desktop_config_paths()
    if not paths:
        return [{"status": "skipped", "detail": "Claude Desktop não encontrado"}]
    for path in paths:
        try:
            config = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
        except json.JSONDecodeError as exc:
            results.append({"status": "error", "path": str(path), "detail": f"JSON inválido, não alterei: {exc}"})
            continue
        servers = config.setdefault("mcpServers", {})
        entry = {"command": command[0], "args": command[1:]}
        if servers.get(MCP_NAME) == entry:
            results.append({"status": "unchanged", "path": str(path)})
            continue
        saved = backup(path)
        servers[MCP_NAME] = entry
        path.write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")
        results.append({"status": "registered", "path": str(path), "backup": str(saved) if saved else None})
    return results


def run_setup(yes: bool, evaluation: bool, skip_models: bool, skip_mcp: bool) -> int:
    from .engines.chatterbox_mtl import MODEL_SIZE_GB, cached_snapshot, download_models

    steps = []
    if not skip_models:
        steps.append(f"descarregar o modelo Chatterbox Multilingual (~{MODEL_SIZE_GB} GB, licença MIT) do Hugging Face")
    if evaluation:
        steps.append("descarregar o modelo de avaliação faster-whisper 'small' (~0,5 GB, MIT)")
    if not skip_mcp:
        steps.append("registar o servidor MCP no Claude Code e no Claude Desktop (com cópia de segurança das configurações)")
    print("O setup vai:\n  - " + "\n  - ".join(steps), flush=True)
    if not yes:
        if input("Continuar? [s/N] ").strip().lower() not in {"s", "sim", "y", "yes"}:
            print("Cancelado.")
            return 1
    report: dict = {}
    if not skip_models:
        if cached_snapshot():
            report["models"] = "já instalados"
        else:
            print("A descarregar modelos… (pode demorar)", flush=True)
            report["models"] = str(download_models())
        print("A carregar o modelo uma vez para confirmar que funciona… (1–2 min no CPU)", flush=True)
        from .engines import get_engine

        eng = get_engine("chatterbox-mtl")
        eng.load()
        report["engine_check"] = f"modelo carregado ({eng.device})"
    if evaluation:
        from .evaluation import download_whisper, whisper_status

        report["evaluation"] = "já instalado" if whisper_status()["installed"] else download_whisper()
    if not skip_mcp:
        cmd = mcp_command()
        report["claude_code"] = register_claude_code(cmd)
        report["claude_desktop"] = register_claude_desktop(cmd)
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)
    print("Setup concluído. Próximo passo: openvoice-studio doctor", flush=True)
    return 0
