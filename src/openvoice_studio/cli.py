"""Command line: openvoice-studio <command>."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

from . import __version__


def _utf8_stdout() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except (AttributeError, ValueError):
            pass


def cmd_doctor(_: argparse.Namespace) -> int:
    from .doctor import run_doctor

    report = run_doctor()
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if report["ok"] else 1


def cmd_selftest(_: argparse.Namespace) -> int:
    from .selftest import run_selftest

    report = run_selftest()
    for step in report["steps"].values():
        if step["ok"]:
            step.pop("trace", None)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if report["ok"] else 1


def cmd_setup(a: argparse.Namespace) -> int:
    from .setup_cmd import run_setup

    return run_setup(a.yes, a.evaluation, a.skip_models, a.skip_mcp)


def _serve(a: argparse.Namespace, open_browser: bool) -> int:
    import httpx

    from .config import DEFAULT_HOST, load_settings, ui_link
    from .server import port_in_use, run

    port = a.port or load_settings()["port"]
    link = ui_link(port)
    if port_in_use(DEFAULT_HOST, port):
        try:
            same = httpx.get(f"http://{DEFAULT_HOST}:{port}/api/health", timeout=2).json().get("product") == "openvoice-studio"
        except Exception:  # noqa: BLE001
            same = False
        if not same:
            print(f"A porta {port} está ocupada por outro programa. Usa --port <outra>.", file=sys.stderr)
            return 1
        print(f"O OpenVoice Studio já está a correr.\nWeb UI: {link}", flush=True)
        if open_browser:
            import webbrowser

            webbrowser.open(link)
        return 0
    print(f"OpenVoice Studio {__version__}", flush=True)
    print(f"Web UI: {link}", flush=True)
    print(f"API:    http://{DEFAULT_HOST}:{port}/api  (token em Authorization: Bearer)", flush=True)
    print("Ctrl+C para parar.", flush=True)
    if open_browser:
        import threading
        import webbrowser

        threading.Timer(1.5, webbrowser.open, [link]).start()
    run(DEFAULT_HOST, port, quiet=a.quiet, preload=not a.no_preload)
    return 0


def cmd_ui(a: argparse.Namespace) -> int:
    return _serve(a, open_browser=not a.no_browser)


def cmd_serve(a: argparse.Namespace) -> int:
    a.quiet = True
    return _serve(a, open_browser=False)


def cmd_mcp(_: argparse.Namespace) -> int:
    from .mcp_server import run

    run()
    return 0


def cmd_voices(a: argparse.Namespace) -> int:
    from .voices import VoiceStore

    store = VoiceStore()
    if a.rebuild:
        voice = store.rebuild(a.rebuild)
        print(f"Voz {a.rebuild} reconstruída a partir das gravações guardadas: {json.dumps(voice['processing'], ensure_ascii=False)}")
        print("O motor prepara-a de novo na próxima geração.")
        return 0
    voices = [store.summary(v) for v in store.list()]
    if a.json:
        print(json.dumps({"voices": voices}, indent=2, ensure_ascii=False))
        return 0
    if not voices:
        print("Ainda não há vozes. Cria uma em `openvoice-studio ui`, secção 1.")
        return 0
    for v in voices:
        print(f"{v['voice_id']:<28} {v['name']:<24} {v['status']:<11} qualidade: {v['quality']:<11} "
              f"{v['speech_seconds']} s de fala  ({v['created_at']})")
    return 0


def cmd_generate(a: argparse.Namespace) -> int:
    from .studio import Studio

    text = Path(a.text_file).read_text(encoding="utf-8") if a.text_file else a.text
    if not text:
        print("Indica --text ou --text-file.", file=sys.stderr)
        return 2
    studio = Studio()
    job = studio.submit_generate(a.voice, text, a.style, a.language)
    print(f"Tarefa {job['job_id']} em curso…", file=sys.stderr, flush=True)
    job = studio.jobs.wait(job["job_id"], 24 * 3600)
    if job["status"] != "done":
        print(json.dumps(job, indent=2, ensure_ascii=False))
        return 1
    if a.out:
        shutil.copyfile(job["result"]["audio_path"], a.out)
        job["result"]["copied_to"] = str(Path(a.out).resolve())
    print(json.dumps(job["result"], indent=2, ensure_ascii=False))
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="openvoice-studio", description="Clonagem de voz e narração 100% local.")
    p.add_argument("--version", action="version", version=f"openvoice-studio {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    sub.add_parser("doctor", help="verificar o ambiente (JSON)").set_defaults(fn=cmd_doctor)
    sub.add_parser("selftest", help="teste ponta a ponta sem modelos (JSON)").set_defaults(fn=cmd_selftest)

    s = sub.add_parser("setup", help="descarregar modelos e registar o MCP no Claude")
    s.add_argument("--yes", "-y", action="store_true", help="não pedir confirmação")
    s.add_argument("--evaluation", action="store_true", help="instalar também o modelo de avaliação (faster-whisper)")
    s.add_argument("--skip-models", action="store_true")
    s.add_argument("--skip-mcp", action="store_true")
    s.set_defaults(fn=cmd_setup)

    for name, fn, help_ in [("ui", cmd_ui, "abrir a interface web"), ("serve", cmd_serve, "servidor HTTP sem abrir o browser")]:
        s = sub.add_parser(name, help=help_)
        s.add_argument("--port", type=int, default=None)
        s.add_argument("--no-browser", action="store_true")
        s.add_argument("--no-preload", action="store_true", help="não carregar o modelo ao arrancar")
        s.add_argument("--quiet", action="store_true")
        s.set_defaults(fn=fn)

    sub.add_parser("mcp", help="servidor MCP (stdio) para o Claude").set_defaults(fn=cmd_mcp)

    s = sub.add_parser("voices", help="listar vozes")
    s.add_argument("--json", action="store_true")
    s.add_argument("--rebuild", metavar="VOICE_ID", help="refazer a referência de clonagem a partir das gravações guardadas")
    s.set_defaults(fn=cmd_voices)

    s = sub.add_parser("generate", help="gerar narração pela linha de comandos")
    s.add_argument("--voice", required=True)
    s.add_argument("--text")
    s.add_argument("--text-file")
    s.add_argument("--style", default="documental")
    s.add_argument("--language", default=None)
    s.add_argument("--out")
    s.set_defaults(fn=cmd_generate)
    return p


def main(argv: list[str] | None = None) -> int:
    _utf8_stdout()
    args = build_parser().parse_args(argv)
    return args.fn(args)
