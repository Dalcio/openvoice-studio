"""Text preparation for narration: Portuguese number/currency expansion, sentence chunking, styles."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# ---------------------------------------------------------------- styles


@dataclass(frozen=True)
class Style:
    key: str
    label: str
    description: str
    exaggeration: float  # expressiveness (Chatterbox emotion_adv)
    cfg_weight: float  # lower = slower, more deliberate pacing
    temperature: float
    pause_sentence: float  # seconds between sentences
    pause_paragraph: float
    extra: dict = field(default_factory=dict)


STYLES: dict[str, Style] = {
    s.key: s
    for s in [
        Style("documental", "Documental", "Calmo, claro e pausado, como um documentário.", 0.40, 0.40, 0.70, 0.50, 0.90),
        Style("natural", "Natural", "Conversa normal, ritmo do dia a dia.", 0.50, 0.50, 0.80, 0.35, 0.70),
        Style("didactico", "Didáctico", "Mais lento e articulado, para explicar passo a passo.", 0.35, 0.30, 0.70, 0.60, 1.00),
        Style("energetico", "Enérgico", "Mais expressivo e rápido, para anúncios e redes sociais.", 0.70, 0.30, 0.80, 0.25, 0.50),
    ]
}
DEFAULT_STYLE = "documental"


def get_style(key: str | None) -> Style:
    key = (key or DEFAULT_STYLE).strip().lower()
    aliases = {"documentario": "documental", "documentary": "documental", "didatico": "didactico",
               "energico": "energetico", "conversa": "natural", "conversational": "natural"}
    key = aliases.get(key, key)
    if key not in STYLES:
        raise ValueError(f"Estilo desconhecido '{key}'. Opções: {', '.join(STYLES)}")
    return STYLES[key]


# ---------------------------------------------------------------- numbers (pt-PT)

_UNITS = ["zero", "um", "dois", "três", "quatro", "cinco", "seis", "sete", "oito", "nove", "dez",
          "onze", "doze", "treze", "catorze", "quinze", "dezasseis", "dezassete", "dezoito", "dezanove"]
_TENS = ["", "", "vinte", "trinta", "quarenta", "cinquenta", "sessenta", "setenta", "oitenta", "noventa"]
_HUNDREDS = ["", "cento", "duzentos", "trezentos", "quatrocentos", "quinhentos", "seiscentos",
             "setecentos", "oitocentos", "novecentos"]


def _below_1000(n: int) -> str:
    if n < 20:
        return _UNITS[n]
    if n < 100:
        t, u = divmod(n, 10)
        return _TENS[t] + (f" e {_UNITS[u]}" if u else "")
    if n == 100:
        return "cem"
    h, rest = divmod(n, 100)
    return _HUNDREDS[h] + (f" e {_below_1000(rest)}" if rest else "")


def number_to_words(n: int) -> str:
    if n < 0:
        return "menos " + number_to_words(-n)
    if n < 1000:
        return _below_1000(n)
    millions, rest = divmod(n, 10**6)
    thousands, units = divmod(rest, 1000)
    parts: list[str] = []
    if millions:  # 1000 milhões -> "mil milhões" (pt-PT long scale)
        parts.append("um milhão" if millions == 1 else f"{number_to_words(millions)} milhões")
    if thousands:
        parts.append("mil" if thousands == 1 else f"{_below_1000(thousands)} mil")
    if units:
        # "mil e um", "mil e cem", but "mil duzentos e trinta"
        joiner = " e " if (units < 100 or units % 100 == 0) else " "
        return " ".join(parts) + joiner + _below_1000(units)
    if len(parts) > 1:
        return " ".join(parts[:-1]) + " e " + parts[-1]
    return parts[0]


def _num(s: str) -> str:
    s = s.replace(" ", "").replace(" ", "")
    if re.fullmatch(r"\d{1,3}(\.\d{3})+", s):  # 1.000.000 (pt thousands separator)
        s = s.replace(".", "")
    if "," in s:
        whole, frac = s.split(",", 1)
        frac_words = number_to_words(int(frac)) if len(frac) <= 2 and not frac.startswith("0") else " ".join(_UNITS[int(d)] for d in frac)
        return f"{number_to_words(int(whole or 0))} vírgula {frac_words}"
    return number_to_words(int(s))


_CURRENCIES = [
    (r"€", "euro", "euros"),
    (r"(?:Kz|AOA|kwanzas?)", "kwanza", "kwanzas"),
    (r"(?:R\$)", "real", "reais"),
    (r"(?:US\$|\$|USD)", "dólar", "dólares"),
]
_NUM_RE = r"\d{1,3}(?:\.\d{3})+(?:,\d+)?|\d+(?:,\d+)?"


def normalize_text(text: str, language: str = "pt") -> str:
    """Expand things a TTS model reads badly. Only Portuguese gets number expansion."""
    text = text.replace("\r\n", "\n").strip()
    if not language.lower().startswith("pt"):
        return text

    def money(m: re.Match, sing: str, plur: str) -> str:
        raw = m.group("n")
        words = _num(raw)
        one = raw.replace(" ", "") in {"1", "1,0", "1,00"}
        return f"{words} {sing if one else plur}"

    for sym, sing, plur in _CURRENCIES:
        text = re.sub(rf"(?P<n>{_NUM_RE})\s?{sym}(?![\w])", lambda m, s=sing, p=plur: money(m, s, p), text)
        text = re.sub(rf"{sym}\s?(?P<n>{_NUM_RE})", lambda m, s=sing, p=plur: money(m, s, p), text)
    text = re.sub(rf"(?P<n>{_NUM_RE})\s?%", lambda m: f"{_num(m.group('n'))} por cento", text)
    text = re.sub(r"\b(\d{1,2})h(\d{2})\b", lambda m: f"{number_to_words(int(m.group(1)))} e {number_to_words(int(m.group(2)))}", text)
    text = re.sub(rf"(?<![\w/]){_NUM_RE}(?![\w/])", lambda m: _num(m.group(0)), text)
    return text


# ---------------------------------------------------------------- chunking

_SENT_END = re.compile(r"(?<=[.!?…])\s+")


def split_paragraphs(text: str) -> list[str]:
    return [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]


def split_sentences(paragraph: str, max_chars: int = 240, min_chars: int = 25) -> list[str]:
    """Sentence chunks small enough for one model call (~15 s of speech each)."""
    raw = [s.strip() for s in _SENT_END.split(paragraph.replace("\n", " ")) if s.strip()]
    out: list[str] = []
    for sent in raw:
        while len(sent) > max_chars:
            cut = max(sent.rfind(",", 0, max_chars), sent.rfind(";", 0, max_chars), sent.rfind(" ", 0, max_chars))
            cut = cut if cut > max_chars // 3 else max_chars
            out.append(sent[: cut + 1].strip())
            sent = sent[cut + 1 :].strip()
        if sent:
            out.append(sent)
    merged: list[str] = []
    for s in out:  # very short fragments sound odd on their own
        if merged and (len(s) < min_chars or len(merged[-1]) < min_chars) and len(merged[-1]) + len(s) < max_chars:
            merged[-1] = f"{merged[-1]} {s}"
        else:
            merged.append(s)
    return merged


def plan_chunks(text: str, language: str = "pt") -> list[dict]:
    """[{text, pause_after: 'sentence'|'paragraph'|None}]"""
    plan: list[dict] = []
    paragraphs = split_paragraphs(normalize_text(text, language))
    for pi, para in enumerate(paragraphs):
        sentences = split_sentences(para)
        for si, sent in enumerate(sentences):
            last_in_para = si == len(sentences) - 1
            last = last_in_para and pi == len(paragraphs) - 1
            plan.append({"text": sent, "pause_after": None if last else ("paragraph" if last_in_para else "sentence")})
    return plan


def expected_seconds(text: str) -> float:
    """Rough duration of Portuguese narration (~14 characters per second)."""
    return max(0.8, len(text) / 14.0)


# ---------------------------------------------------------------- recording script

RECORDING_SCRIPT = [
    "Olá! Estou a gravar esta amostra para criar a minha voz digital. Vou ler com calma e naturalidade, "
    "como se estivesse a explicar algo a um amigo. Hoje o dia está tranquilo e a luz entra pela janela da sala.",
    "Os números também contam: um, dois, três, quatro, cinco. Às vezes pergunto-me: será que a tecnologia "
    "nos aproxima ou nos afasta? Acho que depende de como a usamos. Por isso, prefiro ferramentas simples, "
    "honestas e úteis.",
    "Agora, com um pouco mais de expressão: que bom ver-te por aqui! Vamos começar? Primeiro, respiramos fundo. "
    "Depois, lemos cada frase com atenção, sem pressa, e terminamos com um sorriso. Obrigado por ouvires até ao fim.",
]
