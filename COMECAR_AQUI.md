# Começar aqui

O OpenVoice Studio cria uma cópia digital da tua voz a partir de 1 a 3 minutos de gravação e gera narrações com ela. Tudo corre no teu computador: nenhum áudio sai daqui e o custo é 0 €.

## 1. Instalar

Windows (PowerShell, dentro desta pasta):

```powershell
$env:OVS_SOURCE="."; powershell -ExecutionPolicy ByPass -File .\install.ps1
```

macOS / Linux (dentro desta pasta):

```bash
OVS_SOURCE=. sh install.sh
```

O instalador faz o seguinte:

- instala o `uv`, se faltar;
- instala o comando `openvoice-studio`;
- descarrega o modelo de voz (cerca de 3,2 GB, só da primeira vez);
- regista o servidor MCP no Claude Code e no Claude Desktop, com cópia de segurança das configurações;
- corre `openvoice-studio doctor`.

O resultado do `doctor` deve mostrar `"ok": true`.

Se faltar o `ffmpeg`:

- Windows: `winget install --id Gyan.FFmpeg -e`
- macOS: `brew install ffmpeg`
- Ubuntu: `sudo apt-get install -y ffmpeg`

## 2. Abrir a interface

```bash
openvoice-studio ui
```

Abre no browser o link `http://127.0.0.1:8765/#token=…`. O token protege a API local.

## 3. Criar a tua voz (secção "1. Voz")

Escolhe uma de duas opções:

- **Opção A, carregar:** 1 a 3 minutos de gravação tua (WAV, MP3 ou M4A), só com uma pessoa a falar e sem música.
- **Opção B, gravar:** carrega em **● Gravar** e lê o texto mostrado com voz natural, num local silencioso e com o microfone a 15–20 cm. Grava 2 a 3 vezes, até perto de 1 minuto no total. O Chrome pede autorização para usar o microfone: carrega em **Permitir**.

Depois:

1. Carrega em **Verificar gravação**. Se aparecer "insuficiente", grava mais ou num sítio mais silencioso.
2. Marca o consentimento.
3. Carrega em **Criar voz**.

A primeira vez demora mais porque carrega o modelo. O `voice_id` aparece na secção "2. As tuas vozes" e também com `openvoice-studio voices`.

## 4. Gerar uma narração (secção "4. Narração")

Escolhe a voz e o estilo (secção 3), escreve o texto e carrega em **Gerar narração**. No CPU de um portátil, cada 10 s de áudio demoram cerca de 1 a 2 minutos. Para teres métricas de semelhança e de clareza, carrega em **Avaliar**.

## 5. Usar a partir do Claude (MCP)

Confirma que o servidor está registado:

```bash
claude mcp list
```

Deve aparecer `openvoice-studio … Connected`. Depois abre uma **sessão nova** do Claude Code, ou reinicia o Claude Desktop, e pede, por exemplo:

> Usa o servidor MCP openvoice-studio: lista as vozes e gera com a voz `<voice_id>` a frase "Olá a todos" em estilo documental, com wait_seconds 300. Copia o áudio para o Desktop com get_audio_artifact.

## Problemas comuns

| Sintoma | O que fazer |
| --- | --- |
| `openvoice-studio` não é reconhecido | Abre um terminal novo, ou corre `uv tool update-shell` |
| `doctor` diz `engine_models: false` | `openvoice-studio setup --yes` |
| A gravação não começa | Carrega no cadeado da barra de endereço e permite o microfone |
| A voz não se parece contigo | Grava mais 30–60 s num local mais silencioso e cria a voz outra vez |
| A porta 8765 está ocupada | `openvoice-studio ui --port 8777` |

## Uso responsável

Só cries vozes tuas ou de pessoas que te deram autorização expressa. Nunca uses as narrações para enganar ninguém nem para te fazeres passar por outra pessoa. O motor por omissão marca os áudios com uma marca de água imperceptível.
