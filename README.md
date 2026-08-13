<img src="assets/cac-logo.png" width="100" />

# AnyCloudLLM — Chase a Cloud

> *We build universes.*


Scan the host hardware, pick a GGUF model that actually fits it, download it once, and serve it
over an OpenAI-compatible HTTP API on localhost.

After the first download there is **no cloud dependency** — no API keys, no outbound calls, no
telemetry. It is a standalone app: nothing here requires Hermes.

## Install

```bash
pip install anycloudllm
```

Optional extras:

```bash
pip install "anycloudllm[gpu]"      # pynvml, for NVIDIA VRAM detection
pip install "anycloudllm[hermes]"   # only if you want the Hermes bridge
```

`llama-cpp-python` builds a native library at install time. If you want GPU offload, install it
first with the right backend flags (CUDA, Metal, ROCm) — see the llama-cpp-python README — then
install this package.

## Run

```bash
python -m anycloudllm
```

or, via the console script:

```bash
anycloudllm --port 8080
```

One process gives you both a chat page and an API. It scans the hardware, picks and
downloads a model if needed, loads it, then serves:

```
Scanning hardware...
  15.9 GB RAM, 6 CPUs, no GPU detected
Selected model: bartowski/Meta-Llama-3.1-8B-Instruct-GGUF/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf (reason: 0.0 GB VRAM / 15.9 GB RAM fits an 8B model at Q4_K_M)
Already cached: C:\Users\you\.cache\anycloudllm\models\Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf
Starting server at http://127.0.0.1:8081
  Loading the model into memory - this takes a minute on CPU.

  Chat in your browser:  http://127.0.0.1:8081
  OpenAI-compatible API: http://127.0.0.1:8081/v1

  Memory: low free memory (1.0 GB free vs 4.9 GB model): streaming weights via mmap, no pinning or repacking
```

Open the first URL and type. The chat page is a single static file served from `/` on the
same origin as the API — no separate frontend, no build step, no configuration.

**The model loads before the port opens.** On CPU that takes roughly a minute for an 8B
model; the browser will refuse the connection until `Uvicorn running on ...` appears. Wait
for that line rather than assuming it hung.

> **Port 8080 is a bad default on this machine.** It is already taken by the newsletter
> project's Vite dev server. Pass `--port 8081` (or set `ANY_COMPANY_LLM_PORT`).

### Standalone executable

`anycloudllm.spec` builds a single self-contained `.exe` with PyInstaller — Python,
llama.cpp, and the chat UI all inside it. Double-click it and a console window opens,
loads the model, and prints the URL to open.

```bash
python -m PyInstaller --noconfirm anycloudllm.spec
```

The exe unpacks itself into `%TEMP%` on every launch, so the drive holding `%TEMP%` needs
a few hundred MB free. Redirect `TEMP`/`TMP` before building if the system drive is tight.

### Flags

| Flag | Meaning |
| --- | --- |
| `--port N` | Bind port (1–65535). Env fallback: `ANYCLOUDLLM_PORT`. Default `8080`. |
| `--host H` | Bind host. Env fallback: `ANYCLOUDLLM_HOST`. Default `127.0.0.1`. |
| `--no-download` | Exit non-zero instead of downloading a model that is not cached yet. |
| `--model-path PATH` | Serve this GGUF; skip hardware-based selection entirely. |
| `--n-ctx N` | Context window. Default `4096`. |
| `--airllm` | Print AirLLM layer-streaming options for large models, then exit. |
| `--airllm-disk-bw GB_S` | Disk bandwidth assumption for `--airllm` estimates (default `3.0` = NVMe). |

Model cache: `~/.cache/anycloudllm/models/` (override with `ANYCLOUDLLM_CACHE_DIR`).

## Chat UI

`GET /` serves `anycloudllm/web/index.html`: one file, no framework, no npm. It streams
tokens as they generate, keeps the conversation in memory (reload clears it), shows the
loaded GGUF and context size in the corner, and has a Stop button — which matters, because
CPU generation is slow enough that you will want to interrupt it.

It calls `/v1/chat/completions` on its own origin, so there is no CORS setup and no base-URL
to configure. If the file is missing from an install, the server logs a warning and serves
the API alone.

## API

The server is llama-cpp-python's built-in OpenAI-compatible app, so the usual endpoints work:

- `POST /v1/chat/completions` (supports `"stream": true`)
- `POST /v1/completions`
- `GET /v1/models`
- `GET /health` (added by this package — upstream does not ship one)
- `GET /api/info` (added by this package — model file, context size, GPU flag; what the UI displays)

```bash
curl http://127.0.0.1:8081/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model": "local", "messages": [{"role": "user", "content": "hi"}]}'
```

With the OpenAI SDK:

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8081/v1", api_key="not-needed")
client.chat.completions.create(model="local", messages=[{"role": "user", "content": "hi"}])
```

`logits_all` is forced off. llama-cpp-python's server defaults it to `True`, which sizes the
logits scratch array `(n_ctx, n_vocab)` — 2 GB of contiguous float32 at 4096 context on a
128k-vocab Llama. Only `/v1/completions` with `echo` + `logprobs` ever reads it.

## How the model is chosen

`hardware_scanner` reads RAM and CPU count from `psutil`, then probes VRAM via `pynvml`, falling
back to `torch.cuda`. Both GPU probes are fail-silent — a machine with no driver, no CUDA and
neither library installed simply reports `0.0` VRAM.

`model_selector` walks a first-match-wins ladder:

| Condition | Model | Quant |
| --- | --- | --- |
| VRAM ≥ 8 GB | `bartowski/Meta-Llama-3.1-8B-Instruct-GGUF` | `Q8_0` |
| VRAM ≥ 4 GB **or** RAM ≥ 15 GB | `bartowski/Meta-Llama-3.1-8B-Instruct-GGUF` | `Q4_K_M` |
| RAM ≥ 8 GB | `bartowski/Llama-3.2-3B-Instruct-GGUF` | `Q5_K_M` |
| otherwise | `bartowski/Llama-3.2-1B-Instruct-GGUF` | `Q4_K_M` |

Every selection carries a `reason` string explaining the match, which the CLI prints.

Use it directly:

```python
from anycloudllm import scan_hardware, select_model

profile = scan_hardware()
selection = select_model(profile)
print(selection.label, "-", selection.reason)
```

## AirLLM: running larger models with limited RAM

[AirLLM](https://github.com/lyogavin/airllm) streams transformer layers from disk one at a time,
letting you run 12B–70B models with only a few GB of RAM. Throughput is bounded by disk speed
rather than memory bandwidth (think ~1–4 tokens/minute on NVMe), but it works on any machine.

```bash
# See available models and speed estimates on your disk type:
anycloudllm --airllm
anycloudllm --airllm --airllm-disk-bw 0.5   # SATA SSD
anycloudllm --airllm --airllm-disk-bw 0.1   # HDD
```

To run a downloaded AirLLM model:
```bash
pip install airllm
anycloudllm --model-path /path/to/model.gguf
```

## Memory behaviour

llama.cpp can spend memory two ways beyond the mapped model, and llama-cpp-python's server
enables both by default. On a host with headroom they are worth having; on a tight one they
turn a slow-but-working load into a hard failure. `plan_memory()` measures free RAM against
the model file and enables each only when it fits:

| Feature | Extra cost | Enabled when |
| --- | --- | --- |
| Weight repacking (ggml extra buffer types) | ~1x model size, not mmap-backed | free RAM ≥ 1.25x model |
| `mlock` (pin weights, never evictable) | whole model resident | free RAM ≥ 2x model |

With both off, weights stream from disk via mmap and the process needs only the KV cache and
compute buffers. Slower, but it runs. The chosen plan is printed at startup. Without this,
a tight host dies with:

```
alloc_tensor_range: failed to allocate CPU_REPACK buffer of size 3359637504
llama_model_load: error loading model: unable to allocate CPU_REPACK buffer
```

## Error handling

- **Disk space**: checked before every download; a clear error is printed if the volume is too full.
- **Network errors**: wrapped with an actionable message — no raw tracebacks.
- **Missing `llama-cpp-python[server]`**: detected before the model download starts, not after.
- **Port conflicts**: the port is validated (1–65535 range) before any work begins.
- **Low memory**: repacking and pinning are disabled rather than allowed to fail the load.

## Optional Hermes bridge

`hermes_bridge` is inert unless `hermes_agents` is installed. It never affects the standalone path.

```python
from anycloudllm.hermes_bridge import get_hermes_adapter

adapter = get_hermes_adapter(port=8080)  # None if Hermes is not installed
```

## Tests

```bash
python -m pytest tests/ -v
```

The suite is offline: `psutil`, `pynvml`, `torch` and `huggingface_hub` are all mocked, and the
model cache is redirected to a temp directory.

## Layout

```
anycloudllm/
├── hardware_scanner.py   # detect RAM + VRAM
├── model_selector.py     # choose model + quant level + download
├── server.py             # memory planning + llama-cpp-python server + UI route
├── web/index.html        # the chat UI, served at /
├── cli.py                # python -m anycloudllm entry point
└── hermes_bridge.py      # optional Hermes AnyCloudLLMAdapter wiring
```
