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

Output looks like:

```
Scanning hardware...
  31.9 GB RAM, 16 CPUs, 8.0 GB VRAM
Selected model: bartowski/Meta-Llama-3.1-8B-Instruct-GGUF/Meta-Llama-3.1-8B-Instruct-Q8_0.gguf (reason: 8.0 GB VRAM (>= 8 GB) fits an 8B model at Q8_0 near-lossless quality)
Downloading bartowski/Meta-Llama-3.1-8B-Instruct-GGUF/Meta-Llama-3.1-8B-Instruct-Q8_0.gguf...
  This may take several minutes. Progress is shown below.
Starting server at http://127.0.0.1:8080
```

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

## API

The server is llama-cpp-python's built-in OpenAI-compatible app, so the usual endpoints work:

- `POST /v1/chat/completions`
- `POST /v1/completions`
- `GET /v1/models`
- `GET /health` (added by this package — upstream does not ship one)

```bash
curl http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model": "local", "messages": [{"role": "user", "content": "hi"}]}'
```

With the OpenAI SDK:

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8080/v1", api_key="not-needed")
client.chat.completions.create(model="local", messages=[{"role": "user", "content": "hi"}])
```

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

## Error handling

- **Disk space**: checked before every download; a clear error is printed if the volume is too full.
- **Network errors**: wrapped with an actionable message — no raw tracebacks.
- **Missing `llama-cpp-python[server]`**: detected before the model download starts, not after.
- **Port conflicts**: the port is validated (1–65535 range) before any work begins.

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
├── server.py             # launch llama-cpp-python server
├── cli.py                # python -m anycloudllm entry point
└── hermes_bridge.py      # optional Hermes AnyCloudLLMAdapter wiring
```
