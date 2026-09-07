# Input notebook audit and synthesis

The three supplied notebooks were treated as untrusted technical inputs. Their
instructions were reviewed, but none were automatically executed and no embedded
credential was copied. The current `ARahim3/kaggle-tpu-lab` `main` branch at
commit `10897e5799c0d911d5c81b4f2f932619f02620bd` is the runtime baseline.

## Findings

| Input | What it actually runs | Useful ideas retained | Why it is not the TPU implementation |
|---|---|---|---|
| `qwen-3-8-27b-2x-t4-gpus.ipynb` (`266de1b9…`) | Q4 GGUF with a downloaded CUDA `llama.cpp` binary on dual T4 GPUs | Scratch storage, explicit cleanup, direct model download | CUDA and GPU layer offload do not run on TPU. Its markdown says 250k context while the command uses 120k, and its Kaggle metadata has no accelerator or Internet enabled. |
| `qwen3-8-27b.ipynb` (`ed1d1865…`) | Q4 GGUF through `llama-cpp-python`, optionally exposed through a named Cloudflare tunnel | OpenAI-compatible API shape, health checks, logs, explicit stop cell | It is a 32k dual-GPU recipe. A test cell contains a fixed low-entropy API key and the tunnel flow asks for long-lived Cloudflare credentials; neither pattern was copied. |
| `qwen3-8-27b-bf16-on-kaggle-tpu-130-tok-s-api.ipynb` (`f7055f30…`) | bf16 vLLM TPU serving with eight-way tensor parallelism | The correct TPU architecture, public datasets, XLA cache, MTP safety patch, generated API key, auto-stop | This is the right family but is an older snapshot. The current upstream version contains corrected startup estimates and documentation. |

## Synthesis decisions

- Use bf16 safetensors and vLLM TPU; do not mix in GGUF/CUDA commands.
- Attach the upstream public weights and environment datasets. Falling back to a
  Hugging Face download remains supported when the weights dataset is unavailable.
- Generate the notebook from `kernel/serve_qwen38.py` using
  `tools/sync_notebook.py`; the source and published UI notebook must match.
- Default the public guide to `text_only` plus `fast_start` and a 30-minute
  keepalive to reduce accidental quota use. Users can opt into images or longer
  serving windows in the documented config cell.
- Keep MTP enabled only with the bundled state-rollback patch; the server disables
  MTP if that patch cannot be applied.
- Generate a fresh inference API key for every run. Never place a Kaggle access
  token, Cloudflare account secret, or fixed inference key in a cell or commit.
- Attribute performance figures to the original project. This fork does not claim
  an independent benchmark until a full Kaggle TPU run completes.

## Reproduction checks

```bash
python tools/sync_notebook.py --check
python -m py_compile launch.py kernel/serve_qwen38.py tools/*.py
jq empty notebook/qwen38-tpu-serve.ipynb notebook/kernel-metadata.json
```
