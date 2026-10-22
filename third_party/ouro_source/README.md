# Official Ouro modeling files

Copied unchanged from Hugging Face [`ByteDance/Ouro-2.6B-Thinking`](https://huggingface.co/ByteDance/Ouro-2.6B-Thinking),
revision `f1edd81e7ac41355db670500ceaf204e0f73af68`, licensed under Apache-2.0 (see `LICENSE`).

| File | SHA256 |
|---|---|
| `modeling_ouro.py` | `263c097a2ed6870ce549a65ef6d6bb9e8e167423f87d2c494e0988b9a4f63f4b` |
| `configuration_ouro.py` | `ba60253335cecc5bd9cb3e5f8cfbd214aa7e865fc97a98208976fb2517811de6` |

`rltt_baseline.prepare_model` accepts a source model directory only if its `modeling_ouro.py`
matches this file. The tests use these files to build a tiny random Ouro model.
