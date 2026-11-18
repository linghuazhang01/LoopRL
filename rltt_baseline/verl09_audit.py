"""Native engine audit: reuse the legacy statistics and persistent shard format."""
from pathlib import Path

import numpy as np
import torch

from rltt_baseline import plc
from rltt_baseline.loop_audit import LoopAudit, distribution_stats, row_key
from rltt_baseline.ouro_rltt.rltt_per_loop import exit_pdf_from_gates


class NativeLoopAudit(LoopAudit):
    def __init__(self, module, config, rank, settings):
        modules = dict(module.named_modules())
        models = [m for m in modules.values() if type(m).__name__ == "OuroModel"]
        heads = [m for name, m in modules.items() if name.split(".")[-1] == "lm_head"]
        if len(models) != 1 or len(heads) != 1:
            raise ValueError("Native audit requires exactly one OuroModel and lm_head")
        self.model, self.lm_head = models[0], heads[0]
        if type(self.lm_head).__name__ == "FullyShardedDataParallel":
            raise ValueError("Native audit cannot issue extra lm_head FSDP collectives")
        self.rank = rank
        self.directory = Path(config["dir"])
        self.every_n_steps = int(config["every_n_steps"])
        self.top_k = int(config["top_k"])
        self.chunk_tokens = int(config["chunk_tokens"])
        self.save_tokens = bool(config["save_tokens"])
        self.loop_weights = bool(config.get("loop_weights", False))
        self.plc_settings = plc.PLCSettings.from_mapping(settings.get("plc", {})) if settings.get("method") == "plc" else None
        self._active = False
        self._records, self._error = [], None

    def begin(self, data):
        from verl.utils import tensordict_utils as tu
        step = tu.get(data, "global_steps", None)
        if step is None or int(step) < 1:
            raise ValueError("Native audit requires trainer global_steps metadata")
        self.step = int(step)
        self._active = (self.step - 1) % self.every_n_steps == 0
        self._temperature = float(tu.get(data, "temperature", 1.0))
        self._records, self._error = [], None

    def end(self):
        if self._active:
            self._write(self.step)
        self._active = False

    def hook(self, data, inputs):
        if not self._active:
            return None
        def capture(module, args, output):
            try:
                self._records.append(self.measure(data, inputs, output[1], output[2]))
            except Exception as exc:
                self._error = repr(exc)
        return self.model.register_forward_hook(capture)

    @torch.no_grad()
    def measure(self, data, inputs, hidden, gates):
        # Native dense engine right-pads whole sequences, unlike the legacy
        # left-padded prompt/right-padded response representation.
        prompt_lengths = data["prompts"].offsets().diff()
        response_lengths = data["responses"].offsets().diff()
        rows = torch.repeat_interleave(torch.arange(len(prompt_lengths), device=hidden[0].device), response_lengths)
        positions = torch.cat([torch.arange(int(p) - 1, int(p + r) - 1, device=rows.device)
                               for p, r in zip(prompt_lengths, response_lengths)])
        labels = data["responses"].values()
        stats = distribution_stats([h[rows, positions] for h in hidden], self.lm_head, labels,
                                   self._temperature, self.top_k, self.chunk_tokens)
        record = {name: value.numpy() for name, value in stats.items()}
        id_type = np.min_scalar_type(self.lm_head.out_features - 1)
        record["topk_ids"] = record["topk_ids"].astype(id_type)
        record["sampled_token"] = labels.cpu().numpy().astype(id_type)
        record["response_tokens"] = response_lengths.cpu().numpy()
        record["prompt_tokens"] = prompt_lengths.cpu().numpy()
        record["row_keys"] = np.array([row_key(ids.cpu().numpy(), np.ones(len(ids), dtype=bool))
                                       for ids in data["input_ids"].unbind()])
        if self.loop_weights:
            if len(gates) != len(hidden):
                raise ValueError("Native audit exit gate count differs from loop count")
            record["exit_pdf"] = exit_pdf_from_gates([g[rows, positions] for g in gates]).cpu().numpy()
            if self.plc_settings is not None:
                ell = stats["sampled_log_prob"]
                record["plc_weight"] = plc.compute_loop_weights(
                    ell, self.plc_settings.weight_mode, self.plc_settings.trunc_cap).cpu().numpy()
        return record
