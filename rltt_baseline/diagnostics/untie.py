"""Give each Ouro loop its own copy of the shared parameters.

The copies hold the same values, so the forward pass is unchanged; after ``backward`` the
gradient on loop t's copy is that loop's contribution g_t, and sum_t g_t equals the gradient
the shared parameter would have received.

Shared modules and how each call is attributed to a loop:
- decoder layers: called with ``current_ut`` by ``OuroModel.forward``;
- ``model.norm`` and ``model.early_exit_gate``: called once per loop, in loop order, so a
  per-forward call counter gives the loop index.

Gradient checkpointing is done here per (layer, loop) call, because the layer's own
checkpointing would recompute with the original (shared) parameters.
"""

from typing import Dict, Iterator, List, Tuple

import torch
from torch import nn
from torch.func import functional_call
from torch.utils.checkpoint import checkpoint


def _copies(module: nn.Module, loops: int) -> List[Dict[str, torch.Tensor]]:
    return [
        {name: p.detach().clone().requires_grad_(True) for name, p in module.named_parameters()}
        for _ in range(loops)
    ]


class _LoopLayer(nn.Module):
    """Runs the wrapped decoder layer with the parameter copy of ``current_ut``."""

    def __init__(self, layer: nn.Module, params, use_checkpoint: bool):
        super().__init__()
        self.layer = layer
        self.loop_params = params
        self.use_checkpoint = use_checkpoint
        self.attention_type = getattr(layer, "attention_type", "full_attention")

    def forward(self, hidden_states, *args, current_ut: int = 0, **kwargs):
        params = self.loop_params[current_ut]

        def run(h):
            return functional_call(self.layer, params, (h, *args),
                                   {"current_ut": current_ut, **kwargs})

        if self.use_checkpoint and torch.is_grad_enabled():
            return checkpoint(run, hidden_states, use_reentrant=False)
        return run(hidden_states)


class _CountedModule(nn.Module):
    """Uses the parameter copy of the n-th call since the last ``reset``."""

    def __init__(self, module: nn.Module, params):
        super().__init__()
        self.module = module
        self.loop_params = params
        self.calls = 0

    def reset(self):
        self.calls = 0

    def forward(self, *args, **kwargs):
        if self.calls >= len(self.loop_params):
            raise RuntimeError("shared module called more times than there are loops")
        params = self.loop_params[self.calls]
        self.calls += 1
        return functional_call(self.module, params, args, kwargs)


class LoopUntie:
    """Context-free handle; call ``restore()`` to put the original modules back."""

    def __init__(self, model: nn.Module, use_checkpoint: bool = True):
        inner = model.model
        self.model = model
        self.inner = inner
        self.loops = int(getattr(inner, "total_ut_steps", None) or inner.config.total_ut_steps)
        self._orig_layers = list(inner.layers)
        self._orig_norm = inner.norm
        self._orig_gate = inner.early_exit_gate
        self._orig_requires_grad = {id(p): p.requires_grad for p in model.parameters()}
        for p in model.parameters():
            p.requires_grad_(False)

        self.groups: Dict[str, List[Dict[str, torch.Tensor]]] = {}
        wrapped = []
        for idx, layer in enumerate(self._orig_layers):
            if hasattr(layer, "gradient_checkpointing"):
                layer.gradient_checkpointing = False
            params = _copies(layer, self.loops)
            self.groups[f"layers.{idx}"] = params
            wrapped.append(_LoopLayer(layer, params, use_checkpoint))
        inner.layers = nn.ModuleList(wrapped)

        self.groups["norm"] = _copies(self._orig_norm, self.loops)
        self.groups["early_exit_gate"] = _copies(self._orig_gate, self.loops)
        self._norm = _CountedModule(self._orig_norm, self.groups["norm"])
        self._gate = _CountedModule(self._orig_gate, self.groups["early_exit_gate"])
        inner.norm = self._norm
        inner.early_exit_gate = self._gate
        self._hook = inner.register_forward_pre_hook(self._reset_counters)

    def _reset_counters(self, module, args):
        self._norm.reset()
        self._gate.reset()

    def named_loop_params(self) -> Iterator[Tuple[str, List[torch.Tensor]]]:
        """(``group.param_name``, [copy for loop 1..T]) for every shared parameter."""
        for group, per_loop in self.groups.items():
            for name in per_loop[0]:
                yield f"{group}.{name}", [per_loop[t][name] for t in range(self.loops)]

    def loop_parameters(self) -> List[torch.Tensor]:
        return [p for _, per_loop in self.named_loop_params() for p in per_loop]

    def zero_grad(self):
        for p in self.loop_parameters():
            p.grad = None

    def restore(self):
        self._hook.remove()
        self.inner.layers = nn.ModuleList(self._orig_layers)
        self.inner.norm = self._orig_norm
        self.inner.early_exit_gate = self._orig_gate
        for p in self.model.parameters():
            p.requires_grad_(self._orig_requires_grad[id(p)])
