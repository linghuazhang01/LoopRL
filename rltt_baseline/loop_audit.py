"""Per-loop next-token distributions on every generated token.

Rollouts sample from the last loop only, but every loop t has a distribution of its own,
P^(t) = softmax(lm_head(h_t) / temperature), with h_t the normed hidden state after loop t. On
audited steps this module records for each response token, given the prompt and the response so
far (loops are compared on the same context):

- ``kl`` (n, T, T): KL(P^(t) || P^(s)) in nats for every ordered loop pair;
- ``entropy``, ``sampled_log_prob`` (n, T): per loop, the latter for the token actually sampled;
- ``topk_ids``, ``topk_probs`` (n, T, k): each loop's k most likely tokens, most likely first.
  Full distributions are not stored: one vocabulary of float32 is about 200 KB per token and loop.

With ``loop_weights`` the audit also records the loop weights of each token (n, T):

- ``exit_pdf``: the exit gate's distribution over loops, ``q_t = lambda_t prod_{s<t}(1 - lambda_s)``
  with the last loop taking the rest (all methods; RLTT's ``loop_weighting: exit_pdf`` weight);
- ``plc_weight``: the earlier-loop weights that ``PLCActor.update_policy`` applies to this step
  (``plc.compute_loop_weights``: ``alpha_t * min(trunc_cap, rho_t)`` for t < T and 0 for the
  final loop with ``is_trunc``, ``alpha_t`` with ``early_prior``), computed from the actor's own
  per-loop log-probs of the same pass. PLC runs only.

The probability of the sampled token at each loop is ``exp(sampled_log_prob)``. The summary
also carries, for each loop t < T, the cross-loop ratio ``rho = P^(t)(y) / P^(T)(y)`` of the
sampled token (``xloop_*``: log rho, min(1, rho), share above 1 and below 0.1), and ``kl/T_t``
is the exact ``KL(P^(T) || P^(t))``, the trust region of the PLC loop term (``plc.kl_coef``).

Distributions come from the training forward (full per-loop KV), also with decode_last rollouts.

Steps are numbered by the trainer's ``global_steps`` when the batch carries it, else by counting
``compute_log_prob`` calls from ``first_step`` (1, or the loaded checkpoint's step + 1 when the run
resumes; set by ``rltt_baseline.config.add_audit_overrides``). A step whose ``summary.json``
already exists is never overwritten.

No forward pass is added. A temporary hook on ``OuroModel`` reads the per-loop hidden states of
the ``compute_log_prob`` pass that training runs anyway and applies ``lm_head`` there. That
head belongs to the root FSDP unit, which stays unsharded during the root forward, so these
calls are not collectives and ranks may make different numbers of them. An extra forward would
be a collective for every decoder layer and would need the same micro-batch count on all ranks.

Files of an audited step, in ``<output_dir>/audit/step_<step>/``:
    summary.json          mean, p5 and p95 of each token metric, pooled over the step
    responses.jsonl       one line per response: uid, lengths, reward, the same statistics
    tokens_rank<r>.npz    the per-token arrays above, read with ``read_response``

``last_loop_log_prob_gap`` in the summary compares the audited last loop with verl's
``old_log_probs``; it is close to 0 when tokens and distributions are aligned.
"""

import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch

from rltt_baseline import plc
from rltt_baseline.ouro_rltt.rltt_per_loop import exit_pdf_from_gates

PREFIX = "loop_audit/"
TOKEN_ARRAYS = ("sampled_token", "sampled_log_prob", "entropy", "kl", "topk_ids", "topk_probs")
WEIGHT_ARRAYS = ("exit_pdf", "plc_weight")  # with loop_weights; plc_weight in PLC runs only


def step_dir_name(step: int) -> str:
    return f"step_{step:06d}"


def response_positions(attention_mask: torch.Tensor, response_length: int, hidden_length: int):
    """Locate the hidden state that predicts each response token.

    Args:
        attention_mask: (B, S); prompts are left-padded and responses right-padded.
        response_length: the last ``response_length`` columns are the response.
        hidden_length: positions in the model output: B * S for a padded forward, or the
            number of attended tokens for verl's ``use_remove_padding`` forward, which packs
            the attended tokens of all rows into one sequence.

    Returns:
        (rows, tokens, positions): token ``tokens[i]`` of the response in row ``rows[i]`` is
        predicted by position ``positions[i]`` of the flattened model output.
    """
    batch, seq = attention_mask.shape
    mask = attention_mask.bool()
    rows, tokens = mask[:, -response_length:].nonzero(as_tuple=True)
    flat = rows * seq + seq - response_length - 1 + tokens
    if hidden_length == batch * seq:
        return rows, tokens, flat
    if hidden_length != int(mask.sum()):
        raise ValueError(f"model output has {hidden_length} positions; expected {batch * seq} "
                         f"(padded) or {int(mask.sum())} (remove_padding)")
    return rows, tokens, (mask.flatten().cumsum(0) - 1)[flat]


@torch.no_grad()
def distribution_stats(
    hidden_list: Sequence[torch.Tensor],
    lm_head: torch.nn.Module,
    labels: torch.Tensor,
    temperature: float = 1.0,
    top_k: int = 5,
    chunk_tokens: int = 512,
) -> Dict[str, torch.Tensor]:
    """Per-token statistics of every loop's distribution.

    Args:
        hidden_list: T tensors (n, H), the hidden state predicting each of the n tokens.
        labels: (n,) the sampled tokens.
        chunk_tokens: tokens per ``lm_head`` call; a chunk holds T * chunk_tokens * vocabulary
            float32 log-probabilities and as many probabilities.

    Returns:
        CPU tensors ``kl`` (n, T, T), ``entropy`` and ``sampled_log_prob`` (n, T), ``topk_ids``
        and ``topk_probs`` (n, T, k), float32 except the ids.
    """
    loops = len(hidden_list)
    parts = defaultdict(list)
    for start in range(0, labels.numel(), chunk_tokens):
        log_p = torch.stack(
            [torch.log_softmax(lm_head(h[start:start + chunk_tokens]).float() / temperature, -1)
             for h in hidden_list], dim=1)  # (c, T, V)
        p = log_p.exp()
        kl = log_p.new_zeros(log_p.shape[0], loops, loops)
        for t in range(loops):
            for s in range(loops):
                if s != t:  # summed term by term: close distributions keep their small KL
                    kl[:, t, s] = (p[:, t] * (log_p[:, t] - log_p[:, s])).sum(-1).clamp_min(0)
        top_log_p, top_ids = log_p.topk(top_k, dim=-1)
        picked = labels[start:start + chunk_tokens].view(-1, 1, 1).expand(-1, loops, 1)
        parts["kl"].append(kl.cpu())
        parts["entropy"].append(-(p * log_p).sum(-1).cpu())
        parts["sampled_log_prob"].append(log_p.gather(-1, picked).squeeze(-1).cpu())
        parts["topk_ids"].append(top_ids.cpu())
        parts["topk_probs"].append(top_log_p.exp().cpu())
    return {name: torch.cat(chunks) for name, chunks in parts.items()}


def row_key(input_ids: np.ndarray, attention_mask: np.ndarray) -> str:
    """Identity of a rollout, the same in the workers and in the driver."""
    tokens = input_ids[attention_mask.astype(bool)].astype(np.int64)
    return hashlib.sha1(tokens.tobytes()).hexdigest()[:16]


class LoopAudit:
    """Wraps a verl actor's ``compute_log_prob``; steps count from 1, as the trainer's do."""

    def __init__(self, actor, config, rank: int):
        modules = dict(actor.actor_module.named_modules())
        models = [m for m in modules.values() if type(m).__name__ == "OuroModel"]
        heads = [m for name, m in modules.items() if name.split(".")[-1] == "lm_head"]
        if len(models) != 1 or len(heads) != 1:
            raise ValueError(f"expected one OuroModel and one lm_head, found {len(models)} and "
                             f"{len(heads)}")
        if type(heads[0]).__name__ == "FullyShardedDataParallel":
            raise ValueError("lm_head has its own FSDP unit; its calls would be collectives")
        self.model, self.lm_head = models[0], heads[0]
        self.rank = rank
        self.directory = Path(config["dir"])
        self.every_n_steps = int(config["every_n_steps"])
        # first training step of this run: 1, or checkpoint step + 1 when resuming
        self.first_step = int(config.get("first_step", 1))
        self.top_k = int(config["top_k"])
        self.save_tokens = bool(config["save_tokens"])
        self.chunk_tokens = int(config["chunk_tokens"])
        self.loop_weights = bool(config.get("loop_weights", False))
        settings = getattr(actor, "settings", None)
        self.plc_settings = settings if isinstance(settings, plc.PLCSettings) else None
        self._calls = 0
        self._active = False
        self._micro_batch = None
        self._temperature = 1.0
        self._records: List[dict] = []
        self._error: Optional[str] = None

        self._compute_log_prob = actor.compute_log_prob
        actor.compute_log_prob = self.compute_log_prob
        # verl / vendor RLTT log-prob passes use the first two; PLCActor.compute_log_prob uses
        # _forward_micro_batch_plc (padded per-loop forward)
        for name in ("_forward_micro_batch", "_forward_micro_batch_rltt",
                     "_forward_micro_batch_plc"):
            if hasattr(actor, name):
                setattr(actor, name, self._remember_micro_batch(getattr(actor, name),
                                                                name == "_forward_micro_batch_plc"))

    def _remember_micro_batch(self, forward, returns_plc_tensors: bool = False):
        def wrapped(micro_batch, *args, **kwargs):
            if not self._active:
                return forward(micro_batch, *args, **kwargs)
            self._micro_batch = micro_batch
            records = len(self._records)
            result = forward(micro_batch, *args, **kwargs)
            if (returns_plc_tensors and self.loop_weights and self.plc_settings is not None
                    and self._error is None and len(self._records) == records + 1):
                try:
                    self._add_plc_weights(self._records[-1], micro_batch, result[1])
                except Exception as exc:  # the audit must not stop training
                    self._error = repr(exc)
            return result
        return wrapped

    def compute_log_prob(self, data, *args, **kwargs):
        self._calls += 1
        step = int(data.meta_info.get("global_steps", self.first_step + self._calls - 1))
        if (step - 1) % self.every_n_steps:
            return self._compute_log_prob(data, *args, **kwargs)
        self._temperature = float(data.meta_info["temperature"])
        self._records, self._error = [], None
        self._active = True
        handle = self.model.register_forward_hook(self._on_forward)
        try:
            result = self._compute_log_prob(data, *args, **kwargs)
        finally:
            handle.remove()
            self._active = False
            self._micro_batch = None
        self._write(step)
        return result

    def _on_forward(self, module, inputs, output):
        micro_batch, self._micro_batch = self._micro_batch, None
        if micro_batch is None or self._error is not None:
            return
        try:
            self._records.append(self._measure(micro_batch, output[1], output[2]))
        except Exception as exc:  # the audit must not stop training
            self._error = repr(exc)

    def _measure(self, micro_batch: dict, hiddens: Sequence[torch.Tensor],
                 gates: Sequence[torch.Tensor]) -> dict:
        input_ids = micro_batch["input_ids"]
        attention_mask = micro_batch["attention_mask"]
        responses = micro_batch["responses"]
        response_length = responses.size(-1)
        width = hiddens[0].shape[-1]
        rows, tokens, positions = response_positions(attention_mask, response_length,
                                                     hiddens[0].numel() // width)
        labels = responses[rows, tokens]
        stats = distribution_stats([h.reshape(-1, width)[positions] for h in hiddens],
                                   self.lm_head, labels, self._temperature, self.top_k,
                                   self.chunk_tokens)
        id_type = np.min_scalar_type(self.lm_head.out_features - 1)
        record = {name: value.numpy() for name, value in stats.items()}
        record["topk_ids"] = record["topk_ids"].astype(id_type)
        record["sampled_token"] = labels.cpu().numpy().astype(id_type)
        record["response_tokens"] = attention_mask[:, -response_length:].sum(-1).cpu().numpy()
        record["prompt_tokens"] = attention_mask[:, :-response_length].sum(-1).cpu().numpy()
        ids, mask = input_ids.cpu().numpy(), attention_mask.cpu().numpy()
        record["row_keys"] = np.array([row_key(i, m) for i, m in zip(ids, mask)])
        if self.loop_weights:
            if len(gates) != len(hiddens):
                raise ValueError(f"{len(gates)} exit gates for {len(hiddens)} loops")
            with torch.no_grad():
                pdf = exit_pdf_from_gates([g.reshape(-1, 1)[positions] for g in gates])
            record["exit_pdf"] = pdf.cpu().numpy()
        return record

    @torch.no_grad()
    def _add_plc_weights(self, record: dict, micro_batch: dict, ell: torch.Tensor) -> None:
        """``record["plc_weight"]``: the weights ``PLCActor.update_policy`` forms from ``ell``
        ``[B, L, T]`` (the old-policy per-loop log-probs of this pass)."""
        settings = self.plc_settings
        response_length = micro_batch["responses"].size(-1)
        rows, tokens = micro_batch["attention_mask"][:, -response_length:].bool().nonzero(
            as_tuple=True)
        weights = plc.compute_loop_weights(ell.float(), settings.weight_mode, settings.trunc_cap)
        record["plc_weight"] = weights[rows, tokens].cpu().numpy()

    def _write(self, step: int) -> None:
        step_dir = self.directory / step_dir_name(step)
        if (step_dir / "summary.json").exists():
            # a finished step of an earlier run in this output_dir: keep it and leave a note (the
            # run was restarted without resuming, or its first_step is wrong)
            (step_dir / f"skipped_rank{self.rank:02d}.txt").write_text(
                f"step {step} already finalized; this run's records were not written\n")
            self._records = []
            return
        step_dir.mkdir(parents=True, exist_ok=True)
        try:
            if self._error is not None:
                raise RuntimeError(self._error)
            names = TOKEN_ARRAYS + ("response_tokens", "prompt_tokens", "row_keys")
            names += tuple(name for name in WEIGHT_ARRAYS
                           if self._records and all(name in r for r in self._records))
            arrays = {name: np.concatenate([r[name] for r in self._records]) for name in names}
            arrays["offsets"] = np.concatenate([[0], np.cumsum(arrays["response_tokens"])])
            arrays["temperature"] = np.array(self._temperature)
            save = np.savez_compressed if self.save_tokens else np.savez
            partial = step_dir / f"partial_rank{self.rank:02d}.npz"
            save(partial, **arrays)
            partial.rename(step_dir / f"tokens_rank{self.rank:02d}.npz")
        except Exception as exc:  # the audit must not stop training
            (step_dir / f"error_rank{self.rank:02d}.txt").write_text(repr(exc))
        self._records = []


def attach_loop_audit(actor, config, rank: int) -> Optional[LoopAudit]:
    """Install the audit on ``actor``; ``config`` is ``actor_rollout_ref.loop_audit`` or None."""
    return LoopAudit(actor, config, rank) if config else None


def token_metrics(arrays: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """Per-token metric columns, one array of n values each; loops are numbered from 1."""
    entropy, kl = arrays["entropy"], arrays["kl"]
    ids, probs = arrays["topk_ids"], arrays["topk_probs"]
    columns = {}
    loops = entropy.shape[1]
    steps = np.arange(1, loops + 1, dtype=np.float32)
    for name in WEIGHT_ARRAYS:
        if name in arrays:
            weights = arrays[name]
            for t in range(loops):
                columns[f"{name}/loop_{t + 1}"] = weights[:, t]
            columns[f"{name}/expected_loop"] = weights @ steps
    log_prob = arrays["sampled_log_prob"]
    for t in range(loops - 1):
        # cross-loop ratio rho = P^(t)(y) / P^(T)(y) of the token the last loop sampled: how far
        # the rollouts are off-policy for loop t (every method, so runs compare directly)
        log_rho = log_prob[:, t] - log_prob[:, -1]
        columns[f"xloop_log_rho/loop_{t + 1}"] = log_rho
        columns[f"xloop_rho_min1/loop_{t + 1}"] = np.exp(np.minimum(log_rho, 0.0))
        columns[f"xloop_rho_gt_1/loop_{t + 1}"] = (log_rho > 0).astype(np.float32)
        columns[f"xloop_rho_lt_0.1/loop_{t + 1}"] = (log_rho < np.log(0.1)).astype(np.float32)
    for t in range(loops):
        columns[f"entropy/loop_{t + 1}"] = entropy[:, t]
        columns[f"sampled_log_prob/loop_{t + 1}"] = log_prob[:, t]
        columns[f"top1_prob/loop_{t + 1}"] = probs[:, t, 0]
        columns[f"topk_mass/loop_{t + 1}"] = probs[:, t].sum(-1)
        for s in range(loops):
            if s != t:
                columns[f"kl/{t + 1}_{s + 1}"] = kl[:, t, s]
            if s > t:
                pair = f"{t + 1}_{s + 1}"
                columns[f"top1_agree/{pair}"] = (ids[:, t, 0] == ids[:, s, 0]).astype(np.float32)
                shared = (ids[:, t, :, None] == ids[:, s, None, :]).any(-1)
                columns[f"topk_overlap/{pair}"] = shared.mean(-1, dtype=np.float32)
    return columns


def statistics(columns: Dict[str, np.ndarray]) -> Dict[str, Dict[str, float]]:
    out = {}
    for name, values in columns.items():
        if values.size:
            p5, p95 = np.percentile(values, [5, 95])
            out[name] = {"mean": float(values.mean(dtype=np.float64)),
                         "p5": float(p5), "p95": float(p95)}
    return out


def batch_rows(batch, correct_threshold: float = 0.5) -> Dict[str, List[dict]]:
    """Rows of a verl ``DataProto`` by ``row_key``; equal keys mean identical rollouts."""
    tensors = batch.batch
    input_ids = tensors["input_ids"].cpu().numpy()
    attention_mask = tensors["attention_mask"].cpu().numpy()
    scores = tensors["token_level_scores"].sum(-1).float().cpu().numpy()
    old_log_probs = tensors["old_log_probs"].float().cpu().numpy()
    capacity = tensors["responses"].shape[-1]
    non_tensor = batch.non_tensor_batch
    uids = non_tensor["uid"] if "uid" in non_tensor else [None] * len(scores)
    rows = defaultdict(list)
    for ids, mask, score, log_probs, uid in zip(input_ids, attention_mask, scores, old_log_probs,
                                                uids):
        rows[row_key(ids, mask)].append({
            "uid": None if uid is None else str(uid),
            "reward": float(score),
            "correct": bool(score >= correct_threshold),
            "truncated": bool(mask[-capacity:].sum() >= capacity),
            "old_log_probs": log_probs,
        })
    return rows


def finalize_step(step_dir: Path, rows: Dict[str, List[dict]], save_tokens: bool = True) -> dict:
    """Write ``responses.jsonl`` and ``summary.json`` from the rank shards; return wandb metrics."""
    shards = sorted(step_dir.glob("tokens_rank*.npz"))
    pooled = defaultdict(list)
    gaps = []
    responses = unmatched = tokens = 0
    temperature = None
    with open(step_dir / "responses.partial", "w") as fh:
        for shard in shards:
            with np.load(shard) as data:
                arrays = {name: data[name] for name in data.files}
            temperature = float(arrays["temperature"])
            columns = token_metrics(arrays)
            for name, values in columns.items():
                pooled[name].append(values)
            offsets = arrays["offsets"]
            for index, key in enumerate(arrays["row_keys"].tolist()):
                start, stop = int(offsets[index]), int(offsets[index + 1])
                row = rows[key].pop() if rows.get(key) else None
                line = {"rank": int(shard.stem[-2:]), "index": index,
                        "prompt_tokens": int(arrays["prompt_tokens"][index]),
                        "response_tokens": stop - start, "uid": None, "reward": None,
                        "correct": None, "truncated": None}
                if row is None:
                    unmatched += 1
                else:
                    old_log_probs = row.pop("old_log_probs")
                    line.update(row)
                    gaps.append(np.abs(arrays["sampled_log_prob"][start:stop, -1]
                                       - old_log_probs[:stop - start]))
                line["stats"] = statistics({n: v[start:stop] for n, v in columns.items()})
                fh.write(json.dumps(line) + "\n")
                responses += 1
                tokens += stop - start
    (step_dir / "responses.partial").rename(step_dir / "responses.jsonl")

    stats = statistics({name: np.concatenate(parts) for name, parts in pooled.items()})
    errors = sorted(path.name for path in step_dir.glob("error_rank*.txt"))
    gap = float(np.concatenate(gaps).mean()) if gaps else None
    summary = {"responses": responses, "tokens": tokens, "ranks": len(shards),
               "temperature": temperature, "unmatched_responses": unmatched,
               "last_loop_log_prob_gap": gap, "errors": errors, "metrics": stats}
    (step_dir / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    if not save_tokens:
        for shard in shards:
            shard.unlink()

    metrics = {f"{PREFIX}{name}/{stat}": value for name, values in stats.items()
               for stat, value in values.items()}
    metrics.update({PREFIX + "responses": float(responses), PREFIX + "tokens": float(tokens),
                    PREFIX + "unmatched_responses": float(unmatched),
                    PREFIX + "error_ranks": float(len(errors))})
    if gap is not None:
        metrics[PREFIX + "last_loop_log_prob_gap"] = gap
    return metrics


def finalize_pending(audit_dir, batch, correct_threshold: float = 0.5,
                     save_tokens: bool = True) -> dict:
    """Finalize steps the workers have written; called by the driver once per training step."""
    metrics = {}
    for step_dir in sorted(Path(audit_dir).glob("step_*")):
        if (step_dir / "summary.json").exists():
            continue
        try:
            metrics = finalize_step(step_dir, batch_rows(batch, correct_threshold), save_tokens)
        except Exception as exc:  # the audit must not stop training
            (step_dir / "summary.json").write_text(json.dumps({"error": repr(exc)}) + "\n")
            metrics = {PREFIX + "error": 1.0}
    return metrics


def read_response(step_dir, rank: int, index: int, tokenizer=None) -> Dict[str, object]:
    """Per-token arrays of one response; ``rank`` and ``index`` as in ``responses.jsonl``.

    With a tokenizer the result also has ``topk_tokens``: for each token and loop, the top-k
    tokens as strings.
    """
    with np.load(Path(step_dir) / f"tokens_rank{rank:02d}.npz") as data:
        start, stop = data["offsets"][index], data["offsets"][index + 1]
        out = {name: data[name][start:stop] for name in TOKEN_ARRAYS + WEIGHT_ARRAYS
               if name in data.files}
    if tokenizer is not None:
        out["topk_tokens"] = [[tokenizer.convert_ids_to_tokens(ids.tolist()) for ids in loops]
                              for loops in out["topk_ids"]]
    return out
