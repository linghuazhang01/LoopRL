"""Names and the token rule of the `decode_last` rollout KV layout (no vLLM import).

The vLLM side is ``rltt_baseline.vllm_decode_last``; ``rltt_baseline.config`` uses the names to
build the verl settings for ``rollout.kv_layout=decode_last``.
"""

import numpy as np

ARCHITECTURE = "OuroDecodeLastKVForCausalLM"
LIB = "rltt_baseline.vllm_decode_last"
# the model reads attention metadata in Python: no torch.compile; cudagraphs for decode batches
COMPILATION = {"mode": 0, "cudagraph_mode": "FULL_DECODE_ONLY"}


def generated_mask(positions: np.ndarray, num_scheduled: np.ndarray,
                   num_prompt_tokens: np.ndarray) -> np.ndarray:
    """Per scheduled token: True when it is a generated token (position >= its prompt length).

    ``positions`` lists the scheduled tokens request by request in input-batch order;
    ``num_scheduled`` and ``num_prompt_tokens`` are per request in the same order.
    """
    owner = np.repeat(np.arange(len(num_scheduled)), num_scheduled)
    if owner.shape[0] != positions.shape[0]:
        raise ValueError("decode_last: scheduled token counts do not match the positions")
    return positions >= num_prompt_tokens[owner]
