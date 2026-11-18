import inspect
from rltt_baseline.prepare_model import MASK_COMPAT, compat_base_source, vendor_modeling


def test_mask_signatures():
    for modern in (False, True):
        calls = []
        if modern:
            def mask(config, inputs_embeds, attention_mask, past_key_values, position_ids):
                calls.append(inputs_embeds)
                return attention_mask
        else:
            def mask(config, input_embeds, attention_mask, cache_position, past_key_values, position_ids):
                calls.append(input_embeds)
                assert cache_position == 7
                return attention_mask
        scope = dict(_inspect=inspect, create_causal_mask=mask,
                     create_sliding_window_causal_mask=mask)
        exec(MASK_COMPAT, scope)
        for name in ('create_causal_mask', 'create_sliding_window_causal_mask'):
            assert scope[name](config=None, input_embeds=3, attention_mask=5,
                               cache_position=7, past_key_values=None, position_ids=None) == 5
        assert calls == [3, 3]


def test_generated_base_compiles():
    source = compat_base_source(vendor_modeling().read_text())
    compile(source, 'modeling_ouro_base.py', 'exec')
    assert 'isinstance(cache_position, int)' in source
