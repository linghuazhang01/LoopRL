"""RLTT baseline (Williams & Tureci 2026, arXiv 2602.10520) on top of the vendored source.

The RLTT source (``code/vendor/rltt_source`` or ``third_party/RLTT``) stays unmodified.
This package adds what that source lacks or what this project needs: the per-loop Ouro
model (``ouro_rltt/``, ``prepare_model``),
YAML configuration with command-line overrides (``config``, ``launch``), reasoning metrics
(``metrics``) and a worker that logs per-loop actor metrics (``workers``).
"""
