"""Per-loop gradient probe for looped (Ouro) models.

Ouro applies the same decoder stack (and the same final norm and exit gate) once per loop, so
the gradient of a shared parameter is the sum of four per-loop contributions g_1 + ... + g_4.
``untie`` gives every loop its own copy of the shared parameters (forward unchanged) so each
g_t is read directly from its copy; ``losses`` defines the GRPO, final-loop NLL and Ouro
SFT-style objectives; ``stats`` turns the per-loop gradients of two half batches into norms,
cosines, cancellation ratios and the split-half noise baseline.
"""
