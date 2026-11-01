"""Match free-form math answers: plain-text form first, then math-verify.

Two answers match when their plain-text forms are equal (this covers Yes/No, option letters and
answers math-verify cannot parse, such as ``\\aleph_0``), or when math-verify finds them equivalent
(fractions, radicals, intervals, sets, expressions). Needs ``math-verify``.
"""

import logging
import re
import warnings

GRADER_VERSION = "last-boxed-plain-or-math-verify-natural-eos-v1"
_WRAPPERS = re.compile(r"\\(?:text|textbf|mathrm|mbox|operatorname)\s*\{([^{}]*)\}")


def plain(answer: str) -> str:
    """Lower-case text form: unwrap \\text{}, drop $, spaces, a final period and outer brackets."""
    text = _WRAPPERS.sub(r"\1", answer)
    text = text.replace("$", "").replace("\\,", "").replace("\\!", "").replace("\\ ", "")
    text = re.sub(r"\s+", "", text).rstrip(".").lower()
    while len(text) > 2 and text[0] + text[-1] in ("()", "[]") and "," not in text:
        text = text[1:-1]
    return text


def _math_verify_match(gold: str, predicted: str) -> bool:
    from math_verify import parse, verify

    logging.getLogger("math_verify").setLevel(logging.ERROR)
    logging.getLogger("latex2sympy2_extended").setLevel(logging.ERROR)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gold_parsed = parse(f"${gold}$")
        predicted_parsed = parse(f"${predicted}$")
        return bool(gold_parsed) and bool(predicted_parsed) and verify(
            gold_parsed, predicted_parsed)


def answers_match(gold: str, predicted: str) -> bool:
    if plain(gold) == plain(predicted):
        return True
    try:
        return _math_verify_match(gold, predicted)
    except Exception:  # math-verify raises on some malformed LaTeX; treat as no match
        return False
