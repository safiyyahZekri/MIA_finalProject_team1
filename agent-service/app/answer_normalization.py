"""Presentation normalization shared with the offline answer replay."""

import re


def normalize_value(value):
    """Retain units, signs, precision, scale and range punctuation."""
    if not isinstance(value, str):
        return value
    value = re.sub(r"\u2212(?=\d)", "-", value)
    value = " ".join(value.split())
    return re.sub(r"(?<=\d)\s+%", "%", value)


def normalize_answer(answer: dict) -> dict:
    result = {**answer, "params": dict(answer.get("params", {}))}
    params = result["params"]
    if "value" in params:
        params["value"] = normalize_value(params["value"])
    if "values" in params:
        params["values"] = [normalize_value(v) for v in params["values"]]
    return result
