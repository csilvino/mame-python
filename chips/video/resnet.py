"""
Resistor network color DACs (MAME: src/emu/video/resnet.cpp)

Arcade boards turn color PROM bits into voltages through resistor ladders.
compute_resistor_weights() gives the contribution of each bit, combine_weights() sums them.
"""

from typing import List, Tuple


def compute_resistor_weights(maxval: int, scaler: float, networks) -> Tuple[List[List[float]], float]:
    """
    Port of MAME's compute_resistor_weights() (minval = 0).
    networks: list of (resistances, pulldown, pullup).
    Returns (weights per network, scale used).
    """
    raw = []
    for resistances, pulldown, pullup in networks:
        w = []
        for n in range(len(resistances)):
            r0 = 1.0 / 1e12 if pulldown == 0 else 1.0 / pulldown
            r1 = 1.0 / 1e12 if pullup == 0 else 1.0 / pullup
            for j, r in enumerate(resistances):
                if j == n:
                    r1 += 1.0 / r
                else:
                    r0 += 1.0 / r
            r0 = 1.0 / r0
            r1 = 1.0 / r1
            vout = maxval * r0 / (r1 + r0)
            w.append(min(max(vout, 0.0), maxval))
        raw.append(w)

    if scaler < 0.0:
        scale = maxval / max(sum(w) for w in raw)
    else:
        scale = scaler
    return [[scale * v for v in w] for w in raw], scale


def combine_weights(weights: List[float], *bits: int) -> int:
    return int(sum(w * b for w, b in zip(weights, bits)) + 0.5)
