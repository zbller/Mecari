#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import Dict, Tuple


def to_katakana(s) -> str:
    """Robust hiragana->katakana conversion for str or sequence.

    Accepts str, list, tuple; concatenates string elements when a sequence is given.
    Non-string inputs are stringified; None becomes empty string.
    """
    if isinstance(s, (list, tuple)):
        s = "".join(x for x in s if isinstance(x, str))
    elif not isinstance(s, str):
        s = str(s) if s is not None else ""
    out = []
    for ch in s:
        if not ch:
            continue
        o = ord(ch)
        if 0x3041 <= o <= 0x3096:
            out.append(chr(o + 0x60))
        else:
            out.append(ch)
    return "".join(out)


def signature_key(m: Dict) -> Tuple:
    """Stable deduplication key for a morpheme dict (POS up to pos1)."""
    surface = m.get("surface", "")
    pos = m.get("pos", "*")
    pos1 = m.get("pos_detail1", "*")
    base = m.get("base_form") or m.get("lemma") or ""
    read = to_katakana(m.get("reading") or "")
    st = m.get("start_pos", 0)
    ed = m.get("end_pos", st + len(surface))
    return (st, ed, surface, pos, pos1, base, read)
