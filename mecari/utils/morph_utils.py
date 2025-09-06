#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import Dict, List, Any, Optional

from mecari.utils.signature import signature_key


def dedup_morphemes(morphs: List[Dict]) -> List[Dict]:
    seen = set()
    out: List[Dict] = []
    for m in morphs:
        key = signature_key(m)
        if key in seen:
            continue
        seen.add(key)
        out.append(m)
    out.sort(key=lambda m: (
        m.get("start_pos", 0),
        -(m.get("end_pos", 0) - m.get("start_pos", 0)),
        m.get("surface", ""),
        m.get("reading", ""),
        m.get("pos", "*"),
    ))
    return out


def build_adjacent_edges(morphs: List[Dict]) -> List[Dict]:
    edges: List[Dict] = []
    for i, s in enumerate(morphs):
        for j, t in enumerate(morphs):
            if i >= j:
                continue
            if s.get("end_pos", 0) == t.get("start_pos", 0):
                edges.append({"source_idx": i, "target_idx": j, "edge_type": "forward"})
    return edges


def normalize_mecab_candidates(candidates: List[Dict]) -> List[Dict]:
    """Normalize MeCab candidates consistently for preprocessing/inference.

    - If surface is digit-only and base_form is empty/missing, set base_form = surface.
    Modifies candidates in place and returns the list for convenience.
    """
    for c in candidates:
        surf = c.get("surface", "")
        bf = c.get("base_form")
        if (bf is None or bf == "") and surf and surf.isdigit():
            c["base_form"] = surf
    return candidates
