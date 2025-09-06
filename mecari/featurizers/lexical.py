import hashlib
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


# -------- Basic data structures --------
@dataclass
class Morpheme:
    surf: str  # surface
    lemma: str  # lemma (base form)
    pos: str  # POS (coarse)
    pos1: str = "*"  # POS (fine)
    ctype: str = "*"  # conjugation type
    cform: str = "*"  # conjugation form
    reading: str = "*"  # reading (if any)


# -------- Utilities --------
def _stable_hash(s: str, dim: int) -> int:
    # md5 stable hash -> lower 8 bytes -> modulo by dim
    d = hashlib.md5(s.encode("utf-8")).digest()
    return int.from_bytes(d[:8], "little") % dim


def _charclass(ch: str) -> str:
    # Simple character classes (for boundary features)
    if not ch:
        return "O"
    try:
        o = ord(ch)
    except Exception:
        return "O"
    if 0x3040 <= o <= 0x309F:
        return "H"  # hiragana
    if 0x30A0 <= o <= 0x30FF:
        return "K"  # katakana
    if 0x4E00 <= o <= 0x9FFF or 0x3400 <= o <= 0x4DBF:
        return "C"  # kanji
    if 0x0030 <= o <= 0x0039 or 0xFF10 <= o <= 0xFF19:
        return "D"  # digits
    if 0x0041 <= o <= 0x007A or 0xFF21 <= o <= 0xFF5A:
        return "A"  # letters
    if ch.isspace():
        return "S"
    return "O"  # other


def _affix(s: str, n: int) -> str:
    return s[:n] if len(s) >= n else s


def _suffix(s: str, n: int) -> str:
    return s[-n:] if len(s) >= n else s


# -------- Lexical n-gram featurizer --------
class LexicalNGramFeaturizer:
    """Build unigram + boundary features as (index, value) pairs."""

    def __init__(self, dim: int = 1_000_000, add_bias: bool = True):
        self.dim = dim
        self.add_bias = add_bias

    def _push(self, feats: List[Tuple[int, float]], key: str, val: float = 1.0):
        feats.append((_stable_hash(key, self.dim), val))

    def unigram_feats(self, m: Morpheme, prev_char: Optional[str], next_char: Optional[str]) -> List[Tuple[int, float]]:
        f: List[Tuple[int, float]] = []
        # POS
        self._push(f, f"U:POS={m.pos}")
        self._push(f, f"U:POS1={m.pos}:{m.pos1}")
        # Lexicalized (surface/lemma) + POS
        self._push(f, f"U:LEM={m.lemma}")
        self._push(f, f"U:SURF={m.surf}")
        self._push(f, f"U:LEM+POS={m.lemma}|{m.pos}")
        self._push(f, f"U:SURF+POS1={m.surf}|{m.pos}:{m.pos1}")
        # Conjugation
        self._push(f, f"U:CFORM={m.ctype}:{m.cform}")
        # Reading (coarse)
        if m.reading and m.reading != "*":
            self._push(f, f"U:READ={m.reading}")
        # Prefix/Suffix (string n-grams)
        self._push(f, f"U:PREF2={_affix(m.surf, 2)}")
        self._push(f, f"U:SUF2={_suffix(m.surf, 2)}")
        # Boundary char types (1 char left/right)
        if prev_char:
            self._push(f, f"U:BTYPE_L={_charclass(prev_char)}->{_charclass(m.surf[:1])}")
        if next_char:
            self._push(f, f"U:BTYPE_R={_charclass(m.surf[-1:])}->{_charclass(next_char)}")
        if self.add_bias:
            self._push(f, "U:BIAS")
        return f

    def featurize_sequence(
        self, morphs: List[Morpheme], raw_sentence: Optional[str] = None
    ) -> List[Dict[str, List[Tuple[int, float]]]]:
        if raw_sentence is None:
            raw_sentence = "".join(m.surf for m in morphs)
        spans = []
        cur = 0
        for m in morphs:
            st, ed = cur, cur + len(m.surf)
            spans.append((st, ed))
            cur = ed

        for i, m in enumerate(morphs):
            st, ed = spans[i]
            prev_char = raw_sentence[st - 1] if st > 0 else None
            next_char = raw_sentence[ed] if ed < len(raw_sentence) else None
            feats_node = self.unigram_feats(m, prev_char, next_char)

        return feats_node


if __name__ == "__main__":
    pass
