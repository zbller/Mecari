#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import shutil
import subprocess
import tempfile
from typing import Dict, List

from mecari.utils.signature import signature_key


def _byte_to_char_map(text: str) -> dict[int, int]:
    mapping: dict[int, int] = {}
    cpos = 0
    bpos = 0
    for ch in text:
        mapping[bpos] = cpos
        bpos += len(ch.encode("utf-8"))
        cpos += 1
    mapping[bpos] = cpos
    return mapping


class MeCabAnalyzer:
    """Obtain morpheme candidates for building graph.

    Args:
        jumandic_path: Filesystem path to the JUMANDIC dictionary used by MeCab.
        mecab_bin: Optional MeCab binary name or full path. If None, resolves
            from the MECAB_BIN environment variable or defaults to "mecab".

    Methods:
        version(): Return the MeCab version string, or an empty string on error.
        get_morpheme_candidates(text): Analyze text and return a list of
            morpheme dicts with fields such as:
              - surface, base_form, reading
              - pos, pos_detail1/2/3
              - inflection_type, inflection_form
              - start_pos, end_pos (character offsets)
            Unknown or unavailable values are filled with "*" or empty strings.
    """

    def __init__(
        self,
        jumandic_path: str | None = None,
        mecab_bin: str | None = None,
    ) -> None:
        # Prefer JUMANDIC if present; otherwise fall back to IPADIC
        if jumandic_path is None:
            candidates = [
                "/var/lib/mecab/dic/juman-utf8",
                "/usr/lib/mecab/dic/juman-utf8",
                "/usr/lib/x86_64-linux-gnu/mecab/dic/juman-utf8",
            ]
            ipadic_candidates = [
                "/var/lib/mecab/dic/ipadic",
                "/usr/lib/x86_64-linux-gnu/mecab/dic/ipadic",
            ]
            chosen = next((p for p in candidates if os.path.isdir(p)), None)
            if chosen is None:
                chosen = next((p for p in ipadic_candidates if os.path.isdir(p)), None)
            self.jumandic_path = chosen  # may be None; handled below
        else:
            self.jumandic_path = jumandic_path

        # Resolve MeCab binary path robustly
        if mecab_bin:
            self.mecab_bin = mecab_bin
        elif os.getenv("MECAB_BIN"):
            self.mecab_bin = os.getenv("MECAB_BIN")  # type: ignore[assignment]
        else:
            resolved = shutil.which("mecab")
            if resolved:
                self.mecab_bin = resolved
            else:
                for cand in ["/usr/bin/mecab", "/usr/local/bin/mecab", "/bin/mecab"]:
                    if os.path.exists(cand):
                        self.mecab_bin = cand
                        break
                else:
                    # Fallback; will be validated at runtime
                    self.mecab_bin = "mecab"

    def version(self) -> str:
        try:
            out = subprocess.run([self.mecab_bin, "-v"], capture_output=True, text=True)
            return (out.stdout or out.stderr).strip()
        except Exception:
            return ""

    def get_morpheme_candidates(self, text: str) -> List[Dict]:
        """Return a flat list of JUMANDIC candidates (robust %H format)."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8") as f:
            f.write(text)
            temp_file = f.name
        try:
            fmt = "%pi\t%m\t%H\t%ps\t%pe\n"
            # Validate mecab binary is available
            if not (os.path.isabs(self.mecab_bin) and os.path.exists(self.mecab_bin)):
                if shutil.which(self.mecab_bin) is None:
                    raise FileNotFoundError(
                        "MeCab binary not found. Ensure packages.txt installs 'mecab' and set MECAB_BIN if needed."
                    )

            cmd = [self.mecab_bin]
            # Pass dictionary only if we have a resolvable path
            if isinstance(self.jumandic_path, str) and os.path.isdir(self.jumandic_path):
                cmd += ["-d", self.jumandic_path]
            cmd += ["-F", fmt, "-E", "", "-a", temp_file]
            result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="ignore")
            stdout = result.stdout
        finally:
            try:
                os.unlink(temp_file)
            except Exception:
                pass
        if result.returncode != 0:
            return []
        byte_to_char = _byte_to_char_map(text)
        out: list[dict] = []
        seen = set()
        for line in stdout.strip().split("\n"):
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) < 5:
                continue
            node_id, surface, features, sb, eb = parts[0], parts[1], parts[2], parts[3], parts[4]
            if surface in ("BOS", "EOS"):
                continue
            
            try:
                start_byte = int(sb)
                end_byte = int(eb)
            except ValueError:
                continue
            start_pos = byte_to_char.get(start_byte, 0)
            end_pos = byte_to_char.get(end_byte, len(text))
            fs = features.split(",")
            pos = fs[0] if len(fs) > 0 else "*"
            pos1 = fs[1] if len(fs) > 1 else "*"
            is_conj = pos in ("動詞", "形容詞", "助動詞")
            ctype = fs[2] if len(fs) > 2 and fs[2] != "*" and is_conj else "*"
            cform = fs[3] if len(fs) > 3 and fs[3] != "*" and is_conj else "*"
            pos2 = (fs[2] if len(fs) > 2 else "*") if not is_conj else "*"
            pos3 = (fs[3] if len(fs) > 3 else "*") if not is_conj else "*"
            base = fs[4] if len(fs) > 4 and fs[4] != "*" else ""
            reading = fs[5] if len(fs) > 5 and fs[5] != "*" else ""
            m = {
                "surface": surface,
                "pos": pos,
                "pos_detail1": pos1,
                "pos_detail2": pos2,
                "pos_detail3": pos3,
                "base_form": base,
                "reading": reading,
                "inflection_type": ctype,
                "inflection_form": cform,
                "start_pos": start_pos,
                "end_pos": end_pos,
                "annotation": "?",
                "node_id": node_id,
            }
            key = signature_key(m)
            if key in seen:
                continue
            seen.add(key)
            out.append(m)
        # Synthesize whitespace nodes for gaps if MeCab didn't output them
        if text:
            existing_ws_spans = set()
            for m in out:
                surf = m.get("surface", "")
                if surf.strip() == "":
                    cs = m.get("start_pos", 0)
                    ce = m.get("end_pos", 0)
                    existing_ws_spans.add((cs, ce))

            i = 0
            n = len(text)
            while i < n:
                ch = text[i]
                if ch.isspace():
                    j = i + 1
                    while j < n and text[j].isspace():
                        j += 1
                    span = (i, j)
                    if span not in existing_ws_spans:
                        ws = text[i:j]
                        wm = {
                            "surface": ws,
                            "pos": "特殊",
                            "pos_detail1": "空白",
                            "pos_detail2": "*",
                            "pos_detail3": "*",
                            "base_form": "",
                            "reading": "",
                            "inflection_type": "*",
                            "inflection_form": "*",
                            "start_pos": i,
                            "end_pos": j,
                            "annotation": "?",
                            "node_id": f"WS_{i}_{j}",
                        }
                        key = signature_key(wm)
                        if key not in seen:
                            seen.add(key)
                            out.append(wm)
                    i = j
                else:
                    i += 1
        return out
