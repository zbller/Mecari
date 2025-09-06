#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Build training graphs from KWDLC with JUMANDIC.

Pipeline:
1) Read gold morphemes from KNP files
2) Parse text with MeCab (JUMANDIC) to get candidate morphemes
3) Match candidates to gold and assign annotations ('+', '-', '?')
4) Save graph data as .pt
"""

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import torch
import yaml
from tqdm import tqdm

from mecari.analyzers.mecab import MeCabAnalyzer
from mecari.data.data_module import DataModule
from mecari.featurizers.lexical import LexicalNGramFeaturizer as LexicalFeaturizer
from mecari.featurizers.lexical import Morpheme
from mecari.utils.morph_utils import build_adjacent_edges, dedup_morphemes, normalize_mecab_candidates


def add_lexical_features(morphemes: List[Dict], text: str, feature_dim: int = 100000) -> List[Dict]:
    """Add lexical (index, value) pairs to morphemes. Not used when saving JSON.

    Kept for backward-compatibility and test equivalence.
    """
    featurizer = LexicalFeaturizer(dim=feature_dim, add_bias=True)
    for m in morphemes:
        surf = m.get("surface", "")
        morph_obj = Morpheme(
            surf=surf,
            lemma=m.get("base_form", surf),
            pos=m.get("pos", "*"),
            pos1=m.get("pos_detail1", "*"),
            ctype="*",
            cform="*",
            reading=m.get("reading", "*"),
        )
        st = m.get("start_pos", 0)
        ed = m.get("end_pos", st + len(surf))
        prev_char = text[st - 1] if st > 0 and st <= len(text) else None
        next_char = text[ed] if ed < len(text) else None
        feats = featurizer.unigram_feats(morph_obj, prev_char, next_char)
        m["lexical_features"] = feats
    return morphemes


def hiragana_to_katakana(text: str) -> str:
    """Convert hiragana to katakana."""
    return "".join([chr(ord(c) + 96) if "ぁ" <= c <= "ん" else c for c in text])


def _load_gold_with_kyoto(knp_path: Path) -> List[Dict]:
    """Load sentences and morphemes from a KNP file using kyoto-reader (required)."""
    try:
        from kyoto_reader import KyotoReader  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError("kyoto-reader is required for gold loading. Install it (pip install kyoto-reader).") from e

    try:
        try:
            reader = KyotoReader(str(knp_path), n_jobs=0)
        except TypeError:
            reader = KyotoReader(str(knp_path))
        sents: List[Dict] = []
        for doc in reader.process_all_documents(n_jobs=0):
            if doc is None:
                continue
            for sent in doc.sentences:
                text = sent.surf
                morphemes: List[Dict] = []
                pos = 0
                for mrph in sent.mrph_list():
                    surf = getattr(mrph, "midasi", "") or ""
                    read = getattr(mrph, "yomi", surf) or surf
                    lemma = getattr(mrph, "genkei", surf) or surf
                    pos_main = getattr(mrph, "hinsi", "*") or "*"
                    pos1 = getattr(mrph, "bunrui", "*") or "*"
                    st = pos
                    ed = st + len(surf)
                    pos = ed
                    morphemes.append(
                        {
                            "surface": surf,
                            "reading": read,
                            "base_form": lemma,
                            "pos": pos_main,
                            "pos_detail1": pos1,
                            "pos_detail2": "*",
                            "pos_detail3": "*",
                            "start_pos": st,
                            "end_pos": ed,
                        }
                    )
                sents.append({"text": text, "morphemes": morphemes})
        return sents
    except Exception as e:
        raise RuntimeError(f"Failed to parse KNP with kyoto-reader: {knp_path}") from e


def match_morphemes_with_gold(candidates: List[Dict], gold_morphemes: List[Dict], text: str) -> List[Dict]:
    """Match candidate morphemes to gold and assign annotations ('?', '+', '-').

    Policy:
      - Initialize every candidate as '?'
      - Mark '+' for candidates that strictly match gold (surface, POS, base, reading)
      - Mark '-' for candidates that overlap any '+' span
    """
    # Reconstruct gold spans in character offsets
    gold_details = []
    cur = 0
    for g in gold_morphemes:
        surf = g.get("surface", "")
        st, ed = cur, cur + len(surf)
        cur = ed
        gold_details.append(
            {
                "start_pos": st,
                "end_pos": ed,
                "surface": surf,
                "pos": g.get("pos", "*"),
                "pos_detail1": g.get("pos_detail1", "*"),
                "pos_detail2": g.get("pos_detail2", "*"),
                "base_form": g.get("base_form", ""),
                "reading": hiragana_to_katakana(g.get("reading", "")),
            }
        )

    # Initialize all candidates with '?'
    annotated: List[Dict] = []
    for cand in candidates:
        a = {**cand}
        a["annotation"] = "?"
        if "inflection_type" not in a:
            a["inflection_type"] = "*"
        if "inflection_form" not in a:
            a["inflection_form"] = "*"
        annotated.append(a)

    # Match by strict equality first; allow reading mismatch as fallback
    span_to_cands: dict[tuple[int, int], list[Dict]] = {}
    for a in annotated:
        cs = a.get("start_pos", 0)
        ce = a.get("end_pos", cs + len(a.get("surface", "")))
        span_to_cands.setdefault((cs, ce), []).append(a)

    matched_spans: List[tuple[int, int]] = []
    for g in gold_details:
        span = (g["start_pos"], g["end_pos"])
        cands = span_to_cands.get(span, [])
        if not cands:
            continue
        strict = []
        fallback = []
        for a in cands:
            if a.get("surface", "") != g["surface"]:
                continue
            if a.get("pos", "*") != g["pos"]:
                continue
            if a.get("pos_detail1", "*") != g.get("pos_detail1", "*"):
                continue
            if a.get("base_form", "") != g["base_form"]:
                continue
            if hiragana_to_katakana(a.get("reading", "")) == g["reading"]:
                strict.append(a)
            else:
                fallback.append(a)
        chosen_list = strict if strict else fallback
        if chosen_list:
            for a in chosen_list:
                a["annotation"] = "+"
            matched_spans.append(span)
            for a in cands:
                if (a not in chosen_list) and a.get("annotation") != "+":
                    a["annotation"] = "-"

    # Demote any morpheme that overlaps (by at least 1 char) with any '+' span.
    plus_spans = []
    for a in annotated:
        if a.get("annotation") == "+":
            cs = a.get("start_pos", 0)
            ce = a.get("end_pos", cs + len(a.get("surface", "")))
            plus_spans.append((cs, ce))

    def _strict_overlap(st1: int, ed1: int, st2: int, ed2: int) -> bool:
        # overlap only if intersection length > 0 (touching is not overlap)
        return max(st1, st2) < min(ed1, ed2)

    for a in annotated:
        if a.get("annotation") == "+":
            continue
        cs = a.get("start_pos", 0)
        ce = a.get("end_pos", cs + len(a.get("surface", "")))
        for ms, me in plus_spans:
            if _strict_overlap(cs, ce, ms, me):
                a["annotation"] = "-"
                break
    return annotated


def main():
    parser = argparse.ArgumentParser(description="Create training data from KWDLC (JUMANDIC)")
    parser.add_argument("--input-dir", type=str, default="KWDLC/knp", help="Directory containing KNP files")
    parser.add_argument("--config", type=str, default="configs/gat.yaml", help="Path to config file")
    parser.add_argument("--limit", type=int, help="Max number of files to process")
    parser.add_argument("--test-only", action="store_true", help="Process only test split IDs")
    parser.add_argument("--jumandic-path", type=str, default="/var/lib/mecab/dic/juman-utf8", help="Path to JUMANDIC")
    args = parser.parse_args()

    config = {}
    if args.config and Path(args.config).exists():
        with open(args.config, "r") as f:
            config = yaml.safe_load(f)

        if "extends" in config:
            parent_config_path = Path(args.config).parent / config["extends"]
            if parent_config_path.exists():
                with open(parent_config_path, "r") as f:
                    parent_config = yaml.safe_load(f)

                def deep_merge(base, override):
                    for key, value in override.items():
                        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                            deep_merge(base[key], value)
                        else:
                            base[key] = value
                    return base

                config = deep_merge(parent_config, config)

    features_config = config.get("features", {})
    feature_dim = features_config.get("lexical_feature_dim", 100000)
    training_config = config.get("training", {})

    if training_config.get("annotations_dir"):
        output_dir = Path(training_config.get("annotations_dir"))
    else:
        output_dir = Path("annotations_kwdlc_juman")
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Lexical features: using {feature_dim} dims")
    print(f"Output directory: {output_dir}")

    analyzer = MeCabAnalyzer(
        jumandic_path=args.jumandic_path,
    )

    knp_files = []

    if args.test_only:
        test_id_file = Path("KWDLC/id/split_for_pas/test.id")
        if test_id_file.exists():
            with open(test_id_file, "r") as f:
                test_ids = [line.strip() for line in f if line.strip()]

            knp_base_dir = Path(args.input_dir)
            for file_id in test_ids:
                dir_name = file_id[:13]
                file_name = f"{file_id}.knp"
                knp_path = knp_base_dir / dir_name / file_name
                if knp_path.exists():
                    knp_files.append(knp_path)
    else:
        knp_dir = Path(args.input_dir)
        knp_files = sorted(knp_dir.glob("**/*.knp"))

    if args.limit:
        knp_files = knp_files[: args.limit]

    print(f"Files to process: {len(knp_files)}")
    print(f"JUMANDIC: {args.jumandic_path}")
    print(f"Output to: {output_dir}")

    total_stats = defaultdict(int)
    annotation_idx = 0

    dm = DataModule(
        annotations_dir=str(output_dir),
        lexical_feature_dim=int(feature_dim),
        use_bidirectional_edges=bool(config.get("edge_features", {}).get("use_bidirectional_edges", True)),
    )

    # Save .pt files directly under the output_dir

    for knp_path in tqdm(knp_files, desc="processing"):
        try:
            sentences = _load_gold_with_kyoto(knp_path)
            if not sentences:
                continue

            doc_id = knp_path.stem
            for s in sentences:
                s["source_id"] = doc_id

            for sent_idx, sentence in enumerate(sentences):
                text = sentence["text"]
                gold_morphemes = sentence["morphemes"]
                source_id = sentence.get("source_id", doc_id)

                candidates = analyzer.get_morpheme_candidates(text)
                candidates = normalize_mecab_candidates(candidates)
                candidates = dedup_morphemes(candidates)
                if not candidates:
                    continue

                annotated_morphemes = match_morphemes_with_gold(candidates, gold_morphemes, text)

                edges = build_adjacent_edges(annotated_morphemes)

                for m in annotated_morphemes:
                    if "lexical_features" in m:
                        m.pop("lexical_features", None)

                morphemes_with_feats = dm.compute_lexical_features(annotated_morphemes, text)
                graph = dm.create_graph_from_morphemes_data(
                    morphemes=morphemes_with_feats,
                    edges=edges,
                    text=text,
                    for_training=True,
                )
                if graph is None:
                    continue

                graph_file = output_dir / f"graph_{annotation_idx:04d}.pt"
                payload = {
                    "graph": graph,
                    "source_id": source_id,
                    "text": text,
                }
                torch.save(payload, graph_file)

                total_stats["sentences"] += 1
                total_stats["morphemes"] += len(annotated_morphemes)
                total_stats["positive"] += sum(1 for m in annotated_morphemes if m.get("annotation") == "+")
                total_stats["negative"] += sum(1 for m in annotated_morphemes if m.get("annotation") == "-")

                annotation_idx += 1

            total_stats["files"] += 1

        except Exception as e:
            print(f"Error ({knp_path}): {e}")
            total_stats["errors"] += 1

    print("\n" + "=" * 50)
    print("Processing complete")
    print("=" * 50)
    print(f"Files: {total_stats['files']}")
    print(f"Sentences: {total_stats['sentences']}")
    print(f"Morphemes: {total_stats['morphemes']}")
    print(f"Positive (+): {total_stats['positive']}")
    print(f"Negative (-): {total_stats['negative']}")
    #
    if total_stats["errors"] > 0:
        print(f"Errors: {total_stats['errors']}")


if __name__ == "__main__":
    main()
