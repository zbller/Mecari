#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Unified evaluation for MeCab (JUMANDIC) and the trained model.

Evaluates both systems on the same KWDLC test data and compares results.
"""

import argparse
import subprocess
from pathlib import Path
from typing import Dict, List

import torch
from tqdm import tqdm


def parse_knp_file(knp_file: Path) -> List[Dict]:
    """Extract gold morphemes from a KNP file."""
    sentences = []
    current_sentence = []
    current_text = ""

    with open(knp_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")

            if line.startswith("#"):
                if line.startswith("# S-ID:"):
                    if current_sentence:
                        sentences.append({"morphemes": current_sentence, "text": current_text})
                        current_sentence = []
                        current_text = ""
                continue
            elif line == "EOS":
                if current_sentence:
                    sentences.append({"morphemes": current_sentence, "text": current_text})
                    current_sentence = []
                    current_text = ""
            elif line.startswith("+") or line.startswith("*"):
                continue
            elif line:
                parts = line.split(" ")
                if len(parts) >= 4:
                    surface = parts[0]
                    reading = parts[1]
                    pos = parts[3]

                    current_sentence.append({"surface": surface, "reading": reading, "pos": pos})
                    current_text += surface

    return sentences


def analyze_with_mecab(text: str) -> List[Dict]:
    """Analyze text with MeCab (JUMANDIC) using a simple best-path parse."""
    try:
        result = subprocess.run(
            ["mecab", "-d", "/var/lib/mecab/dic/juman-utf8"],
            input=text,
            capture_output=True,
            text=True,
            encoding="utf-8",
        )

        if result.returncode != 0:
            return []

        morphemes = []
        for line in result.stdout.strip().split("\n"):
            if line == "EOS":
                break
            parts = line.split("\t")
            if len(parts) >= 2:
                surface = parts[0]
                features = parts[1].split(",")
                if len(features) >= 7:
                    pos = features[0]
                    # Do not fallback reading to surface when missing ('*')
                    reading = features[7] if len(features) > 7 and features[7] != "*" else ""

                    morphemes.append({"surface": surface, "reading": reading, "pos": pos})

        return morphemes
    except Exception as e:
        print(f"MeCab error: {e}")
        return []


def analyze_with_jumanpp(text: str) -> List[Dict]:
    """Analyze text with JUMAN++ (optional baseline)."""
    try:
        result = subprocess.run(["jumanpp"], input=text, capture_output=True, text=True, encoding="utf-8")

        if result.returncode != 0:
            return []

        morphemes = []
        for line in result.stdout.strip().split("\n"):
            if line.startswith("@") or line == "EOS":
                continue
            parts = line.split(" ")
            if len(parts) >= 12:
                surface = parts[0]
                reading = parts[1]
                pos = parts[3]

                morphemes.append({"surface": surface, "reading": reading, "pos": pos})

        return morphemes
    except Exception as e:
        print(f"JUMAN++ error: {e}")
        return []


def analyze_with_model(text: str, model, experiment_info) -> List[Dict]:
    """Analyze text with the trained model."""
    try:
        import infer

        results, optimal_morphemes = infer.predict_morphemes_from_text(
            text, model=model, experiment_info=experiment_info, silent=True
        )

        morphemes = []
        for morph in optimal_morphemes:
            morphemes.append(
                {"surface": morph["surface"], "reading": morph.get("reading", ""), "pos": morph.get("pos", "*")}
            )

        return morphemes
    except Exception as e:
        print(f"Model inference error: {e}")
        return []


def evaluate_morphemes(gold_morphemes: List[Dict], pred_morphemes: List[Dict]) -> Dict:
    """Compute segmentation and POS F1 between gold and predictions."""
    gold_spans = []
    pred_spans = []

    # Gold spans (from gold morphemes)
    pos = 0
    for m in gold_morphemes:
        surface = m["surface"]
        end = pos + len(surface)
        gold_spans.append((pos, end, m["pos"]))
        pos = end

    # Predicted spans (from predictions)
    pos = 0
    for m in pred_morphemes:
        surface = m["surface"]
        end = pos + len(surface)
        pred_spans.append((pos, end, m["pos"]))
        pos = end

    # Segmentation accuracy (without POS)
    gold_seg = {(s, e) for s, e, _ in gold_spans}
    pred_seg = {(s, e) for s, e, _ in pred_spans}

    seg_correct = len(gold_seg & pred_seg)
    seg_precision = seg_correct / len(pred_seg) if pred_seg else 0
    seg_recall = seg_correct / len(gold_seg) if gold_seg else 0
    seg_f1 = 2 * seg_precision * seg_recall / (seg_precision + seg_recall) if (seg_precision + seg_recall) > 0 else 0

    # Accuracy with POS
    gold_pos = set(gold_spans)
    pred_pos = set(pred_spans)

    pos_correct = len(gold_pos & pred_pos)
    pos_precision = pos_correct / len(pred_pos) if pred_pos else 0
    pos_recall = pos_correct / len(gold_pos) if gold_pos else 0
    pos_f1 = 2 * pos_precision * pos_recall / (pos_precision + pos_recall) if (pos_precision + pos_recall) > 0 else 0

    return {
        "seg_precision": seg_precision,
        "seg_recall": seg_recall,
        "seg_f1": seg_f1,
        "pos_precision": pos_precision,
        "pos_recall": pos_recall,
        "pos_f1": pos_f1,
    }


def main():
    parser = argparse.ArgumentParser(description="Unified evaluation script")
    parser.add_argument("--kwdlc-dir", type=str, default="KWDLC", help="Path to KWDLC root directory")
    parser.add_argument(
        "--test-ids", type=str, default="KWDLC/id/split_for_pas/test.id", help="File containing test IDs (one per line)"
    )
    parser.add_argument(
        "--max-samples", type=int, default=None, help="Max number of samples to evaluate (default: all)"
    )
    parser.add_argument("--experiment", "-e", type=str, required=True, help="Experiment name to evaluate")

    args = parser.parse_args()

    # Load test document IDs
    test_ids = []
    with open(args.test_ids, "r") as f:
        for line in f:
            test_ids.append(line.strip())

    if args.max_samples is not None:
        test_ids = test_ids[: args.max_samples]

    print(f"Evaluating: {len(test_ids)} files")

    # Load model
    print("\nLoading trained model...")
    import infer

    model_info = infer.load_model(experiment_name=args.experiment)
    if model_info:
        model, experiment_info = model_info
        # Force CPU execution for evaluation
        device = torch.device("cpu")
        model = model.to(device)
        experiment_info["device"] = device
        print(f"Model: {experiment_info['name']}")
    else:
        print("Failed to load model")
        model = None
        experiment_info = None

    mecab_results = []
    model_results = []

    print("\nStart evaluation...")
    for test_id in tqdm(test_ids, desc="evaluating"):
        # Find KNP file
        found = False
        knp_base = Path(args.kwdlc_dir) / "knp"

        for subdir in knp_base.glob("w*"):
            candidate = subdir / f"{test_id}.knp"
            if candidate.exists():
                knp_path = candidate
                found = True
                break

        if not found:
            continue

        # Read gold data
        gold_sentences = parse_knp_file(knp_path)

        for sent_data in gold_sentences:
            text = sent_data["text"]
            gold_morphemes = sent_data["morphemes"]

            # MeCab (JUMANDIC)
            pred_mecab = analyze_with_mecab(text)
            if pred_mecab:
                result = evaluate_morphemes(gold_morphemes, pred_mecab)
                mecab_results.append(result)

            # Trained model
            if model is not None:
                pred_model = analyze_with_model(text, model, experiment_info)
                if pred_model:
                    model_eval = evaluate_morphemes(gold_morphemes, pred_model)
                    model_results.append(model_eval)

    # Aggregate and display results
    print("\n" + "=" * 70)
    print("Evaluation Results (KWDLC test data)")
    print("=" * 70)
    print(f"Num evaluated: MeCab={len(mecab_results)}, Model={len(model_results)}")

    # MeCab (JUMANDIC)
    if mecab_results:
        avg_seg_f1 = sum(r["seg_f1"] for r in mecab_results) / len(mecab_results)
        avg_pos_f1 = sum(r["pos_f1"] for r in mecab_results) / len(mecab_results)
        print("\n[1] MeCab (JUMANDIC):")
        print(f"    Seg F1:     {avg_seg_f1:.4f}")
        print(f"    POS F1:     {avg_pos_f1:.4f}")

    # Trained model
    if model_results:
        avg_seg_f1 = sum(r["seg_f1"] for r in model_results) / len(model_results)
        avg_pos_f1 = sum(r["pos_f1"] for r in model_results) / len(model_results)
        print(f"\n[2] Trained model ({experiment_info['name']}):")
        print(f"    Seg F1:     {avg_seg_f1:.4f}")
        print(f"    POS F1:     {avg_pos_f1:.4f}")


if __name__ == "__main__":
    main()
