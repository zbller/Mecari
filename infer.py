#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Show immediate feedback from the moment the command starts
print("Loading model...", flush=True)

import os
import random
from typing import Any, Dict, Optional, Tuple

# Disable WandB during inference to avoid hanging processes
os.environ["WANDB_MODE"] = "disabled"

from importlib import import_module

import numpy as np
import torch
import yaml

from mecari.analyzers.mecab import MeCabAnalyzer
from mecari.data.data_module import DataModule
from mecari.utils.morph_utils import build_adjacent_edges, dedup_morphemes, normalize_mecab_candidates


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility during inference.

    Args:
        seed: Random seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed(42)


def _find_best_checkpoint(checkpoints_dir: str, prefer_metric: str = "val_error") -> Tuple[Optional[str], float]:
    """Find the best checkpoint file in a directory.

    Args:
        checkpoints_dir: Path to the checkpoints directory.
        prefer_metric: Preferred metric ("val_error" or "val_loss").

    Returns:
        Tuple of (best checkpoint filename, score).
    """
    checkpoint_files = [f for f in os.listdir(checkpoints_dir) if f.endswith(".ckpt")]
    if not checkpoint_files:
        return None, float("inf")

    best_checkpoint = None
    best_score = float("inf")

    # Prefer filenames that include the metric keyword (e.g., val_error=..., val_error_epoch=...)
    for ckpt_file in checkpoint_files:
        if prefer_metric == "val_loss" and ("val_loss=" in ckpt_file or "val_loss_epoch=" in ckpt_file):
            try:
                if "val_loss_epoch=" in ckpt_file:
                    score_str = ckpt_file.split("val_loss_epoch=")[-1].split(".ckpt")[0]
                else:
                    score_str = ckpt_file.split("val_loss=")[-1].split(".ckpt")[0]
                score = float(score_str)
                if score < best_score:
                    best_score = score
                    best_checkpoint = ckpt_file
            except (ValueError, IndexError):
                pass
        elif prefer_metric == "val_error" and ("val_error=" in ckpt_file or "val_error_epoch=" in ckpt_file):
            try:
                if "val_error_epoch=" in ckpt_file:
                    score_str = ckpt_file.split("val_error_epoch=")[-1].split(".ckpt")[0]
                else:
                    score_str = ckpt_file.split("val_error=")[-1].split(".ckpt")[0]
                score = float(score_str)
                if score < best_score:
                    best_score = score
                    best_checkpoint = ckpt_file
            except (ValueError, IndexError):
                pass

    # If not found, try the alternative metric
    if not best_checkpoint:
        other_metric = "val_loss" if prefer_metric == "val_error" else "val_error"
        for ckpt_file in checkpoint_files:
            if other_metric == "val_loss" and "val_loss=" in ckpt_file:
                try:
                    score_str = ckpt_file.split("val_loss=")[1].split("-loss.ckpt")[0]
                    score = float(score_str)
                    if score < best_score:
                        best_score = score
                        best_checkpoint = ckpt_file
                except (ValueError, IndexError):
                    pass
            elif other_metric == "val_error" and "val_error=" in ckpt_file:
                try:
                    score_str = ckpt_file.split("val_error=")[1].split(".ckpt")[0]
                    score = float(score_str)
                    if score < best_score:
                        best_score = score
                        best_checkpoint = ckpt_file
                except (ValueError, IndexError):
                    pass

    # Additional fallback: parse score from filename pattern (model-epoch-score.ckpt)
    if not best_checkpoint:
        for ckpt_file in sorted(checkpoint_files):
            if ckpt_file == "last.ckpt":
                continue
            try:
                stem = ckpt_file[:-5] if ckpt_file.endswith(".ckpt") else ckpt_file
                # Fallback: treat the last hyphen-separated token as a score
                last_tok = stem.split("-")[-1]
                score = float(last_tok)
                if score < best_score:
                    best_score = score
                    best_checkpoint = ckpt_file
            except Exception:
                continue
    # Final fallback: use last.ckpt or the first file
    if not best_checkpoint:
        if "last.ckpt" in checkpoint_files:
            best_checkpoint = "last.ckpt"
        else:
            best_checkpoint = sorted(checkpoint_files)[0]

    return best_checkpoint, best_score


def _load_model_by_type(model_type: str, checkpoint_path: str) -> Any:
    """Load the appropriate model class based on type.

    Args:
        model_type: Model type ("gat" or "gatv2").
        checkpoint_path: Path to the checkpoint file.

    Returns:
        Loaded model instance.
    """
    if model_type == "gatv2":
        cls = getattr(import_module("mecari.models.gatv2"), "MecariGATv2")
    model = cls.load_from_checkpoint(checkpoint_path, strict=False, map_location="cpu")

    model.eval()
    model.cpu()
    return model


def _instantiate_model_from_config(config: Dict[str, Any]):
    """Instantiate a model using config fields (no checkpoint loading)."""
    model_cfg = config.get("model", {})
    training_cfg = config.get("training", {})
    features_cfg = config.get("features", {})

    if model_cfg.get("type") != "gatv2":
        raise ValueError(f"Unsupported model type: {model_cfg.get('type')}")

    MecariGATv2 = getattr(import_module("mecari.models.gatv2"), "MecariGATv2")
    model = MecariGATv2(
        hidden_dim=model_cfg.get("hidden_dim", 64),
        num_classes=model_cfg.get("num_classes", 1),
        learning_rate=training_cfg.get("learning_rate", 1e-3),
        lexical_feature_dim=features_cfg.get("lexical_feature_dim", 100000),
        num_heads=model_cfg.get("num_heads", 4),
        share_weights=model_cfg.get("share_weights", False),
        dropout=model_cfg.get("dropout", 0.1),
        attn_dropout=model_cfg.get("attn_dropout", model_cfg.get("attention_dropout", 0.1)),
        add_self_loops_flag=model_cfg.get("add_self_loops", True),
        edge_dropout=model_cfg.get("edge_dropout", 0.0),
        norm=model_cfg.get("norm", "layer"),
    )
    return model


def _load_model_from_state(config_path: str, state_path: str):
    """Load model from a plain state_dict plus config.yaml."""
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    model = _instantiate_model_from_config(config)
    state = torch.load(state_path, map_location="cpu")
    # Lightning checkpoints saved via export may store under 'state_dict' already
    if (
        isinstance(state, dict)
        and "state_dict" in state
        and all(k.startswith("model.") for k in state["state_dict"].keys())
    ):
        state = state["state_dict"]
    # Remove potential 'model.' prefix if present (depends on save path)
    new_state = {}
    for k, v in state.items():
        nk = k
        if k.startswith("model."):
            nk = k[len("model.") :]
        new_state[nk] = v
    model.load_state_dict(new_state, strict=False)
    model.eval()
    model.cpu()
    return model


def load_model(
    experiment_name: Optional[str] = None, model_type: Optional[str] = None, prefer_metric: str = "val_error"
) -> Optional[Tuple[Any, Dict[str, Any]]]:
    """Load a trained model and its experiment info.

    Default behavior: load the single model under sample_model/.
    If --experiment is provided (or sample_model is unavailable), use experiments/.
    """
    # Default: load from sample_model/
    if not experiment_name:
        root = "sample_model"
        if os.path.exists(root):
            fixed_config = os.path.join(root, "config.yaml")
            state_path = os.path.join(root, "model.pt")
            if os.path.exists(fixed_config) and os.path.exists(state_path):
                try:
                    with open(fixed_config, "r", encoding="utf-8") as f:
                        config = yaml.safe_load(f)
                    model = _load_model_from_state(fixed_config, state_path)
                    experiment_info = {
                        "name": os.path.basename(root),
                        "path": root,
                        "best_metric": None,
                        "best_score": None,
                        "model_type": config.get("model", {}).get("type", "unknown"),
                        "best_model_path": state_path,
                        "config": config,
                    }
                    return model, experiment_info
                except Exception as e:
                    print(f"Failed to load sample model: {e}")
                    return None
            print("sample_model/model.pt or config.yaml not found")
            return None
        else:
            print("sample_model directory not found")
            return None

    # Specific experiment provided
    if experiment_name:
        exp_path = os.path.join("experiments", experiment_name)
        config_path = os.path.join(exp_path, "config.yaml")
        checkpoints_dir = os.path.join(exp_path, "checkpoints")

        if not os.path.exists(config_path) or not os.path.exists(checkpoints_dir):
            print(f"Experiment not found: {experiment_name}")
            return None

        try:
            with open(config_path, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)

            model_type_from_config = config.get("model", {}).get("type", "unknown")
            best_checkpoint, best_score = _find_best_checkpoint(checkpoints_dir, prefer_metric)

            if not best_checkpoint:
                print("No checkpoint found")
                return None

            metric_name = "val_loss" if prefer_metric == "val_loss" else "val_error"

            experiment_info = {
                "name": experiment_name,
                "path": exp_path,
                "val_error": best_score if prefer_metric == "val_error" else None,
                "val_loss": best_score if prefer_metric == "val_loss" else None,
                "best_metric": metric_name,
                "best_score": best_score,
                "model_type": model_type_from_config,
                "best_model_path": os.path.join(checkpoints_dir, best_checkpoint),
                "config": config,
            }
        except Exception as e:
            print(f"Failed to read experiment info: {e}")
            return None

    # Auto-select the best experiment
    else:
        if not os.path.exists(experiments_dir):
            print("Experiments directory does not exist")
            return None

        experiments = []
        for exp_dir in os.listdir(experiments_dir):
            exp_path = os.path.join(experiments_dir, exp_dir)
            config_path = os.path.join(exp_path, "config.yaml")
            checkpoints_dir = os.path.join(exp_path, "checkpoints")

            if not os.path.exists(config_path) or not os.path.exists(checkpoints_dir):
                continue

            try:
                with open(config_path, "r", encoding="utf-8") as f:
                    config = yaml.safe_load(f)

                exp_model_type = config.get("model", {}).get("type", "unknown")

                if model_type and exp_model_type.lower() != model_type.lower():
                    continue

                best_checkpoint, best_score = _find_best_checkpoint(checkpoints_dir, prefer_metric)
                if best_checkpoint:
                    metric_name = "val_loss" if prefer_metric == "val_loss" else "val_error"
                    experiments.append(
                        {
                            "name": exp_dir,
                            "path": exp_path,
                            "val_error": best_score if prefer_metric == "val_error" else None,
                            "val_loss": best_score if prefer_metric == "val_loss" else None,
                            "best_metric": metric_name,
                            "best_score": best_score,
                            "model_type": exp_model_type,
                            "best_model_path": os.path.join(checkpoints_dir, best_checkpoint),
                            "config": config,
                        }
                    )
            except Exception:
                continue

        if not experiments:
            print("No available experiments found")
            return None

        experiment_info = min(experiments, key=lambda x: x["best_score"])

    # Load model
    print(f"Loading model: {experiment_info['best_model_path']}")
    print(f"Experiment: {experiment_info['name']}")

    try:
        model = _load_model_by_type(experiment_info["model_type"], experiment_info["best_model_path"])

        # No BERT features in this pipeline

        return model, experiment_info
    except Exception as e:
        print(f"Model loading error: {e}")
        return None


def viterbi_decode_from_morphemes(logits: torch.Tensor, morphemes: list, edges: list, silent: bool = False) -> list:
    """Edge-based Viterbi decoding.

    Args:
        logits: Logits per morpheme.
        morphemes: List of morpheme records.
        edges: Edge list among morpheme indices.
        silent: If True, suppress debug prints.

    Returns:
        Indices of morphemes on the optimal path.
    """
    if len(logits) != len(morphemes):
        if not silent:
            print(f"Warning: #logits ({len(logits)}) != #morphemes ({len(morphemes)})")
        return list(range(min(len(logits), len(morphemes))))

    if not silent:
        print("\n=== Viterbi Decode ===")
        print(f"#Morphemes: {len(morphemes)}")
        print(f"Using edge info: {len(edges)} edges")

        print("\nNode logits:")
        for idx, (morph, logit) in enumerate(zip(morphemes, logits)):
            print(
                f"  [{idx:3d}] {morph['surface']:10s} ({morph['start_pos']:2d}-{morph['end_pos']:2d}) {morph['pos']:10s} logit={logit:.3f}"
            )

    # Build adjacency from edges (forward edges only)
    n = len(morphemes)
    adj_list = [[] for _ in range(n)]
    for edge in edges:
        source_idx = edge["source_idx"]
        target_idx = edge["target_idx"]
        if 0 <= source_idx < n and 0 <= target_idx < n:
            # Add forward edges only (source.end_pos <= target.start_pos)
            source_end = morphemes[source_idx].get("end_pos", 0)
            target_start = morphemes[target_idx].get("start_pos", 0)
            if source_end <= target_start:
                adj_list[source_idx].append(target_idx)

    # POS to UD mapping (for display)
    pos_to_ud = {
        "名詞": "NOUN",
        "動詞": "VERB",
        "形容詞": "ADJ",
        "副詞": "ADV",
        "助詞": "ADP",  # approximate
        "助動詞": "AUX",
        "接続詞": "CCONJ",
        "連体詞": "DET",
        "感動詞": "INTJ",
        "代名詞": "PRON",
        "形状詞": "ADJ",
        "補助記号": "PUNCT",
        "接頭辞": "PREFIX",
        "接尾辞": "SUFFIX",
    }

    if not silent:
        print("\nMorpheme details:")
        for i, morpheme in enumerate(morphemes):
            start_pos = morpheme.get("start_pos", 0)
            end_pos = morpheme.get("end_pos", 0)
            surface = morpheme.get("surface", "")
            logit = morpheme.get("logit", 0.0)
            pos = morpheme.get("pos", "")
            pos_main = pos.split(",")[0] if "," in pos else pos
            ud_pos = pos_to_ud.get(pos_main, "X")
            print(f"  {i}: {surface} ({start_pos}-{end_pos}) {pos_main}({ud_pos}) logit={logit:.3f}")

    # Dynamic programming
    dp = [-float("inf")] * n  # max score to each node
    parent = [-1] * n  # best predecessor per node

    # Find start nodes (earliest start position)
    start_nodes = []
    min_start_pos = min(m.get("start_pos", 0) for m in morphemes)
    for i, m in enumerate(morphemes):
        if m.get("start_pos", 0) == min_start_pos:
            start_nodes.append(i)

    # Initialize start nodes
    for i in start_nodes:
        dp[i] = morphemes[i].get("logit", 0.0)

    # Process nodes in position order (topological-like)
    node_positions = [(i, morphemes[i].get("start_pos", 0), morphemes[i].get("end_pos", 0)) for i in range(n)]
    node_positions.sort(key=lambda x: (x[1], x[2]))  # sort by start_pos, end_pos

    # Relax edges for each node in order
    for node_idx, _, _ in node_positions:
        if dp[node_idx] == -float("inf"):
            continue  # unreachable node

        # Relax transitions to reachable next nodes
        for next_idx in adj_list[node_idx]:
            new_score = dp[node_idx] + morphemes[next_idx].get("logit", 0.0)
            if new_score > dp[next_idx]:
                dp[next_idx] = new_score
                parent[next_idx] = node_idx

    # Select best end node at the final position
    end_nodes = []
    max_end_pos = max(m.get("end_pos", 0) for m in morphemes)
    for i, m in enumerate(morphemes):
        if m.get("end_pos", 0) == max_end_pos:
            end_nodes.append(i)

    best_end_idx = -1
    best_score = -float("inf")
    for i in end_nodes:
        if dp[i] > best_score:
            best_score = dp[i]
            best_end_idx = i

    # Backtracking with safety cap to avoid infinite loops
    path = []
    current = best_end_idx
    max_iterations = n * 2  # safety cap
    iteration_count = 0
    visited = set()

    while current != -1 and iteration_count < max_iterations:
        if current in visited:
            print(f"Warning: Detected cycle during backtracking (node {current})")
            break
        visited.add(current)
        path.append(current)
        current = parent[current]
        iteration_count += 1

    if iteration_count >= max_iterations:
        print(f"Warning: Backtracking reached max iterations ({max_iterations})")

    path.reverse()

    # Display
    if path:
        total_score = sum(morphemes[idx].get("logit", 0.0) for idx in path)
        if not silent:
            print(f"\nOptimal path (total score: {total_score:.3f}):")
            for idx in path:
                morpheme = morphemes[idx]
                logit = morpheme.get("logit", 0.0)
                print(f"  {morpheme['surface']} (logit: {logit:.3f})")

    return path


##


# Global singletons (lazy initialization)
_analyzer = None
_data_module_cache = {}


def predict_morphemes_from_text(text, model=None, experiment_info=None, silent=False):
    """Predict morpheme boundaries from text.

    Steps:
    1. Analyze with MeCab to get candidates.
    2. Build nodes/edges from morphemes and connections.
    3. Run the model to get per-node scores.
    4. Run Viterbi decoding over nodes and edges.

    Args:
        text: Input text.
        model: Model to use.
        experiment_info: Experiment metadata.
        silent: If True, suppress prints.
    """
    global _analyzer

    if model is None:
        result = load_model()
        if result is None:
            return [], []
        model, experiment_info = result

    if not silent:
        print(f"Input text: {text}")

    # 1) Get morpheme candidates (initialize analyzer on first use)
    if _analyzer is None:
        _analyzer = MeCabAnalyzer()

    # Fetch candidates directly via analyzer and deduplicate
    candidates = _analyzer.get_morpheme_candidates(text)
    candidates = normalize_mecab_candidates(candidates)
    candidates = dedup_morphemes(candidates)

    if not candidates:
        print("Error: Failed to obtain morpheme candidates")
        return [], []

    if not silent:
        print(f"#Candidates: {len(candidates)}")

    # 2) Use candidates as morphemes
    morphemes = candidates

    # Validate type
    if not isinstance(morphemes, list):
        print(f"Warning: morphemes is not a list: {type(morphemes)}")
        morphemes = []

    # Add lexical features using the shared DataModule implementation
    dm_tmp = DataModule(annotations_dir="dummy", batch_size=1, num_workers=0, lexical_feature_dim=100000, silent=True)
    morphemes = dm_tmp.compute_lexical_features(morphemes, text)

    # Build edges (adjacent only)
    edges = build_adjacent_edges(morphemes)

    # Add annotation field as '?' for inference
    for morpheme in morphemes:
        if "annotation" not in morpheme:
            morpheme["annotation"] = "?"

    if not silent:
        print(f"Unified graph: {len(morphemes)} nodes, {len(edges)} edges")

    # 3) Initialize DataModule per experiment settings
    features_config = experiment_info["config"].get("features", {})
    training_config = experiment_info["config"].get("training", {})
    edge_config = experiment_info["config"].get("edge_features", {})

    # Cache DataModule by annotations_dir
    global _data_module_cache
    cache_key = str(training_config.get("annotations_dir", "annotations_kwdlc"))

    if cache_key not in _data_module_cache:
        # Always use lexical features
        _data_module_cache[cache_key] = DataModule(
            annotations_dir=training_config.get("annotations_dir", "annotations_kwdlc"),
            batch_size=1,
            num_workers=0,
            silent=silent,
            lexical_feature_dim=features_config.get("lexical_feature_dim", 100000),
            use_bidirectional_edges=edge_config.get("use_bidirectional_edges", True),
        )

    data_module = _data_module_cache[cache_key]

    # Build graph using the same public API as preprocessing
    graph = data_module.create_graph_from_morphemes_data(
        morphemes=morphemes,
        edges=edges,
        text=text,
        for_training=False,
    )

    if graph is None:
        print("Error: Failed to create PyTorch graph")
        return [], []

    # Inference

    # Device (CPU by default)
    device = torch.device("cpu")

    # Respect explicit device from experiment_info if present
    if experiment_info and "device" in experiment_info:
        device = experiment_info["device"]

    with torch.no_grad():
        # Ensure lexical feature tensors exist
        if not hasattr(graph, "lexical_indices") or graph.lexical_indices is None:
            print("Error: lexical_indices not found")
            return [], []

        logits = model(
            graph.lexical_indices.to(device),  # lexical_indices
            graph.lexical_values.to(device),  # lexical_values
            graph.edge_index.to(device),
            None,
            graph.edge_attr.to(device) if graph.edge_attr is not None else None,
        ).squeeze()

        if logits.dim() == 0:
            logits = logits.unsqueeze(0)
        probabilities = torch.sigmoid(logits)
        predictions = (probabilities >= 0.5).float()

        # Move back to CPU for post-processing
        logits = logits.cpu()
        probabilities = probabilities.cpu()
        predictions = predictions.cpu()

    # Attach predictions to morphemes
    for i, morpheme in enumerate(morphemes):
        if i < len(predictions):
            morpheme["predicted_annotation"] = "+" if predictions[i] == 1 else "-"
            morpheme["logit"] = logits[i].item()
            morpheme["probability"] = probabilities[i].item()

    # 4) Viterbi decode over nodes/edges (no CRF)
    optimal_path = viterbi_decode_from_morphemes(logits, morphemes, edges, silent=silent)

    # Format results
    results = []
    for i, morpheme in enumerate(morphemes):
        is_in_optimal_path = optimal_path and i in optimal_path

        result = {
            "surface": morpheme["surface"],
            "pos": morpheme["pos"],
            "reading": morpheme["reading"],
            "predicted_annotation": morpheme.get("predicted_annotation", "?"),
            "logit": morpheme.get("logit", 0.0),
            "probability": morpheme.get("probability", 0.5),
            "in_optimal_path": is_in_optimal_path,
        }

        results.append(result)

    # Collect morphemes on the optimal path
    optimal_morphemes = []
    if optimal_path:
        # Count candidates per span
        position_candidates = {}
        for i, m in enumerate(morphemes):
            pos_key = (m.get("start_pos", 0), m.get("end_pos", 0))
            if pos_key not in position_candidates:
                position_candidates[pos_key] = []
            position_candidates[pos_key].append(i)

        for idx in optimal_path:
            if idx < len(morphemes):
                morph = morphemes[idx].copy()
                # Add candidate count and selected rank for this span
                pos_key = (morph.get("start_pos", 0), morph.get("end_pos", 0))
                if pos_key in position_candidates:
                    candidates_at_pos = position_candidates[pos_key]
                    morph["num_candidates"] = len(candidates_at_pos)
                    morph["selected_rank"] = candidates_at_pos.index(idx) + 1 if idx in candidates_at_pos else 0
                optimal_morphemes.append(morph)

    return results, optimal_morphemes


def print_results(results, optimal_morphemes=None, verbose: bool = False):
    """Print morphemes in MeCab-like format (surface\tCSV features)."""
    if not results:
        return

    def mecab_features(m):
        pos = m.get("pos", "*")
        pos1 = m.get("pos_detail1", "*")
        pos2 = m.get("pos_detail2", "*")
        ctype = m.get("inflection_type", "*")
        cform = m.get("inflection_form", "*")
        base = m.get("base_form", m.get("lemma", "*")) or "*"
        reading = m.get("reading", "*") or "*"
        return f"{pos},{pos1},{pos2},{ctype},{cform},{base},{reading}"

    items = (
        optimal_morphemes
        if optimal_morphemes
        else [
            {
                "surface": r.get("surface", ""),
                "pos": r.get("pos", "*"),
                "pos_detail1": "*",
                "pos_detail2": "*",
                "inflection_type": "*",
                "inflection_form": "*",
                "base_form": r.get("surface", ""),
                "reading": r.get("reading", "*"),
            }
            for r in results
        ]
    )

    for m in items:
        print(f"{m.get('surface', '')}\t{mecab_features(m)}")
    print("EOS")


def main():
    """Main inference entrypoint."""
    import argparse

    parser = argparse.ArgumentParser(description="Mecari morphological analysis inference")
    parser.add_argument("--text", "-t", help="Input text directly")
    parser.add_argument("--experiment", "-e", help="Experiment name to load (e.g., gat_20250730_145624)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output (include UD POS)")
    args = parser.parse_args()

    if args.experiment:
        result = load_model(experiment_name=args.experiment)
    else:
        result = load_model()

    if result is None:
        return

    model, experiment_info = result

    if args.text:
        result = predict_morphemes_from_text(args.text, model, experiment_info, silent=not args.verbose)
        if result:
            results, optimal_morphemes = result
            print_results(results, optimal_morphemes, verbose=args.verbose)
        else:
            print("Inference failed.")

    else:
        print("\nMecari morphological inference")
        print("Enter text (e.g., Tokyo is nice)")
        print("Type 'quit' or 'exit' to finish.\n")

        while True:
            try:
                user_input = input("Input: ").strip()

                if user_input.lower() in ["quit", "exit", "q"]:
                    print("Exiting.")
                    break

                if not user_input:
                    continue

                print(f"Text: {user_input}")

                result = predict_morphemes_from_text(user_input, model, experiment_info, silent=not args.verbose)
                if result:
                    results, optimal_morphemes = result
                    print_results(results, optimal_morphemes, verbose=args.verbose)
                else:
                    print("Inference failed.")

                print()

            except EOFError:
                print("\nExiting.")
                break
            except KeyboardInterrupt:
                print("\nExiting.")
                break
            except Exception as e:
                import traceback

                print(f"\nAn error occurred: {e}")
                traceback.print_exc()
                continue


if __name__ == "__main__":
    main()
