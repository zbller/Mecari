#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from typing import Dict, List, Optional

import pytorch_lightning as pl
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data, DataLoader

# Required import for lexical feature computation
from mecari.featurizers.lexical import (
    LexicalNGramFeaturizer as LexFeaturizer,
    Morpheme as LexMorpheme,
)


"""Data module for lexical-graph training using prebuilt .pt graphs only."""


# Prebuilt .pt graph dataset
class _PtGraphDataset(Dataset):
    """Prebuilt PyG graph tensors saved as .pt per sentence.

    Each file is expected to be a dict with keys:
      - 'graph': torch_geometric.data.Data
      - 'source_id': str (used for split)
      - optional: 'text'
    """

    def __init__(self, files: List[str]) -> None:
        self.files = files

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> Data:
        path = self.files[idx]
        obj = torch.load(path, map_location="cpu")
        if isinstance(obj, dict) and "graph" in obj:
            data = obj["graph"]
        else:
            data = obj
        if not isinstance(data, Data):
            raise RuntimeError(f"Invalid graph object in: {path}")
        data.data_index = idx
        return data


# Safe globals registration for PyTorch 2.6+
try:
    import torch.serialization
    from torch_geometric.data.data import DataEdgeAttr

    torch.serialization.add_safe_globals([DataEdgeAttr, Data])
except (ImportError, AttributeError):
    pass


class DataModule(pl.LightningDataModule):
    """Loads .pt graphs and builds lexical graph features for training."""

    def __init__(
        self,
        annotations_dir: str = "annotations",
        batch_size: int = 32,
        num_workers: int = 0,
        max_files: Optional[int] = None,
        use_bidirectional_edges: bool = True,
        annotations_override_dir: Optional[str] = None,
        silent: bool = False,
        lexical_feature_dim: int = 100000,
        lexical_max_features: int = 20,
    ) -> None:
        super().__init__()
        self.annotations_dir = annotations_dir
        self.annotations_override_dir = annotations_override_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.max_files = max_files
        self.use_bidirectional_edges = True
        self.silent = silent
        self.lexical_feature_dim = lexical_feature_dim
        self.lexical_max_features = int(lexical_max_features)
        self.use_bidirectional_edges = bool(use_bidirectional_edges)

        # Initialized in setup()
        self.train_dataset = []
        self.val_dataset = []
        self.test_dataset = []
        # Eagerly initialize lexical featurizer (small and picklable)
        self._lex_featurizer = LexFeaturizer(dim=int(self.lexical_feature_dim), add_bias=True)
        # POS mapping for evaluation breakdown
        self.pos_to_id = {
            "名詞": 1,
            "動詞": 2,
            "形容詞": 3,
            "副詞": 4,
            "助詞": 5,
            "助動詞": 6,
            "接続詞": 7,
            "連体詞": 8,
            "感動詞": 9,
            "形状詞": 10,
            "補助記号": 11,
            "接頭辞": 12,
            "接尾辞": 13,
            "特殊": 14,
        }
        self.id_to_pos = {v: k for k, v in self.pos_to_id.items()}

    def create_graph_from_morphemes_data(self, *args, **kwargs) -> Optional[Data]:
        """Create a lexical graph from morpheme data (or candidates)."""
        if "candidates" in kwargs:
            candidates = kwargs.pop("candidates")
            text = kwargs.get("text", "")
            morphemes_edges = self._build_graph_from_candidates(candidates, text)
            if not morphemes_edges:
                return None
            kwargs["morphemes"] = morphemes_edges["morphemes"]
            kwargs["edges"] = morphemes_edges["edges"]
        return self._create_lexical_graph(*args, **kwargs)

    # --- Lexical features helper (for preprocessing) ---
    def compute_lexical_features(self, morphemes: List[Dict], text: str) -> List[Dict]:
        """Add lexical_features to each morpheme using Mecari's lexical featurizer.

        Requires mecari.featurizers.lexical to be importable. Raises a clear error
        if the featurizer is unavailable (training/inference depend on it).
        """
        if not morphemes:
            return morphemes

        for m in morphemes:
            try:
                morph_obj = LexMorpheme(
                    surf=m.get("surface", ""),
                    lemma=m.get("base_form", ""),
                    pos=m.get("pos", "*"),
                    pos1=m.get("pos_detail1", "*"),
                    ctype=m.get("inflection_type", "*"),
                    cform=m.get("inflection_form", "*"),
                    reading=m.get("reading", "*"),
                )
                st = m.get("start_pos", 0)
                ed = m.get("end_pos", st + len(m.get("surface", "")))
                prev_char = text[st - 1] if st > 0 else None
                next_char = text[ed] if ed < len(text) else None
                feats = self._lex_featurizer.unigram_feats(morph_obj, prev_char, next_char)
                m["lexical_features"] = feats
            except Exception:
                # on any failure, leave unchanged
                pass
        return morphemes

    def _create_lexical_graph(
        self, morphemes: List[Dict], edges: List[Dict], text: str, for_training: bool = True
    ) -> Optional[Data]:
        """Build a graph using lexical features."""
        if not morphemes:
            return None

        # Sparse lexical features per node
        all_indices = []
        all_values = []
        all_lengths = []
        annotations = []
        valid_mask = []

        max_features = 0
        for morpheme in morphemes:
            lexical_feats = morpheme.get("lexical_features", [])
            indices = []
            values = []
            for idx, val in lexical_feats:
                if 0 <= idx < self.lexical_feature_dim:
                    indices.append(idx)
                    values.append(val)
            all_lengths.append(len(indices))
            max_features = max(max_features, len(indices))

            all_indices.append(indices)
            all_values.append(values)

            if for_training:
                annotation = morpheme.get("annotation", "?")
                if annotation == "+":
                    annotations.append(1)
                    valid_mask.append(True)
                elif annotation == "-":
                    annotations.append(0)
                    valid_mask.append(True)
                else:
                    annotations.append(0)
                    valid_mask.append(False)

        # Fixed-size padding/truncation for batching
        FIXED_MAX_FEATURES = int(getattr(self, "lexical_max_features", 20))

        padded_indices = []
        padded_values = []
        for indices, values in zip(all_indices, all_values):
            if len(indices) > FIXED_MAX_FEATURES:
                padded_indices.append(indices[:FIXED_MAX_FEATURES])
                padded_values.append(values[:FIXED_MAX_FEATURES])
            else:
                pad_length = FIXED_MAX_FEATURES - len(indices)
                padded_indices.append(indices + [0] * pad_length)
                padded_values.append(values + [0.0] * pad_length)

        edge_index = self._build_edge_index(edges, len(morphemes))

        # POS ids per node (for evaluation breakdown)
        pos_ids = []
        for m in morphemes:
            pos = m.get("pos", "*")
            pos_ids.append(self.pos_to_id.get(pos, 0))

        graph_data = Data(
            lexical_indices=torch.tensor(padded_indices, dtype=torch.long),
            lexical_values=torch.tensor(padded_values, dtype=torch.float32),
            lexical_lengths=torch.tensor(all_lengths, dtype=torch.long),
            edge_index=edge_index,
            num_nodes=len(morphemes),
        )
        graph_data.pos_ids = torch.tensor(pos_ids, dtype=torch.long)
        if for_training:
            graph_data.y = torch.tensor(annotations, dtype=torch.float32)
            graph_data.valid_mask = torch.tensor(valid_mask, dtype=torch.bool)

        return graph_data

    def _build_edge_index(self, edges: List[Dict], num_nodes: int) -> torch.Tensor:
        """Build a PyG edge_index tensor from edge dicts."""
        if not edges:
            return torch.tensor([[], []], dtype=torch.long)

        source_indices = []
        target_indices = []

        for edge in edges:
            source = edge.get("source_idx", 0)
            target = edge.get("target_idx", 0)

            if 0 <= source < num_nodes and 0 <= target < num_nodes:
                source_indices.append(source)
                target_indices.append(target)
                if self.use_bidirectional_edges:
                    source_indices.append(target)
                    target_indices.append(source)

        if not source_indices:
            return torch.tensor([[], []], dtype=torch.long)

        return torch.tensor([source_indices, target_indices], dtype=torch.long)

    def _load_kwdlc_ids(self, ids_file: str) -> set:
        """Load KWDLC ID list (one ID per line)."""
        ids = set()
        if ids_file and os.path.exists(ids_file):
            with open(ids_file, "r") as f:
                for line in f:
                    ids.add(line.strip())
        return ids

    def load_annotation_data(self, max_files: Optional[int] = None) -> List[Dict]:
        """Detect and list available .pt annotation graph files."""
        if os.path.isdir(self.annotations_dir):
            pt_files = [
                os.path.join(self.annotations_dir, fn)
                for fn in sorted(os.listdir(self.annotations_dir))
                if fn.endswith(".pt")
            ]
            if pt_files:
                if max_files is not None:
                    pt_files = pt_files[:max_files]
                return [{"_mode": "pt", "_pt_files": pt_files}]
        raise FileNotFoundError(f"No annotation graphs found under: {self.annotations_dir}")

    def setup(self, stage: Optional[str] = None) -> None:
        """Build train/val/test datasets from discovered .pt files."""
        annotation_data = self.load_annotation_data(max_files=self.max_files)

        if not annotation_data:
            self.train_dataset = []
            self.val_dataset = []
            self.test_dataset = []
            return

        dev_ids = self._load_kwdlc_ids(os.path.join("KWDLC", "id", "split_for_pas", "dev.id"))
        test_ids = self._load_kwdlc_ids(os.path.join("KWDLC", "id", "split_for_pas", "test.id"))

        mode = annotation_data[0].get("_mode")
        if mode == "pt":
            files: List[str] = annotation_data[0]["_pt_files"]
            train_files: List[str] = []
            val_files: List[str] = []
            test_files: List[str] = []

            # Use KWDLC split ids (mandatory)
            dev_ids = self._load_kwdlc_ids(os.path.join("KWDLC", "id", "split_for_pas", "dev.id"))
            test_ids = self._load_kwdlc_ids(os.path.join("KWDLC", "id", "split_for_pas", "test.id"))

            for fp in files:
                sid = None
                try:
                    obj = torch.load(fp, map_location="cpu")
                    if isinstance(obj, dict):
                        sid = obj.get("source_id")
                except Exception:
                    pass
                if sid and (dev_ids or test_ids):
                    if sid in test_ids:
                        test_files.append(fp)
                    elif sid in dev_ids:
                        val_files.append(fp)
                    else:
                        train_files.append(fp)
                else:
                    train_files.append(fp)

            # Build datasets strictly based on KWDLC dev/test ids
            self.train_dataset = _PtGraphDataset(train_files)
            self.val_dataset = _PtGraphDataset(val_files)
            self.test_dataset = _PtGraphDataset(test_files)

            if len(self.val_dataset) == 0 or len(self.test_dataset) == 0:
                raise RuntimeError(
                    "KWDLC dev/test split produced empty val/test datasets. Ensure KWDLC id files exist and source_id is set in .pt files."
                )
        else:
            raise RuntimeError("Unsupported annotation mode; expected pt")

        print(
            f"Data split: train={len(self.train_dataset)}, val={len(self.val_dataset)}, test={len(self.test_dataset)}"
        )

    def _create_dataloader(self, dataset: List[Data], batch_size: int, shuffle: bool = False) -> DataLoader:
        """Create a DataLoader with optional workers/prefetching."""
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=False,
            persistent_workers=True if self.num_workers > 0 else False,
            prefetch_factor=2 if self.num_workers > 0 else None,
        )

    def train_dataloader(self) -> DataLoader:
        """Return train DataLoader."""
        return self._create_dataloader(self.train_dataset, self.batch_size, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        """Return val DataLoader."""
        return self._create_dataloader(self.val_dataset, self.batch_size, shuffle=False)

    def test_dataloader(self) -> DataLoader:
        """Return test DataLoader."""
        return self._create_dataloader(self.test_dataset, self.batch_size, shuffle=False)
