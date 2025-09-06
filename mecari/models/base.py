#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Base model with lexical features only."""

from typing import Optional

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F


class BaseMecariGNN(pl.LightningModule):
    """Base class for Mecari morpheme GNNs."""

    def __init__(
        self,
        hidden_dim: int = 512,
        num_classes: int = 1,
        learning_rate: float = 1e-3,
        lexical_feature_dim: int = 100000,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.lexical_feature_dim = lexical_feature_dim

        self.lexical_embedding = nn.Embedding(
            num_embeddings=lexical_feature_dim, embedding_dim=hidden_dim, padding_idx=0, sparse=False
        )
        nn.init.xavier_uniform_(self.lexical_embedding.weight[1:])
        self.lexical_embedding.weight.data[0].fill_(0)

        self.lexical_norm = nn.LayerNorm(hidden_dim)
        self.lexical_dropout = nn.Dropout(0.2)

        self.residual_proj = nn.Linear(hidden_dim, hidden_dim)

        self.node_classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Dropout(0.2), nn.Linear(hidden_dim, 1)
        )

    def _process_features(
        self, lexical_indices: torch.Tensor, lexical_values: torch.Tensor, bert_features: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Process lexical features."""
        embedded = self.lexical_embedding(lexical_indices)
        weighted = embedded * lexical_values.unsqueeze(-1)
        aggregated = weighted.sum(dim=1)
        processed = self.lexical_dropout(self.lexical_norm(aggregated))
        return processed

    def forward(self, lexical_indices, lexical_values, edge_index, bert_features=None, edge_attr=None):
        """Forward pass (implemented in subclasses)."""
        raise NotImplementedError("Subclasses must implement forward method")

    def training_step(self, batch, batch_idx):
        node_predictions = self(
            batch.lexical_indices,
            batch.lexical_values,
            batch.edge_index,
            None,
            batch.edge_attr if hasattr(batch, "edge_attr") else None,
        ).squeeze()

        valid_mask = batch.valid_mask
        valid_predictions = node_predictions[valid_mask]
        valid_targets = batch.y[valid_mask]

        loss = self._compute_bce_loss(valid_predictions, valid_targets, stage="train")

        with torch.no_grad():
            pred_probs = torch.sigmoid(valid_predictions)
            pred_binary = (pred_probs > 0.5).float()
            correct = (pred_binary == valid_targets).sum()
            accuracy = correct / valid_targets.numel()
            error_rate = 1.0 - accuracy

        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log("train_error", error_rate, prog_bar=True, on_step=True, on_epoch=True)

        if self.trainer and self.trainer.optimizers:
            current_lr = self.trainer.optimizers[0].param_groups[0]["lr"]
            self.log("learning_rate", current_lr, on_step=True, on_epoch=False)

        return loss

    def validation_step(self, batch, batch_idx):
        node_predictions = self(
            batch.lexical_indices,
            batch.lexical_values,
            batch.edge_index,
            None,
            batch.edge_attr if hasattr(batch, "edge_attr") else None,
        ).squeeze()

        valid_mask = batch.valid_mask
        valid_predictions = node_predictions[valid_mask]
        valid_targets = batch.y[valid_mask]

        loss = self._compute_bce_loss(valid_predictions, valid_targets, stage="val")

        with torch.no_grad():
            pred_probs = torch.sigmoid(valid_predictions)
            pred_binary = (pred_probs > 0.5).float()
            correct = (pred_binary == valid_targets).sum()
            accuracy = correct / valid_targets.numel()
            error_rate = 1.0 - accuracy

        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log("val_error", error_rate, prog_bar=True, on_step=True, on_epoch=True)

        self.log("val_loss_epoch", loss, on_step=False, on_epoch=True)
        self.log("val_error_epoch", error_rate, on_step=False, on_epoch=True)

        return loss

    def configure_optimizers(self):
        """Configure optimizer."""
        optimizer_config = getattr(self, "training_config", {}).get("optimizer", {})
        optimizer_type = optimizer_config.get("type", "adamw")

        if optimizer_type == "adamw":
            optimizer = torch.optim.AdamW(
                self.parameters(), lr=self.learning_rate, weight_decay=optimizer_config.get("weight_decay", 0.01)
            )
        else:
            optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        # Optional warmup scheduler (linear warmup to base LR)
        tc = getattr(self, "training_config", {}) or {}
        warmup_steps = int(tc.get("warmup_steps", 0) or 0)
        warmup_start_lr = float(tc.get("warmup_start_lr", 0.0) or 0.0)
        if warmup_steps > 0 and self.learning_rate > 0.0:
            start_factor = max(0.0, min(1.0, warmup_start_lr / float(self.learning_rate)))

            def lr_lambda(step: int):
                if step <= 0:
                    return start_factor
                if step < warmup_steps:
                    return start_factor + (1.0 - start_factor) * (step / float(warmup_steps))
                return 1.0

            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "step",
                    "frequency": 1,
                    "name": "linear_warmup",
                },
            }
        return {"optimizer": optimizer}

    def test_step(self, batch, batch_idx):
        node_predictions = self(
            batch.lexical_indices,
            batch.lexical_values,
            batch.edge_index,
            None,
            batch.edge_attr if hasattr(batch, "edge_attr") else None,
        ).squeeze()

        valid_mask = batch.valid_mask
        valid_predictions = node_predictions[valid_mask]
        valid_targets = batch.y[valid_mask]

        with torch.no_grad():
            pred_probs = torch.sigmoid(valid_predictions)
            pred_binary = (pred_probs > 0.5).float()
            correct = (pred_binary == valid_targets).sum()
            accuracy = correct / valid_targets.numel()
            error_rate = 1.0 - accuracy

        self.log("test_error", error_rate, on_step=False, on_epoch=True)
        self.log("test_accuracy", accuracy, on_step=False, on_epoch=True)

        return error_rate

    def _compute_bce_loss(self, logits: torch.Tensor, targets: torch.Tensor, stage: str = "train") -> torch.Tensor:
        """BCEWithLogits loss with optional label smoothing and pos_weight.

        - label_smoothing: smooth targets toward 0.5 by epsilon.
        - pos_weight: handle class imbalance using ratio (neg/pos) per batch, robustly.
        """
        loss_cfg = getattr(self, "training_config", {}).get("loss", {})
        eps = float(loss_cfg.get("label_smoothing", 0.0) or 0.0)
        use_pos_weight = bool(loss_cfg.get("use_pos_weight", True))

        # Compute pos_weight from unsmoothed targets
        pos = torch.clamp(targets.sum(), min=0.0)
        total = torch.tensor(targets.numel(), device=targets.device, dtype=targets.dtype)
        neg = total - pos
        pos_weight = None
        if use_pos_weight and pos > 0 and neg > 0:
            # pos_weight = neg/pos; clamp to avoid extreme values
            pw = (neg / pos).detach()
            pw = torch.clamp(pw, 0.5, 50.0)  # safety bounds
            pos_weight = pw

        # Apply label smoothing to targets: y' = (1-eps)*y + 0.5*eps
        if eps > 0.0:
            targets = (1.0 - eps) * targets + 0.5 * eps

        loss = F.binary_cross_entropy_with_logits(
            logits,
            targets,
            pos_weight=pos_weight,
        )

        return loss
