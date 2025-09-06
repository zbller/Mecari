#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""GATv2 model for morpheme graph classification."""

import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv
from torch_geometric.utils import add_self_loops, dropout_adj

from .base import BaseMecariGNN


class MecariGATv2(BaseMecariGNN):
    """Graph Attention Network v2 for morpheme analysis"""

    def __init__(
        self,
        hidden_dim: int = 512,
        num_heads: int = 8,
        num_layers: int = 4,
        num_classes: int = 1,
        learning_rate: float = 1e-3,
        lexical_feature_dim: int = 100000,
        share_weights: bool = False,  # share-weights option of GATv2
        # New knobs
        dropout: float = 0.1,
        attn_dropout: float = 0.1,
        add_self_loops_flag: bool = True,
        edge_dropout: float = 0.0,
        norm: str = "layer",
        **kwargs,  # Ignore extra params for config compatibility
    ):
        super().__init__(
            hidden_dim=hidden_dim,
            num_classes=num_classes,
            learning_rate=learning_rate,
            lexical_feature_dim=lexical_feature_dim,
        )
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.share_weights = share_weights
        self.feat_dropout_p = dropout
        self.attn_dropout_p = attn_dropout
        self.add_self_loops_flag = add_self_loops_flag
        self.edge_dropout_p = edge_dropout
        self.norm_type = (norm or "layer").lower()

        # GATv2 layers
        self.gatv2_layers = nn.ModuleList()
        self.layer_norms = nn.ModuleList()

        for i in range(num_layers):
            if i == 0:
                # First layer
                self.gatv2_layers.append(
                    GATv2Conv(
                        hidden_dim,
                        hidden_dim,
                        heads=num_heads,
                        dropout=self.attn_dropout_p,
                        share_weights=share_weights,
                        add_self_loops=False,
                    )
                )
            elif i == num_layers - 1:
                # Last layer - single head
                self.gatv2_layers.append(
                    GATv2Conv(
                        hidden_dim * num_heads,
                        hidden_dim,
                        heads=1,
                        concat=False,
                        dropout=self.attn_dropout_p,
                        share_weights=share_weights,
                        add_self_loops=False,
                    )
                )
            else:
                # Middle layers
                self.gatv2_layers.append(
                    GATv2Conv(
                        hidden_dim * num_heads,
                        hidden_dim,
                        heads=num_heads,
                        dropout=self.attn_dropout_p,
                        share_weights=share_weights,
                        add_self_loops=False,
                    )
                )

            # Layer normalization (all layers)
            if i < num_layers - 1:
                self.layer_norms.append(
                    nn.LayerNorm(hidden_dim * num_heads)
                    if self.norm_type == "layer"
                    else nn.BatchNorm1d(hidden_dim * num_heads)
                )
            else:
                self.layer_norms.append(
                    nn.LayerNorm(hidden_dim) if self.norm_type == "layer" else nn.BatchNorm1d(hidden_dim)
                )

    def forward(self, lexical_indices, lexical_values, edge_index, bert_features=None, edge_attr=None):
        """Forward pass of GATv2"""
        x = self._process_features(lexical_indices, lexical_values, bert_features)

        residual = self.residual_proj(x)

        ei = edge_index
        if self.add_self_loops_flag:
            ei, _ = add_self_loops(ei, num_nodes=x.size(0))
        if self.edge_dropout_p > 0 and self.training:
            ei, _ = dropout_adj(ei, p=self.edge_dropout_p, force_undirected=False, training=True)

        # Apply GATv2 layers
        prev = None
        for i in range(self.num_layers):
            prev = x
            x = self.gatv2_layers[i](x, ei)
            x = self.layer_norms[i](x)

            # Per-layer residual if dimension matches (middle layers)
            if x.shape == prev.shape and i < self.num_layers - 1:
                x = x + prev

            # Add residual at last layer
            if i == self.num_layers - 1:
                x = x + residual

            x = F.elu(x)

            # Dropout except last layer
            if i < self.num_layers - 1:
                x = F.dropout(x, p=self.feat_dropout_p, training=self.training)

        # Classification
        logits = self.node_classifier(x)

        return logits
