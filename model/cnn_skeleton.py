#!/usr/bin/env python3
"""
Dual-head CNN decoder for the (3,3) heavy-hex surface code.

This is the one file you need to complete. Fill in every `TODO` block
below — the rest of the pipeline (dataset generation, training loop,
evaluation, MWPM baseline, hardware run) is already done and will run as
soon as you finish this file.

Input : (B, 2*num_cycles + 1, H, W) uint8 tensor
        - heavyhex: 4x5 diamond embedding of the 8 ancillas
          (dataset_generation/heavyhex33_stim.py, ANC_COORD)
        - surface (--code surface): 4x4 plaquette-vertex grid
          (circuits/rotatedSurface/rotatedSurface3.py, ANC_GRID); the constructor's `code`
          argument selects the grid via model.CODE_SPECS
        - channels alternate [Z-plane, X-plane] per cycle
          (default num_cycles=3 -> in_channels=6, so the tensor the
          model actually sees has 7 channels)
        - the LAST channel holds the final-Z detector bits
          (prepare_features / model.graph.augment_features — the same
          detector set the GNN and MWPM consume). Not label leakage:
          logical Z is not a product of Z-stabilizers, so head-LER's
          label cannot be derived from it.
Output: qubit_logits   (B, 17) — per-qubit X-error head
                                 (diagnostics: ECR, parity_LER)
        logical_logits (B, 1)  — logical Z flip head
                                 (official metric: head-LER)

Keep the interfaces exactly as they are — the training/eval/hardware
scripts call them as-is:
  * HeavyHexCNN(in_channels, num_qubits, ...) with
    forward(x) -> (qubit_logits, logical_logits)
  * compute_loss(...) -> (total_loss, loss_logical, loss_qubit)
    The loss is LER-first: BCE on the logical head is the MAIN loss and
    BCE on the per-qubit head is an AUXILIARY loss scaled by aux_weight
    (default 0.5).

Hints:
  * The grid is small (4x5): 3x3 convs with padding=1 and no pooling work
    well. BatchNorm + ReLU per conv is a good default.
  * Flatten -> a shared fully-connected layer -> two linear heads.
  * Use torch.nn.functional.binary_cross_entropy_with_logits (the heads
    output logits, not probabilities).
"""
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from model.graph import augment_features  # noqa: E402

NUM_QUBITS = 17
GRID_H, GRID_W = 4, 5           # heavy-hex diamond default


def prepare_features(features, final_bits, code="heavyhex"):
    """Dataset tensor (B,2C,H,W) + final bits (B,num_qubits) -> augmented input.

    train.py passes the per-qubit flip labels; hardware/run_hw.py passes
    the measured final data bits. Same detector bits either way (see
    model/graph.py). Provided — do not change."""
    return augment_features(features, final_bits, code)


class HeavyHexCNN(nn.Module):
    def __init__(self, in_channels=6, num_qubits=NUM_QUBITS,
                 conv_channels=64, fc_dim=256, dropout=0.1,
                 code="heavyhex"):
        super().__init__()
        # provided: the tensor grid of the selected code (do not change)
        from model import CODE_SPECS
        grid_h, grid_w = CODE_SPECS[code]["grid"]
        # ------------------------------------------------------------------
        # feature extractor: 4 x [Conv3x3(pad=1) -> BN -> ReLU], no pooling
        # (the grid is tiny; 4 layers give every cell a receptive field
        # covering the whole plane). Input has in_channels + 1 planes: the
        # +1 is the final-Z detector channel added by prepare_features.
        # ------------------------------------------------------------------
        self.grid = (grid_h, grid_w)
        cin = in_channels + 1
        layers = []
        for _ in range(4):
            layers += [nn.Conv2d(cin, conv_channels, 3, padding=1, bias=False),
                       nn.BatchNorm2d(conv_channels),
                       nn.ReLU(inplace=True)]
            cin = conv_channels
        self.features = nn.Sequential(*layers)
        self.shared = nn.Sequential(
            nn.Flatten(),
            nn.Linear(conv_channels * grid_h * grid_w, fc_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout))

        # ------------------------------------------------------------------
        # the two heads
        # ------------------------------------------------------------------
        self.head_qubit = nn.Linear(fc_dim, num_qubits)    # per-qubit X (ECR)
        self.head_logical = nn.Linear(fc_dim, 1)           # logical Z (LER)

    def forward(self, x):
        x = x.float()
        h = self.shared(self.features(x))
        return self.head_qubit(h), self.head_logical(h)


def compute_loss(qubit_logits, logical_logits, y_qubit, y_logical,
                 aux_weight=0.5, qubit_pos_weight=None):
    """Return (total_loss, loss_logical, loss_qubit).

    total = BCE(logical_logits, y_logical) + aux_weight * BCE(qubit_logits,
    y_qubit). qubit_pos_weight (optional, shape (17,)) goes into the
    per-qubit BCE's pos_weight to counter class imbalance
    (e.g. pos_weight=(1-p)/p).
    """
    # cast logits to fp32 so the BCE is numerically safe under AMP autocast
    loss_logical = F.binary_cross_entropy_with_logits(
        logical_logits.float().reshape(-1), y_logical.float().reshape(-1))
    loss_qubit = F.binary_cross_entropy_with_logits(
        qubit_logits.float(), y_qubit.float(), pos_weight=qubit_pos_weight)
    total = loss_logical + aux_weight * loss_qubit
    return total, loss_logical, loss_qubit
