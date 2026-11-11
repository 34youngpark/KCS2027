#!/usr/bin/env python3
"""
Dual-head GNN decoder for the (3,3) heavy-hex surface code.

This is the file you complete for --model gnn. Fill in every `TODO`
block below — the rest of the pipeline (dataset generation, graph
construction, training loop, evaluation, MWPM baseline, hardware run) is
already done and will run as soon as you finish this file.

Graph : nodes = Stim detectors (order of heavyhex33_stim._append_detectors;
        48 nodes for cycles=3 incl. the 8 final-Z detectors — same input
        as MWPM), static edges from model/graph.GraphBuilder (spatial
        support-sharing / temporal / final-Z links). The graph plumbing
        (adjacency, static node features, detector-bit extraction) is
        provided — you only design the network on top of it.
Input : (B, 2*num_cycles + 1, 4, 5) uint8 AUGMENTED syndrome tensor —
        the dataset tensor plus one channel with the 8 final-Z detector
        bits (model/graph.augment_features; train.py / hardware/run_hw.py
        apply it automatically for --model gnn via prepare_features).
Output: qubit_logits   (B, 17) — per-qubit X-error head
                                 (diagnostics: ECR, parity_LER)
        logical_logits (B, 1)  — logical Z flip head
                                 (official metric: head-LER)

Keep the interfaces exactly as they are — the training/eval/hardware
scripts call them as-is:
  * HeavyHexGNN(in_channels, num_qubits, ...) with
    forward(x) -> (qubit_logits, logical_logits)
    (in_channels = 2*num_cycles, same convention as the CNN; the model
     derives num_cycles = in_channels // 2)
  * compute_loss(...) -> (total_loss, loss_logical, loss_qubit)
    The loss is LER-first: BCE on the logical head is the MAIN loss and
    BCE on the per-qubit head is an AUXILIARY loss scaled by aux_weight
    (default 0.5).
  * prepare_features(...) — provided, do not change.

Constraint: pure torch only — no torch_geometric or other graph
libraries. The graph is small (48 nodes), so message passing works well
as a dense matmul with the normalized adjacency buffer `self.adj`:
neighbor aggregation of node states h (B, N, H) is simply `self.adj @ h`.

Hints:
  * Encode each node's 7 features (detector bit + 6 static) with a
    Linear -> hidden_dim, then run 3-4 message-passing rounds, e.g.
        h' = h + ReLU(LayerNorm(W_self h + adj @ (W_neigh h)))
    (residual + LayerNorm keep deeper stacks stable).
  * Pool over nodes (mean and/or max), then a shared FC -> two linear
    heads, exactly like the CNN.
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

from model.graph import GraphBuilder, augment_features, IN_FEATS  # noqa: E402

NUM_QUBITS = 17


def prepare_features(features, final_bits, code="heavyhex"):
    """Dataset tensor (B,2C,4,5) + final bits (B,17) -> augmented input.

    train.py passes the per-qubit flip labels; hardware/run_hw.py passes
    the measured final data bits. Same detector bits either way (see
    model/graph.py). Provided — do not change."""
    return augment_features(features, final_bits, code)


class HeavyHexGNN(nn.Module):
    """in_channels = 2*num_cycles (same convention as the CNN)."""

    def __init__(self, in_channels=6, num_qubits=NUM_QUBITS,
                 hidden_dim=128, num_layers=4, fc_dim=256, dropout=0.1,
                 code="heavyhex"):
        super().__init__()
        # provided: static graph buffers (do not change) ------------------
        num_cycles = in_channels // 2
        gb = GraphBuilder(code, num_cycles)
        self.graph = gb
        self.register_buffer("adj",
                             torch.from_numpy(gb.norm_adjacency))
        self.register_buffer("static_feats",
                             torch.from_numpy(gb.static_feats))
        # ------------------------------------------------------------------
        # network construction
        # ------------------------------------------------------------------
        self.num_nodes = gb.num_nodes
        self.encoder = nn.Linear(IN_FEATS, hidden_dim)
        # message passing: h' = h + ReLU(LN(W_self h + adj @ (W_neigh h)))
        self.mp_layers = nn.ModuleList(
            [_MPLayer(hidden_dim) for _ in range(num_layers)])
        # readout: concat(mean-pool, max-pool, flattened node states).
        # The graph is static and tiny (48 nodes), so keeping the
        # per-node states (position-aware, like the CNN's flatten) on top
        # of the permutation-invariant pools costs little and lets the
        # heads see *which* detector fired, which the pools alone blur.
        self.shared = nn.Sequential(
            nn.Linear(2 * hidden_dim + self.num_nodes * hidden_dim, fc_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout))

        # ------------------------------------------------------------------
        # the two heads
        # ------------------------------------------------------------------
        self.head_qubit = nn.Linear(fc_dim, num_qubits)    # per-qubit X (ECR)
        self.head_logical = nn.Linear(fc_dim, 1)           # logical Z (LER)

    def forward(self, x):
        # provided: detector bits + static features per node --------------
        det = self.graph.node_values(x).float()          # (B, N)
        stat = self.static_feats.unsqueeze(0).expand(x.shape[0], -1, -1)
        h = torch.cat([det.unsqueeze(-1), stat], dim=-1)  # (B, N, 7)
        h = F.relu(self.encoder(h))                        # (B, N, H)
        for layer in self.mp_layers:
            h = layer(h, self.adj)
        pooled = torch.cat([h.mean(dim=1), h.amax(dim=1),
                            h.flatten(1)], dim=-1)
        z = self.shared(pooled)
        return self.head_qubit(z), self.head_logical(z)


class _MPLayer(nn.Module):
    """One dense message-passing round on the normalized adjacency:
    h' = h + ReLU(LayerNorm(W_self h + adj @ (W_neigh h)))."""

    def __init__(self, hidden_dim):
        super().__init__()
        self.w_self = nn.Linear(hidden_dim, hidden_dim)
        self.w_neigh = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, h, adj):
        # adj: (N, N) normalized; h: (B, N, H). Batched aggregation is a
        # plain matmul since adj broadcasts over the batch dimension.
        agg = torch.matmul(adj, self.w_neigh(h))
        return h + F.relu(self.norm(self.w_self(h) + agg))


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
