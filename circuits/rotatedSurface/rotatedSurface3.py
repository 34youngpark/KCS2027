#!/usr/bin/env python3
"""
Rotated surface code d=3 (rotatedSurface3) — code definition + hardware circuit
====================================================================
9 data + 8 ancilla = 17 qubits, 4 Z-stabilizers + 4 X-stabilizers,
memory-Z protocol (initial |0..0>, final Z-basis data readout).

Lattice coordinates (x horizontal, y vertical), data on odd-odd sites:

    y=5   (1,5)   (3,5)   (5,5)          Z26=(2,6)  boundary (top)
    y=3   (1,3)   (3,3)   (5,3)          X02=(0,2), X64=(6,4) boundary
    y=1   (1,1)   (3,1)   (5,1)          Z40=(4,0)  boundary (bottom)

  Z-stabilizers: Z22=(2,2), Z44=(4,4) bulk; Z40=(4,0), Z26=(2,6) boundary
  X-stabilizers: X42=(4,2), X24=(2,4) bulk; X02=(0,2), X64=(6,4) boundary
  logical Z = data column x=1: {(1,1),(1,3),(1,5)}   (3 data, one column)
  logical X = data row    y=1: {(1,1),(3,1),(5,1)}

CX order inside a cycle (hook-error-safe standard order, FIXED constants):
  every stabilizer executes its CXs over 4 shared layers; a stabilizer's
  CX in layer l targets its corner CORNER_ORDER[type][l] (skipped if that
  corner has no data qubit — boundary weight-2 checks keep their slots):
    Z (data->anc): Z_CORNER_ORDER = (-1,-1),(+1,-1),(-1,+1),(+1,+1)
                   ("Z" shape: bottom row left-to-right, then top row)
    X (anc->data): X_CORNER_ORDER = (-1,-1),(-1,+1),(+1,-1),(+1,+1)
                   ("N" shape: left column bottom-to-top, then right)
  Why hook-safe (memory-Z): an X error on an X-ancilla after layer 2
  propagates into the two data of layers 3-4, which under X_CORNER_ORDER
  share the same x — a VERTICAL X pair, parallel to logical Z / adding no
  progress along logical X, so one fault never eats a unit of effective
  distance. Dually, a Z error on a Z-ancilla after layer 2 leaves a
  HORIZONTAL Z pair (same y), benign for logical Z. The schedule is also
  conflict-free: no data qubit is touched twice in one layer.
  verification/verify_rotatedSurface3.py checks the propagation empirically.

Ancillas are NEVER reset (hardware): raw bit = XOR accumulation with the
same ancilla's previous measurement; check_values() (per-ancilla XOR
chain) recovers the check values. The abstract Stim circuit
(dataset_generation/rotatedSurface3_stim.py) uses MR — the two streams agree at the
check-value level, the same contract as heavy-hex (README §4).

ibm_miami embedding (45-degree rotation): verified 12x10 row-major
square lattice (q = row*10 + col; 218 undirected edges). The code
lattice maps by u=(x+y)/2, v=(y-x)/2 — diagonal data-ancilla pairs
become unit lattice steps, so EVERY stabilizer CX is device-adjacent and
no SWAP is ever inserted. The patch occupies a 5x5 block placed at
EMBED_OFFSETS[backend].
"""
import json

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister

DISTANCE = 3
NUM_DATA = 9
N_CHECKS = 8

# ---- lattice ---------------------------------------------------------
DATA_COORDS = [(1, 1), (3, 1), (5, 1),
               (1, 3), (3, 3), (5, 3),
               (1, 5), (3, 5), (5, 5)]
DIDX = {c: i for i, c in enumerate(DATA_COORDS)}       # coord -> data idx

ANC_DEFS = {                                           # name -> (type, coord)
    'Z22': ('Z', (2, 2)), 'Z44': ('Z', (4, 4)),
    'Z40': ('Z', (4, 0)), 'Z26': ('Z', (2, 6)),
    'X42': ('X', (4, 2)), 'X24': ('X', (2, 4)),
    'X02': ('X', (0, 2)), 'X64': ('X', (6, 4)),
}
CYCLE_ORDER = ['Z22', 'Z44', 'Z40', 'Z26', 'X42', 'X24', 'X02', 'X64']
Z_NAMES = [n for n in CYCLE_ORDER if ANC_DEFS[n][0] == 'Z']
X_NAMES = [n for n in CYCLE_ORDER if ANC_DEFS[n][0] == 'X']
Z_POS = [j for j, n in enumerate(CYCLE_ORDER) if n in Z_NAMES]

# fixed hook-safe CX corner orders (see module docstring) --------------
Z_CORNER_ORDER = ((-1, -1), (+1, -1), (-1, +1), (+1, +1))
X_CORNER_ORDER = ((-1, -1), (-1, +1), (+1, -1), (+1, +1))
N_LAYERS = 4


def _layer_slots(name):
    """Check -> ((layer, data_coord), ...) honoring the corner orders."""
    ctype, (ax, ay) = ANC_DEFS[name]
    order = Z_CORNER_ORDER if ctype == 'Z' else X_CORNER_ORDER
    slots = []
    for layer, (dx, dy) in enumerate(order):
        c = (ax + dx, ay + dy)
        if c in DIDX:
            slots.append((layer, c))
    return tuple(slots)


# name -> (type, support data coords, anc coord, layer slots)
CHECK_DEFS = {n: (ANC_DEFS[n][0],
                  tuple(c for _, c in _layer_slots(n)),
                  ANC_DEFS[n][1],
                  _layer_slots(n))
              for n in CYCLE_ORDER}
Z_STABS = {n: CHECK_DEFS[n][1] for n in Z_NAMES}
X_STABS = {n: CHECK_DEFS[n][1] for n in X_NAMES}

LOGICAL_Z = [(1, 1), (1, 3), (1, 5)]                   # one data column
LOGICAL_X = [(1, 1), (3, 1), (5, 1)]
LOGICAL_Z_IDX = [DIDX[c] for c in LOGICAL_Z]           # [0, 3, 6]

# ---- CNN tensor grid: ancillas on the 4x4 plaquette-vertex grid ------
GRID_SHAPE = (4, 4)
ANC_GRID = {n: (ANC_DEFS[n][1][1] // 2, ANC_DEFS[n][1][0] // 2)
            for n in CYCLE_ORDER}                      # (row=y/2, col=x/2)
assert len(set(ANC_GRID.values())) == N_CHECKS

# qubit ordering: data 0..8, ancillas 9..16 in CYCLE_ORDER
ALL_COORDS = DATA_COORDS + [ANC_DEFS[n][1] for n in CYCLE_ORDER]
L = {c: i for i, c in enumerate(ALL_COORDS)}           # coord -> local idx
AIDX = {n: L[ANC_DEFS[n][1]] for n in CYCLE_ORDER}

# sanity: schedule is conflict-free (a data qubit once per layer)
for _l in range(N_LAYERS):
    _touched = [c for n in CYCLE_ORDER
                for (ll, c) in CHECK_DEFS[n][3] if ll == _l]
    assert len(_touched) == len(set(_touched)), f"layer {_l} conflict"


def check_values(syn_rows, num_cycles):
    """No-reset raw bits -> per-cycle check values via per-ancilla XOR
    chains (each rotatedSurface3 ancilla measures the same check every cycle)."""
    syn_rows = np.asarray(syn_rows)
    n = syn_rows.shape[0]
    vals = {name: np.zeros((n, num_cycles), dtype=int)
            for name in CYCLE_ORDER}
    for j, name in enumerate(CYCLE_ORDER):
        prev = np.zeros(n, dtype=int)
        for cyc in range(num_cycles):
            raw = syn_rows[:, cyc * N_CHECKS + j]
            vals[name][:, cyc] = raw ^ prev
            prev = raw.copy()
    return vals


# ==================================================================
# hardware circuit (Qiskit) — no-reset ancillas, memory-Z
# ==================================================================
class RotatedSurface3Hardware:
    """d=3 rotated surface code circuit on 17 qubits (local indices;
    the ibm_miami embedding only enters at transpile initial_layout).

    inject:      (pauli, data_coord, after_cycle) — data error between
                 cycles (verification)
    inject_ops:  list of (pauli, ("data", coord) | ("anc", name), cycle,
                 after_layer) — error right after a CX layer inside a
                 cycle (hook-error verification)
    """

    def __init__(self, num_cycles=3):
        self.num_cycles = num_cycles

    def build_circuit(self, initial_state=0, inject=None, inject_ops=()):
        q = QuantumRegister(17, 'q')
        syn = ClassicalRegister(N_CHECKS * self.num_cycles, 'syn')
        dat = ClassicalRegister(NUM_DATA, 'data')
        qc = QuantumCircuit(q, syn, dat)
        if initial_state == 1:
            for c in LOGICAL_X:
                qc.x(L[c])
        for cyc in range(self.num_cycles):
            for n in X_NAMES:
                qc.h(AIDX[n])
            for layer in range(N_LAYERS):
                for n in CYCLE_ORDER:
                    for ll, c in CHECK_DEFS[n][3]:
                        if ll != layer:
                            continue
                        if CHECK_DEFS[n][0] == 'Z':
                            qc.cx(L[c], AIDX[n])
                        else:
                            qc.cx(AIDX[n], L[c])
                for pauli, tgt, icyc, ilayer in inject_ops:
                    if icyc == cyc and ilayer == layer:
                        idx = (AIDX[tgt[1]] if tgt[0] == "anc"
                               else L[tgt[1]])
                        getattr(qc, pauli.lower())(idx)
            for n in X_NAMES:
                qc.h(AIDX[n])
            for j, n in enumerate(CYCLE_ORDER):        # no reset
                qc.measure(AIDX[n], syn[cyc * N_CHECKS + j])
            qc.barrier()
            if inject is not None and inject[2] == cyc:
                pauli, dc, _ = inject
                getattr(qc, pauli.lower())(L[dc])
                qc.barrier()
        for i, c in enumerate(DATA_COORDS):
            qc.measure(L[c], dat[i])
        return qc


# ==================================================================
# ibm_miami 45-degree embedding
# ==================================================================
# verified device topology (fetch_coupling): 12 rows x 10 cols,
# row-major numbering q = row*10 + col
MIAMI_ROWS, MIAMI_COLS = 12, 10
# top-left device (row, col) of the 5x5 patch block per backend
EMBED_OFFSETS = {"ibm_miami": (4, 2), "ibm_chicago": (4, 2)}


def embedding_for_surface(backend_name):
    """coord (x,y) -> device qubit, via u=(x+y)/2, v=(y-x)/2 (+ offset).

    Diagonal data-ancilla pairs map to unit lattice steps, so every
    stabilizer CX is device-adjacent (no SWAPs)."""
    try:
        r0, c0 = EMBED_OFFSETS[backend_name]
    except KeyError:
        raise RuntimeError(
            f"no rotatedSurface3 embedding registered for backend '{backend_name}'. "
            f"Known: {sorted(EMBED_OFFSETS)}. Add a 5x5-block offset to "
            f"EMBED_OFFSETS in circuits/rotatedSurface/rotatedSurface3.py after checking the "
            f"device is a row-major square lattice "
            f"(validate_backend_surface).")
    emb = {}
    for (x, y) in ALL_COORDS:
        u, v = (x + y) // 2, (y - x) // 2
        row, col = v + 2 + r0, u - 1 + c0
        emb[(x, y)] = row * MIAMI_COLS + col
    return emb


def required_edges_surface():
    """All (coord, coord) pairs a stabilizer CX needs (undirected)."""
    need = []
    for n in CYCLE_ORDER:
        anc = ANC_DEFS[n][1]
        for c in CHECK_DEFS[n][1]:
            need.append((anc, c))
    return need


def validate_backend_surface(coupling_json_path, raise_on_fail=True):
    """Verify the rotatedSurface3 patch fits the backend under its embedding.

    Also re-checks the square-lattice assumption on the patch block: if
    the device is NOT the expected row-major square lattice, this fails —
    stop and report to the user instead of forcing the embedding."""
    cm = json.load(open(coupling_json_path))
    emb = embedding_for_surface(cm.get('name', '?'))
    edges = {tuple(sorted(e)) for e in cm['coupling_map']}
    missing = []
    for u, v in required_edges_surface():
        dev = tuple(sorted((emb[u], emb[v])))
        if dev not in edges:
            missing.append(((u, v), dev))
    if missing and raise_on_fail:
        raise RuntimeError(
            f"backend '{cm.get('name', '?')}' lacks {len(missing)} device "
            f"edges required by the rotatedSurface3 45-degree embedding (coupling "
            f"map differs from the expected square lattice — report to "
            f"the user): {missing}")
    return missing


if __name__ == '__main__':
    qc = RotatedSurface3Hardware(2).build_circuit()
    ops = qc.count_ops()
    print(f"2-cycle: depth={qc.depth()} cx={ops.get('cx')} h={ops.get('h')} "
          f"meas={ops.get('measure')} reset={ops.get('reset', 0)}")
    print("supports:", {n: CHECK_DEFS[n][1] for n in CYCLE_ORDER})
    emb = embedding_for_surface('ibm_miami')
    print("miami embedding:", {f"{c}": q for c, q in sorted(emb.items())})
