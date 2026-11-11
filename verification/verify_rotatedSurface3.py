#!/usr/bin/env python3
"""
Contract gate for the rotated surface code d=3 (rotatedSurface3, --code surface)
=====================================================================
Dataset generation / hardware submission for --code surface must NOT
proceed until this script prints ALL PASS.

  [A] With zero noise, every abstract-Stim detector is deterministic
  [B] Stim raw-stream invariants: Z-check values are 0, X-check
      cycle-to-cycle XORs are 0, final data Z-support parity == last
      Z-check value, logical Z parity is 0
  [C] The hardware circuit (noiseless Aer, no-reset ancillas) satisfies
      the same invariants through rotatedSurface3.check_values(), with the same bit
      layout (cyc*8 + j, CYCLE_ORDER)
  [D] For injected single DATA errors, the set of checks firing at
      cycle 1 is identical between Stim and Aer and equals the stabilizer
      supports
  [E] HOOK errors: an ancilla error injected in the middle of a cycle
      (right after CX layer 2 of 4) must propagate to the data qubits of
      the REMAINING layers exactly as the fixed CX corner orders predict
      (rotatedSurface3.Z_CORNER_ORDER / X_CORNER_ORDER):
        * statically: the residual pair of a bulk X-check is VERTICAL
          (same x) and of a bulk Z-check HORIZONTAL (same y) — the
          hook-safety property for memory-Z
        * dynamically (Stim AND Aer): the detector stream of the
          hook-injected circuit is bit-identical to the circuit with the
          predicted residual data Paulis injected at the corresponding
          layers
  [F] ibm_chicago embedding check (informational): if a fetched
      coupling_ibm_chicago.json exists at the repo root, the registered
      45-degree embedding must fit it (every stabilizer CX
      device-adjacent). Skipped when the file is absent (offline).
  [G] Hardware-shaped rotatedSurface3 Stim circuit (rotatedSurface3_qpu_stim, the surface
      QPU calibration-profile generator; no-reset ancillas):
      (a) with zero noise, every detector is deterministic
      (b) injected single errors fire the same checks as Aer
          (RotatedSurface3Hardware; raw values through the per-ancilla XOR chain)
      (c) the deterministic detector stream (incl. final-Z detectors)
          reconstructed from raw values is bit-identical to the abstract
          rotatedSurface3 Stim circuit's AND actually fires (no vacuous pass)
      Dataset generation from qpu/* surface profiles must NOT proceed
      until this section passes together with the rest (ALL PASS).
"""
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from qiskit_aer import AerSimulator  # noqa: E402

from circuits.rotatedSurface.rotatedSurface3 import (  # noqa: E402
    RotatedSurface3Hardware, check_values, validate_backend_surface,
    CYCLE_ORDER, CHECK_DEFS, Z_NAMES, X_NAMES, Z_STABS, X_STABS,
    DATA_COORDS, DIDX, LOGICAL_Z_IDX, N_CHECKS, NUM_DATA)
from dataset_generation.rotatedSurface3_stim import (  # noqa: E402
    build_rotatedSurface3_stim_circuit, split_rotatedSurface3_sample, check_matrix_from_dict_rotatedSurface3,
    detectors_from_dataset_rotatedSurface3, num_detectors)
from dataset_generation.rotatedSurface3_qpu_stim import (  # noqa: E402
    build_rotatedSurface3_qpu_stim_circuit)

STIM_SHOTS = 512
AER_SHOTS = 400
CYCLES = 3
DET_EPS = 0.002
IDEAL = "ideal/dp0_mf0_rf0_gd0"


def stim_run(inject=None, inject_ops=()):
    c = build_rotatedSurface3_stim_circuit(CYCLES, "X", 0.0, IDEAL, inject=inject,
                                inject_ops=inject_ops)
    raw = c.compile_sampler().sample(shots=STIM_SHOTS)
    syn, dat = split_rotatedSurface3_sample(raw, CYCLES)
    return syn, dat, np.ones(syn.shape[0])


def aer_run(inject=None, inject_ops=()):
    qc = RotatedSurface3Hardware(CYCLES).build_circuit(0, inject, inject_ops)
    counts = AerSimulator().run(qc, shots=AER_SHOTS).result().get_counts()
    syn, dat, w = [], [], []
    for bs, cnt in counts.items():
        d, s = bs.split()
        syn.append([int(b) for b in s[::-1]])
        dat.append([int(b) for b in d[::-1]])
        w.append(cnt)
    syn = np.array(syn, dtype=np.uint8)
    dat = np.array(dat, dtype=np.uint8)
    mat = check_matrix_from_dict_rotatedSurface3(check_values(syn, CYCLES), CYCLES)
    return mat, dat, np.array(w)


def p1(bits, w):
    return float((bits * w).sum() / w.sum())


def invariants(tag, mat, dat, w):
    fails = []
    pos = {name: j for j, name in enumerate(CYCLE_ORDER)}
    for name in Z_NAMES:
        for c in range(CYCLES):
            if (p := p1(mat[:, c * N_CHECKS + pos[name]], w)) > DET_EPS:
                fails.append(f"Z:{name}@c{c}={p:.3f}")
    for name in X_NAMES:
        for c in range(CYCLES - 1):
            x0 = mat[:, c * N_CHECKS + pos[name]]
            x1 = mat[:, (c + 1) * N_CHECKS + pos[name]]
            if (p := p1(x0 ^ x1, w)) > DET_EPS:
                fails.append(f"Xxor:{name}@c{c}={p:.3f}")
    for name, supp in Z_STABS.items():
        par = np.bitwise_xor.reduce(dat[:, [DIDX[q] for q in supp]], axis=1)
        if (p := p1(par ^ mat[:, (CYCLES - 1) * N_CHECKS + pos[name]],
                    w)) > DET_EPS:
            fails.append(f"final:{name}={p:.3f}")
    zl = np.bitwise_xor.reduce(dat[:, LOGICAL_Z_IDX], axis=1)
    if (p := p1(zl, w)) > DET_EPS:
        fails.append(f"Z_L={p:.3f}")
    ok = not fails
    print(f"[{tag}] invariants: {'PASS' if ok else 'FAIL ' + str(fails[:8])}")
    return ok


def fired_at_cycle1(mat, w):
    fired = []
    pos = {name: j for j, name in enumerate(CYCLE_ORDER)}
    for name in CYCLE_ORDER:
        j = pos[name]
        if name in Z_NAMES:
            p = p1(mat[:, 1 * N_CHECKS + j], w)
        else:
            p = p1(mat[:, 1 * N_CHECKS + j] ^ mat[:, 0 * N_CHECKS + j], w)
        if p > 1 - DET_EPS:
            fired.append(name)
        elif p > DET_EPS:
            fired.append(f"{name}?{p:.2f}")
    return sorted(fired)


def stim_det_stream(inject_ops=()):
    """(deterministic?, first detector row), reconstructed from RAW
    measured values.

    NOTE: stim's compile_detector_sampler cannot be used here — it takes
    its reference from a noiseless run of the SAME circuit, so an
    explicitly injected Pauli is absorbed into the reference and never
    fires a detection event. Reconstructing the detectors from raw
    values through detectors_from_dataset_rotatedSurface3 anchors them to the
    |0>_L expectations instead, which is what the pipeline (and MWPM)
    actually consumes."""
    syn, dat, _ = stim_run(inject_ops=inject_ops)
    det = detectors_from_dataset_rotatedSurface3(syn, dat, CYCLES)  # MR values
    return bool((det == det[0]).all()), det[0]


def aer_det_stream(inject_ops=()):
    """Detector reconstruction of the Aer run (must be shot-independent)."""
    mat, dat, w = aer_run(inject_ops=inject_ops)
    det = detectors_from_dataset_rotatedSurface3(mat, dat, CYCLES)
    return bool((det == det[0]).all()), det[0]


def main():
    ok = True

    # [A] Stim detector determinism
    c = build_rotatedSurface3_stim_circuit(CYCLES, "X", 0.0, IDEAL)
    det = np.asarray(c.compile_detector_sampler().sample(
        shots=STIM_SHOTS, append_observables=True), dtype=np.uint8)
    a_ok = det.sum() == 0 and det.shape[1] == num_detectors(CYCLES) + 1
    print(f"[A] Stim noiseless detectors+observable all-zero: "
          f"{'PASS' if a_ok else 'FAIL'} (shape={det.shape}, sum={det.sum()})")
    ok &= a_ok

    # [B] Stim raw-stream invariants (MR values ARE check values)
    syn, dat, w = stim_run()
    ok &= invariants("B:Stim", syn, dat, w)

    # [C] hardware circuit on noiseless Aer (no-reset -> XOR chain)
    aer_mat, aer_dat, aer_w = aer_run()
    assert aer_mat.shape[1] == syn.shape[1] == N_CHECKS * CYCLES
    ok &= invariants("C:Aer", aer_mat, aer_dat, aer_w)

    # [D] single data-error signatures (inject after cycle 0)
    cases = [("X", (1, 1)), ("X", (3, 3)), ("X", (5, 1)), ("X", (1, 5)),
             ("Z", (3, 3)), ("Z", (1, 1)), ("Z", (5, 5)), ("Z", (3, 1))]
    print("[D] single data-error signatures (Stim vs Aer):")
    for pauli, dc in cases:
        s_syn, _, s_w = stim_run(inject=(pauli, dc, 0))
        a_mat, _, a_w = aer_run(inject=(pauli, dc, 0))
        f_stim = fired_at_cycle1(s_syn, s_w)
        f_aer = fired_at_cycle1(a_mat, a_w)
        stabs = Z_STABS if pauli == "X" else X_STABS
        exp = sorted(n for n, s in stabs.items() if dc in s)
        good = f_stim == f_aer == exp
        ok &= good
        print(f"  {pauli} {dc}: stim={f_stim} aer={f_aer} exp={exp} "
              f"{'PASS' if good else 'FAIL'}")

    # [E] hook errors: ancilla error right after CX layer 2 (index 1)
    INJ_LAYER = 1
    print("[E] hook propagation (anc error after CX layer 2, cycle 0):")
    for name in CYCLE_ORDER:
        ctype = CHECK_DEFS[name][0]
        pauli = "X" if ctype == "X" else "Z"   # the propagating error
        residual = [(ll, c) for ll, c in CHECK_DEFS[name][3]
                    if ll > INJ_LAYER]
        # static hook-safety: a bulk residual PAIR must be vertical for
        # X-checks (same x) / horizontal for Z-checks (same y)
        static_ok = True
        if len(residual) == 2:
            (_, c1), (_, c2) = residual
            static_ok = (c1[0] == c2[0]) if ctype == "X" else (c1[1] == c2[1])
        hook_ops = ((pauli, ("anc", name), 0, INJ_LAYER),)
        resid_ops = tuple((pauli, ("data", c), 0, ll)
                          for ll, c in residual)
        s_ok1, s_hook = stim_det_stream(hook_ops)
        s_ok2, s_resid = stim_det_stream(resid_ops)
        a_ok1, a_hook = aer_det_stream(hook_ops)
        a_ok2, a_resid = aer_det_stream(resid_ops)
        dyn_ok = (s_ok1 and s_ok2 and a_ok1 and a_ok2
                  and np.array_equal(s_hook, s_resid)
                  and np.array_equal(a_hook, a_resid)
                  and np.array_equal(s_hook, a_hook))
        good = static_ok and dyn_ok
        ok &= good
        pair = [c for _, c in residual]
        print(f"  {pauli}-hook on {name}: residual={pair} "
              f"{'(vertical)' if ctype == 'X' and len(pair) == 2 else ''}"
              f"{'(horizontal)' if ctype == 'Z' and len(pair) == 2 else ''}"
              f" stim==resid=={np.array_equal(s_hook, s_resid)} "
              f"aer==resid=={np.array_equal(a_hook, a_resid)} "
              f"{'PASS' if good else 'FAIL'}")

    # [G] hardware-shaped rotatedSurface3 Stim circuit (surface QPU-profile path)
    # (a) noiseless detector determinism
    cq = build_rotatedSurface3_qpu_stim_circuit(CYCLES)
    detq = np.asarray(cq.compile_detector_sampler().sample(
        shots=STIM_SHOTS, append_observables=True), dtype=np.uint8)
    ga_ok = (detq.sum() == 0
             and detq.shape[1] == num_detectors(CYCLES) + 1)
    print(f"[G-a] rotatedSurface3 HW-shaped Stim noiseless detectors+observable "
          f"all-zero: {'PASS' if ga_ok else 'FAIL'} "
          f"(shape={detq.shape}, sum={detq.sum()})")
    ok &= ga_ok

    # (b) single-error signatures vs Aer, (c) deterministic
    #     detector-stream equality vs the abstract rotatedSurface3 Stim circuit —
    #     detectors reconstructed from RAW values (stim's own detector
    #     sampler absorbs explicit Paulis into its reference, see the
    #     stim_det_stream note)
    def qpu_value_run(inject=None, shots=64):
        c = build_rotatedSurface3_qpu_stim_circuit(CYCLES, inject=inject)
        raw = np.asarray(c.compile_sampler().sample(shots), dtype=np.uint8)
        syn, dat = split_rotatedSurface3_sample(raw, CYCLES)
        mat = check_matrix_from_dict_rotatedSurface3(check_values(syn, CYCLES), CYCLES)
        return mat, dat

    print("[G-b/c] rotatedSurface3 HW-shaped Stim vs Aer signatures / vs abstract "
          "rotatedSurface3 Stim detector streams:")
    for pauli, dc in cases:
        inject = (pauli, dc, 0)
        q_mat, q_dat = qpu_value_run(inject)
        f_qpu = fired_at_cycle1(q_mat, np.ones(q_mat.shape[0]))
        a_mat, _, a_w = aer_run(inject=inject)
        f_aer = fired_at_cycle1(a_mat, a_w)
        stabs = Z_STABS if pauli == "X" else X_STABS
        exp = sorted(n for n, s in stabs.items() if dc in s)
        s_syn, s_dat, _ = stim_run(inject=inject)
        det_abs = detectors_from_dataset_rotatedSurface3(s_syn, s_dat, CYCLES)
        det_qpu = detectors_from_dataset_rotatedSurface3(q_mat, q_dat, CYCLES)
        stream_ok = (bool((det_abs == det_abs[0]).all())
                     and bool((det_qpu == det_qpu[0]).all())
                     and np.array_equal(det_abs[0], det_qpu[0])
                     and det_abs[0].any())          # must actually fire
        good = (f_qpu == f_aer == exp) and stream_ok
        ok &= good
        print(f"  {pauli} {dc}: qpu={f_qpu} aer={f_aer} exp={exp} "
              f"detstream={'==' if stream_ok else '!='}abstract "
              f"{'PASS' if good else 'FAIL'}")

    # [F] ibm_chicago embedding (informational; needs a fetched coupling map)
    coupling = _ROOT / "coupling_ibm_chicago.json"
    if coupling.exists():
        missing = validate_backend_surface(coupling, raise_on_fail=False)
        f_ok = not missing
        ok &= f_ok
        print(f"[F] ibm_chicago 45-degree embedding fits coupling map: "
              f"{'PASS' if f_ok else 'FAIL ' + str(missing[:5])}")
    else:
        print("[F] ibm_chicago embedding: SKIP (no coupling_ibm_chicago.json — "
              "run circuits/heavyhex/fetch_coupling.py first)")

    print(f"\nOVERALL: {'ALL PASS' if ok else 'FAILURES PRESENT'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
