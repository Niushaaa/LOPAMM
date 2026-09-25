#!/usr/bin/env python3
"""
Validate the Exp 4 lattice engine against Exp 1's dense engine. Both replay the SAME
target-price flow through Exp 1's DesignA/DesignC + sweep_books (bottom-up, shared marg);
the only difference is the structure -- experiment4_kalshi.build_lattice (no 2^M, closure
of traded leg-sets) vs parlay_mm_sim.build_structure (full 2^M lattice). Per-design total
loss must match to floating point.
"""
import numpy as np

import experiment1_loworder as E
import experiment4_kalshi as K
from parlay_mm_sim import build_structure, DesignA, DesignC

M = 5


def make(seed):
    rng = np.random.default_rng(seed)
    legs = list(range(1, M + 1))
    parlays = []
    for _ in range(4):
        k = int(rng.integers(2, M + 1))
        S = tuple(sorted(rng.choice(legs, size=k, replace=False).tolist()))
        parlays.append({"S": S, "conj": tuple(int(x) for x in rng.integers(0, 2, size=k))})
    tape = []
    for _ in range(50):
        p = float(rng.uniform(.05, .95))
        if rng.random() < 0.5:
            tape.append(("base", int(rng.integers(1, M + 1)), p, 1.0, "yes"))
        else:
            tape.append(("parlay", int(rng.integers(0, len(parlays))), p, 1.0, "yes"))
    rbit = {i: int(rng.integers(0, 2)) for i in legs}
    return {"rbit": rbit, "parlays": parlays, "tape": tape}


def replay(game, struct):
    rbit, parlays = game["rbit"], game["parlays"]
    A, I, Bd = DesignA(struct, K.B), DesignC(struct, K.B), DesignC(struct, K.B)
    cash = {"A": 0.0, "I": 0.0, "B": 0.0}; trk = [0.0, 0.0, 0]
    for tr in game["tape"]:
        if tr[0] == "base":
            _, leg, p, _sz, _sd = tr
            S = (leg,); marg = {S: np.array([1.0 - p, p])}
            cash["A"] += E.sweep_books(A, struct, [S], marg, K.B, False, trk)
            cash["I"] += E.sweep_books(I, struct, [S], marg, K.B, False, trk)
            cash["B"] += E.sweep_books(Bd, struct, [S], marg, K.B, True, trk)
        else:
            _, pi, p, _sz, _sd = tr
            S, conj = parlays[pi]["S"], parlays[pi]["conj"]
            marg = K._shared_marg(struct, A, S, struct.out_index[S][conj], p)
            cash["A"] += E.sweep_books(A, struct, struct.subsets[S], marg, K.B, False, trk)
            cash["I"] += E.sweep_books(I, struct, struct.subsets[S], marg, K.B, False, trk)
            cash["B"] += E.sweep_books(Bd, struct, struct.subsets[S], marg, K.B, True, trk)

    def win(S):
        return struct.out_index[S][tuple(rbit[leg] for leg in S)]

    def tot(d, ck):
        return sum(c[win(S)] for S, c in d.contracts.items() if c.any()) - ck
    return {"base": tot(Bd, cash["B"]), "ind": tot(I, cash["I"]), "lopmm": tot(A, cash["A"])}


def main():
    worst = 0.0
    for seed in range(8):
        g = make(seed)
        latt = K.build_lattice([(l,) for l in g["rbit"]] + [p["S"] for p in g["parlays"]])
        d = replay(g, build_structure(M)); l = replay(g, latt)
        for nm in ("base", "ind", "lopmm"):
            worst = max(worst, abs(d[nm] - l[nm]))
        print("seed %d:  " % seed + "  ".join(
            f"{nm} dense={d[nm]:8.3f} latt={l[nm]:8.3f} Δ={d[nm]-l[nm]:+.1e}"
            for nm in ("base", "ind", "lopmm")))
    print(f"\nmax abs diff: {worst:.3e}  -> {'MATCH' if worst < 1e-6 else 'MISMATCH'}")


if __name__ == "__main__":
    main()
