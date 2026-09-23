#!/usr/bin/env python3
"""Validate the sparse checkpointed multi-game engine (experiment4_multi) against the
dense lattice engine (experiment4_kalshi.run_game) on synthetic parlay-only flow.
Both run LOPAMM + ind via the same bottom-up sweep; per-design total loss must match."""
import numpy as np

import experiment4_kalshi as K
import experiment4_multi as MM


def make(seed):
    rng = np.random.default_rng(seed)
    legs = list(range(1, 7))
    parlays = []
    for _ in range(5):
        k = int(rng.integers(2, 6))
        S = tuple(sorted(rng.choice(legs, k, replace=False).tolist()))
        parlays.append((S, tuple(int(x) for x in rng.integers(0, 2, k))))
    tape = [(int(rng.integers(0, len(parlays))), float(rng.uniform(.05, .95)), 1.0, "yes")
            for _ in range(40)]
    rbit = {i: int(rng.integers(0, 2)) for i in legs}
    return parlays, tape, rbit


def sparse_run(parlays, tape, rbit):
    rA, rI, cA, cI = {}, {}, {}, {}
    for pi, p, _sz, _sd in tape:
        S, conj = parlays[pi]
        MM._outs(S); marg = MM._shared_marg(rA, S, MM._OI[S][conj], p)
        for Sp in MM._subsets(S):
            lp = len(Sp)
            qa = MM._QA(rA, Sp); c, dl = MM._match(qa, marg[Sp])
            cA[lp] = cA.get(lp, 0.0) + c
            rA[Sp] = rA.get(Sp, np.zeros(len(MM._OC[Sp]))) + dl
            qi = rI.get(Sp, np.zeros(len(MM._OC[Sp]))); c, dl = MM._match(qi, marg[Sp])
            cI[lp] = cI.get(lp, 0.0) + c; rI[Sp] = qi + dl

    def win(S):
        return MM._OI[S][tuple(rbit[l] for l in S)]

    def tot(r, cash):
        return sum(c[win(S)] for S, c in r.items() if c.any()) - sum(cash.values())
    return tot(rA, cA), tot(rI, cI)


def main():
    worst = 0.0
    for seed in range(8):
        parlays, tape, rbit = make(seed)
        sa, si = sparse_run(parlays, tape, rbit)
        g = {"suffix": "x", "M": 6, "legs": 6, "rbit": rbit, "dropped": 0,
             "parlays": [{"S": S, "conj": c} for S, c in parlays],
             "tape": [("parlay", pi, p, sz, sd) for pi, p, sz, sd in tape]}
        r = K.run_game(g)
        la, li = r["lopamm_total"], r["ind_total"]
        worst = max(worst, abs(sa - la), abs(si - li))
        print(f"seed {seed}: lopamm sparse={sa:9.4f} lattice={la:9.4f} | "
              f"ind sparse={si:9.4f} lattice={li:9.4f}")
    print(f"\nmax abs diff: {worst:.3e}  -> {'MATCH' if worst < 1e-6 else 'MISMATCH'}")


if __name__ == "__main__":
    main()
