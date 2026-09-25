#!/usr/bin/env python3
"""
Experiment 4 -- multi-game (cross-game) Kalshi parlay flow through LOPMM and ind only.

Same model as experiment4_kalshi (Exp 1's bottom-up buys-only sweep, shared target from the
base-leg product, real settlement), but SPARSE and CHECKPOINTED so it scales to ~100k
cross-game parlays on a big-memory machine: residuals are stored per touched leg-set,
sub-lattices are enumerated on demand, and state is snapshotted so a broken run resumes.

Run:  python3 exp4_multi_stage.py && python3 experiment4_multi.py [--ckpt-every N]
"""
import argparse
import itertools
import os
import pickle
import time

import numpy as np

from parlay_mm_sim import lmsr_cost, stable_softmax, T_CLIP

B = 10.0
DATA = "exp4_multi_data/staged.pkl"
OUT = "results_exp4_multi"

_OC, _OI, _SUB, _PJ = {}, {}, {}, {}                 # per-leg-set caches (deterministic)


def _outs(S):
    if S not in _OC:
        _OC[S] = list(itertools.product([0, 1], repeat=len(S)))
        _OI[S] = {o: i for i, o in enumerate(_OC[S])}
    return _OC[S]


def _subsets(S):                                     # non-empty subsets, ascending (size,lex)
    if S not in _SUB:
        subs = []
        for r in range(1, len(S) + 1):
            subs.extend(itertools.combinations(S, r))
        _SUB[S] = subs
    return _SUB[S]


def _proj(S, T):                                     # S-outcome idx -> T-outcome idx
    key = (S, T)
    if key not in _PJ:
        _outs(S); _outs(T)
        pos = {leg: i for i, leg in enumerate(S)}
        _PJ[key] = np.fromiter((_OI[T][tuple(o[pos[l]] for l in T)] for o in _OC[S]),
                               dtype=np.intp, count=len(_OC[S]))
    return _PJ[key]


def _match(Q, tau):
    m = stable_softmax(Q, B)
    lr = B * (np.log(np.clip(tau, T_CLIP, 1.0)) - np.log(np.clip(m, 1e-300, 1.0)))
    d = np.maximum(lr - lr.min(), 0.0)
    return lmsr_cost(Q + d, B) - lmsr_cost(Q, B), d


def _parlay_tau(m, ci, p):
    mc = min(m[ci], 1.0 - 1e-9)
    tau = m * ((1.0 - p) / (1.0 - mc)); tau[ci] = p
    return tau / tau.sum()


def _QA(rA, Sp):                                     # LOPMM Q(Sp)=Σ_{touched T⊆Sp} rA[T]
    O = _outs(Sp); q = np.zeros(len(O))
    for T in _subsets(Sp):                           # (size,lex) order == dense
        v = rA.get(T)
        if v is not None:
            q += v[_proj(Sp, T)]
    return q


def _shared_marg(rA, S, ci, p):
    pl = [float(stable_softmax(rA[(leg,)], B)[1]) if (leg,) in rA else 0.5 for leg in S]
    m = np.array([np.prod([pl[i] if o[i] else 1.0 - pl[i] for i in range(len(S))])
                  for o in _outs(S)])
    tauS = _parlay_tau(m, ci, p)
    return {Sp: (tauS if Sp == S else
                 np.bincount(_proj(S, Sp), weights=tauS, minlength=len(_outs(Sp))))
            for Sp in _subsets(S)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-every", type=int, default=25000)
    ap.add_argument("--kcap", type=int, default=8,          # run-time order cap (O(5^k))
                    help="skip parlay trades with order > kcap (k>8 is very slow)")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    CKPT = f"exp4_multi_data/checkpoint_k{a.kcap}.pkl"     # per-kcap so resumes never mix
    d = pickle.load(open(DATA, "rb"))
    parlays, tape, rbit = d["parlays"], d["tape"], d["rbit"]
    print(f"legs={d['n_legs']} parlays={len(parlays)} trades={len(tape)} "
          f"dropped={d['dropped']} kcap={d['kcap']}", flush=True)

    if os.path.exists(CKPT):
        st = pickle.load(open(CKPT, "rb"))
        rA, rI, cA, cI, start = st["rA"], st["rI"], st["cA"], st["cI"], st["idx"]
        print(f"resumed from checkpoint at trade {start}", flush=True)
    else:
        rA, rI, cA, cI, start = {}, {}, {}, {}, 0

    def save(idx):
        tmp = CKPT + ".tmp"
        pickle.dump({"rA": rA, "rI": rI, "cA": cA, "cI": cI, "idx": idx},
                    open(tmp, "wb"), protocol=4)
        os.replace(tmp, CKPT)

    t0 = time.time()
    for i in range(start, len(tape)):
        pi, p, _sz, _sd = tape[i]
        S, conj = parlays[pi]
        if len(S) > a.kcap:                              # skip too-expensive high-order trades
            continue
        _outs(S); marg = _shared_marg(rA, S, _OI[S][conj], p)
        for Sp in _subsets(S):                       # bottom-up sweep
            lp = len(Sp)
            qa = _QA(rA, Sp); c, dl = _match(qa, marg[Sp])
            cA[lp] = cA.get(lp, 0.0) + c
            rA[Sp] = rA.get(Sp, np.zeros(len(_OC[Sp]))) + dl
            qi = rI.get(Sp)
            if qi is None:
                qi = np.zeros(len(_OC[Sp]))
            c, dl = _match(qi, marg[Sp]); cI[lp] = cI.get(lp, 0.0) + c
            rI[Sp] = qi + dl
        if (i + 1) % a.ckpt_every == 0:
            save(i + 1)
            print(f"  trade {i+1}/{len(tape)}  legsets={len(rA)}  "
                  f"{(i+1-start)/(time.time()-t0):.0f} tr/s", flush=True)
    save(len(tape))

    def win(S):
        return _OI[S][tuple(rbit[leg] for leg in S)]

    def by_level(r, cash):
        pay = {}
        for S, c in r.items():
            if c.any():
                pay[len(S)] = pay.get(len(S), 0.0) + c[win(S)]
        return {j: pay.get(j, 0.0) - cash.get(j, 0.0) for j in set(pay) | set(cash)}

    a_lvl, i_lvl = by_level(rA, cA), by_level(rI, cI)
    res = {"lopmm_by_level": {str(k): v for k, v in a_lvl.items()},
           "ind_by_level": {str(k): v for k, v in i_lvl.items()},
           "lopmm_total": sum(a_lvl.values()), "ind_total": sum(i_lvl.values()),
           "parlays": len(parlays), "trades": len(tape)}
    pickle.dump(res, open(f"{OUT}/result.pkl", "wb"))
    import json
    json.dump({k: (v if not isinstance(v, dict) else v) for k, v in res.items()},
              open(f"{OUT}/result.json", "w"), indent=1, default=float)
    print(f"\nLOPMM total loss = {res['lopmm_total']:.1f}   ind total loss = {res['ind_total']:.1f}")
    print("loss by parlay order |S|:")
    for j in sorted(set(a_lvl) | set(i_lvl)):
        print(f"  |S|={j:>2}:  ind={i_lvl.get(j,0.0):12.1f}   lopmm={a_lvl.get(j,0.0):10.2f}")


if __name__ == "__main__":
    main()
