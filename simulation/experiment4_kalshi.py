#!/usr/bin/env python3
"""
Experiment 4 -- replay real Kalshi NBA parlay order flow through LOPMM / ind / base and
settle at the realized game outcomes, using the Exp 1 engine exactly.

Per trade on leg-set S: build ONE shared target marginal tau_S (pin the conjunction to the
executed price p, keep the other outcomes proportional to the shared base-leg product),
project it to marg[Sp] for every Sp in the sub-lattice, then run Exp 1's bottom-up
`sweep_books`: buys-only `match_to_target` on each Sp ascending by |Sp| (base markets ->
pairs -> ... -> S), charging cash at each level. LOPMM = DesignA (shared residuals), ind =
DesignC (independent at every level), base = DesignC singletons-only. Settle at `realized`.

Runs Exp 1's DesignA/DesignC + sweep_books on a light (no-2^M) lattice; k>KCAP dropped.
Native order-book maker P&L reported alongside (size-weighted). Data: exp4_data/.
"""
import glob
import itertools
import json
import os
import sys

import numpy as np

import experiment1_loworder as E
from parlay_mm_sim import DesignA, DesignC, Structure, match_to_target

B = 10.0
KCAP = 8
PCLIP = 1e-6
DATA = "exp4_data"
OUT = "results_exp4_kalshi"


def _parlay_tau(m, ci, p):
    """Pin conjunction ci to p; scale the rest proportionally (keep their ratios)."""
    mc = min(m[ci], 1.0 - 1e-9)
    tau = m * ((1.0 - p) / (1.0 - mc))
    tau[ci] = p
    return tau / tau.sum()


def build_lattice(maximal):
    """Light Structure (no 2^M atoms) over the downward closure of the given leg-sets."""
    closure = set()
    for S in maximal:
        S = tuple(sorted(S))
        for rr in range(1, len(S) + 1):
            closure.update(itertools.combinations(S, rr))
    legsets = sorted(closure, key=lambda s: (len(s), s))
    outcomes, out_index, pos_in = {}, {}, {}
    for S in legsets:
        outs = list(itertools.product([0, 1], repeat=len(S)))
        outcomes[S] = outs
        out_index[S] = {o: i for i, o in enumerate(outs)}
        pos_in[S] = {leg: i for i, leg in enumerate(S)}
    subsets, proj_idx = {}, {}
    for S in legsets:
        subs = []
        for rr in range(1, len(S) + 1):
            subs.extend(itertools.combinations(S, rr))
        subsets[S] = subs                              # ascending by |Sp|
        posS = pos_in[S]
        for T in subs:
            proj_idx[(S, T)] = np.array(
                [out_index[T][tuple(o[posS[leg]] for leg in T)] for o in outcomes[S]],
                dtype=np.intp)
    legs = set().union(*legsets) if legsets else set()
    return Structure(M=len(legs), legsets=legsets, full=None, outcomes=outcomes,
                     out_index=out_index, pos_in=pos_in, subsets=subsets,
                     proj_idx=proj_idx, atoms=None, consistent=None, atom_signs=None)


def _shared_marg(struct, ref, S, ci, p):
    """One tau_S from the shared base-leg product (pin conj to p), projected to all Sp<=S."""
    pl = [ref.prices((leg,))[1] for leg in S]          # P(leg = yes), shared singletons
    m = np.array([np.prod([pl[i] if o[i] else 1.0 - pl[i] for i in range(len(S))])
                  for o in struct.outcomes[S]])
    tauS = _parlay_tau(m, ci, p)
    marg = {}
    for Sp in struct.subsets[S]:
        marg[Sp] = (tauS if Sp == S else
                    np.bincount(struct.proj_idx[(S, Sp)], weights=tauS,
                                minlength=len(struct.outcomes[Sp])))
    return marg


def load_game(path):
    g = json.load(open(path))
    legs = sorted(g["realized"].keys())
    li = {mt: i + 1 for i, mt in enumerate(legs)}
    rbit = {li[mt]: (1 if g["realized"][mt] == "yes" else 0) for mt in legs}
    parlays, pmap, dropped = [], {}, 0
    for p in g["parlays"]:
        if any(mt not in li for mt, _s in p["legs"]):
            continue
        pl = sorted((li[mt], 1 if side == "yes" else 0) for mt, side in p["legs"])
        S = tuple(x[0] for x in pl)
        if len(S) > KCAP:
            dropped += 1
            continue
        pmap[p["ticker"]] = len(parlays)
        parlays.append({"S": S, "conj": tuple(x[1] for x in pl)})
    tape = []
    for t in g["trades"]:
        tk, pr = t["ticker"], t.get("price")
        if pr is None:
            continue
        pr = float(np.clip(pr, PCLIP, 1 - PCLIP))
        sz, sd = float(t.get("size", 0.0)), t.get("side")
        if t["kind"] == "base" and tk in li:
            tape.append(("base", li[tk], pr, sz, sd))
        elif tk in pmap:
            tape.append(("parlay", pmap[tk], pr, sz, sd))
    return {"suffix": g["gameSuffix"], "M": g["M"], "legs": len(legs), "rbit": rbit,
            "parlays": parlays, "tape": tape, "dropped": dropped}


def _sweep(design, books, marg, cash_lvl, base_only=False):
    """Exp 1's bottom-up buys-only sweep; accumulate cash by book level |Sp|."""
    for Sp in books:                                   # ascending by |Sp|
        if base_only and len(Sp) > 1:
            continue
        c, _, _ = match_to_target(design, Sp, marg[Sp], B)
        cash_lvl[len(Sp)] = cash_lvl.get(len(Sp), 0.0) + c


def run_game(game):
    rbit, parlays = game["rbit"], game["parlays"]
    maximal = [(leg,) for leg in rbit] + [p["S"] for p in parlays]
    struct = build_lattice(maximal)
    A, I, Bd = DesignA(struct, B), DesignC(struct, B), DesignC(struct, B)
    cA, cI, cB = {}, {}, {}                             # cash by book level, per design
    nat_base = nat_par = 0.0

    for tr in game["tape"]:
        if tr[0] == "base":
            _, leg, p, sz, sd = tr
            S = (leg,); marg = {S: np.array([1.0 - p, p])}
            _sweep(A, [S], marg, cA); _sweep(I, [S], marg, cI); _sweep(Bd, [S], marg, cB)
            w = rbit[leg] if sd == "yes" else 1 - rbit[leg]
            nat_base += sz * (w - (p if sd == "yes" else 1.0 - p))
        else:
            _, pi, p, sz, sd = tr
            S, conj = parlays[pi]["S"], parlays[pi]["conj"]
            ci = struct.out_index[S][conj]
            marg = _shared_marg(struct, A, S, ci, p)   # shared across LOPMM/ind
            _sweep(A, struct.subsets[S], marg, cA)
            _sweep(I, struct.subsets[S], marg, cI)
            # base: no parlay markets -> only genuine base trades touch it
            hit = all(rbit[leg] == conj[pos] for pos, leg in enumerate(S))
            w = hit if sd == "yes" else not hit
            nat_par += sz * (int(w) - (p if sd == "yes" else 1.0 - p))

    def win(S):
        return struct.out_index[S][tuple(rbit[leg] for leg in S)]

    def pay_lvl(design):                               # payout by book level (real outcome)
        d = {}
        for S, c in design.contracts.items():
            if c.any():
                d[len(S)] = d.get(len(S), 0.0) + c[win(S)]
        return d

    pA, pI, pB = pay_lvl(A), pay_lvl(I), pay_lvl(Bd)
    lvls = range(1, max((len(p["S"]) for p in parlays), default=1) + 1)
    a_lvl = {j: pA.get(j, 0.0) - cA.get(j, 0.0) for j in lvls}      # loss by level, LOPMM
    i_lvl = {j: pI.get(j, 0.0) - cI.get(j, 0.0) for j in lvls}      # loss by level, ind
    lopmm_t = sum(a_lvl.values()); ind_t = sum(i_lvl.values())
    base_t = sum(pB.get(j, 0.0) - cB.get(j, 0.0) for j in cB)       # base: level 1 only
    return {
        "suffix": game["suffix"], "M": game["M"], "legs": game["legs"],
        "maxK": max((len(p["S"]) for p in parlays), default=0),
        "parlayTrades": sum(1 for t in game["tape"] if t[0] == "parlay"),
        "dropped": game["dropped"],
        "base_total": base_t, "ind_total": ind_t, "lopmm_total": lopmm_t,
        "ind_parlay": ind_t - base_t, "lopmm_parlay": lopmm_t - base_t,
        "ind_by_level": i_lvl, "lopmm_by_level": a_lvl,
        "native_parlay": nat_par, "native_total": nat_base + nat_par,
    }


def main():
    os.makedirs(OUT, exist_ok=True)
    files = sorted(glob.glob(f"{DATA}/*.json"))
    if len(sys.argv) > 1:
        files = [f for f in files if any(s in f for s in sys.argv[1:])]
    rows = []
    for f in files:
        rows.append(run_game(load_game(f)))
        r = rows[-1]
        print(f"{r['suffix']:>16} M={r['M']:>3} k={r['maxK']:>2} pTr={r['parlayTrades']:>4} "
              f"drop={r['dropped']} | parlay  ind={r['ind_parlay']:9.3f} "
              f"lopmm={r['lopmm_parlay']:9.3f} | total base={r['base_total']:8.1f}", flush=True)
    json.dump(rows, open(f"{OUT}/per_game.json", "w"), indent=1)
    ap = np.array([r["lopmm_parlay"] for r in rows])
    ind = np.array([r["ind_parlay"] for r in rows])
    print(f"\n{len(rows)} games | corpus parlay P&L  ind={ind.sum():.1f}  lopmm={ap.sum():.1f}"
          f"  | LOPMM<=ind in {int((ap <= ind + 1e-9).sum())}/{len(rows)}")


if __name__ == "__main__":
    main()
