#!/usr/bin/env python3
"""
Stage Kalshi NBA game data for Experiment 4 into ./exp4_data/ (self-contained; the
kalshi/ directory is used ONLY as a data source, never its code).

For each selected game we keep the relevant order flow -- base trades on the legs that
appear in parlays, plus every parlay trade -- in time order, together with the parlay
leg-sets and the realized base-market outcomes (settlement is real; no 2^M joint needed).
Also prints the M / k feasibility check.
"""
import json
import os
from collections import Counter

KALSHI = "/Users/niusha/Documents/Projects/LOPMM/kalshi"
GAMES = f"{KALSHI}/data/games"
OUT = "exp4_data"


def side_of(trade):
    return trade.get("takerOutcomeSide") or trade.get("takerSide")


def main():
    os.makedirs(OUT, exist_ok=True)
    sel = json.load(open(f"{KALSHI}/data/selection.json"))
    rows = []
    tot_parlay_tr = tot_base_tr = 0
    for r in sel:
        suf = r["gameSuffix"]
        path = f"{GAMES}/{suf}.json"
        if not os.path.exists(path):
            print(f"  MISSING {suf}")
            continue
        g = json.load(open(path))
        realized = g["realized"]                      # ticker -> 'yes'/'no'
        parlays = []
        active = set()
        legc = Counter()
        for p in g["parlays"]:
            legs = [[L["marketTicker"], L["side"]] for L in p["legs"]]
            parlays.append({"ticker": p["ticker"], "legs": legs,
                            "settlement": p.get("settlement")})
            legc[len(legs)] += 1
            for mt, _s in legs:
                active.add(mt)
        ptickers = {p["ticker"] for p in parlays}
        # keep base trades on active legs + all parlay trades, in time order
        kept = []
        for t in g["trades"]:
            tk, kind = t["ticker"], t.get("kind")
            if kind == "parlay" and tk in ptickers:
                kept.append([tk, "parlay", side_of(t), t["size"], t["timestampMs"],
                             t.get("yesPrice")])
            elif kind == "base" and tk in active:
                kept.append([tk, "base", side_of(t), t["size"], t["timestampMs"],
                             t.get("yesPrice")])
        kept.sort(key=lambda x: x[4])
        n_par = sum(1 for x in kept if x[1] == "parlay")
        n_base = len(kept) - n_par
        tot_parlay_tr += n_par
        tot_base_tr += n_base
        maxk = max(legc) if legc else 0
        missing_real = [mt for mt in active if mt not in realized]
        json.dump({"gameSuffix": suf, "M": g["M"], "activeLegs": len(active),
                   "maxK": maxk, "realized": {mt: realized[mt] for mt in active
                                              if mt in realized},
                   "parlays": parlays,
                   "trades": [{"ticker": x[0], "kind": x[1], "side": x[2],
                               "size": x[3], "ts": x[4], "price": x[5]} for x in kept]},
                  open(f"{OUT}/{suf}.json", "w"))
        rows.append((suf, g["M"], len(active), maxk, n_par, n_base,
                     dict(sorted(legc.items())), len(missing_real)))

    rows.sort(key=lambda x: x[1])
    print(f"\n{'game':>16} {'M':>4} {'actLegs':>7} {'maxK':>4} "
          f"{'parlayTr':>8} {'baseTr':>8} {'missReal':>8}")
    for suf, M, al, mk, npar, nbase, legc, miss in rows:
        print(f"{suf:>16} {M:>4} {al:>7} {mk:>4} {npar:>8} {nbase:>8} {miss:>8}")
    print(f"\ngames staged: {len(rows)}   parlay trades: {tot_parlay_tr}   "
          f"active-leg base trades: {tot_base_tr}")
    print(f"max M={max(r[1] for r in rows)}  max activeLegs={max(r[2] for r in rows)}  "
          f"max maxK={max(r[3] for r in rows)}  "
          f"max trades/game={max(r[4]+r[5] for r in rows)}")
    allk = Counter()
    for r in rows:
        for kk, c in r[6].items():
            allk[kk] += c
    print("parlay leg-count distribution (markets):", dict(sorted(allk.items())))


if __name__ == "__main__":
    main()
