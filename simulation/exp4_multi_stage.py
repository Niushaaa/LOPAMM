#!/usr/bin/env python3
"""
Stage the multi-game (cross-game) Kalshi parlay collection for Experiment 4 into
exp4_multi_data/staged.pkl. Parlay-only file: every trade is a parlay. We keep parlays
with all legs resolved and order k<=KCAP, mapped to integer leg indices, plus the
time-ordered trade tape (parlay-index, yesPrice, size, side) and the realized outcomes.
"""
import json
import os
import pickle

KALSHI = "/Users/niusha/Documents/Projects/LOPAMM/kalshi"
SRC = f"{KALSHI}/data/games/SPORTSMULTIJUN16.json"
OUT = "exp4_multi_data"
KCAP = 13
PCLIP = 1e-6


def main():
    os.makedirs(OUT, exist_ok=True)
    g = json.load(open(SRC))
    realized = g["realized"]
    legs = sorted(realized.keys())                 # only resolved base markets
    li = {mt: i + 1 for i, mt in enumerate(legs)}
    rbit = {li[mt]: (1 if realized[mt] == "yes" else 0) for mt in legs}

    parlays, pmap, dropped = [], {}, 0
    for p in g["parlays"]:
        L = p.get("legs", [])
        if not L or any(l["marketTicker"] not in li for l in L):
            continue
        pl = sorted((li[l["marketTicker"]], 1 if l["side"] == "yes" else 0) for l in L)
        S = tuple(x[0] for x in pl)
        if len(S) > KCAP:
            dropped += 1
            continue
        pmap[p["ticker"]] = len(parlays)
        parlays.append((S, tuple(x[1] for x in pl)))

    raw = []
    for t in g["trades"]:
        tk, pr = t["ticker"], t.get("yesPrice")
        if pr is None or tk not in pmap:
            continue
        pr = float(min(max(pr, PCLIP), 1 - PCLIP))
        raw.append((t["timestampMs"], pmap[tk], pr,
                    float(t.get("size", 0.0)), t.get("takerOutcomeSide")))
    raw.sort(key=lambda x: x[0])                    # time order
    tape = [(pi, pr, sz, sd) for _ts, pi, pr, sz, sd in raw]

    pickle.dump({"rbit": rbit, "parlays": parlays, "tape": tape,
                 "n_legs": len(legs), "dropped": dropped, "kcap": KCAP},
                open(f"{OUT}/staged.pkl", "wb"), protocol=4)
    from collections import Counter
    kc = Counter(len(S) for S, _ in parlays)
    print(f"legs(resolved)={len(legs)}  parlays(k<={KCAP})={len(parlays)}  dropped(k>{KCAP})={dropped}")
    print(f"parlay trades={len(tape)}")
    print("parlay order distribution:", dict(sorted(kc.items())))


if __name__ == "__main__":
    main()
