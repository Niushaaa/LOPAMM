#!/usr/bin/env python3
"""
Experiment 3 -- Informed-trader deal comparison at the Exp 2 center point
(M=10, joint settlement).

Both designs fill the SAME flow. Per arriving trader (full-trade on leg-set S):
    premium = sum of sub-trade cash paid
    E[payout] = sum_{S'<=S} Delta_{S'} . mu(S'),  mu(S') = marginal of final joint on S'
    p_eff = E[payout] / premium                      (effective payout-per-premium)
We report the PER-TRADER ratio  q = p_eff(LOPAMM) / p_eff(ind), broken down by parlay
order |S|. q ~ 1 means the informed trader gets the same effective deal under both
designs -- so LOPAMM's lower operator loss is a genuine design efficiency, not value
extracted from traders' pockets.

Reuses the Exp 1 engine (cached joint flows). See EXPERIMENT3_PLAN.md.
"""
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import experiment1_loworder as E
from parlay_mm_sim import DesignA, DesignC

OUT = "results_exp3_trader"
M = 15
MODELS = [("LOPAMM", DesignA), ("ind", DesignC)]

CFG = dict(k=6, k_auto=False, support="pyramid", decay=1.25, b=10.0, rho=1.0,
           rho_0=5.0, n_steps=M * 10, steps_per_M=10, n_settle=2000, seeds=100,
           base_seed=12345, M_list=[M])


def main():
    os.makedirs(OUT, exist_ok=True)
    struct = E.build_sparse_structure(M, CFG["k"])
    k, b = CFG["k"], CFG["b"]

    qbyS = {l: [] for l in range(1, k + 1)}            # per-trader p_eff_LOPAMM / p_eff_ind
    effA = {l: [] for l in range(1, k + 1)}            # per-trader p_eff (LOPAMM)
    effI = {l: [] for l in range(1, k + 1)}            # per-trader p_eff (ind)
    for seed in range(CFG["seeds"]):
        flow = E.load_or_build_flow(struct, CFG, seed)
        needed = set()                                 # only the traded books (closure of D)
        for (S, _m) in flow["trades"]:
            needed.update(struct.subsets[S])
        # marginal mu(S') of the final joint on each needed book (for E[payout])
        mu = {S: np.bincount(E._outcome_idx(struct, S), weights=flow["joint_p"],
                             minlength=len(struct.outcomes[S])) for S in needed}
        designs = {nm: klass(struct, b) for nm, klass in MODELS}    # both fill same flow
        for (S, marg) in flow["trades"]:
            rec = {}
            for nm, d in designs.items():
                premium, epay = 0.0, 0.0
                for Sp in struct.subsets[S]:
                    cash, delta, _ = E.match_to_target(d, Sp, marg[Sp], b)
                    premium += cash
                    epay += float(delta @ mu[Sp])      # expected payout
                rec[nm] = (premium, epay)
            (pA, eA), (pI, eI) = rec["LOPAMM"], rec["ind"]
            if pA > 1e-9 and pI > 1e-9 and eI > 1e-12:
                rA, rI = eA / pA, eI / pI
                qbyS[len(S)].append(rA / rI)
                effA[len(S)].append(rA)
                effI[len(S)].append(rI)
        print(f"  seed {seed} done", flush=True)
    for l in range(1, k + 1):
        qbyS[l] = np.array(qbyS[l])
        effA[l] = np.array(effA[l])
        effI[l] = np.array(effI[l])

    ls = list(range(1, k + 1))
    mean = np.array([qbyS[l].mean() if qbyS[l].size else np.nan for l in ls])
    med = np.array([np.median(qbyS[l]) if qbyS[l].size else np.nan for l in ls])
    qmin = np.array([qbyS[l].min() if qbyS[l].size else np.nan for l in ls])
    qmax = np.array([qbyS[l].max() if qbyS[l].size else np.nan for l in ls])

    # --- comparability statistics (#1 bootstrap CI on mean q, #3 fraction in +-10%) ---
    rng = np.random.default_rng(0)

    def boot_ci(x, B=2000):
        n = len(x)
        if n < 2:
            return (np.nan, np.nan)
        m = x[rng.integers(0, n, size=(B, n))].mean(axis=1)
        return tuple(np.percentile(m, [2.5, 97.5]))

    def band(x):                                   # fraction within +-10% of equal deal
        return float(np.mean((x >= 0.9) & (x <= 1.1))) if len(x) else np.nan

    qpool = np.concatenate([qbyS[l] for l in ls])
    print("\nper-trader p_eff_LOPAMM / p_eff_ind  (expected payout, no settlement):")
    print(f"  {'|S|':>4} {'mean':>7} {'CI95_lo':>8} {'CI95_hi':>8} "
          f"{'in±10%':>7} {'median':>7} {'#':>7}")
    for l in ls + ["pool"]:
        q = qpool if l == "pool" else qbyS[l]
        if not len(q):
            continue
        lo, hi = boot_ci(q)
        tag = "pool" if l == "pool" else f"{l}"
        print(f"  {tag:>4} {q.mean():>7.3f} {lo:>8.3f} {hi:>8.3f} "
              f"{band(q):>7.3f} {np.median(q):>7.3f} {len(q):>7}")

    # --- #4 per-trader correlation of p_eff (LOPAMM vs ind) ---
    from scipy.stats import spearmanr

    def corr(sel):                                 # sel = list of |S| to pool
        a = np.concatenate([effA[l] for l in sel])
        i = np.concatenate([effI[l] for l in sel])
        return float(np.corrcoef(a, i)[0, 1]), float(spearmanr(a, i).correlation), len(a)

    for name, sel in [("all |S|", ls), ("|S|>=2", [l for l in ls if l >= 2])]:
        pear, spear, n = corr(sel)
        print(f"per-trader p_eff correlation (LOPAMM vs ind), {name:>7}: "
              f"Pearson={pear:.4f}  Spearman={spear:.4f}  (n={n})")
    print("  (|S|=1 base sub-trades are identical under both designs -> exact corr)")

    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    ax.errorbar(ls, mean, yerr=[mean - qmin, qmax - mean], fmt="o-",
                color="tab:blue", lw=2, ms=5, capsize=3, elinewidth=1.1,
                label="mean (min–max over traders)")
    ax.plot(ls, med, "s--", color="0.35", lw=1.4, ms=4, label="median")
    ax.axhline(1, color="tab:red", lw=1.2, ls=":", label="equal deal (=1)")
    ax.set_xticks(ls)
    ax.set_xlabel("parlay order  |S|  (number of legs)")
    ax.set_ylabel(r"$p_{\mathrm{eff}}^{\mathrm{LOPAMM}} / p_{\mathrm{eff}}^{\mathrm{ind}}$")
    ax.legend(frameon=False)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(f"{OUT}/payout_premium_ratio_by_legs.{ext}", dpi=200)
    print(f"\nsaved {OUT}/payout_premium_ratio_by_legs.png and .pdf")


if __name__ == "__main__":
    main()
