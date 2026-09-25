#!/usr/bin/env python3
"""
Paper figure: operator net loss vs M for ind / LOPMM / base.

- ind, base, LOPMM plotted as mean-over-seeds lines.
- LOPMM additionally shows the EXACT per-seed range (min..max band) across all seeds.
- Overlay c*M^2 + d, least-squares fit to LOPMM's worst-case (max-over-seeds) loss
  then raised so it strictly upper-bounds every point.

Reads per-seed net loss from the cached results in results_exp1_loworder/result_cache.
Parameterized by support / decay / seeds / M-range so it targets a specific run;
run that sweep first, e.g.:
  python3 experiment1_loworder.py --k_auto --steps_per_M 10 --rho 1 --rho_0 5 \
      --support pyramid --decay 1.25 --seeds 40 --M 2 3 4 5 6 7 8 9 10
"""
import argparse
import glob
import pickle

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
# --- paper figure typography: 1.5x matplotlib's defaults (font.size 10 -> 15) ---
plt.rcParams.update({
    "font.size": 15, "axes.titlesize": 18, "axes.labelsize": 15,
    "xtick.labelsize": 15, "ytick.labelsize": 15, "legend.fontsize": 15,
})


RC = "results_exp1_loworder/result_cache"
OUT = "results_exp1_loworder"
MODELS = ["ind", "LOPMM", "base"]
COL = {"ind": "tab:red", "LOPMM": "tab:blue", "base": "tab:green"}


def decay_tag(support, decay):
    return "" if (support != "pyramid" or decay == 2.0) else f"_dec{decay}"


def load_per_seed(support, decay, seeds, M_list, steps_per_M):
    """M -> model -> np.array of per-seed net loss, from the cached result bundles."""
    dtag = decay_tag(support, decay)
    data = {}
    for M in M_list:
        pat = (f"{RC}/resjs_M{M}_k*_{support}{dtag}_b10.0_rho1.0_r05.0_"
               f"T{steps_per_M*M}_ns2000_seeds{seeds}_bs{12345}.pkl")
        files = glob.glob(pat)
        if not files:
            print(f"  (no cache for M={M}: {pat})")
            continue
        with open(files[0], "rb") as f:
            bundle = pickle.load(f)
        per = {nm: [] for nm in MODELS}
        for row in bundle["rows"]:            # row = [M,k,model,seed,net_loss,...]
            per[row[2]].append(float(row[4]))
        data[M] = {nm: np.asarray(per[nm]) for nm in MODELS}
    return data


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--support", default="pyramid", choices=["flat", "pyramid"])
    p.add_argument("--decay", type=float, default=1.25)
    p.add_argument("--seeds", type=int, default=40)
    p.add_argument("--steps_per_M", type=int, default=10)
    p.add_argument("--Mmin", type=int, default=2)
    p.add_argument("--Mmax", type=int, default=20)
    a = p.parse_args()

    data = load_per_seed(a.support, a.decay, a.seeds, range(a.Mmin, a.Mmax + 1),
                         a.steps_per_M)
    Ms = np.array([M for M in range(a.Mmin, a.Mmax + 1) if M in data])
    if len(Ms) == 0:
        print("no cached data found for this config"); return
    mean = {nm: np.array([data[M][nm].mean() for M in Ms]) for nm in MODELS}
    amin = np.array([data[M]["LOPMM"].min() for M in Ms])
    amax = np.array([data[M]["LOPMM"].max() for M in Ms])

    Mf = Ms.astype(float)
    c, d0 = np.polyfit(Mf ** 2, amax, 1)              # amax ~ c*M^2 + d0
    resid = amax - (c * Mf ** 2 + d0)
    d = float(d0 + resid.max())                       # raise constant to a strict bound
    bnd = c * Mf ** 2 + d
    mbind = int(Ms[int(np.argmax(resid))])
    print(f"bound c*M^2 + d:  c = {c:.4g}   d = {d:.4g}   (binds at M={mbind})")

    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    ax.errorbar(Ms, mean["LOPMM"], yerr=[mean["LOPMM"] - amin, amax - mean["LOPMM"]],
                fmt="o-", color=COL["LOPMM"], lw=2, ms=4, capsize=3, elinewidth=1.1,
                label="LOPMM (mean, min–max)")
    ax.plot(Ms, mean["ind"], "s-", color=COL["ind"], lw=2, ms=4, label="ind (mean)")
    ax.plot(Ms, mean["base"], "^-", color=COL["base"], lw=1.6, ms=4,
            label="base (floor, mean)")
    ax.plot(Ms, bnd, "--", color="0.25", lw=1.8,
            label=rf"$c\,M^2+d$ bound  ($c={c:.3g},\ d={d:.3g}$)")

    ax.axhline(0, color="0.7", lw=0.8, zorder=0)
    ax.set_xticks(Ms)
    ax.set_xlabel("number of base events $M$")
    ax.set_ylabel("operator net loss")
    ax.legend(frameon=False, fontsize=13.5)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    tag = f"{a.support}{decay_tag(a.support, a.decay)}_M{a.Mmax}"
    for ext in ("png", "pdf"):
        fig.savefig(f"{OUT}/paper_net_loss_{tag}.{ext}", dpi=200)
    print(f"saved {OUT}/paper_net_loss_{tag}.png and .pdf")

    print(f"\n{'M':>3} {'ind':>9} {'LOPMM_mean':>10} {'LOPMM_min':>9} "
          f"{'LOPMM_max':>9} {'cM^2+d':>9} {'base':>8}")
    for i, M in enumerate(Ms):
        print(f"{M:>3} {mean['ind'][i]:>9.2f} {mean['LOPMM'][i]:>10.2f} "
              f"{amin[i]:>9.2f} {amax[i]:>9.2f} {bnd[i]:>9.2f} {mean['base'][i]:>8.2f}")


if __name__ == "__main__":
    main()
