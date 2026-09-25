#!/usr/bin/env python3
"""
Experiment 2 -- Robustness: net loss under parameter sweeps at fixed M=10.

Sweeps one flow parameter at a time around the Exp-1 operating point and plots net
loss (LOPMM / ind / base) vs the swept parameter, with LOPMM's exact per-seed min-max
band. See EXPERIMENT2_PLAN.md.

Axes (paper name -> code knob):
    rho   = --decay        (support count decay, sparsity)   center 1.25
    B_th  = --rho_0        (increment magnitude)             center 5.0
    R     = --steps_per_M  (re-quote frequency, T/M)         center 10
    k     = --k            (order cap)                        center 4
Fixed: M=10, pyramid, b=10, code rho=1 (flat increment), n_settle=2000, seeds=40.

Reuses the Exp-1 engine and result_cache from experiment1_loworder (each point is one
cached M=10 run), so re-runs are instant.
"""
import os
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


import experiment1_loworder as E

OUT = "results_exp2_robustness"
M = 10
COL = {"ind": "tab:red", "LOPMM": "tab:blue", "base": "tab:green"}

# Exp-1 operating point at M=10 (full cfg the engine expects)
CENTER = dict(k=4, k_auto=False, support="pyramid", decay=1.25, b=10.0,
              rho=1.0, rho_0=5.0, n_steps=100, steps_per_M=10, n_settle=2000,
              seeds=40, base_seed=12345, M_list=[M])

# (label, cfg-key, swept values, x-axis title, log-x?, x-transform for plotting)
# paper's rho = 1/decay: sweep code `decay` in 1..2.5, plot at x=1/decay, label rho.
_ID = lambda v: v
SWEEPS = [
    ("rho",  "decay",       [1.0, 1.25, 1.5, 1.75, 2.0, 2.5],      r"$\rho$", False, lambda v: 1.0 / v),
    ("Btheta", "rho_0",     [1, 2, 3, 4, 5],                       r"$B_\theta$ (magnitude)", False, _ID),
    ("R",    "steps_per_M", [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],       r"$R = T/M$ (re-quote freq.)", False, _ID),
    ("k",    "k",           [2, 3, 4, 5, 6, 7, 8],                 r"$k$ (order)", False, _ID),
]


def get_point(cfg):
    """Per-seed net loss {model: array} for one M=10 config; computes+caches on miss."""
    kM = cfg["k"]
    TM = cfg["steps_per_M"] * M
    rpath = E.result_path(M, kM, TM, cfg)
    if not os.path.exists(rpath):                      # compute + cache (mirrors E.run)
        struct = E.build_sparse_structure(M, kM)
        mc = dict(cfg, k=kM, n_steps=TM)
        acc = {nm: [] for nm, _, _ in E.MODELS}
        rows, supp, ntr = [], None, None
        for seed in range(cfg["seeds"]):
            flow = E.load_or_build_flow(struct, mc, seed)
            supp, ntr = len(flow["D"]), len(flow["trades"])
            execs = {nm: E.run_model(nm, kl, bo, struct, flow, cfg["b"])
                     for nm, kl, bo in E.MODELS}
            srng = np.random.default_rng(cfg["base_seed"] + 10_000_000 + seed)
            net = E.settle_flow(struct, execs, flow["joint_p"], cfg["n_settle"], srng)
            for nm, _, _ in E.MODELS:
                acc[nm].append(net[nm][0])
                rows.append([M, kM, nm, seed, f"{net[nm][0]:.6f}"])
        summary = {nm: {"net_loss": E.agg(acc[nm]), "support_size": supp, "k": kM}
                   for nm, _, _ in E.MODELS}
        os.makedirs(E.RESULT_DIR, exist_ok=True)
        with open(rpath, "wb") as f:
            pickle.dump({"summary": summary, "rows": rows, "supp": supp, "ntr": ntr}, f)
    with open(rpath, "rb") as f:
        bundle = pickle.load(f)
    per = {nm: [] for nm, _, _ in E.MODELS}
    for row in bundle["rows"]:
        per[row[2]].append(float(row[4]))
    return {nm: np.asarray(per[nm]) for nm in per}, bundle["supp"]


def run_sweep(key, values):
    xs, stats = [], {nm: {"mean": [], "min": [], "max": []} for nm in COL}
    supps = []
    for v in values:
        cfg = dict(CENTER); cfg[key] = v
        data, supp = get_point(cfg)
        xs.append(v); supps.append(supp)
        for nm in COL:
            stats[nm]["mean"].append(data[nm].mean())
            stats[nm]["min"].append(data[nm].min())
            stats[nm]["max"].append(data[nm].max())
        print(f"    {key}={v:<5} |D|={supp:3d}  "
              f"base={stats['base']['mean'][-1]:8.2f} "
              f"LOPMM={stats['LOPMM']['mean'][-1]:8.2f} "
              f"ind={stats['ind']['mean'][-1]:8.2f}", flush=True)
    return np.array(xs, float), stats, supps


def panel(ax, xs, stats, xlabel, logx):
    amean = np.array(stats["LOPMM"]["mean"])
    amin = np.array(stats["LOPMM"]["min"]); amax = np.array(stats["LOPMM"]["max"])
    # LOPMM mean with a min->max range line (whisker) at each point
    ax.errorbar(xs, amean, yerr=[amean - amin, amax - amean], fmt="o-",
                color=COL["LOPMM"], lw=2, ms=4, capsize=3, elinewidth=1.2,
                label="LOPMM (mean, min–max)")
    ax.plot(xs, stats["ind"]["mean"], "s-", color=COL["ind"], lw=2, ms=4, label="ind")
    ax.plot(xs, stats["base"]["mean"], "^-", color=COL["base"], lw=1.6, ms=4, label="base")
    ax.axhline(0, color="0.7", lw=0.8, zorder=0)
    if logx:
        ax.set_xscale("log"); ax.set_xticks(xs); ax.set_xticklabels([f"{v:g}" for v in xs])
    ax.set_xlabel(xlabel); ax.set_ylabel("operator net loss")
    ax.grid(alpha=0.3)


def main():
    os.makedirs(OUT, exist_ok=True)
    results = {}
    print(f"EXPERIMENT 2  robustness at M={M}  (center: rho=1.25 B_th=5 R=10 k=4)")
    for label, key, values, xlabel, logx, xf in SWEEPS:
        print(f"  sweep {label} ({key}):")
        xs, stats, _ = run_sweep(key, values)
        xplot = np.array([xf(v) for v in xs])          # e.g. rho panel plots at 1/decay
        results[label] = (xplot, stats, xlabel, logx)
        fig, ax = plt.subplots(figsize=(5.2, 4.0))
        panel(ax, xplot, stats, xlabel, logx)
        ax.legend(frameon=False, fontsize=12)
        fig.tight_layout()
        for ext in ("png", "pdf"):
            fig.savefig(f"{OUT}/exp2_sweep_{label}.{ext}", dpi=200)
        plt.close(fig)

    combined = [s for s in SWEEPS if s[0] != "R"]      # rho, B_theta, k (drop R sweep)
    fig, axes = plt.subplots(1, len(combined), figsize=(5 * len(combined), 4.2))
    for axi, (label, key, values, xlabel, logx, xf) in zip(axes, combined):
        xplot, stats, xl, lx = results[label]
        panel(axi, xplot, stats, xl, lx)
    axes[0].legend(frameon=False, fontsize=12)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(f"{OUT}/exp2_all.{ext}", dpi=200)
    print(f"saved {OUT}/exp2_all.png and per-sweep figures")


if __name__ == "__main__":
    main()
