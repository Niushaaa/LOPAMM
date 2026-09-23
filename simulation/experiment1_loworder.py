#!/usr/bin/env python3
"""
Experiment 1 -- Loss scaling under synthetic LOW-ORDER informed flow.
====================================================================
Tests whether realized operator net loss scales as the theory predicts:
    LOPAMM  stays pinned near the base floor,
    ind   grows toward the theoretical 2^M reference,
    base  = absolute lower bound (M independent binary LMSRs, parlays unavailable).

Everything mechanical is imported UNCHANGED from parlay_mm_sim.py; this file only
adds (i) a low-order latent + one-trade-at-a-time flow generator, (ii) a
confined bottom-up executor, and (iii) the run/report layer. See
EXPERIMENT1_PLAN.md for the full spec.

Canonical parameters are scalar Walsh coefficients theta_S (one per leg-set),
anchored at the all-ones corner (atom_signs are +-1 with omega_i=1 -> +1). A
k-order belief change is exact: only theta_S with |S|<=k are ever written, so
theta_S == 0 for |S|>k at every step. Each trade picks one leg-set S from the
support, bumps only theta_S by a bounded increment, and hands the operator the
marginal of the NEW distribution on S. Execution is confined to S's sub-lattice.
"""
import argparse
import itertools
import math
import os
import pickle

import numpy as np

from parlay_mm_sim import (
    DesignA, DesignC, match_to_target,
    entropy_drop_by_level, T_CLIP, Structure,
)

OUTDIR = "results_exp1_loworder"
FLOW_DIR = f"{OUTDIR}/flow_cache"
RESULT_DIR = f"{OUTDIR}/result_cache"      # per-M aggregated results (skip recompute)
CHECK_ROUTE = False        # opt-in eq.route guard for LOPAMM sub-trades (--check_route)


def _decay_tag(cfg):
    """Cache-key suffix for a non-default pyramid decay (keeps default-decay caches)."""
    d = cfg.get("decay", 2.0)
    return "" if (cfg["support"] != "pyramid" or d == 2.0) else f"_dec{d}"


def result_path(M, kM, TM, cfg):
    """Cache path for one M's aggregated result, keyed by everything that affects it."""
    return (f"{RESULT_DIR}/resjs_M{M}_k{kM}_{cfg['support']}{_decay_tag(cfg)}_b{cfg['b']}_"
            f"rho{cfg['rho']}_r0{cfg['rho_0']}_T{TM}_ns{cfg['n_settle']}_"
            f"seeds{cfg['seeds']}_bs{cfg['base_seed']}.pkl")

# model name -> (design class, base_only?)
MODELS = (
    ("LOPAMM", DesignA, False),
    ("ind",  DesignC, False),
    ("base", DesignC, True),
)


def agg(vals):
    """mean, std, 95% CI half-width over a list of per-seed values."""
    a = np.asarray(vals, float)
    n = len(a)
    ci = 1.96 * a.std(ddof=1) / np.sqrt(n) if n > 1 else 0.0
    return a.mean(), (a.std(ddof=1) if n > 1 else 0.0), ci


def kl(p, q):
    """KL(p || q) in nats; q clipped to stay positive (LMSR softmax)."""
    q = np.clip(q, 1e-300, 1.0)
    mask = p > 0
    return float(np.sum(p[mask] * np.log(p[mask] / q[mask])))


# ---------------------------------------------------------------- sparse structure
def build_sparse_structure(M, k):
    """Like parlay_mm_sim.build_structure but holds ONLY leg-sets of size <= k
    (the flow never touches higher orders; ind is clipped at level k), while
    keeping the full 2^M atom space for the joint P. The top/full leg-set is NOT a
    market. This makes large M feasible: O(M^k) leg-sets vs 2^M-1."""
    legs = list(range(1, M + 1))
    legsets = [c for l in range(1, k + 1) for c in itertools.combinations(legs, l)]
    full = tuple(legs)

    outcomes, out_index, pos_in = {}, {}, {}
    for S in legsets:
        outs = list(itertools.product([0, 1], repeat=len(S)))
        outcomes[S] = outs
        out_index[S] = {o: i for i, o in enumerate(outs)}
        pos_in[S] = {leg: i for i, leg in enumerate(S)}

    subsets, proj_idx = {}, {}
    for S in legsets:
        subs = [c for l in range(1, len(S) + 1)
                for c in itertools.combinations(S, l)]
        subsets[S] = subs
        for T in subs:
            posS = pos_in[S]
            proj_idx[(S, T)] = np.array(
                [out_index[T][tuple(o[posS[leg]] for leg in T)]
                 for o in outcomes[S]], dtype=np.intp)

    n_atoms = 1 << M                                       # full 2^M joint space
    ar = np.arange(n_atoms, dtype=np.int64)
    shifts = np.arange(M - 1, -1, -1, dtype=np.int64)      # big-endian: leg 1 = MSB
    atom_signs = (2 * ((ar[:, None] >> shifts[None, :]) & 1) - 1).astype(np.int8)
    # `consistent` is NOT built (O(M^k * 2^M) memory); marginals use bincount instead.
    return Structure(M=M, legsets=legsets, full=full, outcomes=outcomes,
                     out_index=out_index, pos_in=pos_in, subsets=subsets,
                     proj_idx=proj_idx, atoms=ar, consistent={},
                     atom_signs=atom_signs)


# ---------------------------------------------------------------- support
def pick_support(M, k, mode, rng, decay=2.0):
    """Changing support D (theta_S != 0 only for S in D, |S|<=k), with |D|=O(M).
    Always includes the M singletons; higher levels per `mode` (default 'flat'):
      flat    -- pool all level 2..k, pick 4M uniformly at random.
      pyramid -- per level l: round(2M / decay^(l-2)) sets (2M pairs at l=2, then
                 the count decays by factor `decay` each level up)."""
    legs = list(range(1, M + 1))
    D = [(i,) for i in legs]                          # level 1: all singletons
    if mode == "flat":
        pool = [c for l in range(2, k + 1)
                for c in itertools.combinations(legs, l)]
        n = min(len(pool), 4 * M)
        D.extend(pool[j] for j in sorted(rng.choice(len(pool), size=n,
                                                     replace=False)))
    elif mode == "pyramid":
        for l in range(2, k + 1):
            combos = list(itertools.combinations(legs, l))
            n_l = min(len(combos), max(1, round(2 * M / (decay ** (l - 2)))))
            D.extend(combos[j] for j in sorted(rng.choice(len(combos), size=n_l,
                                                          replace=False)))
    else:
        raise ValueError(f"unknown support_mode {mode!r}")
    return D


# ---------------------------------------------------------------- flow
def _chi(struct, S):
    """chi_S over all 2^M atoms:  chi_S(w) = prod_{i in S} s_i(w)  (+-1)."""
    return np.prod(struct.atom_signs[:, [i - 1 for i in S]], axis=1).astype(np.int8)


def _outcome_idx(struct, S):
    """Big-endian outcome index of every atom under leg-set S (no `consistent`)."""
    idx = np.zeros(struct.atom_signs.shape[0], dtype=np.intp)
    for leg in S:
        idx = idx * 2 + (struct.atom_signs[:, leg - 1] > 0)
    return idx


def marginal_from_Q(Q, struct, S, b):
    """Marginal of softmax(Q/b) onto leg-set S, computed DIRECTLY from the potential
    Q -- only the |S|-outcome marginal is formed, the full 2^M joint P is never
    materialized. (bincount(exp-weights)/sum == bincount(softmax), so this is exact.)"""
    z = Q / b
    w = np.exp(z - z.max())                            # unnormalized atom weights
    bins = np.bincount(_outcome_idx(struct, S), weights=w,
                       minlength=len(struct.outcomes[S]))
    return bins / bins.sum()


def build_low_order_flow(struct, k, mode, rho, rho_0, b, n_steps, seed, base_seed,
                         decay=2.0):
    """Generate one low-order flow. Each STEP updates the whole distribution (a
    bounded random-walk increment on EVERY support leg-set), then emits a batch of
    trades by revealing the marginal of the new distribution on a random subset of
    the changed leg-sets. The selection ratio ramps linearly from 0.5 at the first
    step to 1.0 at the last, so a step yields |D|/2 .. |D| trades. The latent walk
    is UNBOUNDED (no convergence). Settlement is not part of the artifact.

    Per-level increment bound: for a leg-set S of level |S| the increment obeys
    |dtheta_S| <= rho^|S| * rho_0 (rho<1), so higher-order updates shrink."""
    rng = np.random.default_rng(base_seed + seed)
    D = pick_support(struct.M, k, mode, rng, decay)
    assert all(len(S) <= k for S in D), "support violates |S|<=k"
    Dlist = list(D)
    nD = len(Dlist)
    chi = {S: _chi(struct, S) for S in Dlist}          # chi_S per support set (+-1)
    bound = {S: rho ** len(S) * rho_0 for S in Dlist}  # level-dependent |dtheta|

    Q = np.zeros(len(struct.atoms))                    # potential over atoms
    trades = []
    for t in range(n_steps):                           # one theta change -> one trade
        S = Dlist[rng.integers(nD)]                    # single leg-set changes this step
        dtheta = float(rng.uniform(-bound[S], bound[S]))
        assert abs(dtheta) <= bound[S] + 1e-12
        Q += dtheta * chi[S]
        marg_top = marginal_from_Q(Q, struct, S, b)    # only S's marginal, no full P
        marg = {}
        for Sp in struct.subsets[S]:                   # sub-marginals from marg_top (cheap)
            m = np.zeros(len(struct.outcomes[Sp]))
            np.add.at(m, struct.proj_idx[(S, Sp)], marg_top)
            marg[Sp] = m
        trades.append((S, marg))                       # one trade on the changed S

    # final CORRELATED joint over the 2^M atoms -- the settlement law
    z = Q / b
    w = np.exp(z - z.max())
    joint_p = w / w.sum()
    return dict(trades=trades, joint_p=joint_p, D=D, k=k, rho=rho, rho_0=rho_0,
                b=b, n_steps=n_steps, seed=seed)


def load_or_build_flow(struct, cfg, seed):
    key = f"flowjs_M{struct.M}_k{cfg['k']}_{cfg['support']}{_decay_tag(cfg)}_b{cfg['b']}_" \
          f"rho{cfg['rho']}_r0{cfg['rho_0']}_steps{cfg['n_steps']}_seed{seed}.pkl"
    path = f"{FLOW_DIR}/{key}"
    if os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    flow = build_low_order_flow(struct, cfg["k"], cfg["support"], cfg["rho"],
                                cfg["rho_0"], cfg["b"], cfg["n_steps"], seed,
                                cfg["base_seed"], cfg.get("decay", 2.0))
    os.makedirs(FLOW_DIR, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(flow, f)
    return flow


# ---------------------------------------------------------------- execution
def _assert_route(design, struct, Sp, tau_Sp, b):
    """Guard (LOPAMM only): right after a sub-trade on Sp, residual[Sp] must equal
    the eq.route canonical value  b*log(tau_Sp) - sum_{T c Sp} r^{(T)}  up to an
    additive constant (the buys-only gauge). Equivalently market Sp quotes tau_Sp."""
    lower = np.zeros(len(struct.outcomes[Sp]))
    for T in struct.subsets[Sp]:                       # proper non-empty subsets
        if T != Sp:
            lower += design.residual[T][struct.proj_idx[(Sp, T)]]
    canonical = b * np.log(np.clip(tau_Sp, T_CLIP, 1.0)) - lower
    diff = design.residual[Sp] - canonical
    spread = float(diff.max() - diff.min())
    tol = 1e-6 * max(1.0, float(np.abs(canonical).max()))
    assert spread < tol, (f"eq.route violated on {Sp}: residual - canonical not "
                          f"constant (spread={spread:.2e}, tol={tol:.2e})")


def sweep_books(design, struct, books, marg, b, base_only, track):
    """Re-quote the given books (bottom-up) to the current true marginal.
    base_only -> singletons only."""
    cash = 0.0
    for Sp in books:                                   # ascending by |Sp|
        if base_only and len(Sp) > 1:
            continue
        tau_Sp = marg[Sp]
        quoted = design.prices(Sp)                     # PRE-match quote
        track[0] += kl(tau_Sp, quoted)
        track[1] += float(np.abs(tau_Sp - quoted).sum())
        track[2] += 1
        c, _delta, _pre = match_to_target(design, Sp, tau_Sp, b)
        cash += c
        if CHECK_ROUTE and hasattr(design, "residual"):
            _assert_route(design, struct, Sp, tau_Sp, b)
    return cash


def run_model(name, klass, base_only, struct, flow, b):
    """Execute the flow on one model. Returns cash + contracts (for settlement)
    and the settlement-independent metrics. Net loss is computed later in
    settle_flow, from shared base-market draws."""
    design = klass(struct, b)
    total_cash = 0.0
    track = [0.0, 0.0, 0]                               # sum_kl, sum_l1, count
    for (S, marg) in flow["trades"]:                    # only S's sub-lattice per trade
        total_cash += sweep_books(design, struct, struct.subsets[S], marg,
                                  b, base_only, track)

    ent_total, ent_level = entropy_drop_by_level(design, struct)
    incons = None                                      # needs struct.consistent (not built)
    cnt = max(track[2], 1)
    return dict(
        cash=total_cash, contracts=design.contracts,
        ent_total=ent_total, ent_level=ent_level,
        kl=track[0] / cnt, l1=track[1] / cnt, incons=incons,
        n_subtrades=track[2],
    )


def settle_flow(struct, execs, joint_p, n_settle, rng):
    """Monte-Carlo expected net loss. Each instance draws a realized atom from the
    final CORRELATED joint (joint_p over the 2^M atoms); that atom settles EVERY
    book (parlays via projecting the realized outcome onto their legs). The same
    draws are used for all models. Returns name -> (mean_net_loss, std_over_draws)."""
    M = struct.M
    names = list(execs)
    # only books actually held (nonzero contracts) contribute to payout
    nz = {nm: [S for S in struct.legsets if execs[nm]["contracts"][S].any()]
          for nm in names}
    atoms = rng.choice(len(joint_p), size=n_settle, p=joint_p)   # correlated draws
    s = {nm: 0.0 for nm in names}
    ss = {nm: 0.0 for nm in names}
    for a in atoms:
        a = int(a)
        omega = [(a >> (M - i)) & 1 for i in range(1, M + 1)]    # atom -> base outcome
        for nm in names:
            c = execs[nm]["contracts"]
            pay = sum(c[S][struct.out_index[S][tuple(omega[leg - 1] for leg in S)]]
                      for S in nz[nm])
            nl = pay - execs[nm]["cash"]
            s[nm] += nl
            ss[nm] += nl * nl
    out = {}
    for nm in names:
        mean = s[nm] / n_settle
        out[nm] = (mean, max(0.0, ss[nm] / n_settle - mean * mean) ** 0.5)
    return out


# ---------------------------------------------------------------- driver
def run(cfg):
    os.makedirs(OUTDIR, exist_ok=True)
    import csv
    per_seed_f = open(f"{OUTDIR}/per_seed.csv", "w", newline="")
    per_seed_w = csv.writer(per_seed_f)
    per_seed_w.writerow(["M", "k", "model", "seed", "net_loss", "net_loss_std",
                         "cash", "ent_total", "kl_track", "l1_track", "incons",
                         "n_subtrades", "support_size"])

    summary = {}                                       # (M, name) -> agg dict
    print("=" * 88)
    kdesc = "ceil(sqrt M)" if cfg.get("k_auto") else cfg["k"]
    tdesc = f"{cfg['steps_per_M']}*M" if cfg.get("steps_per_M") else cfg["n_steps"]
    print("EXPERIMENT 1  (low-order informed flow)   LOPAMM / ind / base")
    sup = cfg['support'] + (f"(decay={cfg['decay']})" if cfg['support'] == "pyramid"
                            else "")
    print(f"k={kdesc} support={sup} b={cfg['b']} "
          f"rho={cfg['rho']} rho_0={cfg['rho_0']} (|dtheta_k|<=rho^k*rho_0) "
          f"T={tdesc} (1 trade/step) n_settle={cfg['n_settle']} "
          f"seeds={cfg['seeds']} M={cfg['M_list']}")
    print("=" * 88)

    def mline(tag, M, kM, supp, ntr):
        print(f"  M={M} k={kM} |D|={supp:3d} trades={ntr:6d} {tag} "
              f"base={summary[(M,'base')]['net_loss'][0]:9.3f}  "
              f"LOPAMM={summary[(M,'LOPAMM')]['net_loss'][0]:9.3f}  "
              f"ind={summary[(M,'ind')]['net_loss'][0]:9.3f}", flush=True)

    for M in cfg["M_list"]:
        kM = max(1, math.ceil(math.sqrt(M))) if cfg.get("k_auto") else cfg["k"]
        TM = cfg["steps_per_M"] * M if cfg.get("steps_per_M") else cfg["n_steps"]
        rpath = result_path(M, kM, TM, cfg)
        if os.path.exists(rpath):                      # cached result -> skip recompute
            with open(rpath, "rb") as f:
                bundle = pickle.load(f)
            for nm, _, _ in MODELS:
                summary[(M, nm)] = bundle["summary"][nm]
            for row in bundle["rows"]:
                per_seed_w.writerow(row)
            per_seed_f.flush()
            mline("[cached]", M, kM, bundle["supp"], bundle["ntr"])
            continue

        mc = dict(cfg, k=kM, n_steps=TM)               # per-M config (k, T depend on M)
        print(f"  [running M={M} k={kM} T={TM} x {cfg['seeds']} seeds ...]",
              flush=True)
        struct = build_sparse_structure(M, kM)         # only leg-sets of size <= kM
        acc = {nm: {kk: [] for kk in
                    ["net_loss", "cash", "ent_total", "kl", "l1"]}
               for nm, _, _ in MODELS}
        incacc = {nm: [] for nm, _, _ in MODELS}
        supp_size = ntrades = None
        rows = []
        for seed in range(cfg["seeds"]):
            flow = load_or_build_flow(struct, mc, seed)
            supp_size = len(flow["D"])
            ntrades = len(flow["trades"])
            execs = {nm: run_model(nm, klass, base_only, struct, flow, cfg["b"])
                     for nm, klass, base_only in MODELS}
            srng = np.random.default_rng(cfg["base_seed"] + 10_000_000 + seed)
            net = settle_flow(struct, execs, flow["joint_p"], cfg["n_settle"], srng)
            for nm, _, _ in MODELS:
                ex = execs[nm]
                acc[nm]["net_loss"].append(net[nm][0])
                acc[nm]["cash"].append(ex["cash"])
                acc[nm]["ent_total"].append(ex["ent_total"])
                acc[nm]["kl"].append(ex["kl"])
                acc[nm]["l1"].append(ex["l1"])
                if ex["incons"] is not None:
                    incacc[nm].append(ex["incons"])
                row = [M, kM, nm, seed, f"{net[nm][0]:.6f}",
                       f"{net[nm][1]:.6f}", f"{ex['cash']:.6f}",
                       f"{ex['ent_total']:.6f}", f"{ex['kl']:.6f}",
                       f"{ex['l1']:.6f}",
                       "" if ex["incons"] is None else f"{ex['incons']:.6f}",
                       ex["n_subtrades"], supp_size]
                rows.append(row)
                per_seed_w.writerow(row)
            per_seed_f.flush()

        msum = {}
        for nm, _, _ in MODELS:
            s = {kk: agg(acc[nm][kk]) for kk in acc[nm]}
            s["incons"] = agg(incacc[nm]) if incacc[nm] else (float("nan"),) * 3
            s["support_size"] = supp_size
            s["k"] = kM
            summary[(M, nm)] = s
            msum[nm] = s
        os.makedirs(RESULT_DIR, exist_ok=True)
        with open(rpath, "wb") as f:
            pickle.dump({"summary": msum, "rows": rows, "supp": supp_size,
                         "ntr": ntrades}, f)
        mline("", M, kM, supp_size, ntrades)

    per_seed_f.close()
    write_outputs(cfg, summary)
    return summary


def write_outputs(cfg, summary):
    M_list = cfg["M_list"]
    lines = []

    def emit(s=""):
        lines.append(s)
        print(s)

    emit()
    emit("=" * 92)
    emit("SUMMARY  net loss (mean +/- std, 95% CI)   [base=floor]")
    emit("=" * 92)
    hdr = f"{'M':>2} {'|D|':>4} {'model':>6} | {'net_loss':>12} {'95%CI':>8} | " \
          f"{'KL_track':>9} | {'incons':>9}"
    emit(hdr)
    emit("-" * len(hdr))
    for M in M_list:
        for nm, _, _ in MODELS:
            s = summary[(M, nm)]
            nl = s["net_loss"]
            inc = "n/a" if np.isnan(s["incons"][0]) else f"{s['incons'][0]:.4f}"
            emit(f"{M:>2} {s['support_size']:>4} {nm:>6} | {nl[0]:>12.4f} "
                 f"{nl[2]:>8.4f} | {s['kl'][0]:>9.5f} | {inc:>9}")
        emit("-" * len(hdr))

    # H1-H4 verdicts
    emit()
    gap = [summary[(M, "ind")]["net_loss"][0] - summary[(M, "LOPAMM")]["net_loss"][0]
           for M in M_list]
    floor_ok = all(summary[(M, "base")]["net_loss"][0]
                   <= summary[(M, "LOPAMM")]["net_loss"][0] + 1e-6
                   <= summary[(M, "ind")]["net_loss"][0] + 1e-6 for M in M_list)
    kl_ok = all(summary[(M, "LOPAMM")]["kl"][0]
                <= 1.10 * summary[(M, "ind")]["kl"][0] for M in M_list)
    emit(f"  H1 LOPAMM near floor, gap(ind-LOPAMM) widens: {[round(g,2) for g in gap]}"
         f" -> {'YES' if gap[-1] > gap[0] else 'NO'}")
    emit(f"  H3 KL(LOPAMM) <~ KL(ind): {'YES' if kl_ok else 'NO'}")
    emit(f"  H4 base <= LOPAMM <= ind pointwise: {'YES' if floor_ok else 'NO'}")

    with open(f"{OUTDIR}/results.txt", "w") as fh:
        fh.write("\n".join(lines) + "\n")

    _write_summary_csv(cfg, summary)
    try:
        make_plots(cfg, summary)
    except Exception as e:                             # plotting is best-effort
        print(f"(plot skipped: {e})")


def _write_summary_csv(cfg, summary):
    import csv
    with open(f"{OUTDIR}/summary_aggregate.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["M", "k", "model", "support_size", "net_loss_mean",
                    "net_loss_std", "net_loss_ci95",
                    "kl_track_mean", "incons_mean"])
        for M in cfg["M_list"]:
            for nm, _, _ in MODELS:
                s = summary[(M, nm)]
                w.writerow([M, s["k"], nm, s["support_size"],
                            f"{s['net_loss'][0]:.6f}", f"{s['net_loss'][1]:.6f}",
                            f"{s['net_loss'][2]:.6f}",
                            f"{s['kl'][0]:.6f}",
                            "" if np.isnan(s['incons'][0]) else
                            f"{s['incons'][0]:.6f}"])


def make_plots(cfg, summary):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    M_list = cfg["M_list"]
    colors = {"LOPAMM": "tab:blue", "ind": "tab:red", "base": "tab:green"}

    fig, ax = plt.subplots(figsize=(7, 5))
    for nm, _, _ in MODELS:
        y = [summary[(M, nm)]["net_loss"][0] for M in M_list]
        ci = [summary[(M, nm)]["net_loss"][2] for M in M_list]
        ax.errorbar(M_list, y, yerr=ci, marker="o", capsize=3,
                    color=colors[nm], label=nm)
    ax.axhline(0, color="0.7", lw=0.8)                 # net loss can be negative
    ax.set_xlabel("M (base events)")
    ax.set_ylabel("net operator loss (mean, log)")
    ax.set_title("Net loss vs M")
    ax.legend()
    ax.grid(True, which="both", alpha=.3)
    fig.tight_layout()
    fig.savefig(f"{OUTDIR}/plot_net_loss_lowM.png", dpi=130)
    print(f"\nPlot saved to {OUTDIR}/plot_net_loss_lowM.png")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--smoke", action="store_true",
                   help="quick check: 100 trades, M in {2,3,4}, 1 seed")
    p.add_argument("--check_route", action="store_true",
                   help="assert eq.route holds after each LOPAMM sub-trade")
    p.add_argument("--M", type=int, nargs="+", default=[2, 3, 4, 5, 6, 7, 8])
    p.add_argument("--k", type=int, default=2)
    p.add_argument("--k_auto", action="store_true",
                   help="set k = ceil(sqrt M) per M")
    p.add_argument("--support", default="pyramid", choices=["flat", "pyramid"])
    p.add_argument("--decay", type=float, default=1.25,
                   help="pyramid per-level count decay: n_l = 2M / decay^(l-2)")
    p.add_argument("--b", type=float, default=10.0)
    p.add_argument("--rho", type=float, default=1.0,
                   help="theta-increment per-level decay (retired; keep 1 = flat |dtheta|<=rho_0)")
    p.add_argument("--rho_0", type=float, default=10.0,
                   help="increment scale constant rho_0")
    p.add_argument("--n_steps", type=int, default=500,
                   help="latent update steps (= trades); one theta change per step")
    p.add_argument("--steps_per_M", type=int, default=10,
                   help="if >0, set n_steps = steps_per_M * M per M (e.g. 10 -> T=10M)")
    p.add_argument("--n_settle", type=int, default=2000,
                   help="Monte-Carlo base-market settlement draws")
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--base_seed", type=int, default=12345)
    return p.parse_args()


def main():
    a = parse_args()
    global CHECK_ROUTE
    CHECK_ROUTE = a.check_route
    cfg = dict(k=a.k, k_auto=a.k_auto, support=a.support, decay=a.decay, b=a.b,
               rho=a.rho, rho_0=a.rho_0, n_steps=a.n_steps, steps_per_M=a.steps_per_M,
               n_settle=a.n_settle, seeds=a.seeds, base_seed=a.base_seed,
               M_list=a.M)
    if a.smoke:
        cfg.update(n_steps=20, n_settle=500, M_list=[2, 3, 4], seeds=1)
    run(cfg)


if __name__ == "__main__":
    main()
