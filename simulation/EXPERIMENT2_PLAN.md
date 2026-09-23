# Experiment 2 — Robustness: net loss under parameter sweeps

**Goal.** Stress-test the mechanism by sweeping the flow parameters (order, sparsity,
magnitude of belief updates) and showing LOPAMM's net loss stays low/bounded while ind
responds — i.e. the Exp 1 result is robust across the operating regime, not a
single-point artifact. Fix **M=10** (fast) and sweep **one** parameter at a time around
the Exp 1 operating point.

Reuses the Exp 1 engine unchanged (`experiment1_loworder.py`:
`build_sparse_structure`, `build_low_order_flow`, `sweep_books`/`run_model`,
`settle_flow`, and the per-M `result_cache`). New driver: `experiment2_robustness.py`.

> **Pending confirmation:** the exact ranges below (§3). Mapping is settled.

---

## 1. Parameter mapping (paper name → Exp 1 code knob)

| paper (Exp 2) | code flag | controls | Exp 1 value @ M=10 |
|---|---|---|---|
| **ρ** | `--decay` | pyramid per-level support **count decay** (sparsity): `n_l = 2M/ρ^{l−2}` | **1.25** |
| **B_θ** | `--rho_0` | θ-increment **magnitude** scale (`|Δθ_S| ≤ B_θ`, flat) | **5.0** |
| **R** | `T/M` = `--steps_per_M` | **re-quote frequency** — trades per M | **10** (T=10M) |
| **k** | `--k` | **order** cap (highest leg-set size that changes) | **4** (=⌈√10⌉) |

The code's `--rho` (θ-increment per-level decay) is **fixed at 1** — flat increment,
retired as a knob. The four axes are **order** (k), **sparsity** (ρ), **magnitude**
(B_θ), and **re-quote frequency** (R), the last being the leak/mechanism axis (§5).

---

## 2. Fixed configuration (all sub-experiments)

`M=10`, `support=pyramid`, `b=10`, `T=10M=100` (`--steps_per_M 10`), `n_settle=2000`,
code `--rho=1` (θ-increment flat, retired), `seeds=40`, `base_seed=12345`. Settlement is
the Exp 1 correlated-joint Monte-Carlo (`ω ~ Categorical(joint_p)`). Center point
(Exp 1 @ M=10): `k=4`, `ρ=1.25`, `B_θ=5`, `R=10` (T=10M). When sweeping one knob, the
other three are held at these values.

---

## 3. Sub-experiments (sweep one, hold the rest)

| sweep | code | center | proposed values |
|---|---|---|---|
| **ρ** (sparsity / support decay) | `--decay` | 1.25 | 0.1, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5 |
| **B_θ** (magnitude) | `--rho_0` | 5.0 | 1, 2, 3, 4, 5 |
| **R** (re-quote frequency, T/M) | `--steps_per_M` | 10 | 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 |
| **k** (order) | `--k` (no `--k_auto`) | 4 | 2, 3, 4, 5, 6, 7, 8 |

Notes:
- **ρ<1** (support decay) grows the higher-order counts (`×1/ρ` per level, capped at
  `C(M,l)`); at **ρ=0.1** the counts saturate to `C(M,l)` so |D| ≈ all `≤k` leg-sets
  (~385 at k=4) — probes the explosion. **ρ=2** halves each level.
- Ranges kept near/inside the low-order regime (`k=O(1)`, bounded magnitude): `B_θ≤5`,
  `R≤10`, `k≤8` — beyond these LOPAMM's buys-only leak dominates.
- The LOPAMM curve is drawn as **mean with a min→max range whisker** at each point (not a
  shaded band).
- **R = T/M** is the re-quote frequency: at low R (few trades) LOPAMM's buys-only leak is
  negligible; it grows with R. This is the mechanism axis (§5).
- **k** is swept with a fixed `--k` (not `--k_auto`); `k≤M`.

---

## 4. Outputs

For each swept parameter, a figure **net loss vs the swept parameter** with three curves
(LOPAMM / ind / base means) plus LOPAMM's **exact per-seed min–max band** — the Exp 1 figure
with the x-axis changed from M to the parameter. Written to
`results_exp2_robustness/exp2_sweep_<param>.png/.pdf`, plus a combined 2×2 panel
`exp2_all.png` for the paper.

**Cache reuse:** every sweep point is just an Exp 1 run at M=10 with one knob changed,
and the `result_cache` key already encodes every knob — so each point caches separately
and re-runs load instantly. The center point (`ρ=1.25,B_θ=5,R=10,k=4`) is shared across
all four sweeps (already cached from Exp 1).

---

## 5. Success criteria (robustness)

- **LOPAMM robust to order / sparsity / magnitude:** stays low/flat (near or below the base
  floor) across the **ρ, B_θ, k** sweeps — the headline robustness claim.
- **R (frequency) is the boundary:** LOPAMM's net loss **grows with R** (more re-quotes ⇒
  more buys-only leak) while ind is ~R-insensitive — this exposes the mechanism and
  justifies the low-order/few-trade (`T=10M`) operating point.
- **ind responds** to the low-order axes: grows with `k↑` (more orders to independently
  price), with `ρ↓` (slower support decay → more high-order markets), with `B_θ↑` (bigger
  belief updates); ~flat in R.
- **base** responds to `B_θ` (base increment ≤ B_θ) and to `R` (more re-quotes ratchet the
  base books); ~flat in ρ and k (it only ever trades singletons).
- **gap `ind−LOPAMM`** stays wide across ρ, B_θ, k; it narrows only at large R (the leak).

---

## 6. Implementation

`experiment2_robustness.py`:
1. For each of the 4 sweeps, for each value: run the Exp 1 engine at M=10 with that knob
   set (others at center) — `run(cfg)` with `M_list=[10]` — which computes-or-loads the
   cached result.
2. Read per-seed net loss (LOPAMM/ind/base) from the `result_cache` bundle for each point.
3. Plot net loss vs the swept parameter (means + LOPAMM min–max band), one panel per sweep.

Determinism/caching identical to Exp 1. Runtime: ~1 min/point at M=10/40 seeds; ~4×6 ≈
24 points → a few minutes total, then instant on re-plot.

---

## 7. Suggested additions for the paper

Beyond the four sweeps, these would strengthen the evaluation section:

1. **Re-quote-frequency ablation — now the R sweep (§3), good.** Frame R=T/M as the
   mechanism axis in the paper: LOPAMM bounded at low R, leaking as R grows, ind
   R-insensitive — this is what justifies the `T=10M` operating point. Make sure the R
   panel is discussed as *mechanism*, not just another robustness knob.
2. **Price-accuracy panel (H3).** Show LOPAMM isn't buying its low loss with worse prices:
   KL(quoted ‖ true marginal) vs M (or vs the swept param). If `KL(LOPAMM) ≲ KL(ind)`, it
   rules out "LOPAMM just mis-quotes." One small figure.
3. **Informed-trader-profit histogram.** The trader-profit ↔ operator-loss identity is a
   clean incentive story: the informed trader extracts a fat right tail against ind and a
   much tighter distribution against LOPAMM. A histogram figure makes "LOPAMM is robust to
   informed order flow" visceral.
4. **Support-construction robustness.** Repeat the Exp 1 M-sweep for **both** `pyramid`
   and `flat` (already supported) as a supplementary, to show the separation isn't an
   artifact of one support rule.
5. **Parameter-count / subsidy comparison.** LOPAMM stores `3^M−1` shared parameters (one
   per partial assignment) that every higher market reuses, vs ind's independent
   per-book parameters. A short table/plot of "operator-subsidized parameters" and the
   resulting loss makes the sharing → efficiency argument concrete.
6. **Tighten stats for the final plots.** Bump `seeds` to ~100 for the headline M-sweep
   and Exp 2 to smooth the large-M wobble and report 95% CIs everywhere.
7. **Theory overlay.** If the paper proves a loss bound, overlay its constant instead of
   the empirical `c·M²+d` fit — turning the figure into a theorem-validation.
8. **Exp 3 (Kalshi) linkage.** State explicitly that Exp 2's robustness ranges bracket
   the parameter regime that historical Kalshi flow lands in (measured in Exp 3), tying
   the synthetic and real-data results together.
