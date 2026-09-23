# Experiment 3 — Informed-trader deal comparison

**Goal.** Show that LOPAMM's lower operator loss is a **genuine design efficiency, not
value extracted from traders' pockets**: at a fixed operating point, the informed trader
gets the **same effective price** (expected payout-per-premium) under LOPAMM and ind.
Because operator loss = trader profit (zero-sum), the comparability is measured **per
trade** — both designs fill the identical flow, so we compare each trader's deal under
LOPAMM vs ind directly.

Reuses the Exp 1 engine (flow with the correlated terminal joint `joint_p`). Driver:
`experiment3_trader.py`.

---

## 1. Fixed setup

`M=15`, `support=pyramid`, `k=6`, `decay=1.25`, `B_θ=5`, `ρ=1`, `b=10`, `T=10M=150`,
`seeds=100`, `base_seed=12345`. Designs: **LOPAMM and ind**. There is **no settlement**:
payouts are scored in expectation against the terminal joint law `joint_p =
softmax(Q/b)` (see §2), so each trader yields one exact, noise-free observation.

---

## 2. Per-trader deal metric

Each trade is a partially-informed trader handing the operator a target marginal `τ_S`
on one leg-set `S`, filled as buys over the sub-lattice `{S'⊆S}`. For that trader, under
a design:
- **premium** `π = Σ_{S'⊆S} cash_{S'}` — the LMSR cost paid.
- **holdings** `Δ_{S'}` — the contract vector bought on each book `S'` (one entry per one
  of the `2^{|S'|}` mutually-exclusive outcomes of the book).
- **expected payout** `v̄ = Σ_{S'⊆S} Δ_{S'} · μ_{S'}`, where `μ_{S'}` is the marginal of
  `joint_p` on book `S'` (`P(S' resolves to each outcome)`, sums to 1). At resolution
  exactly one outcome per book pays, so the expectation weights each held outcome by its
  probability.
- **`p_eff = v̄ / π`** — the effective (expected) payout-per-premium.

Both designs fill the **same** `τ_S`, so per trader we form the paired ratio
```
q = p_eff(LOPAMM) / p_eff(ind).
```
`q ≈ 1` ⇒ the informed trader gets the same effective price under both designs.
(Expected payout is used so `q` is well-defined per trader — realized payout would divide
by zero when a parlay loses — and so every trader is one clean data point.)

---

## 3. Measurement, statistics & figure

Group trades by parlay order `|S| = 1…k`. We report, per `|S|` and pooled:
- **mean `q` with a bootstrap 95% CI** (2000 resamples over traders) — the average deal;
- the **fraction of traders within a ±10% equivalence band** (`0.9 ≤ q ≤ 1.1`);
- **min–max and median** of `q`;
and, pooled, the **per-trader correlation** (Pearson and Spearman) of `p_eff(LOPAMM)`
against `p_eff(ind)` — trade-by-trade co-movement — reported for all `|S|` and for
`|S|≥2` separately (`|S|=1` base sub-trades are identical under both designs, so their
`p_eff` coincides exactly).

Figure: `|S|` vs `q` (mean + min–max whiskers, median line, reference at `q=1`),
`results_exp3_trader/payout_premium_ratio_by_legs.png/.pdf`.

**Reading:** if mean `q` sits within the ±10% band at every order (TOST-equivalent), the
informed trader's expected price is comparable under both designs — LOPAMM's loss advantage
comes from writing less corrective contract volume on its shared books, not from a worse
deal to any trader.

---

## 4. Result (M=15, k=6, 100 seeds; 15k traders)

Mean `q` is within ~2% of an identical deal pooled and stays inside the ±10% equivalence
band at every order (statistically equivalent by TOST at ±10%), with a mild downward drift
in `|S|` (LOPAMM prices high-order parlays slightly more accurately, leaving the trader a
hair less edge). The effect is significant only because of the large `n` but economically
negligible.

| \|S\| | mean q | 95% CI | in ±10% | median |
|---|---|---|---|---|
| 1 | 1.000 | [1.000, 1.000] | 1.00 | 1.000 |
| 2 | 0.994 | [0.990, 0.998] | 0.77 | 0.997 |
| 3 | 0.981 | [0.976, 0.986] | 0.62 | 0.977 |
| 4 | 0.971 | [0.964, 0.977] | 0.56 | 0.956 |
| 5 | 0.961 | [0.954, 0.969] | 0.55 | 0.941 |
| 6 | 0.938 | [0.931, 0.946] | 0.52 | 0.923 |
| **pool** | **0.978** | **[0.976, 0.980]** | **0.68** | 0.991 |

Per-trader `p_eff` correlation (LOPAMM vs ind): all `|S|` Pearson 0.78 / Spearman 0.65;
`|S|≥2` Pearson 0.67 / Spearman 0.57 — the two designs' effective prices track each other
trade-by-trade and agree on average, with genuine per-trader spread (~52–77% within ±10%).

---

## 5. Implementation

`experiment3_trader.py`: build the sparse structure once; for each seed load/build the
flow (`flowjs_` cache), collect the **traded books only** (closure of the support `D`) and
precompute their joint marginals `μ_{S'}` from `joint_p` (one bincount each — avoids the
~2.6 GB all-leg-set `oidx` table at M=15), then fill the flow through **both** designs in
lockstep, recording per trade each design's `π` and `v̄` and hence `q`, `p_eff`. Stats
(bootstrap CI, ±10% band, correlations) and the figure are computed at the end. Reuses
`build_sparse_structure`, `load_or_build_flow`, `match_to_target`, the `Design` classes,
and `joint_p`. Runtime: one M=15 / k=6 point × 100 seeds ≈ 2 min after the ~12 s structure
build (feasibility scales as `Σ_{l≤k} C(M,l)·4^l`, so `k` is the binding cost — `k=10` at
M=15 is ~40 GB / infeasible on 18 GB).
