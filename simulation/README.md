# LOPMM evaluation — parlay market-maker simulations

Code and experiments for the evaluation section of the **Low-order Parlay Automated Market Maker
(LOPMM)** paper. We compare three LMSR-based market makers for **same-game parlay**
(conjunction) markets, on synthetic and real order flow.

## Designs

All designs are logarithmic market scoring rules (LMSR) over the Walsh/canonical
parameterization anchored at all-ones; trades are executed **buys-only** via
`match_to_target` (bottom-up `sweep_books`).

- **LOPMM** (`DesignA`) — hierarchical, **shared** residuals: `Q^{(S)}_ω = Σ_{∅≠T⊆S} r^{(T)}_{ω|_T}`.
- **ind** (`DesignC`) — an **independent** LMSR at every leg-set (no sharing).
- **base** — singletons only (no parlays); the operational **floor**.
- **native** (Exp 4 only) — the realized order-book maker (counters every trade at its
  executed price and size); reported for reference.

Core mechanics live in `parlay_mm_sim.py`; the Exp 1 engine (structure, flow, sweep,
settlement) in `experiment1_loworder.py` and is **reused verbatim** by Exp 2–4.

## Experiments

1. **Scaling** (`experiment1_loworder.py`) — synthetic low-order informed flow; operator
   net loss vs `M`. Result: `base ≤ LOPMM ≤ ind` pointwise — LOPMM tracks the floor while
   `ind` grows steeply, the gap widens with `M`, and LOPMM stays under a `c·M²+d` bound.
2. **Robustness** (`experiment2_robustness.py`) — fix `M=10`, sweep support sparsity `ρ`,
   belief magnitude `B_θ`, re-quote rate `R=T/M`, and order `k`. LOPMM is robust; `ind`'s
   loss grows on every axis and **explodes with parlay order**.
3. **Trader fairness** (`experiment3_trader.py`) — per-trader effective payout-per-premium
   `p_eff`; the paired ratio `q = p_eff^{LOPMM}/p_eff^{ind} ≈ 1` (±10% equivalence). LOPMM's
   lower operator loss is a **genuine efficiency gain, not value extracted from traders**.
4. **Real data** (`experiment4_kalshi.py`) — replay actual Kalshi NBA same-game-parlay
   order flow (83 games, `M∈[42,315]`, parlay order `k≤8`), settled at the **real** game
   outcomes. LOPMM offers the full parlay book at **≈ zero incremental cost** (total ≈ base
   floor); `ind`'s cost **explodes** (+174k over the corpus, peaking at 4-leg parlays).
   Runs Exp 1's engine on an atom-free lattice (real settlement ⇒ no `2^M`); validated
   **float-identical** to the dense engine (`exp4_validate.py`, 0.0 diff).

Full methodology per experiment: `EXPERIMENT{1,2,3,4}_PLAN.md`.

## Layout

```
parlay_mm_sim.py          core LMSR, DesignA/DesignC, match_to_target, Structure
experiment1_loworder.py   Exp 1 + shared engine (sparse structure, flow, sweep, settle)
experiment2_robustness.py Exp 2   (reuses Exp 1 engine)
experiment3_trader.py     Exp 3   (reuses Exp 1 engine)
experiment4_kalshi.py     Exp 4   (Exp 1 engine on a light lattice, real Kalshi data)
exp4_stage.py             stage Kalshi game data -> exp4_data/  (gitignored, ~1.5 GB)
exp4_validate.py          Exp 4 engine == dense Exp 1 engine (0.0 diff)
paper_fig.py              Exp 1 figure;   experiment4_fig.py  Exp 4 figures
results_exp*/             cached results and figures
```

## Reproduce

Requires `numpy`, `matplotlib` (`scipy` for Exp 3).

```bash
# Exp 1 (M=2..20) + paper figure
python3 experiment1_loworder.py --k_auto --steps_per_M 10 --rho 1 --rho_0 5 \
    --support pyramid --seeds 40 --M 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20
python3 paper_fig.py --Mmax 20

# Exp 2 and Exp 3
python3 experiment2_robustness.py
python3 experiment3_trader.py

# Exp 4 (needs the kalshi/ dataset present alongside this repo)
python3 exp4_stage.py && python3 experiment4_kalshi.py && python3 experiment4_fig.py
```

Per-`M`/flow results are cached under `results_exp1_loworder/`, so reruns load from cache.
Exp 4 uses the `kalshi/` directory **as a data source only** (no code from it).
