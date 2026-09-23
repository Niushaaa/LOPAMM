# Experiment 4 — Real Kalshi order flow

**Goal.** Validate the synthetic-flow results (Exp 1) on **real** market data: replay the
actual Kalshi NBA same-game-parlay order flow through LOPAMM, ind, and base, settle at the
**realized** game outcomes, and compare operator net loss. Thesis (as in Exp 1): on real
flow, **LOPAMM's parlay loss ≪ ind's**, with base as the floor (no parlays).

Uses the Exp 1 LMSR/canonical mechanics (Python) only — the `kalshi/` directory is a
**data source only**, never its code. Driver: `experiment4_kalshi.py`; data staged in
`exp4_data/` by `exp4_stage.py`.

---

## 1. Data (staged in `exp4_data/`)

83 curated games (`kalshi/data/selection.json`), M∈[42,315], 10 M-buckets. Per game we
keep the **relevant** flow, time-ordered: base trades on the legs that appear in parlays,
plus every parlay trade — together with parlay leg-sets (+sides), and the realized
base-market outcomes. Each staged record: `{gameSuffix, M, activeLegs, maxK, realized,
parlays:[{ticker, legs:[[marketTicker, side]], settlement}], trades:[{ticker, kind, side,
size, ts}]}`.

**M / k check (staged):** active legs 3–257; **maxK mostly 2, up to 32**; 29,257 parlay
trades; 14.1M active-leg base trades; largest game ~787k trades.

---

## 2. Engine — Exp 1's, on a light lattice (no `2^M`)

Exp 4 uses **Exp 1's actual engine** — `parlay_mm_sim.DesignA`/`DesignC` and
`experiment1_loworder.sweep_books`/`match_to_target` — unchanged. The only substitution is
the *structure*: instead of Exp 1's `build_structure(M)` (which also builds the global
`2^M` atom space used solely to draw the joint settlement), we build a light lattice
(`build_lattice`) over the **downward closure of the traded leg-sets** (base singletons +
parlays), with the same per-book fields (`outcomes`, `subsets`, `proj_idx`) but **no
atoms**. Settlement is real (`realized`), so the atom space is never needed. `build_lattice`
is structurally identical to `build_structure` on the same leg-sets, so results are
**float-identical to the dense Exp 1 engine** (§7, 0.0 diff).

The cost is the **bottom-up sweep** (§3): a parlay trade re-quotes every `Sp⊆S`, i.e.
`O(3^k)` work per parlay trade. So `k` is capped at **`k≤8`** (a sweep touches ≤255
sub-markets); parlays with `k>8` are dropped and logged (~0.5% of the 29k parlay trades).
Everything runs local.

---

## 3. Models & trade replay (Exp 1's `sweep_books`)

We replay the tape in time order. For a trade on leg-set `S` we build one shared target
`τ_S` (§5b), project it to `marg[Sp]` for every `Sp⊆S`, and run Exp 1's **bottom-up
`sweep_books`**: buys-only `match_to_target` on each `Sp` ascending by `|Sp|`, charging cash
at each level. The three designs use the **same** `marg`; they differ only in `match`:

- **LOPAMM** = `DesignA`: `Q(Sp)=Σ_{T⊆Sp} residual[T]` — pricing a sub-market reuses the lower
  residuals, so once the base legs are set the higher corrections are tiny.
- **ind** = `DesignC`: `Q[Sp]` is a standalone book (uniform start) at **every level** — each
  sub-market is bought from scratch to `marg[Sp]`, independently, on every trade.
- **base** = only **genuine base-market trades** (real base flow). It does **not** take the
  parlay sweep (not even its singleton level) — parlays are unavailable to base. So `base`
  is the base-markets-only floor; its singletons therefore diverge slightly from LOPAMM/ind's
  (which get parlay-sweep nudges). (This differs from Exp 1, where the synthetic flow has no
  separate base trades, so base's singletons come entirely from the projected sub-marginals
  of every trade — see the note; real data has genuine base flow, so base uses it.)
- **native (order book)** — realized baseline: a maker countering *every* trade at its
  executed price with the *actual size*, P&L `= size·(win − price_paid)`; **size-weighted**,
  so not directly comparable in magnitude to the buys-only LMSR designs.

`sweep_books`, `match_to_target`, `DesignA`/`DesignC` are used **verbatim** from Exp 1; the
only substitution is the atom-free lattice (§2). Cost: the sweep is `O(3^k)` per parlay
trade ⇒ `k≤8` (§2).

---

## 4. Settlement & loss (real outcomes)

Each active leg resolves per `realized`; a parlay's conjunction resolves 1 iff every leg
matches its stated side. For each design, **total loss** `= Σ_S (payout − cash)` over all
its books (sign as in Exp 1: positive = operator loses). We report:
- **total loss** per design (`base_total`, `ind_total`, `lopamm_total`);
- **parlay loss** `= total − base_total` — the incremental cost of offering parlays over
  the genuine-base-only floor (the LOPAMM-vs-ind signal; base parlay loss ≡ 0).

---

## 5. Metrics & figures

Per game: `M`, `maxK`, parlay-trade count, per-level loss, and (base / ind / LOPAMM / native)
parlay and total loss. Figures (LOPAMM vs ind; **native is table-only**):
- **loss by parlay order** (`fig_loss_by_order`, headline) — grouped bars of corpus total
  loss at each `|S|=2..8`, ind vs LOPAMM: ind humps at 4–5 legs (~52k), LOPAMM ≈ 0 everywhere;
- **parlay loss vs M** (`fig_parlay_loss_vs_M`, y clipped, off-scale ind noted);
- **loss gap `ind − LOPAMM` vs M** (`fig_loss_gap_vs_M`) — gap>0 ⇒ LOPAMM loses less
  (sign-robust): gap>0 in 64/83 games, corpus gap +174,081;
- (`fig_lopamm_vs_ind` scatter kept but de-emphasized — the huge dynamic range makes it hard
  to read).

Output: `results_exp4_kalshi/per_game.json`, `fig_parlay_loss_vs_M.png`,
`fig_lopamm_vs_ind.png`, `fig_loss_gap_vs_M.png`, `fig_loss_by_order.png`
(via `experiment4_fig.py`).

---

## 5b. Input mapping (Exp-1 consistent — shared target price)

Each trade's **executed price** is the target, not a raw size. A base trade targets the
leg quote `[1−p, p]`. A parlay trade builds **one shared** `τ_S`: the reference is the
**shared base-leg product** (`m[o]=∏_{leg∈S} P(leg=o_leg)`), with the conjunction pinned to
`p` and the rest kept proportional, `τ[o≠conj]=m[o]·(1−p)/(1−m[conj])`. This one `τ_S` is
projected to `marg[Sp]` for every `Sp⊆S` and fed to **all** designs' sweeps (§3), so their
singleton books stay identical (base excepted — genuine trades only). (An earlier
size-as-quantity mapping saturated the base legs at `b=10`; a per-design `τ` diverged the
singletons under the sweep — both discarded.)

---

## 6. Implementation

`experiment4_kalshi.py`: for each staged game, build one binary LMSR per active leg and
replay the base tape (optionally aggregating base trades between parlay events, since only
cumulative leg quotes matter — cuts the 14M base replays); at each parlay trade, price the
conjunction under ind (own LMSR) and LOPAMM (marginalize touched residuals, §2), record cash
and the contract sold; at end, settle at `realized` and accumulate loss per design.
Reuses the Exp 1 LMSR cost / canonical residual code; adds the sparse single-outcome
marginal pricing. No `2^M`, no `2^k`; runs locally.

**Feasibility recap:** real settlement ⇒ no `2^M`; single-outcome marginalization ⇒ no
`2^k`; sparse residuals ⇒ ~hundreds of entries/game. Dominant cost is the 14M base-trade
replay (full replay, `b=10`). 18 GB is sufficient.

---

## 7. Validation — lattice engine vs dense Exp 1 engine

The engine *is* Exp 1's (`DesignA`/`DesignC` + `sweep_books` + `match_to_target`); only the
structure differs (`build_lattice`, no atoms, vs `build_structure`). We confirm
`build_lattice` is structurally identical (`out_index`/`proj_idx` match, 0 mismatches), and
replay the **same** target-price flow through both structures at `M=5`. **Done**
(`exp4_validate.py`, 8 seeds): per-design total loss (base, ind, LOPAMM) matches the dense
`build_structure` engine to **max abs diff 0.0**. So the Exp 4 numbers are the genuine Exp 1
model on real data, at real M (42–315) and k (≤8).

---

## 8. Result (83 games, 29,046 parlay trades ≤k8, b=10)

The bottom-up sweep (Exp 1's execution) reproduces the thesis at full magnitude:
`parlay P&L = total − base` (base = genuine base trades only).

| metric | ind | LOPAMM |
|---|---|---|
| corpus parlay P&L | **+174,038** (loss) | **−43** (≈ break-even) |
| median per-game | +58.7 | +0.11 |
| max per-game | **+82,494** | +19.0 |
| mean \|P&L\| per game | 2,123 | **3.76** (~560× smaller) |
| **LOPAMM ≤ ind** | — | **64/83 (77%)** |

LOPAMM prices the whole parlay market at **essentially zero incremental cost** (parlay P&L
sum −43, median 0.11): once the base legs are set, its shared residuals make every pair /
triple / … correction tiny. `ind` re-buys each sub-market from uniform on every trade, so
its cost **explodes with parlay order**. Corpus loss by order `|S|` (headline figure
`fig_loss_by_order`):

| `\|S\|` | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---:|---:|---:|---:|---:|---:|---:|
| ind | 16,671 | 33,325 | **51,664** | 43,646 | 21,870 | 6,131 | 743 |
| LOPAMM | −25 | −27 | 3 | 19 | 2 | −2 | 0 |

ind humps at 4–5 legs (per-parlay cost rises with order, but high-order parlay *volume*
falls, so the product peaks mid-order); **LOPAMM ≈ 0 at every order** (|·|≤27). Level 1
(singletons) is identical for ind/LOPAMM and equals the base floor.

**Aggregate total P&L over all 83 games** (single interleaved flow = per-game sum; disjoint
markets):

| model | parlay P&L | total P&L |
|---|---:|---:|
| base (genuine base trades only) | 0 | 18,755 |
| ind | +174,038 (loss) | 192,793 |
| LOPAMM | −43 | **18,712** ≈ floor |
| native (order book) | — | (size-weighted, reported separately) |

**LOPAMM's total ≈ the base floor** (18,712 vs 18,755): offering the full parlay book costs
LOPAMM almost nothing, whereas `ind`'s total (192,793) is ~10× the floor. The **native**
order-book maker (actual sizes) is reported alongside but on a different, size-weighted
scale than the buys-only LMSR designs.
