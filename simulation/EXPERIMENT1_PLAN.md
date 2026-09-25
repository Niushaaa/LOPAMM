# Experiment 1 — Loss scaling under synthetic low-order informed flow

**Goal.** Using synthetic *informed* order flow that satisfies the low-order
assumption, test whether the operator's realized net loss (equivalently, the
informed trader's profit — §5) scales as the theory predicts: **LOPMM** stays pinned
close to the **base** floor while the independent baseline **ind** grows steeply as
more orders enter.

Implemented in `simulation/experiment1_loworder.py` — self-contained apart from the
shared mechanics imported from `parlay_mm_sim.py` (`DesignA`, `DesignC`,
`match_to_target`, `entropy_drop_by_level`, `T_CLIP`, `Structure`). First of three
scripts; the flow generator and executor carry over to Exp 2 (order/sparsity/
magnitude stress) and Exp 3 (Kalshi flow).

---

## 1. Model roster (3 measured curves)

| Name | Class | What it is | Books | Role |
|------|-------|-----------|-------|------|
| **LOPMM** | `DesignA` | hierarchical shared-residual LMSR (eq. Q-agg) | all traded leg-sets | design under study |
| **ind**  | `DesignC` | independent per-market LMSRs | all traded leg-sets | SOTA baseline |
| **base** | `DesignC`, singletons only | M independent binary LMSRs, parlays unavailable | M singletons | lower reference / floor |

**Loss metric:** total net loss `= settlement_payout − cash` over all books,
Monte-Carlo expected (§5). B (monolithic) is dropped. The run report shows raw curves
and H1/H3/H4 verdicts — no exponential fits (those live in the paper figure, §6).

---

## 2. Canonical parameterization — Walsh, anchored at all-ones

Sign `s_i(ω)=2ω_i−1` (ω=1 → +1; all-ones is the reference corner, from `atom_signs`).
Walsh characters `χ_S(ω)=∏_{i∈S} s_i(ω)`. Natural parameters are scalars `θ_S`, one per
leg-set (complete basis): the potential is `Q(ω)=Σ_S θ_S χ_S(ω)` and the joint is
`softmax(Q/b)`. These `θ_S` are LOPMM's `residual[S]` in the Walsh basis. A k-order
belief change is exact: only write `θ_S` for `|S|≤k` ⇒ `θ_S≡0` for `|S|>k`.
(The full joint is never materialized — §3 step 3.)

---

## 3. Flow generation — `build_low_order_flow` (one trade per step)

Settled run (defaults for Exp 1; Exp 2 sweeps these):
```
python3 experiment1_loworder.py --k_auto --steps_per_M 10 --rho 1 --rho_0 5 \
    --support pyramid --seeds 40 --M 2 3 … 20
```

| knob | flag | settled value | code default | meaning |
|------|------|---------------|--------------|---------|
| `M` | `--M` | 2…20 (feasible ~25) | `[2..8]` | base events |
| `k` | `--k_auto` | `⌈√M⌉` per M | fixed `--k 2` | order cap (climbs with M) |
| support | `--support` | `pyramid` | `pyramid` | `pyramid` or `flat` |
| decay | `--decay` | `1.25` | `1.25` | pyramid per-level count decay (§step 1) |
| `ρ` | `--rho` | `1.0` | `0.5` | θ-increment per-level decay |
| `ρ_0` | `--rho_0` | `5.0` | `10.0` | θ-increment scale (`ρ=1,ρ_0=5` ⇒ flat ±5) |
| `T` | `--steps_per_M` | `10` ⇒ `T=10M` | `10` | steps = trades (1 trade/step) |
| `b` | `--b` | `10.0` | `10.0` | LMSR liquidity |
| settles | `--n_settle` | `2000` | `2000` | MC base-market draws (§5) |
| seeds | `--seeds` | `40` | `5` | for tight CIs |

**Step 1 — support `D` (`pick_support`, `|D|=O(M)`).** Always all M singletons, plus
higher levels by `--support`:
- **`pyramid` (default):** per level `l=2…k`, keep `round(2M / decay^{l−2})` sets — `2M`
  pairs, the count decaying by factor `decay` each level up — chosen uniformly at random
  per seed. `decay=1.25` (default) decays slowly → keeps *more* high-order markets;
  `decay=2` halves each level. |D| e.g. M=20,k=5: `decay=1.25 → 138`, `decay=2 → 95`.
- **`flat`:** pool all level-`2…k` subsets and pick `4M` uniformly at random. |D| e.g.
  M=20 → 100.

`θ_S≠0` only for `S∈D`, `|S|≤k`. Concentrating a fixed `O(M)` support is what produces
the clean scaling; spreading over more markets under-trades them and blurs it.

**Step 2 — one θ-change → one trade per step.** `Q=0` at open (uniform P̂). For each of
`T` steps: pick **one** leg-set `S` uniformly from `D`, draw
`Δθ_S ~ U[−ρ^{|S|}ρ_0, +ρ^{|S|}ρ_0]` (with `ρ=1` this is a flat bound `±5`), update
`Q += Δθ_S·χ_S`. **Only Δθ is bounded — θ itself is an unbounded walk.** Few re-quotes
(small `T`) keep LOPMM's buys-only leak negligible (§7).

**Step 3 — the trade (marginal computed directly, no full `P`).** After the θ-update,
`marginal_from_Q(Q, struct, S, b)` forms **only** `S`'s marginal directly from `Q` —
exp-weights over the `2^M` atoms then `bincount` onto `S`'s `2^|S|` outcomes; the
normalized joint `P` is never built. Sub-marginals on `S'⊆S` are derived from that one
top-`S` marginal via `proj_idx`. Emit `(S, {S'⊆S : τ_{S'}})`. `T` trades total.

**Step 4 — settlement law.** Store `joint_p` = the final correlated joint `softmax(Q/b)`
over the `2^M` atoms (§5).

**Caching.** Flows → `flow_cache/…pkl`; per-M aggregated results → `result_cache/…pkl`
(loaded and printed as `[cached]`, skipping recompute). **Cache keys encode every knob
that affects the result** (M, k, support + non-default decay tag, ρ, ρ_0, b, T,
n_settle, seeds, base_seed), so changing any knob makes a fresh key — no manual
clearing, and the output dir is **not** wiped between runs.

**Sparse structure (`build_sparse_structure`).** Builds only leg-sets of size `≤k` (no
full leg-set; ind clipped at k) with the full `2^M` atom space. Marginals use `bincount`
(no `consistent` dict). Cost `~T·2^M` per seed and **doubles per +1 M** (memory too:
`atom_signs` is `2^M·M` bytes). Feasible to ~M=25 on 18 GB RAM, ~M=30 with large RAM;
`2^M` is the hard wall. `incons` metric dropped (needed `consistent`).

---

## 4. Trade execution — confined to S's sub-lattice (`sweep_books`)

Each trade re-quotes only `subsets(S)` bottom-up to the current marginals, buys-only
(`match_to_target`):
- **LOPMM:** matching `Sp` writes `residual[Sp]` to the eq. route value
  `b·log τ_{Sp} − Σ_{T⊊Sp} r^{(T)}` (up to the loss-neutral buys-only gauge); lower
  orders propagate up for free. Guarded by `--check_route`.
- **ind:** same sub-trades on independent books (each re-learns).
- **base:** singleton sub-trades only.

---

## 5. Settlement & loss — `settle_flow` (Monte-Carlo, correlated-joint draws)

`n_settle` instances, each draws a realized atom from the final correlated joint
`ω ~ Categorical(joint_p)` over the `2^M` atoms; that one realization settles **every**
book (parlays by projecting the realized outcome onto their legs). Per model
`net = payout − cash`, averaged over draws (same draws for all models; only
nonzero-contract books iterated). Per-seed = MC mean; aggregate over seeds = mean ± 95%
CI. `joint_p` is a `2^M` vector carried in the flow (cacheable to ~M≈20–22; beyond that
it hits the same `2^M` wall as the structure).

**Trader ↔ operator (zero-sum):** the informed trader pays the premiums (`cash`) and
receives the payout, so **trader net profit = payout − cash = operator net loss** — the
reported net-loss curves *are* the informed trader's profit against each design. (A
histogram of per-run trader profit shows a fat right tail against ind and a much tighter
distribution against LOPMM.)

---

## 6. Outputs

`results_exp1_loworder/`: `flow_cache/`, `result_cache/` (per-M, reused across runs),
`per_seed.csv`, `summary_aggregate.csv`, `results.txt` (summary table + H1/H3/H4),
`plot_net_loss_lowM.png`. Live per-M `[running …]` / `[cached]` progress prints.

**Paper figure — `paper_fig.py`.** Reads `result_cache` and draws net loss vs M for
LOPMM/ind/base with LOPMM's **exact per-seed min–max band** and a **`c·M²+d` upper bound**
on LOPMM's worst-case (max-over-seeds) loss (least-squares fit raised to a strict bound).
Parameterized `--support --decay --steps_per_M --Mmin --Mmax --seeds`; writes
`paper_net_loss_<config>.png/.pdf` (linear axis, no title).

**Success criteria:** H1 LOPMM near floor, gap `ind−LOPMM` widens with M; H3
`KL(LOPMM) ≲ KL(ind)`; H4 `base ≤ LOPMM ≤ ind` pointwise.

---

## 7. Key finding — the leak scales with re-quote frequency

LOPMM's buys-only "leak" (shared low-order residuals, re-quoted to a drifting target,
forcing corrective buys on sibling books) accumulates with the **number of re-quotes**,
not with book inconsistency. Confirmed: many trades ⇒ LOPMM leaks and loses to ind; an
active-set *consistent* full-menu sweep did **not** fix it; **few trades (one per step,
`T=10M`) fixes it** — LOPMM's sharing advantage then dominates. So `T` (via
`steps_per_M`) is the lever, not the executor.

---

## 8. Validated result (pyramid, decay=1.25, k=⌈√M⌉, 40 seeds, M=2…20)

- **`base ≤ LOPMM ≤ ind` at every M** (H4 = YES) — the clean theoretical ordering; all
  curves positive.
- **LOPMM tracks the floor closely while ind pulls away:** LOPMM grows slowly, ind fast.
  E.g. M=10: base 9.9, LOPMM 53.7, ind 228.2.
- **Gap `ind − LOPMM` widens monotonically** with M (H1 = YES); H3 (KL) holds — LOPMM prices
  at least as accurately as ind.
- ind's rise steepens where `k` steps up (`k=⌈√M⌉`: 2→3 at M=5, 3→4 at M=10, 4→5 at M=17),
  each new order adding a stratum of parlay books that ind prices independently.

**Support / decay / k levers on the separation** (all preserve `base ≤ LOPMM ≤ ind` and
steepen ind):
- smaller `decay` (ρ) keeps more high-order markets (larger |D|) → steeper ind.
- `flat` support (4M) pulls in the most high-order books → steepest ind.
- larger `k` admits higher orders → steeper ind; `k=⌈√M⌉` makes k climb with M so higher
  orders enter as M grows.

---

## 9. Reuse & guards

**Imported unchanged from `parlay_mm_sim.py`:** `DesignA`, `DesignC`, `match_to_target`,
`entropy_drop_by_level`, `T_CLIP`, `Structure`. Everything else (sparse structure,
support, flow, `marginal_from_Q`, executor, settlement, `agg`, `kl`, result caching,
reporting) lives in `experiment1_loworder.py`. **Guard:** `--check_route` asserts eq.
route after each LOPMM sub-trade. **Determinism:** one `default_rng(base_seed+seed)` for
the flow, a separate stream for settlement; flows and per-M results cached; runs
reproducible.
