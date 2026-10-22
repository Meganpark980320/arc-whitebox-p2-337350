# Code map, guards and cost engineering (write-up Appendix E)

## E.1 Code map

File: `experiments/claude_merge_20261014/package_c2eko10S2H8D3ahF10CR3GsHV4RxDnoT78cf4g065/package/estimator.py` (sha8 dd17f7f4, 21,067 lines). Later sections re-bind earlier functions.

| piece | lines |
|---|---|
| entry points | `Estimator` 7971 → `_V25Core` 6235; outermost setup 19412 (of 16 bindings), predict 21066 (of 9) → … → `_predict_core` 6345 (Box 1) |
| O1; ACC | store 17457 (call 7221), fold 17561, τ 17440; γ 17765–17768 |
| K31F; GDK31 | 7859 → 18230 → 18189 (GDK31) → 17633 (O1) → 17222 (K31F); τ 17193 |
| C2E; feedback-birth | 17351 (call 7254), τ 17335; 7255–7265 |
| gk v3; tmd-lite | 20082 (call 7216); 6968–6972, 7317–7324 |
| heads; corrector; last-layer head | 8022 (fits 8049, 8052, 8085); 14947 (call 12512); 17163 |
| learned κ4 parts | A14 6110; quartic star 15538; κ4-diagonal MLP 18846; K31 slot 7063–7095 |
| gains; λ | 1892–1899; table 1772, per-network rule 6952–6961 |
| guards; safety net | E.3; 14044 → 14009 (Gaussian covariance, else mean field, else zeros) |

**Switches.** Unset, the 246 environment variables give the submitted build. The ablation (§5) uses `CI_O1P_TAU`, `CI_O1P_ACC`, `CI_K31F_TAU`, `CI_C2E_TAU`, `ARC_GK_GATE`, `CI_TMD_LITE`, `CI_K4C_TAU`, `CI_TR_GAIN`, `CF_TRLZ_G`, `V25_BETA`, `CI_GK` and, after import, `NO_CORR=True` (heads, corrector), `est.core._law_arrays=None` (A14), `_CF_RG_ON=False` and `_CF_DL_TRIGGER_S=_CF_DL_TRIGGER2_S=1e9` (guards). `V17_NO_CORR=1` makes setup raise (line 9871).

**Dead code.** Unused: the header's matched-Monte-Carlo blend fallback and comparison arm (lines 25–67), the blend's constants (371–381), the comparison chain (385–1436) and its switch `_V25_STACK` (1467).

*Sources:* the estimator; `docs/writeup/v9/ABLATION-SPEC-v9.md`; `docs/writeup/v9/ABLATION-RUN-NOTES-v9.md`.

## E.2 Internal names

| name | meaning |
|---|---|
| `c2eko10S2H8D3ahF10CR3GsHV4RxDnoT78cf4g065` | #337350's leaderboard description: our build name, chaining its ancestors' tags |
| old tier, Q_c; lazy join | rank-352 basis of sources over four layers old; lazy joins (from layer 11) reuse it |
| slot; K31, K22 | one layer's stored D21; κ4 slices (i,i,i,j), (i,i,j,j) |
| K31 slot; GDK31 | fitted K31 stand-in; §4.6 |
| quartic star; A14 law | §4.2; learned κ4-diagonal correction |
| Mehler q3 | C³/6 term of the covariance's Mehler series |
| trace repair; lazy-join trace | re-inject κ3 trace lost to compression (layers 10–14), to lazy joins (13–14) |
| alive, hub masks | public V34 masks; skip low-weight columns of young D21 products |

*Sources:* `docs/writeup/evidence/official/lineage/LINEAGE-RECON-20261021.md`; `docs/writeup/v9/APPENDIX-I-INVENTORY-v9.md`.

## E.3 Time guards and #337451

- Near the 0.4 s residual limit, a residual guard switches on *levers*, lighter paths, from layer 3 (20869–20962); a deadline guard drops one fast-product step at a projected 111 s (13786–13797).
- Three levers, like the designed light paths of a setup's first two predicts (13739–13809), only reorder floating-point operations; the *head lever* (20953) skips later learned mean corrections, +18 to +32 % MSE on that network [unverified].
- The *sticky rule* (20873–20876, 21039–21057) carries a late verdict into the worker's next predict.
- Officially only network 4 shows a lever-sized compute rise (+1.9 %), with no limit flag; whether the head lever fired is unrecorded [official bundle].

#337451 scored 1.952e-9, 1.78 % worse than #337350 [official bundle]. Our hypothesis (its reproduction was stopped): the sticky rule kept re-arming, as the levers save too little real time to disarm it, so one normal-speed worker stayed on lighter paths for 18 networks and carried most of the loss.

*Sources:* the estimator; `experiments/claude_setupbox_20261020/post337451/GUARD-MARGIN-REPORT-20261021_v3.md`; `experiments/claude_pair_20261014/fresh/official/evaluation_337451.json`.

## E.4 Cost engineering

- Compute fell 38.3 % from #332675 to #337350 [official bundle].
- Largest cuts: alive-neuron masks −6.5 % (MSE +4.6 %), thin feedback-leg ranks −4.9 % (MSE +3.0 %), exact root-product folding −1.8 % [official single change]; shared-buffer Strassen–Winograd routing −5.8 %, lazy joins at layers 13–14 −2.1 %, PLinOpt's programs −1.2 % [official bundle].
- Other pieces batch requests, reuse buffers or move work into setup; predictions change only by rounding.
- flopscope bills an m×k by k×n matmul 2mkn − mn FLOPs; fast programs split it into smaller ones plus billed additions.

*Sources:* `docs/writeup/evidence/official/LEDGER-STEPS-20261021.md`; `.venv/lib/python3.10/site-packages/flopscope/_flops.py`.
