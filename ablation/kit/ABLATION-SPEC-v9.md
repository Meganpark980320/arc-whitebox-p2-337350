# Ablation of the fitted parts of #337350: runnable specification (v9)

*22 October 2026. A specification only: nothing here has been run. It answers the reviewers' request to measure how much of the linked submission's accuracy rests on parts fitted to Monte Carlo labels or set by hand. All switches were read from the linked `estimator.py` (sha256 dd17f7f4…); line numbers refer to that file. Run it on a rented CPU machine, never on the WSL workstation.*

## 0. Question and comparisons

One question: what is the final-layer MSE of #337350 when its fitted parts are removed layer by layer, at the compute each arm actually uses, and how does each arm compare with Monte Carlo at equal compute?

| comparison | what it isolates |
|---|---|
| B / A | the learned mean corrections (C2, C3, Track A, the learned corrector, the last-layer head), with everything else as shipped |
| C / B | the learned κ4 suppliers (the A14 law and the k4c corrector) |
| D / C | the hand-set strengths at their derived values and the fitted gain tables at their neutral value 1 |
| D′ / D | the inherited fitted κ4 constant λ against its physical value |
| E / B | our six derived mechanisms (O1, ACC, K31F, C2E, gk v3, tmd-lite), on a base without learned mean corrections |
| F / A | the whole path from the starting point #332675 (offline counterpart of the official −38.5 % MSE) |

D′ is the closest to a derived-only engine that switches can reach. It still contains three things: the fitted K31-slot weights θ and the fitted star-D4 mixing Ω (neither has a clean switch, §8), the inherited pruning of the Wick tables, and our lossy compression gates.

## 1. Pinned inputs

| input | path | sha256 |
|---|---|---|
| linked estimator (arms A–E) | `experiments/claude_merge_20261014/package_c2eko10S2H8D3ahF10CR3GsHV4RxDnoT78cf4g065/package/estimator.py` | dd17f7f47195ae4063c3f7d91d273f2fb2bf31c0dac865eaadcfdf76acd45b41 |
| starting point #332675 (arm F) | `research/corpaci/submission/estimator.py` | 5eae6e1f76cd9f7137fba7ae23c800cd5474dd84d6be7ffb5a5114de7c33625f (equals `research/corpaci/METRICS.json`) |
| runner template | `experiments/claude_pair_20261014/fresh/podj164/run_net12.py` | e66fed9a51df0d02b783c1ff907eb9e9dbcb672fec39b68f7edcef8765041072 |
| SDK | whestbench 0.16.1 with flopscope 0.12.3 (the version of the official #337350 run; #332675 was graded on 0.12.1). Use one version for every arm and record the vendor-tree hash (`run_net12.py sdk`). | — |
| data | Hugging Face `aicrowd/arc-whestbench-public-2026`, revision `v2-phase2` (commit aa99830f…), files `data/mini-0000{0..6}-of-00007.parquet`. The workstation's Hugging Face cache holds a prepared Arrow copy of the mini split (6.3 GB); download again on the rented machine. | record per file |

Columns read per network: `mlp_id`, `mlp_name`, `mlp_seed`, `weights` (16 × 1024 × 1024 float32), `all_layer_means` (16 × 1024, the truth) and `avg_variance` (σ², the column the grader uses for the sampling reference). The schema of a cached public shard (`full-00000-of-00067.parquet`) has all six columns. Check `avg_variance` in the mini files on download. If it is missing, use σ² = 1.132e-6 × 65,374.4 = 0.0740, the value for the 50 official public networks, and say so in the report.

## 2. Networks

**Primary set: the public mini split, 100 networks (`mlp_id` 0–99).**
- Truth uses 10⁹ samples (dataset `n_samples`; whestbench `PHASE2_ROUND.n_samples`). Label noise is σ²/10⁹ ≈ 7.4e-11, about 0.55 % of the linked MSE, the same in every arm.
- The mini split's networks are distinct from the 1,000 networks of the public "full" split (dataset metadata: 100 + 1,000 = 1,100 networks).
- No learned part was fitted on this split, as far as the training records go: the C2 split file lists only full-split IDs, and the corrector's training shards are full-split shards. The C3, Track A, last-layer-head, k4c and star-D4 training sets are not recorded [unverified].
- The split was used in many earlier decisions [held-out, reused]. That does not bias an ablation, which selects nothing.
- The 50 official public networks are not available with truth in the repository. Their names appear only in files derived from the official evaluations.

**Do not use:**
- the sealed 64-network designation set (`experiments/claude_pair_20261014/fresh/SEALS.md`);
- the public shards that trained the corrector or tuned constants (shards 34 and 39 = networks 544–559 and 624–639; the shards listed in the corrector's training set);
- any set with 2²⁴-sample labels. There, label noise is about 33 % of the linked MSE.

**Optional confirmation, only if the participant releases them:** the public shards 9, 10, 11 and 60 (64 networks, 10⁹-sample truth). They were kept blind in the C2 split and lie outside the corrector's training shards. Their cleanliness for the other heads is [unverified].

## 3. Arms and exact switches

Two kinds of switch exist:
- environment variables, read when the module is imported, written to the arm's `env.txt`;
- post-setup flips, applied in the client process after `Estimator.setup` returns and before the first predict.

`m` is the imported estimator module and `est` the `Estimator` instance.

| arm | build | `env.txt` | post-setup flips |
|---|---|---|---|
| A | linked | empty | none |
| Z (null check) | linked | the shipped defaults written out (block Z below) | none; must reproduce A byte for byte |
| B | linked | empty | `m.NO_CORR = True` |
| C | linked | `CI_K4C_TAU=0` | `m.NO_CORR = True`; `est.core._law_arrays = None` |
| C0 (optional; the reviewers' C) | linked | empty | as C |
| D | linked | block D | as C |
| D′ | linked | block D′ | as C |
| E | linked | block E | `m.NO_CORR = True` |
| E′ (optional) | linked | block D′ followed by block E (later lines win) | as C |
| F | starting point | empty | none |
| F′ (optional) | copy of the starting point with line 1159 `_V25_STACK = True` changed to `False` (record its sha256) | empty | none |

**Why each switch does what it says.**
- `m.NO_CORR = True` is exactly what the guard's head lever does (line 20953). Line 7193 then skips `_cb_head_alive`, which holds C2 (line 8049), C3 (8052), Track A (8084–8085) and the learned corrector with the last-layer head (8088 → 12485–12515). The guard's predict wrapper restores `NO_CORR` only to a value it saved itself (lines 20951–20953, 21035–21037), so the flip persists.
- Do not set `V17_NO_CORR=1` instead. The Track A setup then raises `ValueError` (lines 9871–9872), and no wrapper of setup catches it (e.g. line 13917).
- `est.core._law_arrays = None` makes the A14 law return its inputs unchanged (line 6111).
- `CI_K4C_TAU=0` turns k4c off (lines 18800–18803).
- `CI_O1P_TAU=0` turns O1 off (lines 17440–17445).
- `CI_O1P_ACC=0` turns ACC off (lines 17753, 17768).
- `CI_K31F_TAU=0` turns K31F off (line 17197).
- `CI_C2E_TAU=0` turns C2E off (lines 17335–17349).
- `ARC_GK_GATE=0` turns gk v3 off (line 20080).
- `CI_TMD_LITE=0` turns tmd-lite off (lines 19339–19342).
- `V25_BETA=0` removes the per-network λ rule (line 6952).
- `CI_GK` replaces the default multiplier string (line 1893); each token `name:a-b:x` multiplies `_CI_G[name][a..b]` by x (lines 1894–1899).

**Block Z** (shipped defaults, written out):
```
CI_O1P_TAU=1.5
CI_K31F_TAU=1.1
CI_C2E_TAU=0.65
CI_TR_GAIN=0.4
CF_TRLZ_G=2.0
CI_K4C_TAU=0.5
V25_BETA=1.0
CI_GK=fbb:1-7:1.0282,fbb:8-15:1.0319,q3:1-7:1.0252,lam:8-15:0.9646,c2:1-7:1.0349
```

**Block D.** Every hand-set strength goes to its derived value: O1 τ, K31F τ, C2E τ_birth, the trace-repair gain and the lazy-join trace gain to 1. The `CI_GK` string undoes the shipped multipliers and the 26 fitted gains of `_CI_G` (line 1892). It also puts the feedback-birth coefficient 0.35 × gain at its derived ½ and the Mehler q3 factor 1.5 × gain at its derived 1. The K31F gain g_s is the star gain, so it becomes 1 too.
```
CI_K4C_TAU=0
CI_O1P_TAU=1
CI_K31F_TAU=1
CI_C2E_TAU=1
CI_TR_GAIN=1
CF_TRLZ_G=1
CI_GK=fbb:0:1.428571429,fbb:1-5:1.342740754,fbb:6-10:1.306738931,fbb:11-14:1.252695488,fbb:15:1.428571429,q3:0:0.6666666667,q3:1-5:0.6654947304,q3:6-10:0.69178683,q3:11-14:0.6784368679,q3:15:0.6666666667,lam:0-4:0.9964943329,lam:5-9:0.976484305,lam:10-14:0.9475556381,a14:1-5:1.010082645,a14:6-10:0.9083576168,a14:11-15:0.9988553118,k31:6-10:0.8745421947,k31:11-14:0.8239354433,c2:1-5:1.092413841,c2:6-10:1.106230178,c2:11-15:1.045510005,c3:1-5:0.9713641839,c3:6-10:1.013326254,c3:11-15:1.062695868,ta:1-5:1.005801463,ta:6-10:0.9488179152,ta:11-15:0.9462591066,star:3-5:1.037509065,star:6-10:1.033675066,star:11-15:1.103736922
```

**Block D′.** As D, plus `V25_BETA=0`. The λ tokens now set λ_ℓ = 2.2e-3 at every layer: the physical value measured on one network with 8 × 10⁴ samples, 2.1–2.6e-3 (`research/corpaci/research/astra_k4_overshoot.md` §0) [simulation]. Multiplier = 2.2e-3 / (0.95 · LAM[ℓ] · gain[ℓ]), with LAM at line 1772.
```
CI_K4C_TAU=0
CI_O1P_TAU=1
CI_K31F_TAU=1
CI_C2E_TAU=1
CI_TR_GAIN=1
CF_TRLZ_G=1
V25_BETA=0
CI_GK=fbb:0:1.428571429,fbb:1-5:1.342740754,fbb:6-10:1.306738931,fbb:11-14:1.252695488,fbb:15:1.428571429,q3:0:0.6666666667,q3:1-5:0.6654947304,q3:6-10:0.69178683,q3:11-14:0.6784368679,q3:15:0.6666666667,a14:1-5:1.010082645,a14:6-10:0.9083576168,a14:11-15:0.9988553118,k31:6-10:0.8745421947,k31:11-14:0.8239354433,c2:1-5:1.092413841,c2:6-10:1.106230178,c2:11-15:1.045510005,c3:1-5:0.9713641839,c3:6-10:1.013326254,c3:11-15:1.062695868,ta:1-5:1.005801463,ta:6-10:0.9488179152,ta:11-15:0.9462591066,star:3-5:1.037509065,star:6-10:1.033675066,star:11-15:1.103736922,lam:0:0.4625054789,lam:1:0.2853344734,lam:2:0.2347794902,lam:3:0.2187573312,lam:4:0.2126689786,lam:5:0.2088411595,lam:6:0.2135548281,lam:7:0.2250529533,lam:8:0.2343762191,lam:9:0.2465473261,lam:10:0.2501241733,lam:11:0.2614238334,lam:12:0.2724026283,lam:13:0.2858329259,lam:14:0.3023005693,lam:15:0.3190319989
```

**Block E** (our six mechanisms off):
```
CI_O1P_TAU=0
CI_O1P_ACC=0
CI_K31F_TAU=0
CI_C2E_TAU=0
ARC_GK_GATE=0
CI_TMD_LITE=0
```

To recompute the multipliers, decode `_CI_G` from line 1892, where each value is a hex float64 array read by `_hxl_n` (line 1749), and LAM from line 1772, then apply the formulas above. Values are rounded to 10 significant digits, a relative error below 1e-9.

## 4. Runner and procedure

**Runner.** Write `run_net_abl_v1.py` as a new file, derived from `run_net12.py`; do not edit the template. Keep its official flow:
- a fresh flopscope server and one client process per line;
- the client loads one build, calls `setup` once, then predicts the listed networks in order, with no warm-up;
- one receipt per network: `pred_sha`, F, request count, guard, deadline and residual-guard logs, VmPeak/VmHWM of client and server, src/runner/SDK hashes, CPU model.

Add four things:
1. `--kind mini`, reading `mini-{i//16:05d}-of-00007.parquet` with the filter `mlp_id = i`, as in `experiments/claude_pair_20261014/fresh/run_net5.py`, including `avg_variance`.
2. `--env FILE`, applied with `os.environ.update` before the module is imported, as `run_net5.py` does.
3. `--post-setup nocorr,a14off`, applied in the client right after `est.setup(ctx)` returns (§3).
4. After every predict, record `m.NO_CORR`, `est.core._law_arrays is None` and `getattr(est, '_cf_rg_last', None)`. Stop the line if a flip no longer holds, which is the sign of a safety-net re-setup.

**Layout.** Each arm runs five lines, as five official workers: line w ∈ {0, …, 4} predicts networks w, w + 5, …, w + 95 in that order. The layout is identical in every arm, so each network sits at the same position (first predict, second, steady) everywhere. Launch the lines interleaved across arms (A0, B0, C0, …, then A1, …), so machine drift hits all arms alike.

**Order of work.**
1. Smoke test, about 15 minutes: arms A, Z and B–F on networks 0 and 5 (line 0 only).
2. Gate before the full run:
   - A run twice gives identical `pred_sha`;
   - Z equals A byte for byte;
   - every other arm differs from A on both networks;
   - no receipt shows a guard event;
   - client VmPeak < 8,190 MiB.
3. Full run: every arm, all 100 networks.
4. Analysis (§7). Report every arm, whatever the outcome.

## 5. Hygiene rules

1. **Paired.** The same 100 networks, positions, SDK and machine type for every arm. All statistics are per-network ratios against A.
2. **Same machine.** One machine, or several of the same CPU model whose A receipts for networks 0 and 5 give identical `pred_sha`. Pods give bit-identical bytes to each other but not to the workstation (about ±0.6 % per network). Never mix receipts from different machine types within a comparison.
3. **Prediction-hash checks.** A is deterministic across reruns; Z reproduces A; every ablation arm differs from A (the switch took effect).
4. **One build per process.** The server's array store holds 100,000 arrays. A third build in one process silently falls back to a Gaussian answer, and two builds per process is the upper limit. This design uses one.
5. **Trust the CPU share, not `nproc`.** Read the cgroup CPU quota. With fewer cores than running lines, predicts slow down until the time guards fire.
6. **Guards as shipped.** Keep official conditions and verify that no guard fired. The designed lighter second predict (layer 15) is allowed; any residual-guard lever, deadline promotion or safety-net entry is not. On such a predict, rerun the whole line; if it repeats, report it. In arms B–E a safety-net restore re-runs setup, which raises in the Track A check when `NO_CORR` is set and re-creates the A14 arrays, so a guard event invalidates the rest of that line.
7. **Label noise.** Report the raw MSE and the noise-corrected MSE, MSE − σ²/10⁹, for every arm.
8. **Frozen before the run.** Hash this file, the `env.txt` files and the runner before the full run, and add no arm after seeing results.

## 6. Time, memory and cost

- **Predict time.** A sibling build took 78–81 s per predict on a rented pod, in one process at one BLAS thread [tuning networks]. Official workers took a median of 103 s and at most 109 s for #337350 [official bundle]. Arm F bills about 1.6 times the FLOPs (C 0.231 against 0.142) and needs about 130 s [unverified]. Arms B–E bill about the same F as A, because the features of the removed parts are still computed.
- **Total.** Seven arms × 100 networks = 700 predicts, about 17 line-hours; with the three optional arms (C0, E′, F′), about 1,000 predicts and 25 line-hours [unverified].
- **Memory per line.** Client VmPeak about 7.3 GiB, the official cap being 8 GiB. In-process runs reached VmHWM 7.8 GB. Budget 12 GB of RAM per line, client and server together.
- **Machine.** 10 concurrent lines need about 20 physical cores at a matching CPU share and 128 GB of RAM. Wall time is then about 2 hours, or 3 hours with the optional arms.
- **Cost.** A held-out run of 640 predicts on the same pods cost about $8–10 (`experiments/claude_pair_20261014/fresh/G05-PREP.md`). Expect $10–20.

## 7. Outputs and analysis

Write a new analysis script, `abl_an_v1.py`, that reads the receipts. Take the paired set: networks that are OK in every arm (report n). Per arm:

| column | definition |
|---|---|
| GF | mean and maximum of F/10⁹ per predict |
| C | mean of max(0.1, F/2^41) |
| MSE | mean final-layer MSE (layer ℓ = 15), with a 95 % bootstrap interval over networks (10,000 resamples, seed 0) |
| MSE, noise-corrected | mean of (MSE_i − σ²_i/10⁹) |
| ratio to A | Σ MSE_arm / Σ MSE_A, with a 95 % bootstrap interval, wins (networks better than A), first and second half (networks 0–49, 50–99), worst network |
| Monte Carlo samples to match | N_match = Σ σ²_i / Σ MSE_i, also in budgets (N_match / 65,374.4) |
| error ratio vs Monte Carlo at equal compute | (Σ σ²_i / (65,374.4 · C)) / Σ MSE_i |
| per-layer MSE | 16 values per arm (to see where an arm loses accuracy) |
| guard events, hash checks | counts; must be 0 and pass |

Here 65,374.4 is the number of Monte Carlo samples one budget buys: 2^41 FLOPs over 33,637,376 FLOPs per sample (whestbench `mc_samples_at_budget`). The ratio to A is the quantity for the write-up. Absolute MSEs on the mini split are not the official values.

## 8. Reading rules (fixed before the run) and known traps

**Reading.**
- Report each comparison of §0 with its interval and halves. A change counts only if both halves have the same sign.
- The arms are nested, so the shares are sequence-order shares: B, C, D, D′ in that order. Do not add them as if they were independent.
- If D′ comes out far worse than A, the write-up says plainly that the score rests on the fitted parts. If D′ comes out close to A, it says the derived engine carries the accuracy. Either result goes in §1.3 and Table 1.2.
- E/B measures our mechanisms on a base without learned mean corrections, the reverse of the official order. Set it beside the official sequence-order 3.1 % (§5).

**Known traps.**
- `V17_NO_CORR=1` at import breaks setup (lines 9871–9872). Use the post-setup flip.
- `CI_STAR=0`, which would remove star-D4 and its fitted Ω, also switches off K31F: K31F reads star-D4's Ω, and its context becomes empty (lines 6397–6398, 6934). The K31-slot θ has no switch (line 1881). Both therefore stay on in D and D′. A fully derived arm needs a code-edited build with its own hash, which is outside this specification.
- `CI_GK` replaces the whole default string. An empty value turns all five multipliers off; it does not restore them.
- Deadline level 2 does nothing in this build: its row cap of 640 equals the largest cap of the band ladder (lines 11632–11634). Only level 1 (one fast-product level off) acts.
- The starting point runs its V29 branch at 1024 × 16 (coupling 16/1024 ≤ 0.05). It falls back to a sampled blend only after an exception (`research/corpaci/submission/estimator.py` lines 2905–2930, 3018–3045). Any such fallback shows as an F far from 0.23 × 2^41 and a non-repeating `pred_sha`.
- F of arms B–D overstates the cost of a fit-free build, because the removed parts' features are still computed (`_c_edge_pre` and `_cc_ta_prepare` run regardless of `NO_CORR`). Report the F as measured, with this caveat.

*Sources:* linked `estimator.py` (lines cited above); `research/corpaci/submission/estimator.py`; `research/corpaci/research/astra_k4_overshoot.md`; `experiments/claude_pair_20261014/fresh/podj164/run_net12.py`; `experiments/claude_pair_20261014/fresh/run_net5.py`; `experiments/claude_pair_20261014/fresh/SEALS.md`; `experiments/claude_corrector_c1/fit_v2_record/c2_split_v2.json`; `experiments/claude_pair_20261014/fresh/G05-PREP.md`; `experiments/claude_setupbox_20261020/post337451/guard_margin_v2.out`; `docs/REVIEW-RG-c2eko7R3-R.md`; `.venv/lib/python3.10/site-packages/whestbench/budget.py`; `docs/writeup/v9/SECTION-1-DERIVED-VS-FITTED-v9.md` (Tables 1.1, 1.2).
