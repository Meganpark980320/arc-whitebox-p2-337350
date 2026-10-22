# Ablation of #337350: how the run differs from the specification

*22 October 2026. The specification (`ABLATION-SPEC-v9.md`) was written before the run and is left unchanged. These are the deviations, decided before any full-run result was read.*

1. **Harness: in-process flopscope, not the IPC server.** Each line is one process that imports the build, calls `setup` once and predicts its 20 networks in order (`experiments/claude_abl_20261022/run_abl_v1.py`). The values are computed by the same numpy calls. Request counts and residual time are not comparable with official records, and the ablation does not use them.
2. **SDK:** flopscope 0.12.3 (the evaluator version since 17 October), whestbench 0.16.1 and numpy 2.2.6 (the official runtime of #337350), for every arm.
3. **Time guards disabled in every arm of the linked build** (`noguard`): `_CF_RG_ON = False`, and `_CF_DL_TRIGGER_S = _CF_DL_TRIGGER2_S = 1e9`.
   - *Why.* Zen 4/5 boxes were out of stock, and on slower hosts the guards would fire on wall time. They would then change values differently in different arms.
   - *What it switches off.* All residual-guard levers, including the head lever and the sticky rule, act through `_cf_rg_step`, which is called only when `_CF_RG_ON` is true. The deadline guard compares projected time with the two triggers.
   - *What stays.* The position-based light first two predicts after setup and the exception safety net stay as shipped.
   - *Cost to fidelity.* In the official run of #337350 no limit flag fired, and one network showed a lever-sized compute rise. Arm A is therefore the shipped build without time-dependent behaviour.
4. **Layout: one box per official worker.** Box w predicts networks w, w + 5, …, w + 95, and runs all nine arms on them. Every comparison of arms is thus a same-machine pair, whatever the box's CPU family. Boxes of different CPU families may differ in the last bits; cross-box byte identity is not required.
5. **Arms run:** A, B, C, D, D′, E, E′ (optional in the specification), F and F′ (optional). Z and a repeat of A run in the smoke test only. C0 was not run.
6. **Gate before the full run** (smoke test, networks 0 and 5 on box 1):
   - A twice gives the same `pred_sha`;
   - Z equals A;
   - every other arm differs from A;
   - no guard event;
   - client VmPeak below 8,190 MiB;
   - every flip holds after every predict.
