# ARC White-Box Estimation Challenge 2026, Phase 2 — megan_park, submission #337350

This is the code release that accompanies our write-up, *How errors survive depth in cumulant propagation* (`writeup/`). The estimator predicts the mean activation of every neuron of a random ReLU network from its weights alone. It draws no samples.

## Contents

| path | what it is |
|---|---|
| `submission/estimator.py`, `submission/manifest.json` | The graded submission #337350, byte-identical to the uploaded tar (tar sha256 `f44bb724…`, estimator sha256 `dd17f7f4…`). |
| `LICENSE` | MIT. Copyright holders: AIcrowd (starter kit) and meganpark (this work). |
| `LICENSE-arc`, `LICENSE-public-k3`, `LICENSE-corpaci`, `LICENSE-acco-cyber`, `NOTICE-acco-cyber.md` | Licences of the prior work the estimator builds on (all MIT): ARC's cumulant propagation, 504aldo's V29 engine, corpaci's Phase 2 package #332675, and the public V34 alive-neuron masks. |
| `Licence_CeCILL-B_V1-en.txt`, `AUTHORS.md`, `README-plinopt-336-40.md` | Licence and attribution of the PLinOpt fast matrix-multiplication programs (CeCILL-B). |
| `INVENTORY.md` | Every constant and fitted part that changes the output, with its line in `estimator.py`. The write-up's Table 1.1 groups these rows. |
| `READING-GUIDE.md` | Glossary and code map: where each part of the write-up lives in the estimator (write-up Appendix E). |
| `scripts/survival/` | §2: the error-survival law against the stored error ledger, the closed forms, and the fresh-network simulations. They run in minutes on a laptop. |
| `scripts/supply/` | §4 and Appendix B: numeric checks of the supply-term derivations. |
| `scripts/ablation/`, `results/ablation/` | §5, Table 5.2: the ablation kit, its specification and run notes, the 900 per-network receipts and the table. |
| `scripts/depth/` | §3 checks. They read Monte Carlo receipts that are too large for this release; the write-up's Appendix F.7 says where they are kept. |

## Running the estimator

The estimator follows the starter-kit interface (`Estimator.setup`, `Estimator.predict`). It was graded with whestbench 0.16.1, flopscope 0.12.x and numpy 2.2.6 under Python 3.10. `scripts/ablation/run_abl_v1.py` shows a minimal in-process driver: setup once, then predict networks of the public `mini` split in order.

## Read this before reusing the code

- **No sampling.** The predict path draws no random numbers. Its low-rank sketches are fixed slices of the weight matrices.
- **Dead code in the header.** The header comment (lines 25–67) is inherited from the starting submission #332675. It describes a matched-Monte-Carlo fallback branch that this file does not contain. These parts of the file are also unused:
  - the constants at lines 371–381;
  - `_V25_STACK` (line 1467);
  - `_kprop3_layer_means` (line 1380).
- **Time guards.** A residual guard and a deadline guard can change the output when a predict runs late (write-up §1.4). For outputs that do not depend on timing, set these after import, as the ablation does: `_CF_RG_ON = False` and `_CF_DL_TRIGGER_S = _CF_DL_TRIGGER2_S = 1e9`.
- **Environment switches.** The estimator reads many of them. Their defaults are the graded configuration, and `INVENTORY.md` lists those that change values.
- **Fitted parts.** About 55,000 constants were fitted to Monte Carlo labels or set by hand. The ablation in §5 measures what they carry.

## How this code was written

The participant, megan_park, set the research direction and proposed most of the ideas that were tested. AI coding agents implemented them and ran the checks: Claude models via Claude Code, and OpenAI Codex. The write-up's §8 says who did what and how each claim was checked.
