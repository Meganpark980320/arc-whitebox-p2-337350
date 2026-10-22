# ARC White-Box Estimation Challenge 2026, Phase 2 — megan_park, submission #337350

Code release for the write-up *How errors survive depth in cumulant propagation*. The estimator predicts the mean activation of every neuron of a random ReLU network from its weights alone. It draws no samples at prediction time.

**Start here:** [`writeup/`](writeup/) has the write-up as a PDF and as Markdown.

## Layout

| folder | what it holds |
|---|---|
| [`writeup/`](writeup/) | The write-up: PDF and Markdown. |
| [`submission/`](submission/) | The graded submission #337350, byte-identical to the uploaded tar (tar sha256 `f44bb724…`, estimator sha256 `dd17f7f4…`). |
| [`docs/`](docs/) | `INVENTORY.md`: every constant and fitted part that changes the output, with its line in `estimator.py`. `READING-GUIDE.md`: code map, time guards and cost engineering. |
| [`checks/`](checks/) | Scripts behind the write-up's checks. `depth-law/`: §2, survival law, closed forms, simulations, Figure 2. `depth-error/`: §3. `supply-terms/`: §4 and Appendix B. |
| [`ablation/`](ablation/) | §5's ablation. `kit/`: the runner, driver, setup, analyses, arm settings, specification and run notes. `results/`: the per-network receipts of both rounds and the tables. |
| [`feedback/`](feedback/) | §6.3 and Appendix H: scripts that reproduce four flopscope issues (a tuple `axis` in client–server mode, einsum's operand-sized float64 buffers, `out=` identity, `weights.csv`), a verified patch for the einsum one, and our outputs. |
| [`licenses/`](licenses/) | Licences and notices of the prior work the estimator builds on (see below). |
| `LICENSE` | MIT. Copyright holders: AIcrowd (starter kit) and meganpark (this work). |
| `SHA256SUMS` | Checksums of every file. |

## Prior work and licences

The estimator builds on:
- ARC's cumulant propagation (`licenses/LICENSE-arc`, MIT);
- 504aldo's V29 engine (`licenses/LICENSE-public-k3`, MIT);
- corpaci's Phase 2 package #332675 (`licenses/LICENSE-corpaci`, MIT);
- the public V34 alive-neuron masks (`licenses/LICENSE-acco-cyber` and `licenses/NOTICE-acco-cyber.md`, MIT);
- PLinOpt's fast matrix-multiplication programs (`licenses/Licence_CeCILL-B_V1-en.txt`, `licenses/AUTHORS-plinopt.md`, `licenses/README-plinopt.md`, CeCILL-B).

## Running the estimator

The estimator follows the starter-kit interface (`Estimator.setup`, `Estimator.predict`). It was graded with whestbench 0.16.1, flopscope 0.12.x and numpy 2.2.6 under Python 3.10. `ablation/kit/run_abl_v2.py` is a minimal in-process driver: it calls setup once, then predicts networks of the public splits in order.

## Read this before reusing the code

- **No sampling.** The predict path draws no random numbers. Its low-rank sketches are fixed slices of the weight matrices.
- **Dead code in the header.** The header comment (lines 25–67) is inherited from the starting submission #332675. It describes a matched-Monte-Carlo fallback that this file does not contain. These are also unused: the constants at lines 371–381, `_V25_STACK` (line 1467) and `_kprop3_layer_means` (line 1380).
- **Time guards.** A residual guard and a deadline guard can change the output when a predict runs late. For outputs that do not depend on timing, set `_CF_RG_ON = False` and `_CF_DL_TRIGGER_S = _CF_DL_TRIGGER2_S = 1e9` after import, as the ablation does.
- **Fitted parts.** About 55,000 constants were fitted to Monte Carlo labels or set by hand. `docs/INVENTORY.md` lists them, and §5 of the write-up measures what they carry.

## How this code was written

The participant, megan_park, set the research direction and proposed most of the ideas that were tested. AI coding agents implemented them and ran the checks: Claude models via Claude Code, and OpenAI Codex. §8 of the write-up says who did what and how each claim was checked.
