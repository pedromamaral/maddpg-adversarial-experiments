# Student 1 — Gonçalo Martins — MSc Thesis on FGSM Robustness

**Task.** Write the MSc thesis on the FGSM (observation-space) adversarial-attack
results that **already exist** — no new experiments are required. The data is
collected, the figures are rendered and committed here, and the narrative is written
up in the guide. Your job is to turn this into a thesis.

## What's in this directory
| file | what it is |
|---|---|
| `msc_fgsm_robustness_guide.md` | **start here** — the full 3-act narrative, the concepts to explain, a suggested chapter structure, the key numbers, and the pitfalls to avoid |
| `figures/T1..T7 .pdf/.png` | the finished thesis figures, committed so you can drop them straight into your document (PDF for LaTeX, PNG for previews) |

## The thesis in one line
> FGSM flips ~26% of routing decisions (3× a random control) but the service-provider
> topology's K-path redundancy absorbs the flips, so delivery barely moves (≤2.8pp of a
> 21pp damage ceiling); under link failures delivery does collapse, but a random
> perturbation is as damaging as the gradient, so that fragility is noise, not adversary.

The guide expands each clause with the mechanism, the numbers, and the figure that
shows it.

## The figures (already rendered)
| fig | shows |
|---|---|
| T1 | flip-rate vs ε, gradient vs random |
| T2 | decisions changed vs delivery lost, 7 variants |
| T3 | adversarial-specific gap by variant, 95% CI |
| T4 | failure fragility: gradient vs random with CI bands (the key figure) |
| T5 | flip-rate vs number of failures |
| T6 | GNN decision-robustness (with the CC-Duelling-GNN exception) |
| T7 | damage ceiling vs achieved damage |

To **regenerate** them from the raw result JSON (only if you change something):
```bash
# inside the maddpg-exp docker image, from the repo root
python tools/plot_thesis.py         # writes here, into figures/
```
The underlying results live in `host_data/results/fgsm_tighten/` (per-variant, the
15-episode paired run) — obtain that directory from the server if you need to re-derive
numbers; see the root `README.md` §Attacks.

## What you do NOT need
- No GPU, no docker, and no server access are needed **just to write** — the figures
  and numbers are here. You only need them if you want to regenerate or extend.
- No new attack runs. The companion paper `paper/paper2_fgsm.tex` is the reference
  write-up of the same results; align terminology with it.

## Related
- Root `README.md` — how the whole pipeline works, where weights/results live.
- `../miguel-chen-learned-adversary/` — the *next* attack (learned adversary); not your
  scope, but your Act-3 result is exactly what motivates it.
