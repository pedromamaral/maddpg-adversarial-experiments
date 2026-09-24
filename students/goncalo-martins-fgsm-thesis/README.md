# Student 1 — Gonçalo Martins — MSc Thesis on FGSM Robustness

**Task.** Write the MSc thesis on the FGSM (observation-space) adversarial-attack
results that **already exist** — no new experiments are required. The data is
collected, the figures are rendered and committed here, and the narrative is written
up in the guide. Your job is to turn this into a thesis.

## What's in this directory
| file | what it is |
|---|---|
| `RESULTS_OUTLINE.md` | **start here** — the current, section-by-section guide to Chapter 8 (PT-PT): figure, numbers, how to explain, what not to claim. Updated 24/09/2026 |
| `FEEDBACK_2026-09-23.md` | feedback on the 17 Sep draft, including answers to the Chapter 6 doubts |
| `msc_fgsm_robustness_guide.md` | the original July guide. Background only: its GNN (T6) and damage-ceiling (T7) statements are superseded by `RESULTS_OUTLINE.md` |
| `figures/T*.pdf/.png` | the finished thesis figures, committed so you can drop them straight into your document (PDF for LaTeX, PNG for previews) |

## The thesis in one line
> FGSM flips ~26% of routing decisions (3× a random control), but the K-path redundancy
> absorbs most of them: FGSM extracts between ~0 and ~30% of each victim's damage ceiling
> (7–21 pp). Part of that weakness is FGSM's own objective, whose gradient the saturated
> policy masks; a single step on the logits does more damage on three victims, yet no
> gradient attack tested extracts more than ~35% of the ceiling. Under link failures a
> random perturbation is as damaging as FGSM, so that fragility belongs to the network.

`RESULTS_OUTLINE.md` expands each clause with the mechanism, the numbers, and the figure
that shows it.

## The figures (already rendered)
| fig | shows |
|---|---|
| T1 | flip-rate vs ε, gradient vs random |
| T2 | decisions changed vs delivery lost, 7 variants |
| T3 | adversarial-specific gap by variant, 95% CI (appendix only; use T3b) |
| T3b | the same gap on three independently trained victims per variant |
| T4 | failure fragility: gradient vs random with CI bands |
| T5 | flip-rate vs number of failures |
| T6 | decisions flipped by random noise, by FGSM with the GNN encoder bypassed, and by FGSM through the encoder |
| T7 | damage ceiling vs load: policy between random routing and the worst path |
| T8 | iterated attacks (PGD, MI-FGSM) against single-step FGSM |
| T9 | single-step attack on the logits against FGSM |

On the GNN variants, every FGSM number in these figures comes from the attack aimed
through the victim's real decision path (encoder included); see `RESULTS_OUTLINE.md` §8.7.

To **regenerate** them from the raw result JSON (only if you change something):
```bash
# from the repo root, with matplotlib available; PAPER_MODE drops the in-figure titles
PAPER_MODE=1 python tools/plot_thesis.py            # all figures, into figures/
PAPER_MODE=1 python tools/plot_thesis.py t6 t9      # only the named ones
```
The underlying results live in `host_data/results/` (`fgsm_tighten/` for the 15-episode
paired run, `logit_attack/` for the GNN-corrected and logit runs) — obtain them from the
server if you need to re-derive numbers; see the root `README.md` §Attacks.

## What you do NOT need
- No GPU, no docker, and no server access are needed **just to write** — the figures
  and numbers are here. You only need them if you want to regenerate or extend.
- No new attack runs. The companion paper `paper/paper2_fgsm.tex` is the reference
  write-up of the same results; align terminology with it.

## Related
- Root `README.md` — how the whole pipeline works, where weights/results live.
- `../miguel-chen-learned-adversary/` — the *next* attack (learned adversary); not your
  scope, but your Act-3 result is exactly what motivates it.
