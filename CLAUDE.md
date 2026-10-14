# maddpg-adversarial-experiments

Multi-agent RL routing (MADDPG / DQN, 14 PE agents) on an 86-node service-provider topology
(+ Deltacom). Paper 1 v2 (TNSM): credit assignment, not architecture, decides whether MARL
routing learns; what is learned is a demand-robust static table. Paper 2 (attacks) on hold.

**Start every session by reading `.mem/STATE.md`, then `.mem/INDEX.md`; read only what it routes.**
Skills: research-memory (init/state/tidy), research-think (analyse/ideate/paper), research-do
(implement/deploy/monitor/fetch/figures).

## Context rules
1. STATE.md + INDEX.md first; code by line range (`rg -n` first). src/standalone_experiment_runner.py
   is ~3000 lines — never read whole.
2. Recall by grep: `rg -i "<kw>" .mem/avenues.md .mem/experiments/LEDGER.md .mem/conclusions.md`.
3. Raw outputs never enter context: tools/summarize_run.py, tools/runs_status.sh, `tail -n 30`,
   `rg -m 20`. Never cat logs or result JSONs; never stream a job.
4. Bulk reading -> subagent returning <=20 lines with file:line refs.
5. Replies <=25 lines unless asked; tables over prose.
6. Write decisions / conclusions / avenues to .mem/ immediately.
7. One phase per session; end with research-memory `state`, then /clear.
8. Think phases on the strongest model; do phases on a cheaper one.

## Hard rules
- Deploy code via git push + `git pull` on the server; never scp code.
- Mount src at /workspace/src in docker (baked copy is stale).
- Never delete exited containers or other projects' images on the shared server.
- Paper numbers only from tools/paper1_tables.py (host_data/results).
- No secrets in .mem/.
