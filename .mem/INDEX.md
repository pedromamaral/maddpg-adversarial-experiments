# Routing — task -> read these (and nothing else by default)

| task | read | skip |
|---|---|---|
| any session start | STATE.md, this file | *-archive.md, import-archive.md |
| "have we tried X?" | rg -i "X" avenues.md experiments/LEDGER.md conclusions.md | reading whole files |
| analyse results / next experiment (research-think) | conclusions.md, avenues.md, project.md, results/E###.json | raw logs |
| paper planning / writing | paper.md, conclusions.md, strategy.md; numbers via tools/paper1_tables.py | archive_v1 |
| implement (research-do) | codebase.md, the experiment card; src by line range (rg -n first) | standalone_experiment_runner.py whole (2985 lines) |
| deploy / monitor / fetch | infra.md, runs.md; tools/runs_status.sh | streaming logs |
| summarise a finished run | tools/summarize_run.py host_data/results/<dir> --id E### | cat of JSON/logs |
| decisions / why | decisions.md | — |
| v1 / Paper 2 / theses history | host_data/archive_v1, paper/PAPER2_ROADMAP.md (only if asked) | — |

Note: results/E###.json "sha" = repo HEAD at summary time, not the training commit (see LEDGER).
