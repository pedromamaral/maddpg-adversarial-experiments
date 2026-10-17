# Run registry — active on top
# id | host | container | command | started | ETA | output | status

E026 | server | maddpg-E026-sp86 | tools/static_table_search.py --config configs/v2_pilot4_lc.json --load 3 --episodes 16 --eps 0.5 --workers 6 --max-evals 1800 --start sp/E025 table/random:1/random:2 (docker -d --rm --cpus 6) | 10-15 21:30 | ~40 h (est, 4 starts) | host_data/results/E026_sp (server) | running
E026 | server | maddpg-E026-dc | same, configs/v2_deltacom.json --load 8, E025 table_v2_deltacom.json | 10-15 21:30 | ~40-80 h (21 s/J at M=8,8w) | host_data/results/E026_dc (server) | running
E027 | server | maddpg-E027-exp then run_eval containers E027_{B_s601,A_s602,B_s602}_v2_dc_lc_LC-Simple_{0,4,8} (--rm, --cpus 4, sequential) | tools/run_E027.sh (nohup, ~/idqn/run_E027.out; .done marks per cell) | 10-17 17:30 | ~2-4 h (est; E025 evals ran ~1 h/cell) | host_data/results/E027 (table_B, ss) + v2_dc_lc/E027_*/n_*/ (server) | running

## Finished (results on disk server + host_data/results locally, except own_flow/ server-only)
E021,E022 | server | dc_{local,central}_{fix,rhf}_s* | ~/idqn/run_dc*.sh | 10-12 | — | dc_ac_* | done
E020 | server | dc_step_train | ~/idqn/run_dc*.sh | 10-12 | — | v2_dc_lc | done
E023 | server | — | tools/own_flow_effect.py | 10-13 | — | own_flow/ | done
E024 | server | maddpg-E024-{v2_pilot4_lc,v2_deltacom} (--rm) | ~/idqn/run_E024.sh | 10-14 | 16:44 / 17:16 | supervised_signal/ (fetched) | done
