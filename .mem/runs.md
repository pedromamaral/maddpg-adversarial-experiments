# Run registry — active on top
# id | host | container | command | started | ETA | output | status

E025 | server | maddpg-E025-exp-{v2_deltacom,v2_pilot4_lc} then run_eval (EVAL_NAME E025_<cfg>) | ~/idqn/run_E025.sh (export table -> run_eval.sh, RULES=table random+flow greedy greedy_stale2 sp, n=0/4/8, policy dirs v2_dc_lc / v2_p3_lc_w05 LC-Simple, rules only) | 10-14 | ~1h | host_data/results/E025/ + v2_dc_lc/E025_v2_deltacom, v2_p3_lc_w05/E025_v2_pilot4_lc | running

## Finished (results on disk server + host_data/results locally, except own_flow/ server-only)
E021,E022 | server | dc_{local,central}_{fix,rhf}_s* | ~/idqn/run_dc*.sh | 10-12 | — | dc_ac_* | done
E020 | server | dc_step_train | ~/idqn/run_dc*.sh | 10-12 | — | v2_dc_lc | done
E023 | server | — | tools/own_flow_effect.py | 10-13 | — | own_flow/ | done
E024 | server | maddpg-E024-{v2_pilot4_lc,v2_deltacom} (--rm) | ~/idqn/run_E024.sh | 10-14 | 16:44 / 17:16 | supervised_signal/ (fetched) | done
