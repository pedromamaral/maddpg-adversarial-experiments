# Run registry — active on top
# id | host | container | command | started | ETA | output | status


## Finished (results on disk server + host_data/results locally, except own_flow/ server-only)
E021,E022 | server | dc_{local,central}_{fix,rhf}_s* | ~/idqn/run_dc*.sh | 10-12 | — | dc_ac_* | done
E020 | server | dc_step_train | ~/idqn/run_dc*.sh | 10-12 | — | v2_dc_lc | done
E023 | server | — | tools/own_flow_effect.py | 10-13 | — | own_flow/ | done
E024 | server | maddpg-E024-{v2_pilot4_lc,v2_deltacom} (--rm) | ~/idqn/run_E024.sh | 10-14 | 16:44 / 17:16 | supervised_signal/ (fetched) | done
