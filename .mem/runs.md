# Run registry — active on top
# id | host | container | command | started | ETA | output | status

E028 | server | maddpg-E028-{dc,sp}-s{0,1,2} (--rm, 2 cpus each) | tools/run_E028.sh (dqn, sp-pretrain 50, margin 0.05, hotspot+failures 0-8; dc then sp per seed) | 10-23 17:15 | unknown (dc first) | E028/{dc,sp}_s* ; done markers ~/idqn/run_E028.done | running
E031 | server | maddpg-E031-search-{dc,sp} (--rm, 5 cpus) then run_eval | tools/run_E031.sh (A037, --failures 0,4 from E026 table_best; eval n=0/4/8) | 10-23 17:15 | unknown (warm restart; dc first, then sp) | E031_{dc,sp} + v2_dc_lc/E031_dc, v2_p4_lc/E031_sp ; ~/idqn/run_E031.done | running

## Finished (results on disk server + host_data/results locally, except own_flow/ server-only)
E026 | server | maddpg-E026-{sp86,dc} (--rm) | tools/static_table_search.py (4 starts each) | 10-15 21:30 | sp 10-19 09:55 / dc 10-21 09:40 | E026_{sp,dc} (fetched 10-21) | done, analyse next
E027 | server | E027_<cell>_v2_dc_lc_LC-Simple_<n> (--rm) | tools/run_E027.sh | 10-17 17:30 | ~21:50 | v2_dc_lc/E027_* (fetched) | done, analysed
E021,E022 | server | dc_{local,central}_{fix,rhf}_s* | ~/idqn/run_dc*.sh | 10-12 | — | dc_ac_* | done
E020 | server | dc_step_train | ~/idqn/run_dc*.sh | 10-12 | — | v2_dc_lc | done
E023 | server | — | tools/own_flow_effect.py | 10-13 | — | own_flow/ | done
E024 | server | maddpg-E024-{v2_pilot4_lc,v2_deltacom} (--rm) | ~/idqn/run_E024.sh | 10-14 | 16:44 / 17:16 | supervised_signal/ (fetched) | done
