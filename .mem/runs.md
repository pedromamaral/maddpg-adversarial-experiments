# Run registry — active on top
# id | host | container | command | started | ETA | output | status


## Finished (results on disk server + host_data/results locally, except own_flow/ server-only)
E026b | server | maddpg-E026b-credit-*, E026b_* (--rm) | tools/run_E026b.sh | 10-22 17:45 | credit 17:58, eval 19:17 | E026b/credit_{dc,sp}.json; v2_dc_lc/E026b_dc, v2_p4_lc/E026b_sp | finished, fetch next
E026 | server | maddpg-E026-{sp86,dc} (--rm) | tools/static_table_search.py (4 starts each) | 10-15 21:30 | sp 10-19 09:55 / dc 10-21 09:40 | E026_{sp,dc} (fetched 10-21) | done, analyse next
E027 | server | E027_<cell>_v2_dc_lc_LC-Simple_<n> (--rm) | tools/run_E027.sh | 10-17 17:30 | ~21:50 | v2_dc_lc/E027_* (fetched) | done, analysed
E021,E022 | server | dc_{local,central}_{fix,rhf}_s* | ~/idqn/run_dc*.sh | 10-12 | — | dc_ac_* | done
E020 | server | dc_step_train | ~/idqn/run_dc*.sh | 10-12 | — | v2_dc_lc | done
E023 | server | — | tools/own_flow_effect.py | 10-13 | — | own_flow/ | done
E024 | server | maddpg-E024-{v2_pilot4_lc,v2_deltacom} (--rm) | ~/idqn/run_E024.sh | 10-14 | 16:44 / 17:16 | supervised_signal/ (fetched) | done
