# Run registry — active on top
# id | host | container | command | started | ETA | output | status

E030 | server | maddpg-E030-{dcW,spW,dcD,dcC}_s{0,1,2} (--rm, 2 cpus) | tools/run_E030.sh @5599288 (2 chains per seed: dcW->spW, dcD->dcC) | 11-05 14:58 | ~12-24 h (estimate) | E030/<arm>_s<seed>; ~/idqn/run_E030.done (6 lines) | running
E034+E035 | server | maddpg-E034-{dqn,lc,cc}-s*-{base,path,all} -> maddpg-E035-{p4f,lc,cc}-s*-{lure,repel,margin,rnd03,rnd10} (--rm, 1.5 cpus) | tools/run_E034.sh @5599288 (3 chains; ends with worst rule n=0/4/8) | 11-05 14:58 | ~1 day (estimate) | E034/, E035/, v2_p4_lc/E034_tab_*, v2_p4_lc/E035_worst_n*; ~/idqn/run_E034.done (7 lines) | running
E033 | server | maddpg-E033-nofb-sp-s0 (+nofb-dc, train-sp, s0-2; --rm 4 cpus) | tools/run_E033.sh (A: E028 ckpts eval no fallback n=0/4/8; B: E028 recipe w/o --fallback-margin, SP-86, 750 eps) | 10-30 18:43 | done 10-30 23:12 | E033/{nofb_sp,nofb_dc,train_sp}_s* ; ~/idqn/run_E033.done | done, fetched + analysed 11-04 (results/E033.json)
E028 | server | maddpg-E028-{dc,sp}-s{0,1,2} (--rm, 2 cpus each) | tools/run_E028.sh (dqn, sp-pretrain 50, margin 0.05, hotspot+failures 0-8; dc then sp per seed) | 10-23 17:15 | done 10-24 04:50 | E028/{dc,sp}_s* ; done markers ~/idqn/run_E028.done | done, fetched 10-28 (results/E028_{dc,sp}.json), analyse next
E031 | server | maddpg-E031-* | tools/run_E031.sh (A037, --failures 0,4 from E026 table_best; eval n=0/4/8) | 10-23 17:15 | dc search+eval done 10-24 11:19; sp search running (47.8 s/J) | E031_{dc,sp} + v2_dc_lc/E031_dc, v2_p4_lc/E031_sp ; ~/idqn/run_E031.done | done 10-25 09:54 (sp search+eval), fetched 10-28 (results/E031_{dc,sp,eval_dc,eval_sp}.json), analyse next

## Finished (results on disk server + host_data/results locally, except own_flow/ server-only)
E026 | server | maddpg-E026-{sp86,dc} (--rm) | tools/static_table_search.py (4 starts each) | 10-15 21:30 | sp 10-19 09:55 / dc 10-21 09:40 | E026_{sp,dc} (fetched 10-21) | done, analyse next
E027 | server | E027_<cell>_v2_dc_lc_LC-Simple_<n> (--rm) | tools/run_E027.sh | 10-17 17:30 | ~21:50 | v2_dc_lc/E027_* (fetched) | done, analysed
E021,E022 | server | dc_{local,central}_{fix,rhf}_s* | ~/idqn/run_dc*.sh | 10-12 | — | dc_ac_* | done
E020 | server | dc_step_train | ~/idqn/run_dc*.sh | 10-12 | — | v2_dc_lc | done
E023 | server | — | tools/own_flow_effect.py | 10-13 | — | own_flow/ | done
E024 | server | maddpg-E024-{v2_pilot4_lc,v2_deltacom} (--rm) | ~/idqn/run_E024.sh | 10-14 | 16:44 / 17:16 | supervised_signal/ (fetched) | done
