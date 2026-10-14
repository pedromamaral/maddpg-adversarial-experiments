# Strategy (2026-10-14)

- Target venue: IEEE Transactions on Network and Service Management (Paper 1 v2).
- Novelty claim: first controlled study showing per-decision credit (not algorithm or
  architecture) determines whether MARL routing learns; ablations show the result is a
  demand-robust static table; second topology separates static vs state-dependent headroom.
- Gaps: G1 MARL-routing papers rarely compare to ECMP/greedy under realistic (pinned, stale)
  conditions; G2 no observation ablations showing state is actually used; G3 credit assignment
  over sparse per-flow decisions largely unaddressed.
- Strongest baselines (ours): greedy (fresh telemetry), greedy_stale2/4, ECMP (random+flow), SP.
  Literature baselines with numbers: TODO(ask) — no lit-graph digest yet for this topic.
- Lit-graph topic: drl-routing-adversarial (OneDrive-FCTNOVA/Docencia/Research/lit-graph/drl-routing-adversarial/).
  Folder NOT YET CREATED: the 2026-09-29 harvest (76 landmark papers, raw/seeds_s2.json) was never
  committed and is not on disk or server (searched 2026-10-14) -> must be redone. Lit access: publisher
  sites via Claude in Chrome (VPN).
- Risk: reviewers read C3/C4 as a negative result; framing must be "what is learned + why".

## Gate 2026-10-14 -> ITERATE
Blockers: A027 (no lit digest: novelty, venue fit, literature numbers unverified), A028 (no published
method in our setting), A024/E024 (C4 rests on it). C1-C3 evidence sufficient (effects >> 1.3 pp noise).
