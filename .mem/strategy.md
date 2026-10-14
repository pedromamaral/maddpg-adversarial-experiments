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
- Lit-graph topic folder: TODO(ask) — OneDrive-FCTNOVA/Docencia/Research/lit-graph has no topic
  folder yet; earlier partial harvest (76 landmark papers) was lit-graph/drl-routing-adversarial/
  in the repo (no longer present). Lit access: publisher sites via Claude in Chrome (VPN).
- Risk: reviewers read C3/C4 as a negative result; framing must be "what is learned + why".
