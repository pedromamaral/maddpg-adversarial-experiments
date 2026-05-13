#!/usr/bin/env bash
# Show training progress: best-checkpoint mean_reward and per-epoch pkt_loss.
# Usage: ./check_progress_training.sh          # summary snapshot
#        ./check_progress_training.sh -f        # live follow (filtered)

SERVER="pedroamaral@10.26.110.14"

if [[ "$1" == "-f" ]]; then
  ssh "$SERVER" "
    docker logs maddpg_training 2>&1 | grep -E 'best checkpoint|epoch.*pkt_loss'
    docker logs -f maddpg_training 2>&1 | grep -E 'best checkpoint|epoch.*pkt_loss'
  "
else
  LOGS=$(ssh "$SERVER" "docker logs maddpg_training 2>&1")
  PY_SCRIPT=$(cat <<'PYEOF'
import sys, re

# Join wrapped lines (docker wraps long lines with leading whitespace)
raw = sys.stdin.read().splitlines()
lines = []
for l in raw:
    if lines and l.startswith(' '):
        lines[-1] += l.rstrip()
    else:
        lines.append(l.rstrip())

best_reward = {}; best_epoch = {}
last_epoch  = {}; last_pktloss = {}; order = []

for line in lines:
    m = re.search(r'\[TRAIN\] (\S+).*new best checkpoint at epoch (\d+) \(mean_reward=([^\)]+)\)', line)
    if m:
        v, ep, rw = m.group(1), m.group(2), m.group(3)
        best_reward[v] = float(rw); best_epoch[v] = int(ep)
        if v not in order: order.append(v)
    m = re.search(r'\[TRAIN\] (\S+)\s+epoch\s+(\d+)\s+reward=\s*[0-9.\-]+\s+pkt_loss=\s*([0-9.]+)%', line)
    if m:
        v, ep, pl = m.group(1), m.group(2), m.group(3)
        last_epoch[v] = int(ep); last_pktloss[v] = float(pl)
        if v not in order: order.append(v)

print()
print(f"{'Variant':<22} {'CurEpoch':>8}  {'BestMeanRew':>12}  {'@Epoch':>7}  {'PktLoss%':>9}")
print(f"{'-'*22} {'-'*8}  {'-'*12}  {'-'*7}  {'-'*9}")
for v in order:
    ep  = last_epoch.get(v, '?')
    br  = f"{best_reward[v]:.4f}" if v in best_reward else '--'
    bep = best_epoch.get(v, '--')
    pl  = f"{last_pktloss[v]:.2f}%" if v in last_pktloss else '--'
    print(f"{v:<22} {str(ep):>8}  {br:>12}  {str(bep):>7}  {pl:>9}")
print()
PYEOF
)
  echo "$LOGS" | python3 -c "$PY_SCRIPT"
fi