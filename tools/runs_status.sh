#!/bin/bash
# One call -> <=20-line status of all active runs listed in .mem/runs.md.
# Active row format: E042-s1 | server | <container> | <command> | started | ETA | <output> | running
# Per host one ssh: our containers (docker ps) + last 2 lines of each run's log (~/idqn/<container>.log)
# + the campaign completion markers (~/idqn/*.done). Flags finished/crashed rows.
#   tools/runs_status.sh
cd "$(dirname "$0")/.." || exit 1
ssh_target() { case $1 in server) echo pedroamaral@10.26.110.14;; *) echo "$1";; esac; }
rows=$(grep -E '\|[[:space:]]*running[[:space:]]*$' .mem/runs.md)
if [ -z "$rows" ]; then echo "no active runs in .mem/runs.md"; exit 0; fi
for host in $(echo "$rows" | awk -F'|' '{gsub(/ /,"",$2); print $2}' | sort -u); do
  ctrs=$(echo "$rows" | awk -F'|' -v h="$host" '{gsub(/ /,"",$2); gsub(/^ +| +$/,"",$3); if ($2==h) print $3}' | sed 's/^ctr //')
  target=$(ssh_target "$host")
  echo "== $host"
  ssh -o ConnectTimeout=8 -o BatchMode=yes "$target" "
    running=\$(docker ps --format '{{.Names}}|{{.Status}}')
    for c in $(echo $ctrs); do
      for n in \$(echo \"\$running\" | cut -d'|' -f1 | grep -E \"^\${c//\\*/.*}\$\" ); do
        echo \"  RUN  \$n  \$(echo \"\$running\" | grep \"^\$n|\" | cut -d'|' -f2)\"
        tail -n 2 ~/idqn/\$n.log 2>/dev/null | cut -c1-110 | sed 's/^/       /'
      done
      if ! echo \"\$running\" | cut -d'|' -f1 | grep -qE \"^\${c//\\*/.*}\$\"; then
        log=\$(ls -t ~/idqn/\${c//\\*/}*.log 2>/dev/null | head -1)
        if [ -n \"\$log\" ] && grep -qiE 'Traceback|Error|Killed' \"\$log\"; then echo \"  CRASH? \$c  (\$log)\"; else echo \"  DONE?  \$c  (no container)\"; fi
      fi
    done
    echo '  .done:'; tail -n 3 \$(ls -t ~/idqn/*.done 2>/dev/null | head -1) 2>/dev/null | sed 's/^/    /'
    echo \"  load \$(cut -d' ' -f1 /proc/loadavg)  gpu \$(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader)\"
  " 2>&1 || echo "  ssh to $host failed (VPN up?)"
done | head -n 20
