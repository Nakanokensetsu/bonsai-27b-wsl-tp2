#!/bin/sh
# Prune the SSD KV tier. vLLM's fs tier has no size limit or eviction of its own.
#   1) delete blocks not used for KV_PRUNE_AGE_MIN minutes (default 120)
#   2) if the total is still above KV_PRUNE_CAP_GB (default 100), delete least recently used first
# "Used" = max(atime, mtime). With BONSAI_KV_FS_TOUCH=1 (bonsai_ext) every store/load refreshes
# mtime, so this order is the real LRU order. A block deleted while being read is just a miss.
# Usage: kvcache_prune.sh <dir>   (serve.sh runs it every KV_PRUNE_INTERVAL_SEC while vLLM is up)
D=${1:?usage: kvcache_prune.sh <kvcache dir>}
AGE_MIN=${KV_PRUNE_AGE_MIN:-120}
CAP_BYTES=$(( ${KV_PRUNE_CAP_GB:-100} * 1024 * 1024 * 1024 ))
[ -d "$D" ] || exit 0
before=$(du -sb "$D" | cut -f1)
# *.tmp are leftovers of interrupted writes
n1=$(find "$D" -type f \( -name '*.bin' -amin +$AGE_MIN -mmin +$AGE_MIN -o -name '*.tmp' -mmin +$AGE_MIN \) -print -delete | wc -l)
n2=$(find "$D" -type f -name '*.bin' -printf '%A@ %T@ %s %p\n' | awk '
  { t = ($1 > $2) ? $1 : $2; sub(/^[^ ]+ [^ ]+ /, ""); print t, $0 }' | sort -n | awk -v cap=$CAP_BYTES '
  { line[NR] = $0; split($0, f, " "); sz[NR] = f[2]; total += f[2] }
  END { i = 1; while (total > cap && i <= NR) { p = line[i]; sub(/^[^ ]+ [^ ]+ /, "", p); print p; total -= sz[i]; i++ } }' |
  while IFS= read -r f; do rm -f -- "$f" && echo x; done | wc -l)
find "$D" -mindepth 1 -type d -empty -delete
after=$(du -sb "$D" | cut -f1)
echo "kvcache_prune: deleted ${n1}(age) + ${n2}(cap) files, $(( (before - after) / 1048576 )) MiB freed, now $(( after / 1048576 )) MiB"
