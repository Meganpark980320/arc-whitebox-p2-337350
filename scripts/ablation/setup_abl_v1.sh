#!/bin/bash
# setup_abl_v1.sh (on the pod, 22 Oct 2026): venv (flopscope 0.12.3 = the evaluator version since 17 Oct, whestbench 0.16.1, numpy 2.2.6 as in
# the official #337350 runtime) and the public mini split (7 shards, sha256 = HF lfs oids of revision v2-phase2, as pair_mini_v1.sh).
# Log /workspace/abl/setup.log, last line SETUP_OK or SETUP_FAIL.  Also arms a money backstop: the pod terminates itself after MAX_H hours.
set -u; cd /workspace; mkdir -p abl hfmini/data
MAX_H=${MAX_H:-5}
{
echo "cgroup cpu.max: $(cat /sys/fs/cgroup/cpu.max 2>/dev/null)  mem.max: $(cat /sys/fs/cgroup/memory.max 2>/dev/null)  nproc: $(nproc)"
lscpu | grep 'Model name' | head -1
# backstop: self-terminate with the pod-scoped key (it can only control this pod)
cat > /workspace/abl/backstop.py <<'PYEOF'
import json, sys, time, urllib.request
time.sleep(float(sys.argv[1]) * 3600)
env = {}
for l in open('/etc/rp_environment'):
    l = l.strip()
    if l.startswith('export '): l = l[7:]
    if '=' in l: k, v = l.split('=', 1); env[k.strip()] = v.strip().strip('"').strip("'")
q = {'query': 'mutation { podTerminate(input: {podId: "%s"}) }' % env['RUNPOD_POD_ID']}
r = urllib.request.Request('https://api.runpod.io/graphql', data=json.dumps(q).encode(),
                           headers={'Content-Type': 'application/json', 'Authorization': 'Bearer ' + env['RUNPOD_API_KEY']})
open('/workspace/abl/backstop.out', 'w').write(urllib.request.urlopen(r, timeout=60).read().decode()[:200])
PYEOF
setsid nohup python3 /workspace/abl/backstop.py $MAX_H > /workspace/abl/backstop.log 2>&1 < /dev/null &
echo "backstop armed: ${MAX_H} h"
for k in 1 2 3; do command -v uv >/dev/null && break; pip install -q --timeout 30 --break-system-packages uv 2>/dev/null || pip install -q --timeout 30 uv; sleep 3; done
[ -x /workspace/venv/bin/python ] || uv venv -q -p 3.10.13 /workspace/venv
for k in 1 2 3; do VIRTUAL_ENV=/workspace/venv uv pip install -q 'flopscope==0.12.3' 'whestbench==0.16.1' 'numpy==2.2.6' 'pyarrow==24.0.0' 'threadpoolctl==3.6.0' && break; sleep 5; done
/workspace/venv/bin/python -c "import whestbench, flopscope, pyarrow, threadpoolctl, numpy; print('KIT_READY flopscope', flopscope.__version__, 'numpy', numpy.__version__)" || { echo "KIT_FAIL"; exit 1; }
declare -A SHA=([0]=7da858bc60616456e7b55a545a91288c1edba612b556be2e6eae3d86df48639c [1]=9c65a4c4c878623032d62ca3dbb1c88fc88e724f84fdbdc9d7b77904fa100e1d
 [2]=f0fe736bdfddf2ebbb9d0e0df7bf68d0938abaaadcc34eeff143bbb26123387c [3]=7c987422c2ad849ff9bce073483d8c10152635398e2c528899af6944d5e0d110
 [4]=fedf0d67479eb7b8743c0d7caa5a4e3c2359e18560c3834f5dbda3232f735541 [5]=6eb84bd50068e8dcb0d8a6a609f2c4f90177d6228229ccb0793b656d7458fd59
 [6]=481bf94b30d81c005181891c9f4dbe540a1eec00dc24ec2da5f5fb45c3973254)
for k in ${SHARDS:-0 1 2 3 4 5 6}; do
  f=mini-$(printf '%05d' $k)-of-00007.parquet; F=/workspace/hfmini/data/$f
  U=https://huggingface.co/datasets/aicrowd/arc-whestbench-public-2026/resolve/v2-phase2/data/$f
  [ -s $F ] && [ "$(sha256sum $F | cut -c1-64)" = "${SHA[$k]}" ] && { echo "mini $k present"; continue; }
  for t in 1 2 3 4 5; do curl -sS -L -C - --retry 5 --retry-delay 5 --speed-limit 200000 --speed-time 60 -o $F.part $U && break; echo "curl retry $t"; sleep 5; done
  [ "$(sha256sum $F.part | cut -c1-64)" = "${SHA[$k]}" ] && mv $F.part $F && echo "mini $k SHA_OK" || { echo "mini $k SHA_BAD"; exit 1; }
done
/workspace/venv/bin/python -c "
import pyarrow.parquet as pq; s = pq.read_schema('/workspace/hfmini/data/mini-00000-of-00007.parquet'); print('columns', s.names)"
} > /workspace/abl/setup.log 2>&1 && echo SETUP_OK >> /workspace/abl/setup.log || echo SETUP_FAIL >> /workspace/abl/setup.log
tail -1 /workspace/abl/setup.log
