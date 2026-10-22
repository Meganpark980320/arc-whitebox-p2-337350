#!/usr/bin/env bash
# Run every feedback script in-process and, where it matters, in client-server mode; write outputs/.
# SERVER_PY: python with flopscope==0.12.3, numpy==2.2.6 and flopscope-server (v0.12.3 tag)
# CLIENT_PY: python with flopscope-client (v0.12.3 tag), no numpy
# PATCHED:   optional directory holding a copy of the flopscope package with einsum-dummies.patch applied
set -u
cd "$(dirname "$0")"
mkdir -p outputs
: "${SERVER_PY:?set SERVER_PY}" "${CLIENT_PY:?set CLIENT_PY}"
SOCK="$(mktemp -u /tmp/flopscope-feedback-XXXX.sock)"
"$SERVER_PY" -m flopscope_server --url "ipc://$SOCK" --timeout 60 2> outputs/server.log &
SERVER=$!
trap 'kill $SERVER 2>/dev/null' EXIT
for _ in $(seq 100); do [ -S "$SOCK" ] && break; "$SERVER_PY" -c 'import time; time.sleep(0.1)'; done
client() { FLOPSCOPE_SERVER_URL="ipc://$SOCK" "$CLIENT_PY" "$@"; }

{ echo "## in-process";    "$SERVER_PY" t1_tuple_axis.py; echo "## client-server"; client t1_tuple_axis.py; } > outputs/t1_tuple_axis.txt 2>&1
{ echo "## in-process";    "$SERVER_PY" t3_out_identity.py; echo "## client-server"; client t3_out_identity.py; } > outputs/t3_out_identity.txt 2>&1
{ echo "## stock";         "$SERVER_PY" t2_einsum_memory.py
  for w in numpy flopscope; do "$SERVER_PY" t2_einsum_rlimit.py $w; done
  "$SERVER_PY" t2_patch_check.py stock
  if [ -n "${PATCHED:-}" ]; then
    echo "## patched ($PATCHED)"
    PYTHONPATH="$PATCHED" "$SERVER_PY" t2_einsum_memory.py
    PYTHONPATH="$PATCHED" "$SERVER_PY" t2_einsum_rlimit.py flopscope
    PYTHONPATH="$PATCHED" "$SERVER_PY" t2_patch_check.py patched
  fi; } > outputs/t2_einsum.txt 2>&1
"$SERVER_PY" t4_weights_csv.py > outputs/t4_weights_csv.txt 2>&1
grep -h -v -E 'Warning|^  ' outputs/t*.txt
