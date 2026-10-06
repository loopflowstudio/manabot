#!/bin/bash
# Started by PID 1 before source/bootstrap. No account credential is accepted.
set -eu
umask 077
: "${MANABOT_DEADLINE:?absolute Unix deadline required}"
: "${RUNPOD_POD_ID:?provider pod identity required}"
# The image supplies the pod-scoped runpodctl configuration. Its ability to
# terminate this pod must be established by a separate short guardian probe.
while [ "$(date +%s)" -lt "$MANABOT_DEADLINE" ]; do sleep 1; done
while true; do
    timeout 20 runpodctl remove pod "$RUNPOD_POD_ID" >/dev/null 2>&1 || true
    sleep 5
done
