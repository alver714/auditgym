#!/usr/bin/env bash
# Isolation checks inside the shipped image (Python 3.11, runs as root, no network), as run by CI job
# `isolation-docker`. The default CI runner is not root, so there the uid sandbox tests are skipped; here they run.
#   scripts/docker_isolation_check.sh [image]      (default image: auditgym)
set -euo pipefail
IMG="${1:-auditgym}"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT
run() { docker run --rm --network none "$@"; }

echo "== python version in the image"
run "$IMG" python -c 'import sys; print(sys.version); assert sys.version_info[:2] == (3, 11)'

echo "== isolation tests (uid sandbox, not skipped: root inside the container)"
run "$IMG" python -m pytest -q -p no:cacheprovider -rs tests/test_isolation.py | tee "$WORK/pytest.txt"
if grep -q "SKIPPED" "$WORK/pytest.txt"; then echo "isolation tests were skipped inside the image"; exit 1; fi

echo "== escape-probe self-test"
run "$IMG" python -m auditgym isolate --self-test

echo "== README command with a fresh 0644 seeds file on a read-only mount: must refuse cleanly"
mkdir -p "$WORK/secrets"
echo '[{"seed": 910001, "config": "holdout"}, {"seed": 910002, "config": "holdout"}]' > "$WORK/secrets/hidden_seeds.json"
chmod 755 "$WORK/secrets"; chmod 644 "$WORK/secrets/hidden_seeds.json"
set +e
run -v "$WORK/secrets:/secrets:ro" "$IMG" python -m auditgym isolate --policy careful_auditor \
    --episodes /secrets/hidden_seeds.json --log-dir /app/reports/isolated_logs 2> "$WORK/refused.txt"
rc=$?
set -e
cat "$WORK/refused.txt"
[ "$rc" -eq 2 ] && grep -q "Read-only file system" "$WORK/refused.txt"
[ "$(stat -c %a "$WORK/secrets/hidden_seeds.json")" = 644 ]

echo "== README command after 'chmod 700 secrets && chmod 600 secrets/hidden_seeds.json'"
chmod 700 "$WORK/secrets"; chmod 600 "$WORK/secrets/hidden_seeds.json"
run -v "$WORK/secrets:/secrets:ro" "$IMG" python -m auditgym isolate --policy careful_auditor \
    --episodes /secrets/hidden_seeds.json --log-dir /app/reports/isolated_logs | tee "$WORK/ok.txt"
grep -q "isolation boundary: yes" "$WORK/ok.txt" && grep -q "seeds private: True" "$WORK/ok.txt"
[ "$(grep -c " submit " "$WORK/ok.txt")" -eq 2 ]

echo "== seeds through stdin"
docker run --rm -i --network none "$IMG" python -m auditgym isolate --policy careful_auditor --episodes - \
    < "$WORK/secrets/hidden_seeds.json" | tee "$WORK/stdin.txt"
[ "$(grep -c " submit " "$WORK/stdin.txt")" -eq 2 ]

echo "== --seed without --public-seeds is refused"
set +e
run "$IMG" python -m auditgym isolate --policy careful_auditor --seed 1 --config train 2> "$WORK/seed.txt"
rc=$?
set -e
[ "$rc" -eq 2 ] && grep -q "cmdline" "$WORK/seed.txt"

echo "== full quick check inside the image (the sandbox self-test is a hard check as root)"
run "$IMG" python -m auditgym check --quick | tee "$WORK/check.txt"
grep -q "PASS  \[6\] isolated agent run" "$WORK/check.txt"

echo "docker isolation checks PASSED"
