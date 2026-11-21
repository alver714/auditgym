# Docker / Linux `check` results

## Current: full `check` inside the shipped image on x86-64

`docker build -t auditgym . && docker run --rm --network none auditgym` (i.e. `python -m auditgym check`, full mode,
no network), run as root inside the container, so the isolation self-test is a hard check:

| | |
|---|---|
| image base | `python:3.11.15-slim` (glibc 2.41) |
| machine | x86_64, Linux |
| Python | 3.11.15 |
| result | **ALL HARD CHECKS PASSED**: 74 PASS, 0 FAIL, 1 WARN (the documented ordering tie `random_explorer` -0.806 vs `reconcile` -0.807), 0 SKIP |
| pytest inside the check | 1079 passed (nothing skipped: the uid-sandbox tests run because the container user is root) |
| isolation | `isolate --self-test` PASS: the escape-probe agent under its own uid finds no token, seed, grader file or protocol hole |
| golden digests | the 6 digests of `tests/golden_hashes.json` (committed from a macOS/arm64 run) **match on x86-64 Linux** |
| wall time | about 230 s for the container run (the check itself 227 s) |

The same image is exercised on every push by the CI job `isolation-docker` (`scripts/docker_isolation_check.sh`):
the isolation tests, the self-test, the README `isolate` commands (fresh `0644` seeds file on a read-only mount must be
refused; after `chmod 700/600` it must run; seeds through stdin; `--seed` without `--public-seeds` refused) and
`check --quick`, all with `--network none`.

## Earlier record: Linux/aarch64

An earlier full run of the same image on Linux 6.8 / aarch64 (before the isolation work: 73 PASS, 0 FAIL, 0 WARN,
1 SKIP for the then-missing config lock file; 1004 tests; about 112 s) matched the same golden digests.

## What this shows

Generator output and deterministic trajectories are byte-identical across macOS/arm64, Linux/aarch64 and
Linux/x86-64 on CPython 3.11.15 (different OS, libc and CPU architecture). `python -m auditgym golden` (or
`make check`) compares the digests on any other machine.
