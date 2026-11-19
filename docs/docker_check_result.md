# Docker / Linux `check` result

Recorded run of the official command inside the shipped image (`docker build -t auditgym . && docker run --rm
--network none auditgym`, i.e. `python -m auditgym check`, full mode, no network):

| | |
|---|---|
| image base | `python:3.11.15-slim` (Linux 6.8, aarch64, glibc 2.41) |
| Python | 3.11.15 |
| result | **ALL HARD CHECKS PASSED**: 73 PASS, 0 FAIL, 0 WARN, 1 SKIP (no `configs/CONFIG_VERSIONS.json` lock file shipped) |
| pytest inside the check | 1004 passed |
| golden digests | the 6 digests of `tests/golden_hashes.json` (episode id, public hash, truth / provenance sha256, `reconcile` and `careful_auditor` trajectory sha256 for train, dev and holdout episodes), committed from a macOS (Darwin) run, **match on Linux** |
| wall time | about 112 s |

What this does and does not show: generator output and deterministic trajectories are byte-identical across
macOS and Linux (different libc and OS, same CPython 3.11.15). Both runs were arm64; an x86-64 run is the one
remaining gap, and `python -m auditgym golden` (or `make check`) compares the digests on any machine.
