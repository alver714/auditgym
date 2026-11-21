"""A built-in policy behind the isolated agent protocol (see auditgym/isolate.py).

    python3 -I auditgym/agents/policy_agent.py careful_auditor [policy_seed]

Reads one observation per line on stdin, writes one {"action": ...} per line on stdout. It imports only the
policy code (auditgym.policies); it has no token, no grader connection and no access to the grader's files.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from auditgym.policies import EXTRA_POLICY_NAMES, POLICY_NAMES, get_policy  # noqa: E402


def main() -> int:
    name = sys.argv[1] if len(sys.argv) > 1 else "careful_auditor"
    if name == "oracle" or name not in POLICY_NAMES + EXTRA_POLICY_NAMES:
        print(f"policy_agent: {name!r} is not an agent-side policy", file=sys.stderr)
        return 2
    policy_seed = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    policy = get_policy(name, policy_seed)
    first = True
    for line in sys.stdin:
        msg = json.loads(line)
        if msg.get("terminated") or msg.get("truncated"):
            break
        obs = msg["obs"]
        if first:
            policy.reset(policy_seed, obs)
            first = False
        sys.stdout.write(json.dumps({"action": policy.act(obs)}) + "\n")
        sys.stdout.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
