"""Content-addressed episode ids.

episode_id = "ep_" + sha256(canonical public content with the manifest's episode_id field removed)[:12].

The id is a pure function of the public records (manifest minus episode_id, policy, ledger, invoices,
payments, fx), so it carries no information about the hidden seed beyond what the records themselves show.
"""
from __future__ import annotations

import copy
import hashlib

from .episode import Episode, render_public


def content_episode_id(ep: Episode) -> str:
    probe = Episode(copy.deepcopy(ep.manifest), ep.policy, ep.ledger, ep.invoices, ep.payments, ep.fx)
    probe.manifest.pop("episode_id", None)
    h = hashlib.sha256()
    for name, blob in sorted(render_public(probe).items()):
        h.update(name.encode()); h.update(b"\0"); h.update(blob.encode()); h.update(b"\0")
    return "ep_" + h.hexdigest()[:12]


def retag(ep: Episode) -> str:
    """Recompute the content id in place and fill it into every place that embeds it."""
    eid = content_episode_id(ep)
    ep.manifest["episode_id"] = eid
    if ep.truth:
        ep.truth["episode_id"] = eid
    if ep.provenance:
        ep.provenance["episode_id"] = eid
    return eid
