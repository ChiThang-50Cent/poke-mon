"""BS0 ORACLE (evaluation only): true-state comparisons for reconstruction checks.

Never imported by the reconstruction path (`reconstruct.py`, `recon_world.py`);
`tests/test_bs0_recon.py` enforces that.
"""

from __future__ import annotations

from typing import Any

IGNORED_REQUEST_KEYS = {"rqid"}
HIDDEN_VOLATILES = {"choicelock", "stall"}  # item lock / protect counter: never shown


def diff(a: Any, b: Any, path: str = "") -> list[str]:
    """Paths where true `a` and reconstructed `b` differ (rqid normalised away)."""
    if isinstance(a, dict) and isinstance(b, dict):
        out = []
        for key in sorted(set(a) | set(b)):
            if key in IGNORED_REQUEST_KEYS and not path:
                continue
            if key not in a or key not in b:
                out.append(f"{path}/{key}:missing")
            else:
                out.extend(diff(a[key], b[key], f"{path}/{key}"))
        return out
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            return [f"{path}:len {len(a)}!={len(b)}"]
        out = []
        for i, (x, y) in enumerate(zip(a, b, strict=True)):
            out.extend(diff(x, y, f"{path}[{i}]"))
        return out
    return [] if a == b else [f"{path}:{a!r}!={b!r}"]


def normalise_view(view: dict[str, Any]) -> dict[str, Any]:
    """Public view with the opponent bench order normalised (name-keyed) and boosts trimmed."""
    out = {k: v for k, v in view.items() if k not in ("p1", "p2")}
    for side in ("p1", "p2"):
        mons = {}
        for m in view[side]["pokemon"]:
            m = dict(m)
            m["boosts"] = {k: v for k, v in m["boosts"].items() if v}
            # internal bookkeeping conditions the learner cannot observe
            m["volatiles"] = ",".join(sorted(
                v for v in m["volatiles"] if v not in HIDDEN_VOLATILES))
            mons[m.pop("name")] = m
        out[side] = {"conditions": view[side]["conditions"], "pokemon": mons}
    return out


def compare_views(
    true: dict[str, Any], recon: dict[str, Any], hp_exempt: frozenset[str] | set[str] = frozenset()
) -> list[str]:
    """`hp_exempt`: opponent bench mons whose HP changed silently (Regenerator switch-out heal)
    depending on a sampled ability; their public HP must agree within 1 percentage point."""
    t, r = normalise_view(true), normalise_view(recon)
    for name in hp_exempt:
        a = t["p2"]["pokemon"].get(name)
        b = r["p2"]["pokemon"].get(name)
        if a and b and abs(_pct(a["hp_public"]) - _pct(b["hp_public"])) <= 1:
            a["hp_public"] = b["hp_public"]
    for side in ("p1", "p2"):  # position: p1 exact, p2 only active-vs-bench is public
        for name in t[side]["pokemon"]:
            if side == "p2" and name in r[side]["pokemon"]:
                t[side]["pokemon"][name].pop("position", None)
                r[side]["pokemon"][name].pop("position", None)
                t[side]["pokemon"][name].pop("shiny", None)
                r[side]["pokemon"][name].pop("shiny", None)
    return diff(t, r)


def true_root(sim: Any, match: Any) -> dict[str, Any]:
    """True learner request + true public view of a restored root (oracle)."""
    from pokeai.cf.restore_check import pending_request

    return {"request": pending_request(match.pending)["p1"], "view": sim.public_view(match.tag)}


def _pct(token: str) -> int:
    head = token.split(" ")[0]
    return 0 if head == "0" else int(head.split("/")[0])
