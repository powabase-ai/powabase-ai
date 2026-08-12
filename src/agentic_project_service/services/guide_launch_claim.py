"""Detects whether an assistant reply claims it launched a guide walkthrough.

Python mirror of the Studio's ``launch-claim.ts`` (components/interfaces/AI/
ProjectCopilot) — same launch verbs, same 60-char same-sentence proximity rule,
same negation exclusions — so both ends of the contract classify the same texts
the same way. The Studio uses it to *observe* a claim with no ``trigger_guide``
event (``guide_launch_missing`` telemetry); the service uses it to *repair* one
(see ``run_project_copilot_chat``'s corrective retry).

Python's ``re`` has no variable-width lookbehind, so the TS regex's negation
lookbehinds are expressed as explicit prefix checks instead: a launch verb
immediately preceded by can't/cannot/won't/unable-to is a denial, not a claim,
and "no walkthrough" is not a walkthrough.
"""

import re

_LAUNCH_VERB = (
    r"(launch|launched|launching|start|started|starting|open|opened|opening|"
    r"show|showed|showing|highlight|highlighted|highlighting)\w*"
)
# Verb … (same sentence, ≤60 chars) … "walkthrough".
_VERB_THEN_WALKTHROUGH = re.compile(
    rf"\b{_LAUNCH_VERB}\b[^.!?\n]{{0,60}}\b(walkthrough)\b", re.IGNORECASE
)
# Claims that don't need the verb+noun proximity form.
_DIRECT_CLAIM = re.compile(
    r"\bi'?ve launched\b|\bwalkthrough (?:is|on) (?:now )?on screen\b", re.IGNORECASE
)
_NEGATED_VERB_PREFIX = re.compile(r"\b(?:can'?t|cannot|won'?t|unable to)\s+$", re.IGNORECASE)
_NO_WALKTHROUGH_PREFIX = re.compile(r"\bno\s+$", re.IGNORECASE)


def looks_like_guide_launch_claim(text: str | None) -> bool:
    if not text:
        return False
    if _DIRECT_CLAIM.search(text):
        return True
    for m in _VERB_THEN_WALKTHROUGH.finditer(text):
        if _NEGATED_VERB_PREFIX.search(text[: m.start(1)]):
            continue
        if _NO_WALKTHROUGH_PREFIX.search(text[: m.start(2)]):
            continue
        return True
    return False
