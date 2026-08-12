"""Detects whether an assistant reply ASSERTS that a guide walkthrough is
launching or launched — the predicate driving the corrective-retry repair in
``run_project_copilot_chat``.

This is deliberately STRICTER than the Studio's ``launch-claim.ts`` telemetry
detector, and not a mirror of it. That detector is tuned for recall because a
false positive there costs an analytics row (``guide_launch_missing``). A false
positive HERE mutates the turn: the repair nudge instructs the model to call
``play_guide``, so matching a permission question ("Would you like me to start
the walkthrough?") would convert the clarifying question the system prompt
solicits into an unrequested launch — a UI action plus an extra billed LLM
call. Precision over recall: anything this predicate misses is still observed
client-side by the telemetry detector; anything it wrongly matches becomes an
unwanted action. Known deliberate misses under that policy: third-person
reported speech, and exotic phrasings no assertion pattern covers.

Semantics, per sentence (questions never match):

- completed/progressive first-person claims — "I've launched…", "I launched…",
  "I'm opening…" — plus subjectless progressives ("Launching the connect
  walkthrough…", markdown-prefixed forms included) and walkthrough-state
  claims ("The walkthrough is now on screen."). For these, offer/permission
  markers veto only when they appear BEFORE the claim: a trailing offer
  ("I've launched X — let me know if you'd like Y too") doesn't unmake a
  completed claim, and models love the trailing-offer dash.
- future first-person claims — "I'll launch the walkthrough now." For these,
  offer/conditional markers anywhere in the sentence veto: "I'll launch the
  walkthrough if you'd like" is an offer, because the condition governs
  whether the action happens at all.

Negations and denials ("can't", "won't", "haven't", "no walkthrough") veto the
whole sentence in either form — the safe direction for a repair guard.
"""

import re

_SENTENCE_SPLIT = re.compile(r"[.!?\n]")

# Offer, permission, ability, or conditional phrasing. Scope of application
# depends on the claim's tense — see module docstring.
_OFFER_OR_CONDITIONAL = re.compile(
    r"\b(?:would you|do you want|want me to|shall i|should i|can i|may i|"
    r"i can\b|i could\b|i would\b|i might\b|if you(?:'d| would)? (?:like|want|prefer)|"
    r"let me know|say the word|just ask|when(?:ever)? you'?re ready)",
    re.IGNORECASE,
)

# A sentence containing any of these denies or negates a launch.
_NEGATION = re.compile(
    r"\b(?:can'?t|cannot|won'?t|couldn'?t|didn'?t|haven'?t|hasn'?t|wasn'?t|"
    r"unable to|not able to|there'?s no|there is no|no walkthrough)\b",
    re.IGNORECASE,
)

_LAUNCH_VERB = r"(?:launch|start|open|show|pull up|bring up|play)"

# Completed / in-progress claims: a trailing offer can't unmake these.
_COMPLETED_ASSERTIONS = [
    # "I've launched…", "I have started…", "I'm opening…", "I (just) launched…"
    re.compile(
        rf"\bi(?:'ve| have|'m| am)?\s+(?:just\s+|now\s+)?{_LAUNCH_VERB}(?:ed|ing)\b"
        rf"[^.!?\n]{{0,60}}\bwalkthrough\b",
        re.IGNORECASE,
    ),
    # Subjectless progressive opening the sentence — allow markdown prefixes
    # (**Launching…**, > Launching…, - Launching…).
    re.compile(
        rf"^[\s*_>#•-]*{_LAUNCH_VERB}ing\b[^.!?\n]{{0,60}}\bwalkthrough\b",
        re.IGNORECASE,
    ),
    # State claim about the walkthrough itself.
    re.compile(
        rf"\bwalkthrough (?:is|has been|has|was) (?:now )?"
        rf"(?:on (?:your )?screen|live|running|{_LAUNCH_VERB}(?:ed|ing))\b",
        re.IGNORECASE,
    ),
    # Past-tense first person with an implied object: "I've launched it for you."
    re.compile(r"\bi'?ve (?:just )?launched\b", re.IGNORECASE),
]

# Future claims: a conditional anywhere in the sentence makes these offers.
_FUTURE_ASSERTIONS = [
    re.compile(
        rf"\bi(?:'ll| will)\s+(?:just\s+|now\s+)?{_LAUNCH_VERB}\b"
        rf"[^.!?\n]{{0,60}}\bwalkthrough\b",
        re.IGNORECASE,
    ),
]


def looks_like_guide_launch_assertion(text: str | None) -> bool:
    if not text:
        return False
    # Walk sentences with their terminators so interrogatives can be skipped.
    pos = 0
    for m in _SENTENCE_SPLIT.finditer(text):
        sentence, terminator = text[pos : m.start()], m.group(0)
        pos = m.end()
        if terminator == "?":
            continue
        if _sentence_asserts_launch(sentence):
            return True
    return _sentence_asserts_launch(text[pos:])  # trailing sentence, no terminator


def _sentence_asserts_launch(sentence: str) -> bool:
    if not sentence.strip():
        return False
    if _NEGATION.search(sentence):
        return False
    for pattern in _COMPLETED_ASSERTIONS:
        m = pattern.search(sentence)
        if m and not _OFFER_OR_CONDITIONAL.search(sentence[: m.start()]):
            return True
    for pattern in _FUTURE_ASSERTIONS:
        if pattern.search(sentence) and not _OFFER_OR_CONDITIONAL.search(sentence):
            return True
    return False
