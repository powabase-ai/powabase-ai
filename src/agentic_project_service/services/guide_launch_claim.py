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
unwanted action.

Semantics, per sentence (questions never match):

- first-person indicative assertions — "I've launched…", "I'm opening…",
  "I'll launch the walkthrough now."
- subjectless progressives — "Launching the connect walkthrough…"
- state assertions — "The walkthrough is now on screen."

Excluded outright: interrogative sentences; offer/ability/conditional phrasing
("I can/could…", "if you'd like", "let me know", "want me to"); negations and
denials ("can't", "won't", "haven't", "no walkthrough").
"""

import re

_SENTENCE_SPLIT = re.compile(r"[.!?\n]")

# A sentence containing any of these is an offer, a request for permission, or
# an ability/conditional statement — not an assertion that a launch happened.
_OFFER_OR_CONDITIONAL = re.compile(
    r"\b(?:would you|do you want|want me to|shall i|should i|can i|may i|"
    r"i can\b|i could\b|i would\b|i might\b|if you(?:'d| would)? like|"
    r"let me know|say the word|just ask)",
    re.IGNORECASE,
)

# A sentence containing any of these denies or negates a launch.
_NEGATION = re.compile(
    r"\b(?:can'?t|cannot|won'?t|couldn'?t|didn'?t|haven'?t|hasn'?t|wasn'?t|"
    r"unable to|not able to|there'?s no|there is no|no walkthrough)\b",
    re.IGNORECASE,
)

_LAUNCH_VERB = r"(?:launch|start|open|show|pull up|bring up|play)"

_ASSERTIONS = [
    # First-person indicative: I've launched / I have started / I'm opening /
    # I'll launch / I just launched … (offers like "I can launch" are excluded
    # above, so bare "I launch" is not needed and not matched).
    re.compile(
        rf"\bi(?:'ve| have|'m| am|'ll| will| just)\s+(?:just\s+|now\s+)?{_LAUNCH_VERB}(?:ed|ing)?\b"
        rf"[^.!?\n]{{0,60}}\bwalkthrough\b",
        re.IGNORECASE,
    ),
    # Subjectless progressive at the start of a sentence: "Launching the
    # connect walkthrough now…"
    re.compile(
        rf"^\s*{_LAUNCH_VERB}ing\b[^.!?\n]{{0,60}}\bwalkthrough\b",
        re.IGNORECASE,
    ),
    # State assertion about the walkthrough itself: "the walkthrough is now on
    # screen", "the walkthrough is launching / has started".
    re.compile(
        rf"\bwalkthrough (?:is|has been|has|was) (?:now )?"
        rf"(?:on (?:your )?screen|live|running|{_LAUNCH_VERB}(?:ed|ing))\b",
        re.IGNORECASE,
    ),
    # Past-tense first person with an implied object: "I've launched it for you."
    re.compile(r"\bi'?ve (?:just )?launched\b", re.IGNORECASE),
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
    if _OFFER_OR_CONDITIONAL.search(sentence) or _NEGATION.search(sentence):
        return False
    return any(p.search(sentence) for p in _ASSERTIONS)
