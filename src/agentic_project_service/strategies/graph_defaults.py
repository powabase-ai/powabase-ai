"""GraphIndex expansion defaults shared by the registry and the search path.

The three defaults are a contract: the registry ships them as a KB's initial
``retrieval_config.graph_expansion``, and ``knowledge_search`` applies the
same values to any KB whose config predates the setting. Defining them twice
would let a registry edit and the fallback drift apart silently, with the
test that compares them updated to match.

They live in their own leaf module because ``registry`` cannot import from
``services``: ``services/__init__`` imports ``knowledge_search``, which
imports ``strategies``, so that direction closes a cycle.

The ceiling is not part of that contract — it bounds what a caller may
configure and is read only by the search path — but it lives here so every
graph_expansion knob has one home.
"""

# Children of a referenced node are opt-in; when asked for, at most this many
# per parent. The document outline that ships instead names the ones omitted.
GRAPH_DEFAULT_INCLUDE_CHILDREN = False
GRAPH_DEFAULT_MAX_CHILDREN = 3
GRAPH_DEFAULT_INCLUDE_DOC_TOC = True

# How many of a hit's explicitly referenced sections are pulled in. This is
# expansion's largest cost by far: measured on a real corpus, one node with 12
# references pulled ~27k tokens of section bodies — more than the whole context
# budget — while the outline standing in for its children was ~1k. GraphIndex
# corpora are typically large and a fan-out of 10 is normal rather than
# pathological, so this bounds the tail without reshaping ordinary retrieval.
GRAPH_DEFAULT_MAX_REFERENCED_NODES = 10

# Hard ceilings on the two caps. Without them, a configured million restores
# exactly the unbounded fan-out this bounding removes — the same reason top_k
# is bounded at the route layer. Set well above any sensible value: they exist
# to refuse absurdity, not to tune.
GRAPH_MAX_CHILDREN_CEILING = 20
GRAPH_MAX_REFERENCED_CEILING = 100
