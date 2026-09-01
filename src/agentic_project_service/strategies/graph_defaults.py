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

# Hard ceiling on the per-parent cap. Without one, a configured cap of a
# million restores exactly the unbounded fan-out this bounding removes — the
# same reason top_k is bounded at the route layer.
GRAPH_MAX_CHILDREN_CEILING = 20
