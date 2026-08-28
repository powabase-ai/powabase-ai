"""GraphIndex expansion defaults shared by the registry and the search path.

These two values are a contract: the registry ships them as a KB's initial
``retrieval_config.graph_expansion``, and ``knowledge_search`` applies the
same numbers to any KB whose config predates the setting. They live in their
own leaf module because ``registry`` cannot import from ``services`` —
``services/__init__`` imports ``knowledge_search``, which imports
``strategies``, so that direction is a cycle.
"""

# Children of a referenced node are opt-in; when asked for, at most this many
# per parent. The document outline that ships instead names the ones omitted.
GRAPH_DEFAULT_MAX_CHILDREN = 3

# Hard ceiling on the per-parent cap. Without one, a configured cap of a
# million restores exactly the unbounded fan-out this bounding removes — the
# same reason top_k is bounded at the route layer.
GRAPH_MAX_CHILDREN_CEILING = 20
