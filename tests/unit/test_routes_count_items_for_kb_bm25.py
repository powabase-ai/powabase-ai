"""``_count_items_for_kb_bm25`` must name the item table's partition key.

The item tables are partitioned by ``knowledge_base_id``. Filtering only
through the ``indexed_sources`` join makes Postgres scan every partition;
a predicate on the item table's own ``knowledge_base_id`` prunes the count to
the KB's partition (or DEFAULT).
"""

from __future__ import annotations

import re
from unittest.mock import patch

import pytest

from agentic_project_service.routes import knowledge_bases as kb_route

R = "agentic_project_service.routes.knowledge_bases"
KB = "3f2504e0-4f89-11d3-9a0c-0305e82c3301"


@pytest.mark.parametrize("item_table", ["chunks", "full_documents", "graph_index_nodes"])
def test_the_count_filters_on_the_item_tables_own_knowledge_base_id(item_table):
    with patch(f"{R}.db") as db:
        db.session.execute.return_value.fetchone.return_value = (4,)
        assert kb_route._count_items_for_kb_bm25(KB, item_table) == 4
    sql = str(db.session.execute.call_args.args[0])
    params = db.session.execute.call_args.args[1]
    alias = re.search(rf"\.{item_table} (\w+)\b", sql).group(1)
    assert re.search(rf"\b{alias}\.knowledge_base_id = :kb\b", sql), sql
    assert params == {"kb": KB}


def test_an_unknown_table_counts_nothing():
    with patch(f"{R}.db") as db:
        assert kb_route._count_items_for_kb_bm25(KB, "no_such_table") == 0
    db.session.execute.assert_not_called()
