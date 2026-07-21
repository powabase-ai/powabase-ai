"""Project service API routes."""

from .sources import sources_bp
from .knowledge_bases import knowledge_bases_bp
from .agents import agents_bp
from .sessions import sessions_bp
from .context_handlers import context_handlers_bp
from .enrichment import enrichment_bp
from .workflows import workflows_bp
from .copilot import copilot_bp
from .internal import internal_bp

__all__ = [
    "sources_bp",
    "knowledge_bases_bp",
    "agents_bp",
    "sessions_bp",
    "context_handlers_bp",
    "enrichment_bp",
    "workflows_bp",
    "copilot_bp",
    "internal_bp",
]
