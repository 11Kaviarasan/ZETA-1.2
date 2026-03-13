"""
mars_ai.py — Bridge between app.py and ProperAI engine.
Resolves proper_ai.py from MARS_DIR env var or current directory.
"""

import sys, os, logging

logger = logging.getLogger("zeta.mars")

MARS_DIR = os.getenv("MARS_DIR", os.path.dirname(os.path.abspath(__file__)))
if MARS_DIR not in sys.path:
    sys.path.insert(0, MARS_DIR)

try:
    from proper_ai import ProperAI
    ai_engine = ProperAI()
    _engine_error = None
except Exception as exc:
    import traceback
    traceback.print_exc()
    _engine_error = str(exc)
    logger.error(f"ProperAI load failed: {exc}")

    class _FallbackAI:
        def ask(self, q, **kw):
            return {"answer": f"AI Engine unavailable: {_engine_error}",
                    "source": "error", "tokens": 0, "intent": "error",
                    "language": "unknown", "cache_hit": False}
    ai_engine = _FallbackAI()


def query(question: str, history=None, user_id=None, conv_id=None, model="auto") -> dict:
    """Called by app.py for every chat request."""
    try:
        result = ai_engine.ask(question, history=history or [])
        if not isinstance(result, dict):
            result = {"answer": str(result)}
        result.setdefault("source",    "mars_ai")
        result.setdefault("tokens",    0)
        result.setdefault("intent",    "general")
        result.setdefault("language",  "en")
        result.setdefault("cache_hit", False)
        return result
    except Exception as e:
        logger.error(f"query() error: {e}")
        return {"answer": f"Query error: {e}", "source": "error",
                "tokens": 0, "intent": "error", "language": "unknown", "cache_hit": False}


def get_engine_status() -> dict:
    ok = _engine_error is None
    return {
        "status":  "online" if ok else "degraded",
        "engine":  "ProperAI (Zeta Engine)",
        "path":    MARS_DIR,
        "error":   _engine_error,
        "pinecone": bool(getattr(getattr(ai_engine, '_pc_index', None), 'describe_index_stats', None)),
    }
