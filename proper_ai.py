"""
proper_ai.py — Zeta AI Core Engine
Priority chain: Conversational → Gemini → OpenAI → Pinecone RAG → LiveBrain (Wikipedia)
Vectors stored in Pinecone. Embeddings via sentence-transformers.
"""

import os, re, logging, time
from datetime import datetime

logger = logging.getLogger("zeta.engine")

# ─── Optional heavy imports (graceful degrades) ───────────────────────────────

try:
    from pinecone import Pinecone, ServerlessSpec
    _pinecone_ok = True
except ImportError:
    _pinecone_ok = False
    logger.warning("pinecone-client not installed — vector search disabled.")

try:
    from sentence_transformers import SentenceTransformer
    _st_ok = True
except ImportError:
    _st_ok = False
    logger.warning("sentence-transformers not installed — embeddings disabled.")

try:
    import google.generativeai as genai
    _gemini_ok = True
except ImportError:
    _gemini_ok = False
    logger.warning("google-generativeai not installed.")

try:
    from openai import OpenAI as _OpenAI
    _openai_ok = True
except ImportError:
    _openai_ok = False
    logger.warning("openai not installed.")

try:
    import wikipedia
    _wiki_ok = True
except ImportError:
    _wiki_ok = False
    logger.warning("wikipedia not installed — LiveBrain disabled.")


# ─── Conversational patterns ──────────────────────────────────────────────────

_GREETINGS = {
    r"\b(hi|hello|hey|sup|howdy|yo)\b": "Hey there! 👋 I'm Zeta AI, your intelligent assistant. What can I help you with today?",
    r"\bhow are you\b":                  "I'm running at full power! 🚀 Ready to help. What's your question?",
    r"\bthank(s| you)\b":               "You're welcome! 😊 Feel free to ask anything else.",
    r"\bbye|goodbye|see you\b":          "Goodbye! Come back anytime. 👋",
    r"\bwhat('s| is) your name\b":       "I'm **Zeta AI** — your RAG-powered assistant built on the Zeta Engine. 🌟",
    r"\bwho (made|built|created) you\b": "I was built by the Zeta AI team. 💡 Anything else you'd like to know?",
}


class ProperAI:
    """Main AI engine for Zeta AI."""

    def __init__(self):
        self._embedder    = None
        self._pc_index    = None
        self._gemini      = None
        self._openai      = None
        self._initialized = False
        self._init()

    # ── Initialization ────────────────────────────────────────────────────────

    def _init(self):
        # Sentence-transformer embedder
        if _st_ok:
            try:
                model_name = os.getenv("EMBED_MODEL", "all-MiniLM-L6-v2")
                self._embedder = SentenceTransformer(model_name)
                logger.info(f"Embedder loaded: {model_name}")
            except Exception as e:
                logger.error(f"Embedder init failed: {e}")

        # Pinecone
        if _pinecone_ok:
            try:
                pc    = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
                iname = os.getenv("PINECONE_INDEX_NAME", "zeta-memory")
                existing = [i.name for i in pc.list_indexes()]
                if iname not in existing:
                    pc.create_index(
                        name=iname,
                        dimension=384,          # all-MiniLM-L6-v2
                        metric="cosine",
                        spec=ServerlessSpec(cloud="aws", region="us-east-1")
                    )
                    logger.info(f"Pinecone index '{iname}' created.")
                self._pc_index = pc.Index(iname)
                logger.info(f"Pinecone connected: {iname}")
            except Exception as e:
                logger.error(f"Pinecone init failed: {e}")

        # Gemini
        if _gemini_ok:
            gkey = os.getenv("GEMINI_API_KEY")
            if gkey:
                try:
                    genai.configure(api_key=gkey)
                    self._gemini = genai.GenerativeModel(os.getenv("GEMINI_MODEL", "gemini-2.5-pro"))
                    logger.info("Gemini configured.")
                except Exception as e:
                    logger.error(f"Gemini init: {e}")

        # OpenAI
        if _openai_ok:
            okey = os.getenv("OPENAI_API_KEY")
            if okey:
                try:
                    self._openai = _OpenAI(api_key=okey)
                    logger.info("OpenAI configured.")
                except Exception as e:
                    logger.error(f"OpenAI init: {e}")

        self._initialized = True
        logger.info("ProperAI engine ready.")

    # ── Embedding ─────────────────────────────────────────────────────────────

    def _embed(self, text: str) -> list[float] | None:
        if not self._embedder:
            return None
        try:
            return self._embedder.encode(text, normalize_embeddings=True).tolist()
        except Exception as e:
            logger.error(f"Embed error: {e}")
            return None

    # ── Pinecone — store ──────────────────────────────────────────────────────

    def _vector_store(self, question: str, answer: str, meta: dict = {}):
        if not self._pc_index or not self._embedder:
            return
        try:
            vec = self._embed(question)
            if not vec:
                return
            vid = f"q_{int(time.time()*1000)}"
            self._pc_index.upsert(vectors=[{
                "id":       vid,
                "values":   vec,
                "metadata": {"question": question[:500], "answer": answer[:1000],
                             "stored_at": datetime.utcnow().isoformat(), **meta}
            }])
        except Exception as e:
            logger.error(f"Vector store error: {e}")

    # ── Pinecone — search ─────────────────────────────────────────────────────

    def _vector_search(self, question: str, top_k: int = 3) -> list[dict]:
        if not self._pc_index or not self._embedder:
            return []
        try:
            vec     = self._embed(question)
            if not vec:
                return []
            results = self._pc_index.query(vector=vec, top_k=top_k, include_metadata=True)
            hits    = []
            for m in results.get("matches", []):
                if m["score"] > 0.75:
                    hits.append({
                        "question": m["metadata"].get("question",""),
                        "answer":   m["metadata"].get("answer",""),
                        "score":    m["score"],
                    })
            return hits
        except Exception as e:
            logger.error(f"Vector search error: {e}")
            return []

    # ── Conversational layer ──────────────────────────────────────────────────

    def _check_small_talk(self, text: str) -> str | None:
        low = text.lower().strip()
        for pattern, reply in _GREETINGS.items():
            if re.search(pattern, low):
                return reply
        return None

    # ── Gemini ────────────────────────────────────────────────────────────────

    def _ask_gemini(self, question: str, context: str = "", history: list = []) -> str | None:
        if not self._gemini:
            return None
        try:
            sys_prompt = (
                "You are Zeta AI, a helpful and intelligent assistant. "
                "Answer concisely and accurately. "
                "If context is provided, prioritise it in your answer.\n"
            )
            parts = [sys_prompt]
            if context:
                parts.append(f"[Relevant context]\n{context}\n")
            for msg in history[-6:]:            # last 3 turns
                role = "User" if msg["role"] == "user" else "Zeta"
                parts.append(f"{role}: {msg['content']}")
            parts.append(f"User: {question}\nZeta:")
            resp = self._gemini.generate_content("\n".join(parts))
            return resp.text.strip()
        except Exception as e:
            logger.warning(f"Gemini error: {e}")
            return None

    # ── OpenAI ────────────────────────────────────────────────────────────────

    def _ask_openai(self, question: str, context: str = "", history: list = []) -> str | None:
        if not self._openai:
            return None
        try:
            messages = [{"role": "system", "content":
                         "You are Zeta AI, a helpful intelligent assistant. "
                         + (f"Use this context if relevant:\n{context}" if context else "")}]
            for msg in history[-6:]:
                messages.append({"role": msg["role"], "content": msg["content"]})
            messages.append({"role": "user", "content": question})
            resp = self._openai.chat.completions.create(
                model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
                messages=messages,
                max_tokens=int(os.getenv("MAX_TOKENS", 1024)),
                temperature=0.7,
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            logger.warning(f"OpenAI error: {e}")
            return None

    # ── LiveBrain (Wikipedia) ─────────────────────────────────────────────────

    def _livebrain(self, question: str) -> str | None:
        if not _wiki_ok:
            return None
        try:
            wikipedia.set_lang("en")
            results = wikipedia.search(question, results=2)
            if not results:
                return None
            page    = wikipedia.page(results[0], auto_suggest=False)
            summary = page.summary[:800]
            return f"{summary}\n\n[Source: Wikipedia — {page.url}]"
        except Exception as e:
            logger.debug(f"LiveBrain: {e}")
            return None

    # ── Public ask() ──────────────────────────────────────────────────────────

    def ask(self, question: str, history: list = [], **kwargs) -> dict:
        """
        Returns dict with: answer, source, tokens, intent, language, cache_hit
        Priority: small_talk → pinecone_cache → gemini → openai → livebrain → fallback
        """
        # 1. Small talk
        small = self._check_small_talk(question)
        if small:
            return self._result(small, "conversational", cache_hit=True)

        # 2. Pinecone semantic cache
        hits = self._vector_search(question)
        rag_context = ""
        cache_hit   = False
        if hits:
            best = hits[0]
            if best["score"] > 0.92:            # High confidence — return directly
                return self._result(best["answer"], "pinecone_cache", cache_hit=True)
            # Medium confidence — use as context for cloud model
            rag_context = "\n\n".join(
                f"Q: {h['question']}\nA: {h['answer']}" for h in hits
            )

        # 3. Gemini (primary cloud)
        answer = self._ask_gemini(question, rag_context, history)
        source = "gemini"

        # 4. OpenAI fallback
        if not answer:
            answer = self._ask_openai(question, rag_context, history)
            source = "openai"

        # 5. LiveBrain (Wikipedia deep fallback)
        if not answer:
            answer = self._livebrain(question)
            source = "livebrain"

        # 6. Hard fallback
        if not answer:
            answer = ("I'm sorry, I couldn't find a reliable answer right now. "
                      "Please try rephrasing your question.")
            source = "fallback"

        # 7. Learn — store good answers in Pinecone for future cache hits
        if source in ("gemini", "openai", "livebrain"):
            self._vector_store(question, answer, {"source": source})

        return self._result(answer, source, cache_hit=cache_hit)

    @staticmethod
    def _result(answer: str, source: str, tokens: int = 0, cache_hit: bool = False) -> dict:
        return {
            "answer":    answer,
            "source":    source,
            "tokens":    tokens,
            "intent":    "general",
            "language":  "en",
            "cache_hit": cache_hit,
        }
