"""
Agent Memory Service — Three-tier memory orchestrator.

Tier A: Structured long-term memory (PostgreSQL CRUD)
Tier B: Unstructured vector memory (pgvector RAG with decay)
Tier C: Short-term session context (in-memory)
"""

import uuid
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, List, Optional

from sqlalchemy import select, update, and_, text, func
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from db.engine import create_db_engine, get_async_session_factory, init_db, test_connection, close_engine
from db.models import Strategy, Signal, Decision, UserPreference, ChatMessage, AgentMemory

logger = logging.getLogger(__name__)


# ============================================================
# Tier C: Session Context (in-memory)
# ============================================================

@dataclass
class SessionContext:
    """Short-term session state kept in memory."""
    session_id: uuid.UUID
    turns: List[Dict[str, str]] = field(default_factory=list)
    daily_summary: Dict[str, Any] = field(default_factory=dict)
    active_symbol: str = "QQQ"
    last_prediction: Dict[str, Any] = field(default_factory=dict)
    max_turns: int = 20

    def add_turn(self, role: str, content: str):
        self.turns.append({"role": role, "content": content})
        if len(self.turns) > self.max_turns:
            self.turns = self.turns[-self.max_turns:]

    def format_context(self) -> str:
        """Format session turns for LLM prompt injection."""
        if not self.turns:
            return ""
        lines = []
        for t in self.turns[-6:]:
            prefix = "User" if t["role"] == "user" else "Assistant"
            lines.append(f"**{prefix}**: {t['content'][:500]}")
        return "\n".join(lines)


class AgentMemoryService:
    """Orchestrates all three memory tiers."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.memory_config = config.get("memory", {})
        self.enabled = self.memory_config.get("enabled", True)

        # Services (injected after async init)
        self._session_factory: Optional[async_sessionmaker] = None
        self._embedding_service = None

        # Tier C: in-memory session store
        self._sessions: Dict[uuid.UUID, SessionContext] = {}

        # Tier B config
        self._recall_limit = self.memory_config.get("recall_limit", 5)
        self._decay_threshold_days = self.memory_config.get("decay_threshold_days", 90)
        self._decay_factor = self.memory_config.get("decay_factor", 0.95)
        self._max_session_turns = self.memory_config.get("max_session_turns", 20)

    # ----------------------------------------------------------
    # Lifecycle
    # ----------------------------------------------------------

    async def initialize(self) -> bool:
        """Create engine, test DB, init tables, load embedding model."""
        if not self.enabled:
            logger.info("Agent memory disabled in config")
            return False

        try:
            engine = await create_db_engine(self.config)
            db_ok = await test_connection(engine)
            if not db_ok:
                logger.warning("Database connection failed — memory service unavailable")
                return False

            await init_db(engine)
            self._session_factory = get_async_session_factory(engine)

            # Load embedding service
            try:
                from services.embedding_service import EmbeddingService
                self._embedding_service = EmbeddingService(self.config)
                if self._embedding_service.is_available():
                    logger.info("Embedding model loaded for memory service")
                else:
                    logger.warning("Embedding model unavailable — vector search disabled")
                    self._embedding_service = None
            except Exception as e:
                logger.warning(f"Embedding service init failed: {e}")
                self._embedding_service = None

            logger.info(
                f"Agent memory service initialized "
                f"(db=True, embeddings={self._embedding_service is not None})"
            )
            return True

        except Exception as e:
            logger.error(f"Agent memory service init failed: {e}")
            return False

    def is_available(self) -> bool:
        return self._session_factory is not None

    def embeddings_available(self) -> bool:
        return self._embedding_service is not None and self._embedding_service.is_available()

    async def shutdown(self):
        await close_engine()
        logger.info("Agent memory service shut down")

    def _get_session(self) -> async_sessionmaker:
        if self._session_factory is None:
            raise RuntimeError("Memory service not initialized")
        return self._session_factory

    # ===========================================================
    # Tier A — Structured CRUD
    # ===========================================================

    # --- Signals ---

    async def store_signal(
        self,
        symbol: str,
        signal_type: str,
        confidence: Optional[float] = None,
        prediction_data: Optional[Dict[str, Any]] = None,
        strategy_id: Optional[uuid.UUID] = None,
    ) -> Dict[str, Any]:
        """Store a trading signal with full indicator snapshot."""
        pred = prediction_data or {}
        async with self._get_session()() as session:
            signal = Signal(
                symbol=symbol,
                timeframe_minutes=pred.get("timeframe_minutes", 1),
                signal_type=signal_type,
                confidence=confidence,
                kronos_p_up=pred.get("p_up_30m"),
                kronos_exp_return=pred.get("exp_ret_30m"),
                indicators=pred.get("indicators", {}),
                percentiles=pred.get("percentiles", {}),
                daily_context=pred.get("daily_context", {}),
                market_regime=pred.get("market_regime"),
                llm_analysis=pred.get("llm_analysis"),
                strategy_id=strategy_id,
            )
            session.add(signal)
            await session.commit()
            await session.refresh(signal)
            logger.info(f"Stored signal: {signal}")
            return {"id": str(signal.id), "signal_type": signal_type, "symbol": symbol}

    async def get_recent_signals(
        self, symbol: Optional[str] = None, limit: int = 20
    ) -> List[Dict[str, Any]]:
        async with self._get_session()() as session:
            q = select(Signal).order_by(Signal.created_at.desc()).limit(limit)
            if symbol:
                q = q.where(Signal.symbol == symbol)
            result = await session.execute(q)
            signals = result.scalars().all()
            return [
                {
                    "id": str(s.id),
                    "symbol": s.symbol,
                    "signal_type": s.signal_type,
                    "confidence": float(s.confidence) if s.confidence else None,
                    "kronos_p_up": float(s.kronos_p_up) if s.kronos_p_up else None,
                    "market_regime": s.market_regime,
                    "created_at": s.created_at.isoformat(),
                }
                for s in signals
            ]

    # --- Decisions ---

    async def store_decision(
        self,
        decision_text: str,
        parsed_rule: Optional[Dict] = None,
        source: str = "chat",
    ) -> Dict[str, Any]:
        async with self._get_session()() as session:
            decision = Decision(
                decision_text=decision_text,
                parsed_rule=parsed_rule or {},
                source=source,
            )
            session.add(decision)
            await session.commit()
            await session.refresh(decision)
            logger.info(f"Stored decision: {decision}")
            return {"id": str(decision.id), "text": decision_text}

    async def get_active_decisions(
        self, symbol: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        async with self._get_session()() as session:
            q = select(Decision).where(Decision.is_active == True).order_by(
                Decision.created_at.desc()
            )
            result = await session.execute(q)
            decisions = result.scalars().all()
            rows = []
            for d in decisions:
                # Optional: filter by symbol in parsed_rule
                if symbol and d.parsed_rule.get("symbol"):
                    if d.parsed_rule["symbol"].upper() != symbol.upper():
                        continue
                rows.append({
                    "id": str(d.id),
                    "text": d.decision_text,
                    "parsed_rule": d.parsed_rule,
                    "source": d.source,
                    "created_at": d.created_at.isoformat(),
                })
            return rows

    async def deactivate_decision(self, decision_id: str, superseded_by_id: Optional[str] = None):
        async with self._get_session()() as session:
            stmt = (
                update(Decision)
                .where(Decision.id == uuid.UUID(decision_id))
                .values(
                    is_active=False,
                    superseded_by=uuid.UUID(superseded_by_id) if superseded_by_id else None,
                )
            )
            await session.execute(stmt)
            await session.commit()

    # --- User Preferences ---

    async def set_user_preference(self, category: str, key: str, value: Any):
        async with self._get_session()() as session:
            # Upsert
            existing = await session.execute(
                select(UserPreference).where(
                    and_(UserPreference.category == category, UserPreference.key == key)
                )
            )
            pref = existing.scalar_one_or_none()
            if pref:
                pref.value = value
                pref.updated_at = datetime.now(timezone.utc)
            else:
                pref = UserPreference(category=category, key=key, value=value)
                session.add(pref)
            await session.commit()

    async def get_user_preference(self, category: str, key: str) -> Optional[Any]:
        async with self._get_session()() as session:
            result = await session.execute(
                select(UserPreference).where(
                    and_(UserPreference.category == category, UserPreference.key == key)
                )
            )
            pref = result.scalar_one_or_none()
            return pref.value if pref else None

    async def get_preferences_by_category(self, category: str) -> Dict[str, Any]:
        async with self._get_session()() as session:
            result = await session.execute(
                select(UserPreference).where(UserPreference.category == category)
            )
            prefs = result.scalars().all()
            return {p.key: p.value for p in prefs}

    # --- Chat Messages ---

    async def store_chat_message(
        self,
        session_id: uuid.UUID,
        role: str,
        content: str,
        symbol: Optional[str] = None,
        metadata: Optional[Dict] = None,
    ):
        async with self._get_session()() as session:
            msg = ChatMessage(
                session_id=session_id,
                role=role,
                content=content,
                symbol=symbol,
                metadata_=metadata or {},
            )
            session.add(msg)
            await session.commit()

    async def get_chat_history(
        self, session_id: uuid.UUID, limit: int = 50
    ) -> List[Dict[str, str]]:
        async with self._get_session()() as session:
            result = await session.execute(
                select(ChatMessage)
                .where(ChatMessage.session_id == session_id)
                .order_by(ChatMessage.created_at.desc())
                .limit(limit)
            )
            msgs = result.scalars().all()
            return [
                {"role": m.role, "content": m.content, "created_at": m.created_at.isoformat()}
                for m in reversed(msgs)  # chronological order
            ]

    # ===========================================================
    # Tier B — Vector Memory (RAG)
    # ===========================================================

    async def store_memory(
        self,
        content: str,
        memory_type: str,
        source: str = "chat",
        symbol: Optional[str] = None,
        strategy_id: Optional[uuid.UUID] = None,
        market_regime: Optional[str] = None,
        importance_score: float = 0.50,
        metadata: Optional[Dict] = None,
    ) -> Optional[Dict[str, Any]]:
        """Store an unstructured memory with auto-embedding."""
        if not self.embeddings_available():
            logger.warning("Cannot store memory — embeddings unavailable")
            return None

        embedding = self._embedding_service.embed(content)

        async with self._get_session()() as session:
            mem = AgentMemory(
                content=content,
                embedding=embedding,
                memory_type=memory_type,
                source=source,
                symbol=symbol,
                strategy_id=strategy_id,
                market_regime=market_regime,
                importance_score=importance_score,
                metadata_=metadata or {},
            )
            session.add(mem)
            await session.commit()
            await session.refresh(mem)
            logger.info(f"Stored memory: {mem}")
            return {"id": str(mem.id), "type": memory_type, "content": content[:80]}

    async def recall_memories(
        self,
        query: str,
        limit: int = 0,
        symbol: Optional[str] = None,
        memory_type: Optional[str] = None,
        min_importance: float = 0.0,
    ) -> List[Dict[str, Any]]:
        """Semantic search with cosine similarity weighted by decay and importance."""
        if not self.embeddings_available():
            return []

        limit = limit or self._recall_limit
        query_embedding = self._embedding_service.embed(query)

        async with self._get_session()() as session:
            # Build raw SQL for pgvector cosine distance with weighting
            # Score = (1 - cosine_distance) * decay_weight * importance_score
            filters = ["1=1"]
            params: Dict[str, Any] = {"embedding": str(query_embedding), "lim": limit}

            if symbol:
                filters.append("symbol = :symbol")
                params["symbol"] = symbol
            if memory_type:
                filters.append("memory_type = :mtype")
                params["mtype"] = memory_type
            if min_importance > 0:
                filters.append("importance_score >= :min_imp")
                params["min_imp"] = min_importance

            where_clause = " AND ".join(filters)

            sql = text(f"""
                SELECT id, content, memory_type, source, symbol, market_regime,
                       importance_score, decay_weight, created_at,
                       (1 - (embedding <=> CAST(:embedding AS vector))) * decay_weight * importance_score AS score
                FROM agent_memories
                WHERE {where_clause}
                ORDER BY score DESC
                LIMIT :lim
            """)

            result = await session.execute(sql, params)
            rows = result.fetchall()

            # Update last_accessed for recalled memories
            if rows:
                ids = [row.id for row in rows]
                await session.execute(
                    update(AgentMemory)
                    .where(AgentMemory.id.in_(ids))
                    .values(last_accessed=datetime.now(timezone.utc))
                )
                await session.commit()

            return [
                {
                    "id": str(row.id),
                    "content": row.content,
                    "type": row.memory_type,
                    "source": row.source,
                    "symbol": row.symbol,
                    "importance": float(row.importance_score),
                    "score": float(row.score),
                    "created_at": row.created_at.isoformat(),
                }
                for row in rows
            ]

    async def apply_decay(self):
        """Reduce decay_weight for old memories that haven't been accessed."""
        threshold = datetime.now(timezone.utc) - timedelta(days=self._decay_threshold_days)
        async with self._get_session()() as session:
            sql = text("""
                UPDATE agent_memories
                SET decay_weight = decay_weight * :factor
                WHERE last_accessed < :threshold
                  AND decay_weight > 0.01
            """)
            result = await session.execute(
                sql, {"factor": self._decay_factor, "threshold": threshold}
            )
            await session.commit()
            logger.info(f"Decay applied to {result.rowcount} memories")

    # ===========================================================
    # Tier C — Session Context
    # ===========================================================

    def get_or_create_session(
        self, session_id: Optional[str] = None, symbol: str = "QQQ"
    ) -> SessionContext:
        sid = uuid.UUID(session_id) if session_id else uuid.uuid4()
        if sid not in self._sessions:
            self._sessions[sid] = SessionContext(
                session_id=sid,
                active_symbol=symbol,
                max_turns=self._max_session_turns,
            )
        return self._sessions[sid]

    def add_turn(self, session_id: uuid.UUID, role: str, content: str):
        ctx = self._sessions.get(session_id)
        if ctx:
            ctx.add_turn(role, content)

    def get_session_context(self, session_id: uuid.UUID) -> str:
        ctx = self._sessions.get(session_id)
        return ctx.format_context() if ctx else ""

    # ===========================================================
    # LLM Context Builder
    # ===========================================================

    async def build_memory_context(
        self,
        query: str,
        symbol: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> str:
        """Combine all memory tiers into a single context block for LLM injection."""
        sections = []

        # Tier B: Relevant vector memories
        memories = await self.recall_memories(query, symbol=symbol)
        if memories:
            mem_lines = []
            for m in memories:
                mem_lines.append(
                    f"- [{m['type']}] (score={m['score']:.2f}) {m['content'][:200]}"
                )
            sections.append("### Relevant Memories\n" + "\n".join(mem_lines))

        # Tier A: Active decisions
        decisions = await self.get_active_decisions(symbol=symbol)
        if decisions:
            dec_lines = [f"- {d['text'][:150]}" for d in decisions[:5]]
            sections.append("### Active Trading Decisions\n" + "\n".join(dec_lines))

        # Tier A: User preferences (risk category)
        risk_prefs = await self.get_preferences_by_category("risk")
        if risk_prefs:
            pref_lines = [f"- {k}: {v}" for k, v in risk_prefs.items()]
            sections.append("### User Risk Preferences\n" + "\n".join(pref_lines))

        # Tier C: Session context
        if session_id:
            session_text = self.get_session_context(
                uuid.UUID(session_id) if isinstance(session_id, str) else session_id
            )
            if session_text:
                sections.append("### Recent Conversation\n" + session_text)

        if not sections:
            return ""

        return "## Agent Memory Context\n\n" + "\n\n".join(sections) + "\n"
