"""ThoughtEngine — spreading activation + concept learning from text."""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .concepts import ConceptNode, ConceptTrie, _cosine
from .config import EngineConfig
from .extractor import ConceptExtractor
from .reader import chunk_text, read_file

logger = logging.getLogger(__name__)


class ThoughtEngine:
    """
    A thought engine that reads text, extracts concepts into a trie,
    and answers questions via spreading activation over the concept graph.
    """

    def __init__(self, config: Optional[EngineConfig] = None):
        self.config = config or EngineConfig()
        self.trie = ConceptTrie()
        self.extractor = ConceptExtractor(self.config)
        self.activation: Dict[str, float] = defaultdict(float)
        self._books_read: List[str] = []

    # ── Reading & Learning ───────────────────────────────────────────

    def read(self, path: str, verbose: bool = True):
        """Read a book/document and learn from it."""
        if verbose:
            logger.info("Reading: %s", path)

        text = read_file(path)
        chunks = chunk_text(text, self.config.chunk_size, self.config.chunk_overlap)

        if verbose:
            logger.info("Split into %d chunks", len(chunks))

        for i, chunk in enumerate(chunks):
            try:
                self._learn_from_chunk(chunk, source=f"{path}:chunk_{i}")
            except Exception as e:
                logger.warning("  Chunk %d failed: %s — skipping", i, e)
                continue
            if verbose and (i + 1) % 10 == 0:
                logger.info("  Processed %d/%d chunks (%d concepts so far)",
                            i + 1, len(chunks), len(self.trie.concepts))
            # Auto-save every 50 chunks to avoid losing progress
            if (i + 1) % 50 == 0:
                self.save()
                if verbose:
                    logger.info("  Auto-saved at chunk %d", i + 1)

        self._books_read.append(path)
        if verbose:
            stats = self.trie.stats()
            logger.info("Done reading %s — %d concepts, %d edges",
                        path, stats["total_concepts"], stats["total_edges"])

    def read_text(self, text: str, source: str = "direct_input"):
        """Learn from raw text directly."""
        chunks = chunk_text(text, self.config.chunk_size, self.config.chunk_overlap)
        for i, chunk in enumerate(chunks):
            self._learn_from_chunk(chunk, source=f"{source}:chunk_{i}")

    def _learn_from_chunk(self, chunk: str, source: str = ""):
        """Extract concepts from a chunk and integrate into the trie."""
        extraction = self.extractor.extract_from_passage(chunk)

        for raw_concept in extraction.get("concepts", []):
            self._integrate_concept(raw_concept, chunk, source)

        for rule in extraction.get("rules", []):
            self._integrate_rule(rule, source)

    def _integrate_concept(self, raw: dict, passage: str, source: str):
        """Integrate an extracted concept into the trie — merge, branch, or create."""
        name = raw.get("name", "").strip().lower()
        if not name:
            return

        domain = raw.get("domain", "general")
        properties = raw.get("properties", {})

        # Get embedding for similarity matching
        embed_text = f"{name}: {' '.join(str(v) for v in properties.values())}"
        embedding = self.extractor.get_embedding(embed_text)

        # Check if concept already exists
        existing_match = self._find_existing_concept(name, embedding, domain)

        if existing_match:
            concept, similarity = existing_match
            if similarity > self.config.merge_similarity_threshold:
                self._merge_concept(concept, raw, embedding, source)
            elif similarity > self.config.branch_similarity_threshold:
                self._branch_concept(concept, raw, embedding, domain, source)
            else:
                self._create_concept(raw, embedding, domain, source)
        else:
            self._create_concept(raw, embedding, domain, source)

        # Process relationships
        for rel in raw.get("relationships", []):
            self._integrate_relationship(name, rel, domain, source)

    def _find_existing_concept(self, name: str, embedding: np.ndarray,
                                domain: str) -> Optional[Tuple[ConceptNode, float]]:
        """Find an existing concept that matches by name or embedding."""
        # Check lexicon first
        lex = self.trie.lookup(name)
        if lex and lex.senses:
            best_id = max(lex.senses, key=lex.senses.get)
            concept = self.trie.concepts.get(best_id)
            if concept and concept.embedding is not None:
                sim = _cosine(embedding, concept.embedding)
                return (concept, sim)

        # Fall back to embedding similarity
        similar = self.trie.find_similar(embedding, top_k=1, domain=domain)
        if similar:
            concept, sim = similar[0]
            if sim > self.config.branch_similarity_threshold:
                return (concept, sim)

        return None

    def _merge_concept(self, existing: ConceptNode, raw: dict,
                       embedding: np.ndarray, source: str):
        """Merge new information into an existing concept."""
        # Consult philosopher only after significant evidence accumulates
        if existing.source_count >= 5:
            advice = self.extractor.consult_philosopher(
                existing.to_dict(),
                json.dumps(raw, indent=2),
            )
            action = advice.get("action", "merge")
            if action == "contradict":
                logger.warning("Contradiction detected for '%s': %s",
                               existing.name, advice.get("reasoning", ""))
                # Store contradiction as a property for review
                existing.add_property(f"_contradiction:{source}", 0.3)
                return
            if action == "branch":
                self._branch_concept(existing, raw, embedding,
                                     raw.get("domain", "general"), source)
                return

        for prop, desc in raw.get("properties", {}).items():
            existing.add_property(prop, 0.5)
        if embedding is not None:
            if existing.embedding is not None:
                n = existing.source_count or 1
                existing.embedding = (existing.embedding * n + embedding) / (n + 1)
            else:
                existing.embedding = embedding
        existing.sources.append(source)
        existing.confidence = min(1.0, existing.confidence + 0.1)

    def _branch_concept(self, parent: ConceptNode, raw: dict,
                        embedding: np.ndarray, domain: str, source: str):
        """Create a child concept branching from a parent."""
        parent_props = set(parent.properties.keys())
        new_props = raw.get("properties", {})
        delta = {k: v for k, v in new_props.items() if k not in parent_props}

        child = self.trie.add_concept(
            name=raw.get("name", "unknown"),
            parent_id=parent.id,
            properties={k: 0.5 for k in new_props},
            delta=delta,
            domain=domain,
            embedding=embedding,
            source=source,
        )
        self.trie.add_edge(child.id, parent.id, "is_a", domain, 0.9, source=source)

    def _create_concept(self, raw: dict, embedding: np.ndarray,
                        domain: str, source: str):
        """Create a brand new root concept."""
        name = raw.get("name", "unknown")
        properties = {k: 0.5 for k in raw.get("properties", {})}

        concept = self.trie.add_concept(
            name=name,
            properties=properties,
            domain=domain,
            embedding=embedding,
            source=source,
        )

        # Link to parent concept if specified
        is_a = raw.get("is_a")
        if is_a:
            lex = self.trie.lookup(is_a.lower())
            if lex and lex.senses:
                parent_id = max(lex.senses, key=lex.senses.get)
                concept.parent_id = parent_id
                if parent_id in self.trie.concepts:
                    self.trie.concepts[parent_id].children_ids.append(concept.id)
                    if concept.id in self.trie.roots:
                        self.trie.roots.remove(concept.id)
                self.trie.add_edge(concept.id, parent_id, "is_a", domain, 0.9, source=source)

    def _integrate_relationship(self, source_name: str, rel: dict,
                                domain: str, source: str):
        """Add an edge between concepts."""
        target_name = rel.get("target", "").strip().lower()
        if not target_name:
            return

        # Find source concept
        source_lex = self.trie.lookup(source_name)
        if not source_lex or not source_lex.senses:
            return
        source_id = max(source_lex.senses, key=source_lex.senses.get)

        # Find or create target concept
        target_lex = self.trie.lookup(target_name)
        if target_lex and target_lex.senses:
            target_id = max(target_lex.senses, key=target_lex.senses.get)
        else:
            target_embedding = self.extractor.get_embedding(target_name)
            target = self.trie.add_concept(
                name=target_name, domain=domain,
                embedding=target_embedding, source=source,
            )
            target_id = target.id

        self.trie.add_edge(
            source_id, target_id,
            relation=rel.get("relation", "related_to"),
            domain=domain,
            weight=0.5,
            context=rel.get("context", ""),
            source=source,
        )

    def _integrate_rule(self, rule: dict, source: str):
        """Store a rule as a special concept with conditional edges."""
        rule_name = f"rule:{rule.get('if', 'unknown')[:50]}"
        embedding = self.extractor.get_embedding(
            f"{rule.get('if', '')} then {rule.get('then', '')}"
        )
        concept = self.trie.add_concept(
            name=rule_name,
            properties={
                "condition": rule.get("if", ""),
                "consequence": rule.get("then", ""),
                "confidence": rule.get("confidence", 0.5),
            },
            domain=rule.get("domain", "general"),
            embedding=embedding,
            source=source,
        )

    # ── Spreading Activation (Thinking) ──────────────────────────────

    def activate(self, concept_id: str, energy: float, depth: int = 0):
        """Activate a concept and spread energy to neighbors."""
        if depth > self.config.max_spread_depth:
            return
        if energy < self.config.activation_threshold:
            return

        self.activation[concept_id] += energy
        concept = self.trie.concepts.get(concept_id)
        if not concept:
            return

        concept.access_count += 1

        # Spread to edges
        for edge in concept.edges:
            neighbor_energy = energy * edge.weight * self.config.spread_factor
            self.activate(edge.target_id, neighbor_energy, depth + 1)

        # Spread to parent (inherit upward)
        if concept.parent_id:
            self.activate(concept.parent_id, energy * 0.3, depth + 1)

        # Spread to children (inherit downward, weaker)
        for child_id in concept.children_ids:
            self.activate(child_id, energy * 0.2, depth + 1)

    def think(self, top_k: int = 20) -> List[Tuple[ConceptNode, float]]:
        """Return the most activated concepts — the current 'thought'."""
        active = sorted(self.activation.items(), key=lambda x: -x[1])[:top_k]
        result = []
        for concept_id, energy in active:
            concept = self.trie.concepts.get(concept_id)
            if concept:
                result.append((concept, energy))
        return result

    def tick(self):
        """Time passes — activation decays."""
        to_remove = []
        for concept_id in self.activation:
            self.activation[concept_id] *= (1 - self.config.activation_decay)
            if self.activation[concept_id] < 0.01:
                to_remove.append(concept_id)
        for cid in to_remove:
            del self.activation[cid]

    def clear_activation(self):
        """Reset all activation — fresh state of mind."""
        self.activation.clear()

    # ── Querying ─────────────────────────────────────────────────────

    def ask(self, question: str, think_cycles: Optional[int] = None) -> str:
        """Ask a question — activate relevant concepts, think, respond."""
        cycles = think_cycles or self.config.default_think_cycles
        self.clear_activation()

        # Get question embedding
        q_embedding = self.extractor.get_embedding(question)

        # Find relevant concepts
        relevant = self.trie.find_similar(q_embedding, top_k=10)

        if not relevant:
            return "I don't have enough knowledge to answer that question."

        # Activate relevant concepts
        for concept, similarity in relevant:
            self.activate(concept.id, similarity)

        # Think — let activation spread for several cycles
        for _ in range(cycles):
            self.tick()
            # Re-boost the most relevant concepts to prevent total decay
            for concept, similarity in relevant[:3]:
                self.activate(concept.id, similarity * 0.3)

        # Gather activated knowledge
        thoughts = self.think(top_k=15)
        knowledge = self._format_knowledge(thoughts)

        # Generate response using LLM
        response = self.extractor.generate_response(knowledge, question)

        # Add confidence annotation
        avg_confidence = np.mean([c.confidence for c, _ in thoughts]) if thoughts else 0
        confidence_label = (
            "HIGH" if avg_confidence > 0.7 else
            "MEDIUM" if avg_confidence > 0.4 else
            "LOW"
        )
        return f"{response}\n\n[Confidence: {confidence_label} | {len(thoughts)} concepts activated | Sources: {self._count_sources(thoughts)}]"

    def _format_knowledge(self, thoughts: List[Tuple[ConceptNode, float]]) -> str:
        """Format activated concepts into context for the LLM."""
        lines = []
        for concept, energy in thoughts:
            props = self.trie.get_inherited_properties(concept.id)
            # Filter to numeric confidence values, coerce strings
            numeric_props = {}
            for k, v in props.items():
                if k.startswith("_"):
                    continue
                try:
                    numeric_props[k] = float(v)
                except (TypeError, ValueError):
                    numeric_props[k] = 0.5  # default confidence for string props
            prop_str = ", ".join(f"{k} ({v:.0%} confident)" for k, v in
                                sorted(numeric_props.items(), key=lambda x: -x[1])[:8])
            rels = []
            for edge in concept.edges[:5]:
                target = self.trie.concepts.get(edge.target_id)
                if target:
                    rels.append(f"{edge.relation} → {target.name}")
            rel_str = "; ".join(rels) if rels else ""

            line = f"- {concept.name} (domain: {concept.domain}, activation: {energy:.2f})"
            if prop_str:
                line += f"\n  Properties: {prop_str}"
            if rel_str:
                line += f"\n  Relationships: {rel_str}"
            lines.append(line)
        return "\n".join(lines)

    def _count_sources(self, thoughts: List[Tuple[ConceptNode, float]]) -> int:
        sources = set()
        for concept, _ in thoughts:
            sources.update(concept.sources)
        return len(sources)

    # ── Persistence ──────────────────────────────────────────────────

    def save(self, name: Optional[str] = None):
        """Save the engine state to disk."""
        expert = name or self.config.expert_name
        base = Path(self.config.storage_path) / expert
        base.mkdir(parents=True, exist_ok=True)
        self.trie.save(str(base / "trie.json"))
        meta = {"books_read": self._books_read, "config": self.config.__dict__}
        with open(base / "meta.json", "w") as f:
            json.dump(meta, f, indent=2)
        logger.info("Saved expert '%s' — %d concepts", expert, len(self.trie.concepts))

    @classmethod
    def load(cls, name: str, storage_path: str = "./thought_engine_data") -> "ThoughtEngine":
        """Load a saved expert."""
        base = Path(storage_path) / name
        if not base.exists():
            raise FileNotFoundError(f"No expert found at {base}")

        with open(base / "meta.json") as f:
            meta = json.load(f)

        config = EngineConfig(**{k: v for k, v in meta["config"].items()
                                 if k in EngineConfig.__dataclass_fields__})
        config.expert_name = name
        config.storage_path = storage_path

        engine = cls(config)
        engine.trie = ConceptTrie.load(str(base / "trie.json"))
        engine._books_read = meta.get("books_read", [])
        logger.info("Loaded expert '%s' — %d concepts", name, len(engine.trie.concepts))
        return engine

    # ── Introspection ────────────────────────────────────────────────

    def stats(self) -> dict:
        trie_stats = self.trie.stats()
        return {
            **trie_stats,
            "books_read": len(self._books_read),
            "active_concepts": len(self.activation),
            "expert_name": self.config.expert_name,
        }

    def explain_concept(self, name: str) -> str:
        """Show what the engine knows about a concept."""
        lex = self.trie.lookup(name.lower())
        if not lex or not lex.senses:
            return f"Unknown concept: {name}"

        lines = [f"Concept: {name}"]
        lines.append(f"Senses: {len(lex.senses)}")

        for concept_id, weight in sorted(lex.senses.items(), key=lambda x: -x[1]):
            concept = self.trie.concepts.get(concept_id)
            if not concept:
                continue
            lines.append(f"\n  [{concept.domain}] {concept.name} (weight: {weight:.2f}, confidence: {concept.confidence:.2f})")
            props = self.trie.get_inherited_properties(concept_id)
            for p, c in props.items():
                if p.startswith("_"):
                    continue
                try:
                    conf = float(c)
                    lines.append(f"    - {p}: {conf:.0%}")
                except (TypeError, ValueError):
                    lines.append(f"    - {p}: {c}")
            for edge in concept.edges[:5]:
                target = self.trie.concepts.get(edge.target_id)
                tname = target.name if target else edge.target_id
                lines.append(f"    → {edge.relation} {tname} ({edge.domain})")

            ancestors = self.trie.get_ancestors(concept_id)
            if ancestors:
                chain = " → ".join(a.name for a in reversed(ancestors))
                lines.append(f"    Inherits from: {chain}")

        return "\n".join(lines)
