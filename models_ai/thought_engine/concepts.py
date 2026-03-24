"""Core concept data structures — the trie, nodes, edges, and lexical index."""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


@dataclass
class Edge:
    target_id: str
    relation: str  # "causes", "part_of", "is_a", "has", "used_in", "produces"
    domain: str  # "finance", "nature", "science", "general"
    weight: float = 0.5
    context: str = ""
    source: str = ""  # which book/passage created this edge

    def to_dict(self) -> dict:
        return {
            "target_id": self.target_id,
            "relation": self.relation,
            "domain": self.domain,
            "weight": self.weight,
            "context": self.context,
            "source": self.source,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Edge":
        return cls(**d)


@dataclass
class ConceptNode:
    id: str
    name: str
    parent_id: Optional[str] = None
    delta: Dict[str, Any] = field(default_factory=dict)  # what's new vs parent
    properties: Dict[str, float] = field(default_factory=dict)  # property → confidence
    embedding: Optional[np.ndarray] = None
    sources: List[str] = field(default_factory=list)  # passage references
    children_ids: List[str] = field(default_factory=list)
    edges: List[Edge] = field(default_factory=list)
    domain: str = "general"
    confidence: float = 0.0
    activation: float = 0.0  # current activation energy
    access_count: int = 0

    @property
    def source_count(self) -> int:
        return len(self.sources)

    def add_property(self, prop: str, confidence: float = 0.5):
        if prop in self.properties:
            # Bayesian-ish update — more sources → higher confidence
            old = self.properties[prop]
            self.properties[prop] = 1 - (1 - old) * (1 - confidence)
        else:
            self.properties[prop] = confidence

    def merge_from(self, other: "ConceptNode"):
        """Merge another concept's knowledge into this one."""
        for prop, conf in other.properties.items():
            self.add_property(prop, conf)
        self.sources.extend(other.sources)
        self.edges.extend(other.edges)
        if other.embedding is not None and self.embedding is not None:
            # Running average of embeddings
            n = self.source_count
            self.embedding = (self.embedding * (n - len(other.sources)) + other.embedding * len(other.sources)) / n
        self.confidence = min(1.0, self.confidence + 0.1)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "parent_id": self.parent_id,
            "delta": self.delta,
            "properties": self.properties,
            "embedding": self.embedding.tolist() if self.embedding is not None else None,
            "sources": self.sources,
            "children_ids": self.children_ids,
            "edges": [e.to_dict() for e in self.edges],
            "domain": self.domain,
            "confidence": self.confidence,
            "access_count": self.access_count,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ConceptNode":
        edges = [Edge.from_dict(e) for e in d.pop("edges", [])]
        emb = d.pop("embedding", None)
        node = cls(**d, edges=edges)
        if emb is not None:
            node.embedding = np.array(emb)
        return node


@dataclass
class LexicalNode:
    """A word/phrase that points to one or more concept senses."""
    word: str
    senses: Dict[str, float] = field(default_factory=dict)  # concept_id → prior weight

    def activate(self, context_embedding: np.ndarray, concept_lookup: Dict[str, ConceptNode]) -> List[Tuple[str, float]]:
        """Reweight senses based on context, return sorted activations."""
        if not self.senses:
            return []
        weighted = {}
        for concept_id, prior in self.senses.items():
            concept = concept_lookup.get(concept_id)
            if concept is None or concept.embedding is None:
                weighted[concept_id] = prior
                continue
            sim = _cosine(context_embedding, concept.embedding)
            weighted[concept_id] = prior * max(0.01, sim)

        total = sum(weighted.values())
        if total > 0:
            weighted = {k: v / total for k, v in weighted.items()}
        return sorted(weighted.items(), key=lambda x: -x[1])

    def add_sense(self, concept_id: str, weight: float = 0.5):
        if concept_id in self.senses:
            # Increase weight for repeated encounters
            self.senses[concept_id] = min(1.0, self.senses[concept_id] + weight * 0.2)
        else:
            self.senses[concept_id] = weight
        # Renormalize
        total = sum(self.senses.values())
        if total > 0:
            self.senses = {k: v / total for k, v in self.senses.items()}

    def to_dict(self) -> dict:
        return {"word": self.word, "senses": self.senses}

    @classmethod
    def from_dict(cls, d: dict) -> "LexicalNode":
        return cls(**d)


class ConceptTrie:
    """The concept knowledge store — trie structure with lexical index."""

    def __init__(self):
        self.concepts: Dict[str, ConceptNode] = {}
        self.lexicon: Dict[str, LexicalNode] = {}  # word → LexicalNode
        self.roots: List[str] = []  # top-level concept IDs

    def add_concept(self, name: str, parent_id: Optional[str] = None,
                    properties: Optional[Dict[str, float]] = None,
                    delta: Optional[Dict[str, Any]] = None,
                    domain: str = "general",
                    embedding: Optional[np.ndarray] = None,
                    source: str = "") -> ConceptNode:
        concept_id = f"{domain}:{name}:{uuid.uuid4().hex[:8]}"
        node = ConceptNode(
            id=concept_id,
            name=name,
            parent_id=parent_id,
            delta=delta or {},
            properties=properties or {},
            embedding=embedding,
            domain=domain,
            sources=[source] if source else [],
            confidence=0.3,
        )
        self.concepts[concept_id] = node

        if parent_id and parent_id in self.concepts:
            self.concepts[parent_id].children_ids.append(concept_id)
        else:
            self.roots.append(concept_id)

        # Register in lexicon
        word = name.lower()
        if word not in self.lexicon:
            self.lexicon[word] = LexicalNode(word=word)
        self.lexicon[word].add_sense(concept_id)

        return node

    def find_similar(self, embedding: np.ndarray, top_k: int = 5,
                     domain: Optional[str] = None) -> List[Tuple[ConceptNode, float]]:
        """Find concepts most similar to an embedding."""
        results = []
        for concept in self.concepts.values():
            if concept.embedding is None:
                continue
            if domain and concept.domain != domain:
                continue
            sim = _cosine(embedding, concept.embedding)
            results.append((concept, sim))
        results.sort(key=lambda x: -x[1])
        return results[:top_k]

    def get_ancestors(self, concept_id: str) -> List[ConceptNode]:
        """Walk up the trie from a concept to its root."""
        ancestors = []
        current = self.concepts.get(concept_id)
        while current and current.parent_id:
            parent = self.concepts.get(current.parent_id)
            if parent:
                ancestors.append(parent)
            current = parent
        return ancestors

    def get_inherited_properties(self, concept_id: str) -> Dict[str, float]:
        """Get all properties including inherited from ancestors."""
        props = {}
        ancestors = self.get_ancestors(concept_id)
        # Start from root, override with more specific
        for ancestor in reversed(ancestors):
            props.update(ancestor.properties)
        concept = self.concepts.get(concept_id)
        if concept:
            props.update(concept.properties)
        return props

    def add_edge(self, source_id: str, target_id: str, relation: str,
                 domain: str = "general", weight: float = 0.5,
                 context: str = "", source: str = ""):
        concept = self.concepts.get(source_id)
        if concept:
            edge = Edge(
                target_id=target_id,
                relation=relation,
                domain=domain,
                weight=weight,
                context=context,
                source=source,
            )
            concept.edges.append(edge)

    def lookup(self, word: str) -> Optional[LexicalNode]:
        return self.lexicon.get(word.lower())

    def stats(self) -> dict:
        return {
            "total_concepts": len(self.concepts),
            "total_words": len(self.lexicon),
            "root_concepts": len(self.roots),
            "total_edges": sum(len(c.edges) for c in self.concepts.values()),
        }

    def save(self, path: str):
        data = {
            "concepts": {k: v.to_dict() for k, v in self.concepts.items()},
            "lexicon": {k: v.to_dict() for k, v in self.lexicon.items()},
            "roots": self.roots,
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path: str) -> "ConceptTrie":
        with open(path) as f:
            data = json.load(f)
        trie = cls()
        trie.concepts = {k: ConceptNode.from_dict(v) for k, v in data["concepts"].items()}
        trie.lexicon = {k: LexicalNode.from_dict(v) for k, v in data["lexicon"].items()}
        trie.roots = data["roots"]
        return trie


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))
