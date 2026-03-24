"""Concept extraction from text using Ollama LLMs."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np

from .config import EngineConfig
from .ollama_client import OllamaClient

logger = logging.getLogger(__name__)

EXTRACTION_SYSTEM = """You are a concept extraction engine. Given a text passage, extract structured knowledge.

Return JSON with this exact structure:
{
  "concepts": [
    {
      "name": "concept name (lowercase, concise)",
      "domain": "the domain (finance, economics, management, science, general, etc.)",
      "properties": {"property_name": "description", ...},
      "is_a": "parent concept or null",
      "relationships": [
        {"target": "other concept name", "relation": "causes|part_of|has|produces|requires|enables|opposes|measures", "context": "brief explanation"}
      ]
    }
  ],
  "rules": [
    {"if": "condition", "then": "consequence", "domain": "domain", "confidence": 0.0-1.0}
  ],
  "key_terms": ["important terms that should be indexed"]
}

Focus on extracting MEANINGFUL concepts, not every noun. Prioritize:
- Causal relationships (X causes Y)
- Compositional relationships (X is part of Y)
- Rules and principles
- Definitions and properties

Be precise. Use specific concept names, not vague ones."""

PHILOSOPHER_SYSTEM = """You are a concept philosopher. Given an existing concept and new information about it, decide how to integrate the new knowledge.

Return JSON:
{
  "action": "merge|branch|refine|contradict",
  "reasoning": "brief explanation",
  "updates": {
    "new_properties": {"prop": "description"},
    "removed_properties": ["prop names that are wrong"],
    "new_relationships": [{"target": "concept", "relation": "type", "context": "why"}],
    "confidence_change": 0.0-0.2
  }
}

Actions:
- merge: new info confirms/extends existing concept
- branch: new info describes a sub-type (create child node)
- refine: new info corrects or sharpens existing understanding
- contradict: new info conflicts with existing knowledge (flag for review)"""


class ConceptExtractor:
    def __init__(self, config: EngineConfig):
        self.config = config
        self.client = OllamaClient(config.ollama_base_url)

    def extract_from_passage(self, passage: str, context: str = "") -> Dict[str, Any]:
        """Extract concepts, relationships, and rules from a text passage."""
        prompt = f"Extract concepts from this passage:\n\n{passage}"
        if context:
            prompt = f"Context: {context}\n\n{prompt}"

        result = self.client.generate_json(
            model=self.config.extraction_model,
            prompt=prompt,
            system=EXTRACTION_SYSTEM,
        )
        if not result:
            return {"concepts": [], "rules": [], "key_terms": []}
        # Ensure expected keys
        result.setdefault("concepts", [])
        result.setdefault("rules", [])
        result.setdefault("key_terms", [])
        return result

    def get_embedding(self, text: str) -> np.ndarray:
        """Get embedding vector for text."""
        raw = self.client.embed(self.config.embedding_model, text)
        return np.array(raw)

    def get_embeddings(self, texts: List[str]) -> List[np.ndarray]:
        """Get embedding vectors for multiple texts."""
        raw = self.client.embed_batch(self.config.embedding_model, texts)
        return [np.array(r) for r in raw]

    def consult_philosopher(self, existing_concept: dict,
                            new_info: str) -> Dict[str, Any]:
        """Ask the philosopher model how to integrate new knowledge."""
        prompt = (
            f"Existing concept:\n{_format_concept(existing_concept)}\n\n"
            f"New information:\n{new_info}\n\n"
            f"How should this new information be integrated?"
        )
        result = self.client.generate_json(
            model=self.config.philosopher_model,
            prompt=prompt,
            system=PHILOSOPHER_SYSTEM,
        )
        if not result or "action" not in result:
            return {"action": "merge", "reasoning": "default", "updates": {}}
        return result

    def generate_response(self, context: str, question: str) -> str:
        """Generate a natural language response given retrieved context."""
        prompt = (
            f"Based on the following knowledge, answer the question.\n\n"
            f"Knowledge:\n{context}\n\n"
            f"Question: {question}\n\n"
            f"Answer concisely and precisely. If the knowledge doesn't cover "
            f"the question, say so. Include confidence level."
        )
        return self.client.generate(
            model=self.config.extraction_model,
            prompt=prompt,
            temperature=0.4,
        )


def _format_concept(concept: dict) -> str:
    lines = [f"Name: {concept.get('name', '?')}"]
    if concept.get("properties"):
        lines.append(f"Properties: {concept['properties']}")
    if concept.get("domain"):
        lines.append(f"Domain: {concept['domain']}")
    if concept.get("edges"):
        rels = [f"  {e.get('relation', '?')} → {e.get('target_id', '?')}" for e in concept["edges"][:10]]
        lines.append("Relationships:\n" + "\n".join(rels))
    return "\n".join(lines)
