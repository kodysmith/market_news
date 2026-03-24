"""Flexible Ollama client — swap models without changing anything else."""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

import requests

logger = logging.getLogger(__name__)


class OllamaClient:
    def __init__(self, base_url: str = "http://localhost:11434"):
        self.base_url = base_url.rstrip("/")

    def generate(self, model: str, prompt: str, system: str = "",
                 temperature: float = 0.3, format: str = "",
                 max_retries: int = 3) -> str:
        """Generate text from a model."""
        payload: Dict[str, Any] = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": temperature},
        }
        if system:
            payload["system"] = system
        if format:
            payload["format"] = format

        import time
        for attempt in range(max_retries):
            try:
                resp = requests.post(f"{self.base_url}/api/generate", json=payload, timeout=120)
                resp.raise_for_status()
                return resp.json()["response"]
            except (requests.exceptions.HTTPError, requests.exceptions.ConnectionError) as e:
                if attempt < max_retries - 1:
                    logger.warning("Ollama request failed (attempt %d/%d): %s", attempt + 1, max_retries, e)
                    time.sleep(2 ** attempt)
                else:
                    raise

    def generate_json(self, model: str, prompt: str, system: str = "",
                      temperature: float = 0.1) -> dict:
        """Generate and parse JSON output."""
        raw = self.generate(model, prompt, system, temperature, format="json")
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            # Try to extract JSON from the response
            start = raw.find("{")
            end = raw.rfind("}") + 1
            if start >= 0 and end > start:
                return json.loads(raw[start:end])
            logger.error("Failed to parse JSON from LLM response: %s", raw[:200])
            return {}

    def embed(self, model: str, text: str) -> List[float]:
        """Get embedding for a single text."""
        resp = requests.post(
            f"{self.base_url}/api/embed",
            json={"model": model, "input": text},
            timeout=60,
        )
        resp.raise_for_status()
        data = resp.json()
        return data["embeddings"][0]

    def embed_batch(self, model: str, texts: List[str]) -> List[List[float]]:
        """Get embeddings for multiple texts."""
        resp = requests.post(
            f"{self.base_url}/api/embed",
            json={"model": model, "input": texts},
            timeout=120,
        )
        resp.raise_for_status()
        return resp.json()["embeddings"]

    def list_models(self) -> List[str]:
        """List available models."""
        resp = requests.get(f"{self.base_url}/api/tags", timeout=10)
        resp.raise_for_status()
        return [m["name"] for m in resp.json().get("models", [])]

    def is_available(self) -> bool:
        try:
            requests.get(f"{self.base_url}/api/tags", timeout=5)
            return True
        except requests.ConnectionError:
            return False
