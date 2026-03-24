from dataclasses import dataclass, field
from typing import Optional


@dataclass
class EngineConfig:
    # Ollama settings
    ollama_base_url: str = "http://localhost:11434"
    extraction_model: str = "llama3.1:8b"
    embedding_model: str = "nomic-embed-text"
    philosopher_model: str = "llama3.1:8b"

    # Activation parameters
    activation_decay: float = 0.05
    spread_factor: float = 0.5
    activation_threshold: float = 0.1
    max_spread_depth: int = 6

    # Concept management
    merge_similarity_threshold: float = 0.85
    branch_similarity_threshold: float = 0.5
    split_contradiction_threshold: float = 0.3

    # Reader settings
    chunk_size: int = 5000
    chunk_overlap: int = 500

    # Persistence
    storage_path: str = "./thought_engine_data"
    expert_name: str = "default"

    # Thinking depth
    default_think_cycles: int = 3
    deep_think_cycles: int = 8
