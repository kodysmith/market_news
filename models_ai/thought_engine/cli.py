"""CLI for the Thought Engine — read books, ask questions, inspect knowledge."""

import argparse
import logging
import sys

from .config import EngineConfig
from .core import ThoughtEngine


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description="Thought Engine — learn from books, think, answer")
    sub = parser.add_subparsers(dest="command")

    # Read a book
    read_p = sub.add_parser("read", help="Read a book/document and learn from it")
    read_p.add_argument("path", help="Path to book file (txt, pdf, epub, md)")
    read_p.add_argument("--expert", default="default", help="Expert name to save as")
    read_p.add_argument("--model", default="llama3.1:8b", help="Ollama model for extraction")
    read_p.add_argument("--embed-model", default="nomic-embed-text", help="Ollama embedding model")
    read_p.add_argument("--ollama-url", default="http://localhost:11434", help="Ollama base URL")

    # Ask a question
    ask_p = sub.add_parser("ask", help="Ask the expert a question")
    ask_p.add_argument("question", help="Your question")
    ask_p.add_argument("--expert", default="default", help="Which expert to ask")
    ask_p.add_argument("--deep", action="store_true", help="Think deeper (more cycles)")
    ask_p.add_argument("--model", default=None, help="Override Ollama model")
    ask_p.add_argument("--ollama-url", default="http://localhost:11434", help="Ollama base URL")

    # Interactive chat
    chat_p = sub.add_parser("chat", help="Interactive chat with an expert")
    chat_p.add_argument("--expert", default="default", help="Which expert to chat with")
    chat_p.add_argument("--ollama-url", default="http://localhost:11434", help="Ollama base URL")

    # Inspect concepts
    inspect_p = sub.add_parser("inspect", help="Inspect a concept")
    inspect_p.add_argument("concept", help="Concept name to inspect")
    inspect_p.add_argument("--expert", default="default", help="Which expert")

    # Stats
    stats_p = sub.add_parser("stats", help="Show expert stats")
    stats_p.add_argument("--expert", default="default", help="Which expert")

    # List experts
    sub.add_parser("list", help="List available experts")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    if args.command == "read":
        config = EngineConfig(
            ollama_base_url=args.ollama_url,
            extraction_model=args.model,
            embedding_model=args.embed_model,
            expert_name=args.expert,
        )
        try:
            engine = ThoughtEngine.load(args.expert, config.storage_path)
            engine.config.extraction_model = args.model
            engine.config.embedding_model = args.embed_model
            engine.config.ollama_base_url = args.ollama_url
            from .extractor import ConceptExtractor
            engine.extractor = ConceptExtractor(engine.config)
        except FileNotFoundError:
            engine = ThoughtEngine(config)

        engine.read(args.path)
        engine.save()
        stats = engine.stats()
        print(f"\nExpert '{args.expert}' now has {stats['total_concepts']} concepts, "
              f"{stats['total_edges']} edges from {stats['books_read']} books")

    elif args.command == "ask":
        engine = ThoughtEngine.load(args.expert)
        if args.model:
            engine.config.extraction_model = args.model
        if args.ollama_url:
            engine.config.ollama_base_url = args.ollama_url
        cycles = engine.config.deep_think_cycles if args.deep else None
        answer = engine.ask(args.question, think_cycles=cycles)
        print(answer)

    elif args.command == "chat":
        engine = ThoughtEngine.load(args.expert)
        if args.ollama_url:
            engine.config.ollama_base_url = args.ollama_url
        print(f"Chatting with expert '{args.expert}' ({len(engine.trie.concepts)} concepts)")
        print("Type 'quit' to exit, 'deep: <question>' for deeper thinking")
        print("Type 'inspect: <concept>' to examine a concept")
        print("-" * 60)
        while True:
            try:
                q = input("\nYou: ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not q:
                continue
            if q.lower() == "quit":
                break
            if q.lower().startswith("inspect:"):
                concept_name = q.split(":", 1)[1].strip()
                print(engine.explain_concept(concept_name))
                continue
            deep = q.lower().startswith("deep:")
            if deep:
                q = q.split(":", 1)[1].strip()
            cycles = engine.config.deep_think_cycles if deep else None
            answer = engine.ask(q, think_cycles=cycles)
            print(f"\nExpert: {answer}")

    elif args.command == "inspect":
        engine = ThoughtEngine.load(args.expert)
        print(engine.explain_concept(args.concept))

    elif args.command == "stats":
        engine = ThoughtEngine.load(args.expert)
        stats = engine.stats()
        for k, v in stats.items():
            print(f"  {k}: {v}")

    elif args.command == "list":
        from pathlib import Path
        storage = Path("./thought_engine_data")
        if not storage.exists():
            print("No experts found.")
            return
        for d in sorted(storage.iterdir()):
            if d.is_dir() and (d / "trie.json").exists():
                try:
                    e = ThoughtEngine.load(d.name)
                    s = e.stats()
                    print(f"  {d.name}: {s['total_concepts']} concepts, "
                          f"{s['total_edges']} edges, {s['books_read']} books")
                except Exception as ex:
                    print(f"  {d.name}: (error loading: {ex})")


if __name__ == "__main__":
    main()
