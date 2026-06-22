"""
CLI for RAG Document Q&A
Author: Aman Mishra
Usage:
  python main.py ingest --files doc1.pdf doc2.txt
  python main.py query --question "What is the main topic?"
  python main.py chat  # interactive mode
"""

import argparse
import json
import os
import sys
from pathlib import Path

# Demo mode — used when heavy deps aren't installed
DEMO_MODE = False
try:
    from src.rag_pipeline import RAGPipeline, FAISS_AVAILABLE, ST_AVAILABLE
    if not FAISS_AVAILABLE or not ST_AVAILABLE:
        DEMO_MODE = True
except (ImportError, ModuleNotFoundError):
    DEMO_MODE = True


# ── Demo data ──────────────────────────────────────────────────────────────────

DEMO_DOCS = {
    "AI_Overview.txt": """
Artificial Intelligence (AI) refers to the simulation of human intelligence in machines.
Machine Learning (ML) is a subset of AI that enables systems to learn from data.
Deep Learning uses neural networks with many layers to learn complex patterns.
Natural Language Processing (NLP) helps machines understand and generate human language.
Large Language Models (LLMs) like GPT-4 are trained on massive text corpora.
Retrieval-Augmented Generation (RAG) combines retrieval systems with LLMs for accuracy.
""".strip(),

    "ML_Concepts.txt": """
Supervised learning trains models on labelled input-output pairs.
Unsupervised learning discovers hidden patterns without labelled data.
Reinforcement learning trains agents through reward signals.
Feature engineering is the process of selecting and transforming input variables.
Model evaluation metrics include accuracy, precision, recall, and F1-score.
Overfitting occurs when a model memorises training data and fails to generalise.
Cross-validation is a technique to assess how models generalise to unseen data.
""".strip(),
}

DEMO_QA = {
    "what is rag": (
        "RAG stands for Retrieval-Augmented Generation. It combines a retrieval "
        "system (that fetches relevant document chunks) with a Large Language Model "
        "to produce accurate, grounded answers. [Source: AI_Overview.txt]"
    ),
    "what is supervised": (
        "Supervised learning trains models on labelled input-output pairs, allowing "
        "them to learn a mapping from inputs to outputs. [Source: ML_Concepts.txt]"
    ),
    "what is overfitting": (
        "Overfitting occurs when a model memorises the training data so closely that "
        "it fails to generalise to new, unseen examples. [Source: ML_Concepts.txt]"
    ),
    "what is deep": (
        "Deep Learning uses neural networks with many layers to learn complex "
        "patterns in data. [Source: AI_Overview.txt]"
    ),
}


def demo_query(question: str) -> dict:
    q = question.lower().strip("?").strip()
    # Match on the most specific/unique word in each key (last word)
    for key, answer in DEMO_QA.items():
        topic = key.split()[-1]  # e.g. "rag", "overfitting", "learning"
        if topic in q:
            return {
                "question": question,
                "answer": answer,
                "sources": [{"rank": 1, "score": 0.92,
                              "source": answer.split("[Source: ")[-1].rstrip("]"),
                              "excerpt": answer[:100]}],
                "latency_ms": 18.4,
            }
            return {
                "question": question,
                "answer": answer,
                "sources": [{"rank": 1, "score": 0.92,
                              "source": answer.split("[Source: ")[-1].rstrip("]"),
                              "excerpt": answer[:100]}],
                "latency_ms": 18.4,
            }
    return {
        "question": question,
        "answer": (
            "In demo mode only a small sample knowledge base is loaded. "
            "Try asking about: RAG, supervised learning, overfitting, or deep learning."
        ),
        "sources": [],
        "latency_ms": 2.1,
    }


# ── CLI commands ───────────────────────────────────────────────────────────────

def cmd_ingest(args) -> None:
    if DEMO_MODE:
        print("[DEMO] Simulating document ingestion...")
        for name, content in DEMO_DOCS.items():
            print(f"  ✓ Processed '{name}' → "
                  f"{max(1, len(content.split()) // 512)} chunk(s)")
        print("\n[DEMO] Index built successfully (demo mode — no real embeddings).")
        return

    pipeline = RAGPipeline(
        embedding_model=args.embedding_model,
        llm_model=args.llm_model,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        top_k=args.top_k,
    )
    pipeline.ingest(args.files)
    pipeline.save_index(args.index_dir)
    print(f"\n✅ Index saved to '{args.index_dir}'")


def cmd_query(args) -> None:
    if DEMO_MODE:
        result = demo_query(args.question)
        _print_result(result, args.output_format)
        return

    pipeline = RAGPipeline(
        embedding_model=args.embedding_model,
        llm_model=args.llm_model,
        top_k=args.top_k,
    )
    pipeline.load_index(args.index_dir)
    response = pipeline.query(args.question)
    _print_result(response.to_dict(), args.output_format)


def cmd_chat(args) -> None:
    print("\n🤖  RAG Document Q&A — Interactive Chat")
    print("    Type 'exit' or 'quit' to stop.\n")

    pipeline = None
    if not DEMO_MODE:
        pipeline = RAGPipeline(
            embedding_model=args.embedding_model,
            llm_model=args.llm_model,
            top_k=args.top_k,
        )
        pipeline.load_index(args.index_dir)
    else:
        print("  ⚠  Running in DEMO mode "
              "(install deps for full functionality).\n")

    while True:
        try:
            question = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not question:
            continue
        if question.lower() in {"exit", "quit"}:
            print("Goodbye!")
            break

        if DEMO_MODE:
            result = demo_query(question)
        else:
            response = pipeline.query(question)
            result = response.to_dict()

        print(f"\nAssistant: {result['answer']}")
        if result.get("sources"):
            print("\nSources:")
            for s in result["sources"]:
                print(f"  [{s['rank']}] {s['source']} "
                      f"(score: {s['score']:.2f})")
        print(f"\n⏱  {result['latency_ms']:.1f} ms\n")


# ── Helpers ────────────────────────────────────────────────────────────────────

def _print_result(result: dict, fmt: str) -> None:
    if fmt == "json":
        print(json.dumps(result, indent=2))
        return

    print(f"\n{'='*60}")
    print(f"Question : {result['question']}")
    print(f"{'='*60}")
    print(f"Answer   : {result['answer']}")
    print(f"\nSources:")
    for s in result.get("sources", []):
        print(f"  [{s['rank']}] {s['source']} — score {s['score']:.2f}")
        print(f"       \"{s['excerpt']}\"")
    print(f"\n⏱  Latency: {result['latency_ms']:.1f} ms")
    if result.get("tokens_used"):
        print(f"🔤  Tokens : {result['tokens_used']}")
    print(f"{'='*60}\n")


# ── Arg parser ─────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="RAG Document Q&A CLI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    shared = argparse.ArgumentParser(add_help=False)
    shared.add_argument("--embedding-model", default="all-MiniLM-L6-v2")
    shared.add_argument("--llm-model", default="gpt-4o-mini")
    shared.add_argument("--index-dir", default="vector_store")
    shared.add_argument("--top-k", type=int, default=5)
    shared.add_argument("--output-format", choices=["pretty", "json"],
                        default="pretty")

    sub = parser.add_subparsers(dest="command", required=True)

    ing = sub.add_parser("ingest", parents=[shared],
                          help="Ingest and index documents")
    ing.add_argument("--files", nargs="+", required=True,
                     help="Paths to .pdf / .txt / .md files")
    ing.add_argument("--chunk-size", type=int, default=512)
    ing.add_argument("--chunk-overlap", type=int, default=64)
    ing.set_defaults(func=cmd_ingest)

    qry = sub.add_parser("query", parents=[shared],
                          help="Run a single query")
    qry.add_argument("--question", required=True)
    qry.set_defaults(func=cmd_query)

    cht = sub.add_parser("chat", parents=[shared],
                          help="Start interactive chat session")
    cht.set_defaults(func=cmd_chat)

    return parser


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)
