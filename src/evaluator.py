"""
RAG Evaluation Module
Author: Aman Mishra
Evaluates RAG pipeline using standard metrics: faithfulness, relevance, answer quality.
"""

import json
import time
from dataclasses import dataclass, field
from typing import List, Dict, Optional
import numpy as np


@dataclass
class EvalSample:
    question: str
    reference_answer: str
    contexts: List[str] = field(default_factory=list)


@dataclass
class EvalMetrics:
    faithfulness: float        # answer grounded in context
    answer_relevance: float    # answer addresses the question
    context_precision: float   # retrieved context is relevant
    context_recall: float      # relevant info retrieved
    latency_ms: float

    def to_dict(self) -> Dict:
        return {
            "faithfulness": round(self.faithfulness, 4),
            "answer_relevance": round(self.answer_relevance, 4),
            "context_precision": round(self.context_precision, 4),
            "context_recall": round(self.context_recall, 4),
            "latency_ms": round(self.latency_ms, 2),
            "overall_score": round(self.overall_score(), 4),
        }

    def overall_score(self) -> float:
        weights = {
            "faithfulness": 0.35,
            "answer_relevance": 0.30,
            "context_precision": 0.20,
            "context_recall": 0.15,
        }
        return (
            self.faithfulness * weights["faithfulness"]
            + self.answer_relevance * weights["answer_relevance"]
            + self.context_precision * weights["context_precision"]
            + self.context_recall * weights["context_recall"]
        )


class RAGEvaluator:
    """
    Lightweight RAG evaluator using token-overlap heuristics.
    For production use, integrate RAGAS or TruLens.
    """

    def __init__(self):
        self.results: List[Dict] = []

    # ── Heuristic metrics ────────────────────────────────────────────────────

    @staticmethod
    def _token_overlap(a: str, b: str) -> float:
        """Jaccard similarity on word sets."""
        a_tokens = set(a.lower().split())
        b_tokens = set(b.lower().split())
        if not a_tokens or not b_tokens:
            return 0.0
        return len(a_tokens & b_tokens) / len(a_tokens | b_tokens)

    def compute_faithfulness(self, answer: str,
                              contexts: List[str]) -> float:
        """Measures how well the answer is grounded in the context."""
        if not contexts:
            return 0.0
        scores = [self._token_overlap(answer, ctx) for ctx in contexts]
        return float(np.max(scores))

    def compute_answer_relevance(self, answer: str, question: str,
                                  reference: str) -> float:
        """Combines relevance to question and similarity to reference."""
        q_sim = self._token_overlap(answer, question)
        ref_sim = self._token_overlap(answer, reference)
        return 0.4 * q_sim + 0.6 * ref_sim

    def compute_context_precision(self, question: str,
                                   contexts: List[str]) -> float:
        """Fraction of retrieved contexts that are relevant to the question."""
        if not contexts:
            return 0.0
        threshold = 0.05
        relevant = sum(
            1 for ctx in contexts
            if self._token_overlap(question, ctx) > threshold
        )
        return relevant / len(contexts)

    def compute_context_recall(self, reference: str,
                                contexts: List[str]) -> float:
        """How much reference answer info is covered in retrieved context."""
        if not contexts:
            return 0.0
        combined = " ".join(contexts)
        return self._token_overlap(reference, combined)

    # ── Evaluation runner ────────────────────────────────────────────────────

    def evaluate_sample(self, sample: EvalSample,
                         predicted_answer: str,
                         retrieved_contexts: List[str],
                         latency_ms: float) -> EvalMetrics:
        metrics = EvalMetrics(
            faithfulness=self.compute_faithfulness(
                predicted_answer, retrieved_contexts),
            answer_relevance=self.compute_answer_relevance(
                predicted_answer, sample.question, sample.reference_answer),
            context_precision=self.compute_context_precision(
                sample.question, retrieved_contexts),
            context_recall=self.compute_context_recall(
                sample.reference_answer, retrieved_contexts),
            latency_ms=latency_ms,
        )
        return metrics

    def run_benchmark(self, samples: List[EvalSample],
                      pipeline) -> Dict:
        """Run full evaluation benchmark against a RAGPipeline instance."""
        print(f"\n[Evaluator] Running benchmark on {len(samples)} samples…\n")
        all_metrics: List[EvalMetrics] = []

        for i, sample in enumerate(samples, start=1):
            t0 = time.perf_counter()
            response = pipeline.query(sample.question, use_cache=False)
            latency = (time.perf_counter() - t0) * 1000

            contexts = [r.chunk.text for r in response.retrieved_chunks]
            metrics = self.evaluate_sample(
                sample=sample,
                predicted_answer=response.answer,
                retrieved_contexts=contexts,
                latency_ms=latency,
            )
            all_metrics.append(metrics)

            print(f"  [{i}/{len(samples)}] Q: {sample.question[:60]}…")
            print(f"    Overall score: {metrics.overall_score():.3f} | "
                  f"Latency: {latency:.1f} ms")

        # Aggregate
        agg = {
            "num_samples": len(samples),
            "avg_faithfulness": float(np.mean([m.faithfulness for m in all_metrics])),
            "avg_answer_relevance": float(np.mean([m.answer_relevance for m in all_metrics])),
            "avg_context_precision": float(np.mean([m.context_precision for m in all_metrics])),
            "avg_context_recall": float(np.mean([m.context_recall for m in all_metrics])),
            "avg_latency_ms": float(np.mean([m.latency_ms for m in all_metrics])),
            "avg_overall_score": float(np.mean([m.overall_score() for m in all_metrics])),
            "per_sample": [m.to_dict() for m in all_metrics],
        }

        print(f"\n{'='*50}")
        print(f"  Benchmark Summary ({len(samples)} samples)")
        print(f"{'='*50}")
        print(f"  Faithfulness      : {agg['avg_faithfulness']:.3f}")
        print(f"  Answer Relevance  : {agg['avg_answer_relevance']:.3f}")
        print(f"  Context Precision : {agg['avg_context_precision']:.3f}")
        print(f"  Context Recall    : {agg['avg_context_recall']:.3f}")
        print(f"  Avg Latency       : {agg['avg_latency_ms']:.1f} ms")
        print(f"  Overall Score     : {agg['avg_overall_score']:.3f}")
        print(f"{'='*50}\n")

        return agg

    def save_results(self, results: Dict, path: str = "eval_results.json") -> None:
        with open(path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"[Evaluator] Results saved to '{path}'")


# ── Quick self-test ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    evaluator = RAGEvaluator()

    # Simulate a sample
    sample = EvalSample(
        question="What is RAG?",
        reference_answer=(
            "RAG is Retrieval-Augmented Generation, a technique that combines "
            "retrieval of relevant documents with language model generation."
        ),
        contexts=[
            "Retrieval-Augmented Generation (RAG) combines retrieval systems "
            "with LLMs to produce accurate, grounded answers."
        ],
    )
    predicted = (
        "RAG stands for Retrieval-Augmented Generation. It enhances LLM "
        "responses by first retrieving relevant document chunks."
    )

    metrics = evaluator.evaluate_sample(
        sample=sample,
        predicted_answer=predicted,
        retrieved_contexts=sample.contexts,
        latency_ms=42.5,
    )
    print("Self-test metrics:", json.dumps(metrics.to_dict(), indent=2))
