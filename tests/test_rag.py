"""
Unit Tests — RAG Document Q&A
Author: Aman Mishra
"""

import unittest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.rag_pipeline import TextProcessor, DocumentChunk, RAGResponse, RetrievalResult
from src.evaluator import RAGEvaluator, EvalSample


# ── TextProcessor ──────────────────────────────────────────────────────────────

class TestTextProcessor(unittest.TestCase):

    def setUp(self):
        self.processor = TextProcessor(chunk_size=10, overlap=2)

    def test_chunk_text_basic(self):
        text = " ".join([f"word{i}" for i in range(25)])
        chunks = self.processor.chunk_text(text, source="test.txt")
        self.assertGreater(len(chunks), 1)
        self.assertIsInstance(chunks[0], DocumentChunk)

    def test_chunk_ids_are_sequential(self):
        text = " ".join(["word"] * 30)
        chunks = self.processor.chunk_text(text, source="test.txt", start_id=5)
        ids = [c.chunk_id for c in chunks]
        self.assertEqual(ids, list(range(5, 5 + len(chunks))))

    def test_chunk_preserves_source(self):
        chunks = self.processor.chunk_text("Hello world test", source="my_doc.pdf")
        for c in chunks:
            self.assertEqual(c.source, "my_doc.pdf")

    def test_chunk_page_number_propagated(self):
        chunks = self.processor.chunk_text("Some text here", source="doc.pdf",
                                            page_number=3)
        for c in chunks:
            self.assertEqual(c.page_number, 3)

    def test_overlap_creates_more_chunks(self):
        text = " ".join(["word"] * 30)
        p_no_overlap = TextProcessor(chunk_size=10, overlap=0)
        p_with_overlap = TextProcessor(chunk_size=10, overlap=5)
        chunks_no = p_no_overlap.chunk_text(text, source="x.txt")
        chunks_with = p_with_overlap.chunk_text(text, source="x.txt")
        self.assertGreater(len(chunks_with), len(chunks_no))

    def test_unsupported_extension_raises(self):
        with self.assertRaises(ValueError):
            self.processor.process_document("file.docx")

    def test_document_chunk_to_dict(self):
        chunk = DocumentChunk(text="hello", source="src.txt",
                               chunk_id=0, page_number=1)
        d = chunk.to_dict()
        self.assertEqual(d["text"], "hello")
        self.assertEqual(d["source"], "src.txt")
        self.assertNotIn("embedding", d)


# ── RAGResponse ────────────────────────────────────────────────────────────────

class TestRAGResponse(unittest.TestCase):

    def _make_response(self):
        chunk = DocumentChunk(text="context text", source="doc.txt", chunk_id=0)
        result = RetrievalResult(chunk=chunk, score=0.88, rank=1)
        return RAGResponse(
            question="What is AI?",
            answer="AI is artificial intelligence.",
            retrieved_chunks=[result],
            latency_ms=55.0,
            tokens_used=120,
        )

    def test_to_dict_keys(self):
        response = self._make_response()
        d = response.to_dict()
        for key in ("question", "answer", "sources", "latency_ms"):
            self.assertIn(key, d)

    def test_sources_contain_rank_and_score(self):
        d = self._make_response().to_dict()
        src = d["sources"][0]
        self.assertEqual(src["rank"], 1)
        self.assertAlmostEqual(src["score"], 0.88, places=2)

    def test_excerpt_truncated(self):
        long_text = "x " * 300
        chunk = DocumentChunk(text=long_text, source="big.txt", chunk_id=0)
        result = RetrievalResult(chunk=chunk, score=0.5, rank=1)
        response = RAGResponse(
            question="q", answer="a",
            retrieved_chunks=[result], latency_ms=10.0
        )
        d = response.to_dict()
        self.assertLessEqual(len(d["sources"][0]["excerpt"]), 210)


# ── RAGEvaluator ──────────────────────────────────────────────────────────────

class TestRAGEvaluator(unittest.TestCase):

    def setUp(self):
        self.evaluator = RAGEvaluator()
        self.reference = "Retrieval-Augmented Generation combines retrieval with LLMs."
        self.contexts = ["RAG combines retrieval systems with large language models."]

    def test_faithfulness_high_for_related_answer(self):
        answer = "RAG combines retrieval with large language models for better answers."
        score = self.evaluator.compute_faithfulness(answer, self.contexts)
        self.assertGreater(score, 0.3)

    def test_faithfulness_low_for_unrelated_answer(self):
        answer = "Bananas are yellow fruits grown in tropical regions."
        score = self.evaluator.compute_faithfulness(answer, self.contexts)
        self.assertLess(score, 0.2)

    def test_faithfulness_empty_context(self):
        score = self.evaluator.compute_faithfulness("Any answer", [])
        self.assertEqual(score, 0.0)

    def test_answer_relevance_perfect_match(self):
        answer = self.reference
        score = self.evaluator.compute_answer_relevance(
            answer, "What is RAG?", self.reference)
        self.assertGreater(score, 0.5)  # reference match should score well

    def test_context_precision_relevant_context(self):
        question = "What is RAG retrieval augmented generation?"
        contexts = ["RAG combines retrieval systems with large language models for generation."]
        precision = self.evaluator.compute_context_precision(question, contexts)
        self.assertGreater(precision, 0.0)

    def test_context_recall_good_coverage(self):
        recall = self.evaluator.compute_context_recall(
            self.reference, self.contexts)
        self.assertGreater(recall, 0.2)

    def test_overall_score_in_range(self):
        sample = EvalSample(
            question="What is RAG?",
            reference_answer=self.reference,
        )
        from src.evaluator import EvalMetrics
        metrics = EvalMetrics(
            faithfulness=0.8,
            answer_relevance=0.75,
            context_precision=0.9,
            context_recall=0.7,
            latency_ms=30.0,
        )
        score = metrics.overall_score()
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)

    def test_evaluate_sample_returns_metrics(self):
        sample = EvalSample(
            question="What is RAG?",
            reference_answer=self.reference,
        )
        from src.evaluator import EvalMetrics
        metrics = self.evaluator.evaluate_sample(
            sample=sample,
            predicted_answer="RAG is retrieval-augmented generation.",
            retrieved_contexts=self.contexts,
            latency_ms=20.0,
        )
        self.assertIsInstance(metrics, EvalMetrics)
        d = metrics.to_dict()
        self.assertIn("overall_score", d)

    def test_token_overlap_symmetric(self):
        a, b = "the cat sat on the mat", "the mat sat on the cat"
        s1 = self.evaluator._token_overlap(a, b)
        s2 = self.evaluator._token_overlap(b, a)
        self.assertAlmostEqual(s1, s2, places=5)

    def test_token_overlap_identical(self):
        s = self.evaluator._token_overlap("hello world", "hello world")
        self.assertAlmostEqual(s, 1.0, places=5)

    def test_token_overlap_disjoint(self):
        s = self.evaluator._token_overlap("cat dog", "fish bird")
        self.assertAlmostEqual(s, 0.0, places=5)


if __name__ == "__main__":
    unittest.main(verbosity=2)
