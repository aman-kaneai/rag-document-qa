"""
RAG (Retrieval-Augmented Generation) Pipeline
Author: Aman Mishra
Description: End-to-end RAG system for intelligent document Q&A using LLMs.
"""

import os
import json
import time
import hashlib
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field

import numpy as np

# ── Optional heavy imports (gracefully skipped in demo mode) ──────────────────
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer
    ST_AVAILABLE = True
except ImportError:
    ST_AVAILABLE = False

try:
    import PyPDF2
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class DocumentChunk:
    """Represents a chunk of text with metadata."""
    text: str
    source: str
    chunk_id: int
    page_number: Optional[int] = None
    embedding: Optional[np.ndarray] = None

    def to_dict(self) -> Dict:
        return {
            "text": self.text,
            "source": self.source,
            "chunk_id": self.chunk_id,
            "page_number": self.page_number,
        }


@dataclass
class RetrievalResult:
    """Holds retrieval results with relevance scores."""
    chunk: DocumentChunk
    score: float
    rank: int


@dataclass
class RAGResponse:
    """Full RAG response with context and answer."""
    question: str
    answer: str
    retrieved_chunks: List[RetrievalResult]
    latency_ms: float
    tokens_used: Optional[int] = None

    def to_dict(self) -> Dict:
        return {
            "question": self.question,
            "answer": self.answer,
            "sources": [
                {
                    "rank": r.rank,
                    "score": round(r.score, 4),
                    "source": r.chunk.source,
                    "excerpt": r.chunk.text[:200] + "...",
                }
                for r in self.retrieved_chunks
            ],
            "latency_ms": round(self.latency_ms, 2),
        }


# ── Text processing ───────────────────────────────────────────────────────────

class TextProcessor:
    """Handles document ingestion and chunking."""

    def __init__(self, chunk_size: int = 512, overlap: int = 64):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def load_text_file(self, path: str) -> str:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()

    def load_pdf(self, path: str) -> List[Tuple[str, int]]:
        """Returns list of (text, page_number) tuples."""
        if not PDF_AVAILABLE:
            raise ImportError("PyPDF2 not installed. Run: pip install PyPDF2")
        pages = []
        with open(path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            for i, page in enumerate(reader.pages):
                text = page.extract_text() or ""
                pages.append((text.strip(), i + 1))
        return pages

    def chunk_text(self, text: str, source: str,
                   start_id: int = 0,
                   page_number: Optional[int] = None) -> List[DocumentChunk]:
        """Splits text into overlapping chunks."""
        words = text.split()
        chunks: List[DocumentChunk] = []
        i = 0
        chunk_id = start_id

        while i < len(words):
            window = words[i: i + self.chunk_size]
            chunk_text = " ".join(window)
            chunks.append(DocumentChunk(
                text=chunk_text,
                source=source,
                chunk_id=chunk_id,
                page_number=page_number,
            ))
            chunk_id += 1
            i += self.chunk_size - self.overlap  # sliding window

        return chunks

    def process_document(self, path: str) -> List[DocumentChunk]:
        """Auto-detects file type and returns chunks."""
        ext = Path(path).suffix.lower()
        source = Path(path).name
        chunks: List[DocumentChunk] = []

        if ext == ".pdf":
            for text, page_num in self.load_pdf(path):
                chunks.extend(self.chunk_text(text, source,
                                               start_id=len(chunks),
                                               page_number=page_num))
        elif ext in {".txt", ".md"}:
            text = self.load_text_file(path)
            chunks.extend(self.chunk_text(text, source))
        else:
            raise ValueError(f"Unsupported file type: {ext}")

        print(f"[TextProcessor] '{source}' → {len(chunks)} chunks")
        return chunks


# ── Vector store ──────────────────────────────────────────────────────────────

class VectorStore:
    """FAISS-backed vector store with cosine similarity retrieval."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        if not ST_AVAILABLE:
            raise ImportError(
                "sentence-transformers not installed.\n"
                "Run: pip install sentence-transformers"
            )
        if not FAISS_AVAILABLE:
            raise ImportError(
                "faiss-cpu not installed.\n"
                "Run: pip install faiss-cpu"
            )
        print(f"[VectorStore] Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.dim = self.model.get_sentence_embedding_dimension()
        self.index = faiss.IndexFlatIP(self.dim)   # inner-product ≈ cosine on L2-normed vecs
        self.chunks: List[DocumentChunk] = []

    def _normalize(self, vecs: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-10
        return vecs / norms

    def add_chunks(self, chunks: List[DocumentChunk]) -> None:
        texts = [c.text for c in chunks]
        embeddings = self.model.encode(texts, batch_size=32,
                                        show_progress_bar=True,
                                        convert_to_numpy=True)
        embeddings = self._normalize(embeddings.astype("float32"))

        for chunk, emb in zip(chunks, embeddings):
            chunk.embedding = emb

        self.index.add(embeddings)
        self.chunks.extend(chunks)
        print(f"[VectorStore] Indexed {len(chunks)} chunks "
              f"(total: {len(self.chunks)})")

    def search(self, query: str, top_k: int = 5) -> List[RetrievalResult]:
        q_emb = self.model.encode([query], convert_to_numpy=True)
        q_emb = self._normalize(q_emb.astype("float32"))

        scores, indices = self.index.search(q_emb, top_k)

        results = []
        for rank, (score, idx) in enumerate(zip(scores[0], indices[0]), start=1):
            if idx < 0:
                continue
            results.append(RetrievalResult(
                chunk=self.chunks[idx],
                score=float(score),
                rank=rank,
            ))
        return results

    def save(self, directory: str) -> None:
        Path(directory).mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, f"{directory}/index.faiss")
        meta = [c.to_dict() for c in self.chunks]
        with open(f"{directory}/chunks.json", "w") as f:
            json.dump(meta, f, indent=2)
        print(f"[VectorStore] Saved to '{directory}'")

    def load(self, directory: str) -> None:
        self.index = faiss.read_index(f"{directory}/index.faiss")
        with open(f"{directory}/chunks.json") as f:
            meta = json.load(f)
        self.chunks = [DocumentChunk(**m) for m in meta]
        print(f"[VectorStore] Loaded {len(self.chunks)} chunks from '{directory}'")


# ── LLM interface ─────────────────────────────────────────────────────────────

class LLMClient:
    """Wraps OpenAI-compatible chat completions."""

    SYSTEM_PROMPT = (
        "You are a precise, helpful document assistant. "
        "Answer questions using ONLY the provided context. "
        "If the answer is not in the context, say so clearly. "
        "Always cite your sources."
    )

    def __init__(self, model: str = "gpt-4o-mini", temperature: float = 0.2):
        if not OPENAI_AVAILABLE:
            raise ImportError("openai not installed. Run: pip install openai")
        self.client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY", ""))
        self.model = model
        self.temperature = temperature

    def build_prompt(self, question: str,
                     retrieved: List[RetrievalResult]) -> str:
        context_blocks = []
        for r in retrieved:
            header = f"[Source: {r.chunk.source}"
            if r.chunk.page_number:
                header += f", Page {r.chunk.page_number}"
            header += f", Relevance: {r.score:.2f}]"
            context_blocks.append(f"{header}\n{r.chunk.text}")

        context = "\n\n---\n\n".join(context_blocks)
        return (
            f"Context:\n{context}\n\n"
            f"Question: {question}\n\n"
            f"Answer (cite sources):"
        )

    def generate(self, question: str,
                 retrieved: List[RetrievalResult]) -> Tuple[str, Optional[int]]:
        user_prompt = self.build_prompt(question, retrieved)
        response = self.client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            messages=[
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
        )
        answer = response.choices[0].message.content.strip()
        tokens = response.usage.total_tokens if response.usage else None
        return answer, tokens


# ── RAG pipeline ──────────────────────────────────────────────────────────────

class RAGPipeline:
    """
    Full RAG pipeline: ingest → embed → retrieve → generate.

    Usage
    -----
    pipeline = RAGPipeline()
    pipeline.ingest(["docs/report.pdf", "docs/notes.txt"])
    response = pipeline.query("What are the key findings?")
    print(response.answer)
    """

    def __init__(self,
                 embedding_model: str = "all-MiniLM-L6-v2",
                 llm_model: str = "gpt-4o-mini",
                 chunk_size: int = 512,
                 chunk_overlap: int = 64,
                 top_k: int = 5):
        self.processor = TextProcessor(chunk_size=chunk_size,
                                        overlap=chunk_overlap)
        self.vector_store = VectorStore(model_name=embedding_model)
        self.llm = LLMClient(model=llm_model)
        self.top_k = top_k
        self._query_cache: Dict[str, RAGResponse] = {}

    def ingest(self, file_paths: List[str]) -> None:
        """Ingest and index documents."""
        all_chunks: List[DocumentChunk] = []
        for path in file_paths:
            chunks = self.processor.process_document(path)
            all_chunks.extend(chunks)
        self.vector_store.add_chunks(all_chunks)

    def query(self, question: str,
              use_cache: bool = True) -> RAGResponse:
        """Run a full RAG query."""
        cache_key = hashlib.md5(question.encode()).hexdigest()
        if use_cache and cache_key in self._query_cache:
            print("[RAGPipeline] Cache hit")
            return self._query_cache[cache_key]

        t0 = time.perf_counter()

        # Retrieve
        retrieved = self.vector_store.search(question, top_k=self.top_k)

        # Generate
        answer, tokens = self.llm.generate(question, retrieved)

        latency_ms = (time.perf_counter() - t0) * 1000

        response = RAGResponse(
            question=question,
            answer=answer,
            retrieved_chunks=retrieved,
            latency_ms=latency_ms,
            tokens_used=tokens,
        )
        self._query_cache[cache_key] = response
        return response

    def save_index(self, path: str = "vector_store") -> None:
        self.vector_store.save(path)

    def load_index(self, path: str = "vector_store") -> None:
        self.vector_store.load(path)
