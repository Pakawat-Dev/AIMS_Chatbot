"""Agentic ISO/IEC 42001 AIMS Chatbot with enhanced search and token optimization.

This module provides a minimal, auditable Agentic RAG AIMS chatbot for
ISO/IEC 42001 (AI Management System) with token usage optimization and tracking.

"""

import argparse
import json
import os
import pickle
import re
import tiktoken
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    Docx2txtLoader,
    PyPDFLoader,
    TextLoader,
    UnstructuredHTMLLoader,
)
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langgraph.graph import END, StateGraph
from pydantic import BaseModel, Field

try:
    import numpy as np
    from rank_bm25 import BM25Okapi
    from sentence_transformers import CrossEncoder
    from sklearn.feature_extraction.text import TfidfVectorizer
    
    _HAS_RERANKER = True
    _HAS_HYBRID = True
except ImportError:
    CrossEncoder = None
    BM25Okapi = None
    TfidfVectorizer = None
    np = None
    _HAS_RERANKER = False
    _HAS_HYBRID = False


# -------------------- Configuration --------------------
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-large")
CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-4o-mini")
FAISS_DIR = os.getenv("FAISS_DIR", "./faiss_iso42001")
KB_FOLDER = os.getenv("KB_FOLDER", r"D:\\AIMS Standard")
TOP_K = int(os.getenv("TOP_K", "12"))
RERANK_KEEP = int(os.getenv("RERANK_KEEP", "6"))
MAX_LOOPS = int(os.getenv("MAX_LOOPS", "2"))
TOKEN_BUDGET = int(os.getenv("TOKEN_BUDGET", "15000"))
MAX_EVIDENCE_CHARS = int(os.getenv("MAX_EVIDENCE_CHARS", "2000"))

# Enhanced search configuration
SEMANTIC_WEIGHT = float(os.getenv("SEMANTIC_WEIGHT", "0.7"))
BM25_WEIGHT = float(os.getenv("BM25_WEIGHT", "0.3"))
MMR_DIVERSITY = float(os.getenv("MMR_DIVERSITY", "0.3"))
QUERY_EXPANSION = os.getenv("QUERY_EXPANSION", "true").lower() == "true"


# -------------------- Token Tracking --------------------
@dataclass
class TokenUsage:
    """Track token usage across conversations."""
    
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    
    def add(self, usage_dict: Dict[str, int]) -> None:
        """Add token usage from OpenAI response."""
        self.prompt_tokens += usage_dict.get("prompt_tokens", 0)
        self.completion_tokens += usage_dict.get("completion_tokens", 0)
        self.total_tokens += usage_dict.get("total_tokens", 0)
    
    def __str__(self) -> str:
        """Return formatted token usage string."""
        return (
            f"Tokens: {self.total_tokens} "
            f"(prompt: {self.prompt_tokens}, "
            f"completion: {self.completion_tokens})"
        )


class TokenTracker:
    """Global token tracker with budget management."""
    
    def __init__(self, budget: int = TOKEN_BUDGET):
        """Initialize token tracker with budget.
        
        Parameters
        ----------
        budget : int
            Maximum number of tokens allowed for the session.
        """
        self.budget = budget
        self.session_usage = TokenUsage()
        self.query_usage = TokenUsage()
        self.encoding = tiktoken.encoding_for_model("gpt-4")
    
    def reset_query(self) -> None:
        """Reset query-level tracking."""
        self.query_usage = TokenUsage()
    
    def add_usage(self, usage_dict: Dict[str, int]) -> None:
        """Add token usage to both session and query trackers.
        
        Parameters
        ----------
        usage_dict : Dict[str, int]
            Dictionary containing token usage from OpenAI API response.
        """
        self.session_usage.add(usage_dict)
        self.query_usage.add(usage_dict)
    
    def estimate_tokens(self, text: str) -> int:
        """Estimate token count for text.
        
        Parameters
        ----------
        text : str
            Text to estimate tokens for.
            
        Returns
        -------
        int
            Estimated number of tokens.
        """
        return len(self.encoding.encode(text))
    
    def check_budget(self, estimated_tokens: int = 0) -> bool:
        """Check if we're within budget.
        
        Parameters
        ----------
        estimated_tokens : int, optional
            Additional tokens to consider for budget check.
            
        Returns
        -------
        bool
            True if within budget, False otherwise.
        """
        return (
            self.session_usage.total_tokens + estimated_tokens
        ) <= self.budget
    
    def budget_remaining(self) -> int:
        """Get remaining token budget.
        
        Returns
        -------
        int
            Number of tokens remaining in budget.
        """
        return max(0, self.budget - self.session_usage.total_tokens)
    
    def get_status(self) -> str:
        """Get formatted status string.
        
        Returns
        -------
        str
            Formatted string showing budget status.
        """
        remaining = self.budget_remaining()
        pct_used = (self.session_usage.total_tokens / self.budget) * 100
        return f"Budget: {remaining}/{self.budget} remaining ({pct_used:.1f}% used)"


# Global tracker instance
token_tracker = TokenTracker()


def get_reranker():
    """Lazy load reranker to avoid SSL issues at startup."""
    global reranker
    if reranker is None and _HAS_RERANKER:
        try:
            reranker = CrossEncoder("BAAI/bge-reranker-v2-m3")
        except Exception:
            pass  # Silently fail if reranker can't be loaded
    return reranker


# -------------------- State Management --------------------
class ISO42001ChatState(BaseModel):
    """Mutable state passed between LangGraph nodes."""
    
    user_question: str
    subqueries: List[str] = Field(default_factory=list)
    docs: List[Dict[str, Any]] = Field(default_factory=list)
    loops: int = 0
    final: Dict[str, Any] = Field(default_factory=dict)


# -------------------- Enhanced Search Classes --------------------
class HybridSearchEngine:
    """Combine semantic (FAISS) and keyword (BM25) search for better retrieval."""
    
    def __init__(self, faiss_db: FAISS, documents: List[Dict[str, Any]]):
        """Initialize hybrid search engine.
        
        Parameters
        ----------
        faiss_db : FAISS
            FAISS vector database for semantic search.
        documents : List[Dict[str, Any]]
            List of documents for BM25 indexing.
        """
        self.faiss_db = faiss_db
        self.documents = documents
        self.bm25 = None
        self.doc_texts = []
        
        if _HAS_HYBRID and documents:
            self._build_bm25_index()
    
    def _build_bm25_index(self):
        """Build BM25 index from documents."""
        try:
            self.doc_texts = [doc['content'] for doc in self.documents]
            # Tokenize for BM25
            tokenized_docs = [doc.lower().split() for doc in self.doc_texts]
            self.bm25 = BM25Okapi(tokenized_docs)
        except Exception:
            self.bm25 = None
    
    def hybrid_search(
        self, 
        query: str, 
        k: int = TOP_K
    ) -> List[Dict[str, Any]]:
        """Perform hybrid semantic + keyword search.
        
        Parameters
        ----------
        query : str
            Search query string.
        k : int, optional
            Number of results to return.
            
        Returns
        -------
        List[Dict[str, Any]]
            List of search results with scores.
        """
        results = []
        
        # 1. Semantic search via FAISS
        semantic_docs = self.faiss_db.similarity_search_with_score(
            query, k=k*2
        )
        semantic_results = []
        for doc, score in semantic_docs:
            semantic_results.append({
                'content': doc.page_content,
                'metadata': doc.metadata,
                'semantic_score': 1.0 / (1.0 + score),
                'bm25_score': 0.0,
                'hybrid_score': 0.0
            })
        
        # 2. Keyword search via BM25 (if available)
        bm25_results = []
        if self.bm25:
            try:
                query_tokens = query.lower().split()
                bm25_scores = self.bm25.get_scores(query_tokens)
                
                # Get top BM25 results
                top_bm25_indices = np.argsort(bm25_scores)[::-1][:k*2]
                
                for idx in top_bm25_indices:
                    if idx < len(self.documents):
                        doc = self.documents[idx]
                        bm25_results.append({
                            'content': doc['content'],
                            'metadata': doc['metadata'],
                            'semantic_score': 0.0,
                            'bm25_score': float(bm25_scores[idx]),
                            'hybrid_score': 0.0
                        })
            except Exception:
                pass
        
        # 3. Combine and deduplicate results
        combined_docs = {}
        
        # Add semantic results
        for doc in semantic_results:
            key = doc['content'][:100]  # Use content prefix as key
            if key not in combined_docs:
                combined_docs[key] = doc
            else:
                combined_docs[key]['semantic_score'] = max(
                    combined_docs[key]['semantic_score'], 
                    doc['semantic_score']
                )
        
        # Add/update BM25 results
        for doc in bm25_results:
            key = doc['content'][:100]
            if key not in combined_docs:
                combined_docs[key] = doc
            else:
                combined_docs[key]['bm25_score'] = max(
                    combined_docs[key]['bm25_score'], 
                    doc['bm25_score']
                )
        
        # 4. Calculate hybrid scores and rank
        max_bm25_score = max(
            [d['bm25_score'] for d in combined_docs.values()], 
            default=1.0
        )
        
        for doc in combined_docs.values():
            # Normalize scores
            semantic_norm = doc['semantic_score']
            bm25_norm = doc['bm25_score'] / max(1.0, max_bm25_score)
            
            # Weighted combination
            doc['hybrid_score'] = (
                SEMANTIC_WEIGHT * semantic_norm + 
                BM25_WEIGHT * bm25_norm
            )
            results.append(doc)
        
        # Sort by hybrid score and return top k
        results.sort(key=lambda x: x['hybrid_score'], reverse=True)
        return results[:k]


class QueryEnhancer:
    """Enhance queries for better retrieval performance."""
    
    def __init__(self, llm: ChatOpenAI):
        """Initialize query enhancer.
        
        Parameters
        ----------
        llm : ChatOpenAI
            Language model for query enhancement.
        """
        self.llm = llm
        self.iso_terms = {
            "risk": ["risk management", "risk assessment", "risk treatment"],
            "control": ["controls", "control measures", "control objectives"],
            "audit": ["auditing", "audit process", "audit criteria"],
            "management": ["management system", "management processes"],
            "AI": ["artificial intelligence", "AI system", "AI model"],
            "data": ["data management", "data governance", "data quality"],
            "performance": ["performance evaluation", "performance monitoring"],
            "improvement": ["continual improvement", "corrective action"],
        }
    
    def expand_query(self, query: str) -> List[str]:
        """Expand query with ISO/IEC 42001 specific terms.
        
        Parameters
        ----------
        query : str
            Original query string.
            
        Returns
        -------
        List[str]
            List of expanded query variants.
        """
        if not QUERY_EXPANSION:
            return [query]
        
        expanded = [query]
        query_lower = query.lower()
        
        # Add domain-specific expansions
        for term, expansions in self.iso_terms.items():
            if term in query_lower:
                for expansion in expansions:
                    if expansion not in query_lower:
                        expanded.append(f"{query} {expansion}")
        
        # Add clause-specific variants
        clause_words = ["clause", "section", "requirement"]
        if any(word in query_lower for word in clause_words):
            expanded.append(
                query.replace("clause", "section").replace(
                    "section", "requirement"
                )
            )
        
        return expanded[:3]  # Limit to 3 variants
    
    def enhance_query_with_context(
        self, 
        query: str, 
        previous_docs: Optional[List[Dict]] = None
    ) -> str:
        """Enhance query using context from previous results.
        
        Parameters
        ----------
        query : str
            Original query string.
        previous_docs : Optional[List[Dict]], optional
            Previous relevant documents for context.
            
        Returns
        -------
        str
            Enhanced query string.
        """
        if not previous_docs:
            return query
        
        # Extract key terms from previous relevant documents
        context_terms = set()
        for doc in previous_docs[:2]:  # Use top 2 docs for context
            content = doc.get('content', '').lower()
            # Extract potential key terms (simple approach)
            words = content.split()
            excluded_words = {
                'that', 'with', 'this', 'shall', 'should'
            }
            for word in words:
                if len(word) > 4 and word not in excluded_words:
                    context_terms.add(word)
        
        if context_terms:
            # Add most relevant context terms
            relevant_terms = list(context_terms)[:3]
            enhanced_query = f"{query} {' '.join(relevant_terms)}"
            return enhanced_query
        
        return query


def mmr_diversify(
    documents: List[Dict[str, Any]], 
    query_embedding, 
    lambda_param: float = MMR_DIVERSITY, 
    top_k: Optional[int] = None
) -> List[Dict[str, Any]]:
    """Apply Maximal Marginal Relevance for diverse results.
    
    Parameters
    ----------
    documents : List[Dict[str, Any]]
        List of candidate documents.
    query_embedding
        Query embedding (not used in current implementation).
    lambda_param : float, optional
        Parameter balancing relevance vs diversity.
    top_k : Optional[int], optional
        Number of results to return.
        
    Returns
    -------
    List[Dict[str, Any]]
        Diversified list of documents.
    """
    if not documents or not _HAS_HYBRID:
        return documents[:top_k] if top_k else documents
    
    try:
        # This is a simplified MMR - in practice you'd need actual embeddings
        # For now, we'll use a heuristic based on content similarity
        selected = []
        remaining = documents.copy()
        
        if top_k is None:
            top_k = len(documents)
        
        # Select first document (highest score)
        if remaining:
            selected.append(remaining.pop(0))
        
        # Iteratively select documents balancing relevance and diversity
        while len(selected) < top_k and remaining:
            best_score = -1
            best_doc = None
            best_idx = -1
            
            for i, doc in enumerate(remaining):
                # Relevance score (from hybrid search)
                relevance = doc.get(
                    'hybrid_score', doc.get('semantic_score', 0)
                )
                
                # Diversity score (simple content overlap check)
                diversity = 1.0
                for selected_doc in selected:
                    # Simple Jaccard similarity on words
                    words1 = set(doc['content'].lower().split())
                    words2 = set(selected_doc['content'].lower().split())
                    if words1 and words2:
                        overlap = (
                            len(words1.intersection(words2)) / 
                            len(words1.union(words2))
                        )
                        diversity = min(diversity, 1.0 - overlap)
                
                # MMR score
                mmr_score = (
                    lambda_param * relevance + 
                    (1 - lambda_param) * diversity
                )
                
                if mmr_score > best_score:
                    best_score = mmr_score
                    best_doc = doc
                    best_idx = i
            
            if best_doc:
                selected.append(remaining.pop(best_idx))
            else:
                break
        
        return selected
    
    except Exception:
        return documents[:top_k] if top_k else documents


# -------------------- Helper Functions --------------------
def iso42001_basename(path: str) -> str:
    """Return the filename with normalized separators.
    
    Parameters
    ----------
    path : str
        File path string.
        
    Returns
    -------
    str
        Basename with normalized separators.
    """
    return os.path.basename(path).replace("\\", "/")


def smart_truncate_evidence(
    docs: List[Dict[str, Any]], 
    max_chars: int = MAX_EVIDENCE_CHARS
) -> str:
    """Intelligently truncate evidence to fit within character limit.
    
    Parameters
    ----------
    docs : List[Dict[str, Any]]
        List of document dictionaries.
    max_chars : int, optional
        Maximum character limit for evidence.
        
    Returns
    -------
    str
        Truncated evidence string.
    """
    evidence_blocks = []
    current_chars = 0
    
    for i, d in enumerate(docs):
        meta = d.get("metadata", {})
        src = meta.get("source", "unknown")
        clause = meta.get("clause", "")
        
        # Create header
        header = f"[{i}] Source: {src}"
        if clause:
            header += f", Clause: {clause}"
        header += "\n"
        
        # Calculate available space
        header_chars = len(header)
        remaining_chars = max_chars - current_chars - header_chars
        
        if remaining_chars <= 100:  # Not enough space for meaningful content
            break
            
        # Truncate content smartly - prefer complete sentences
        content = d['content']
        if len(content) <= remaining_chars:
            truncated_content = content
        else:
            # Find last complete sentence within limit
            truncated = content[:remaining_chars]
            last_period = truncated.rfind('.')
            if last_period > remaining_chars * 0.7:
                truncated_content = truncated[:last_period + 1]
            else:
                truncated_content = truncated + "..."
        
        block = header + truncated_content
        evidence_blocks.append(block)
        current_chars += len(block) + 10  # Add separator chars
        
        if current_chars >= max_chars:
            break
    
    return "\n---\n".join(evidence_blocks)


def call_llm_with_tracking(prompt: str, model: ChatOpenAI) -> str:
    """Call LLM and track token usage.
    
    Parameters
    ----------
    prompt : str
        Prompt string to send to LLM.
    model : ChatOpenAI
        OpenAI model instance.
        
    Returns
    -------
    str
        LLM response content.
    """
    # Estimate input tokens
    estimated_input = token_tracker.estimate_tokens(prompt)
    
    if not token_tracker.check_budget(estimated_input + 500):
        return "ERROR: Token budget exceeded. Please start a new session."
    
    try:
        response = model.invoke(prompt)
        
        # Track actual usage if available
        if (hasattr(response, 'response_metadata') and 
            'token_usage' in response.response_metadata):
            token_tracker.add_usage(response.response_metadata['token_usage'])
        else:
            # Fallback estimation
            estimated_output = token_tracker.estimate_tokens(response.content)
            estimated_usage = {
                "prompt_tokens": estimated_input,
                "completion_tokens": estimated_output,
                "total_tokens": estimated_input + estimated_output
            }
            token_tracker.add_usage(estimated_usage)
        
        return response.content
    except Exception:
        return "ERROR: Failed to generate response."


def iso42001_load_docs(folder: str) -> List[Any]:
    """Recursively load documents from a folder.
    
    Parameters
    ----------
    folder : str
        Root folder path to scan.
        
    Returns
    -------
    List[Any]
        List of LangChain Document objects.
    """
    docs = []
    for root, _, files in os.walk(folder):
        for fn in files:
            p = os.path.join(root, fn)
            lower = fn.lower()
            try:
                if lower.endswith(".pdf"):
                    loader = PyPDFLoader(p)
                elif lower.endswith((".docx", ".doc")):
                    loader = Docx2txtLoader(p)
                elif lower.endswith((".html", ".htm")):
                    loader = UnstructuredHTMLLoader(p)
                elif lower.endswith((".txt", ".md")):
                    loader = TextLoader(p, encoding="utf-8")
                else:
                    continue
                    
                file_docs = loader.load()
                for d in file_docs:
                    d.metadata = d.metadata or {}
                    d.metadata.update({
                        "source": iso42001_basename(p),
                        "path_hidden": True,
                        "standard": "ISO/IEC 42001",
                    })
                docs.extend(file_docs)
            except Exception:
                continue  # Silently skip problematic files
    return docs


def iso42001_detect_content_type(content: str) -> str:
    """Detect the type of content for better categorization.
    
    Parameters
    ----------
    content : str
        Document content to analyze.
        
    Returns
    -------
    str
        Content type classification.
    """
    content_lower = content.lower()
    
    definition_words = ['definition', 'term', 'means', 'refers to']
    requirement_words = ['shall', 'must', 'required']
    guidance_words = ['example', 'note', 'guidance']
    audit_words = ['audit', 'assessment', 'review']
    risk_words = ['risk', 'threat', 'vulnerability']
    process_words = ['process', 'procedure', 'activity']
    
    if any(word in content_lower for word in definition_words):
        return 'definition'
    elif any(word in content_lower for word in requirement_words):
        return 'requirement'
    elif any(word in content_lower for word in guidance_words):
        return 'guidance'
    elif any(word in content_lower for word in audit_words):
        return 'audit'
    elif any(word in content_lower for word in risk_words):
        return 'risk'
    elif any(word in content_lower for word in process_words):
        return 'process'
    else:
        return 'general'


def iso42001_chunk_docs(raw_docs: List[Any]) -> List[Any]:
    """Split raw documents into overlapping chunks with enhanced metadata.
    
    Parameters
    ----------
    raw_docs : List[Any]
        Raw documents from loaders.
        
    Returns
    -------
    List[Any]
        Chunked documents with metadata.
    """
    # Enhanced chunking strategy for better retrieval
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,  # Slightly larger for better context
        chunk_overlap=200,  # Increased overlap for better connectivity
        separators=[
            "\n## ",    # Headers
            "\n# ",     # Main headers  
            "\n### ",   # Sub-headers
            "\n- ",     # List items
            "\n\n",     # Paragraphs
            "\n",       # Lines
            ". ",       # Sentences
            " ",        # Words
        ],
        length_function=len,
    )
    
    chunks = []
    for doc in raw_docs:
        doc_chunks = splitter.split_documents([doc])
        
        for i, chunk in enumerate(doc_chunks):
            md = chunk.metadata or {}
            
            # Enhanced metadata extraction
            content = chunk.page_content
            
            # Extract potential clause/section references
            clause_patterns = [
                r'clause\s+(\d+(?:\.\d+)*)',
                r'section\s+(\d+(?:\.\d+)*)', 
                r'(\d+(?:\.\d+)+)\s+(?:requirements?|controls?)',
                r'requirement\s+(\d+(?:\.\d+)*)'
            ]
            
            clauses = []
            for pattern in clause_patterns:
                matches = re.findall(pattern, content.lower())
                clauses.extend(matches)
            
            # Enhance metadata
            md.update({
                "source": md.get("source") or "unknown",
                "clause": md.get("clause") or md.get("section") or (
                    clauses[0] if clauses else ""
                ),
                "all_clauses": clauses,
                "chunk_index": i,
                "total_chunks": len(doc_chunks),
                "content_type": iso42001_detect_content_type(content),
                "word_count": len(content.split()),
                "has_requirements": bool(
                    re.search(
                        r'\b(?:shall|should|must|required?)\b', 
                        content.lower()
                    )
                ),
                "has_definitions": bool(
                    re.search(
                        r'\b(?:definition|means|refers? to)\b', 
                        content.lower()
                    )
                ),
                "standard": "ISO/IEC 42001",
            })
            
            chunk.metadata = md
            chunks.append(chunk)
    
    return chunks


def iso42001_ingest_faiss(
    chunks: List[Any], 
    emb: OpenAIEmbeddings, 
    persist_dir: str
) -> Tuple[FAISS, List[Dict[str, Any]]]:
    """Build and persist a FAISS index with document metadata for hybrid search.
    
    Parameters
    ----------
    chunks : List[Any]
        Chunked documents to embed and index.
    emb : OpenAIEmbeddings
        Embedding model wrapper.
    persist_dir : str
        Directory where the index will be stored.
        
    Returns
    -------
    Tuple[FAISS, List[Dict[str, Any]]]
        FAISS database and document list for BM25.
    """
    os.makedirs(persist_dir, exist_ok=True)
    
    # Build FAISS index
    db = FAISS.from_documents(chunks, embedding=emb)
    
    # Prepare documents for BM25 indexing
    documents = []
    for chunk in chunks:
        documents.append({
            'content': chunk.page_content,
            'metadata': chunk.metadata
        })
    
    # Save FAISS using native method and documents using pickle
    db.save_local(persist_dir)
    
    with open(os.path.join(persist_dir, "documents.pkl"), "wb") as f:
        pickle.dump(documents, f)
    
    return db, documents


def iso42001_load_faiss(
    emb: OpenAIEmbeddings, 
    persist_dir: str
) -> Tuple[Optional[FAISS], List[Dict[str, Any]]]:
    """Load a persisted FAISS index and documents.
    
    Parameters
    ----------
    emb : OpenAIEmbeddings
        Embedding model wrapper (for consistency).
    persist_dir : str
        Directory where the index is stored.
        
    Returns
    -------
    Tuple[Optional[FAISS], List[Dict[str, Any]]]
        Loaded FAISS database and documents, or None and empty list.
    """
    docs_path = os.path.join(persist_dir, "documents.pkl")
    
    db = None
    documents = []
    
    # Load FAISS using native method
    if os.path.exists(persist_dir) and os.path.exists(os.path.join(persist_dir, "index.faiss")):
        try:
            db = FAISS.load_local(persist_dir, emb, allow_dangerous_deserialization=True)
        except Exception:
            db = None
    
    if os.path.exists(docs_path):
        with open(docs_path, "rb") as f:
            documents = pickle.load(f)
    
    return db, documents


# -------------------- Resource Initialization --------------------
emb = OpenAIEmbeddings(model=EMBED_MODEL, api_key=OPENAI_API_KEY)
llm = ChatOpenAI(model=CHAT_MODEL, temperature=0, api_key=OPENAI_API_KEY)

db, documents = iso42001_load_faiss(emb, FAISS_DIR)
hybrid_search = HybridSearchEngine(db, documents) if db else None
query_enhancer = QueryEnhancer(llm)
reranker = None  # Lazy loaded to avoid SSL issues


# -------------------- Optimized Prompts --------------------
PLANNER_PROMPT = PromptTemplate.from_template(
    "Decompose this ISO/IEC 42001 question into 1-2 focused search queries:\n"
    "{q}\n\nQueries (one per line):"
)

SUFFICIENCY_PROMPT = PromptTemplate.from_template(
    "Question: {q}\nEvidence:\n{snips}\n\n"
    "Is evidence sufficient? JSON: "
    "{{\"sufficient\": true/false, \"refined_query\": \"...\"}}"
)

WRITER_PROMPT = PromptTemplate.from_template(
    "Answer this ISO/IEC 42001 question using only the evidence. "
    "Be concise, cite sources [source:file], map to clauses:\n\n"
    "Q: {q}\nEvidence:\n{ev}\n\n"
    "JSON: {{\"answer_md\": \"...\", \"citations\": [...], \"limits\": \"...\"}}"
)

VERIFIER_PROMPT = PromptTemplate.from_template(
    "Verify this answer against evidence. Remove unsupported claims. "
    "Add disclaimer about not being legal advice:\n\n"
    "Draft: {draft}\nEvidence:\n{ev}\n\n"
    "JSON: {{\"answer_md\": \"...\", \"citations\": [...], \"limits\": \"...\"}}"
)


# -------------------- Node Functions --------------------
def iso42001_planner(state: ISO42001ChatState) -> ISO42001ChatState:
    """Propose up to two subqueries for downstream retrieval.
    
    Parameters
    ----------
    state : ISO42001ChatState
        Current state containing user question.
        
    Returns
    -------
    ISO42001ChatState
        Updated state with subqueries.
    """
    prompt = PLANNER_PROMPT.format(q=state.user_question)
    text = call_llm_with_tracking(prompt, llm)
    if not text.startswith("ERROR"):
        state.subqueries = [
            ln.strip() for ln in text.strip().split("\n") if ln.strip()
        ][:2]
    return state


def iso42001_metadata_filter(
    results: List[Dict[str, Any]], 
    query: str
) -> List[Dict[str, Any]]:
    """Filter and boost results based on metadata relevance.
    
    Parameters
    ----------
    results : List[Dict[str, Any]]
        Search results to filter.
    query : str
        Original query for context.
        
    Returns
    -------
    List[Dict[str, Any]]
        Filtered and boosted results.
    """
    if not results:
        return results
    
    query_lower = query.lower()
    filtered_results = []
    
    for result in results:
        metadata = result.get('metadata', {})
        boost_factor = 1.0
        
        # Content type boosting
        content_type = metadata.get('content_type', 'general')
        if 'definition' in query_lower and content_type == 'definition':
            boost_factor *= 1.5
        elif (any(word in query_lower for word in ['requirement', 'shall', 'must']) 
              and content_type == 'requirement'):
            boost_factor *= 1.4
        elif 'audit' in query_lower and content_type == 'audit':
            boost_factor *= 1.3
        elif 'risk' in query_lower and content_type == 'risk':
            boost_factor *= 1.3
        
        # Clause-specific boosting
        if (any(word in query_lower for word in ['clause', 'section']) 
            and metadata.get('clause')):
            boost_factor *= 1.2
        
        # Requirements vs guidance distinction
        if (metadata.get('has_requirements') and 
            any(word in query_lower for word in ['requirement', 'shall', 'must'])):
            boost_factor *= 1.3
        
        # Apply boost to scores
        if 'hybrid_score' in result:
            result['hybrid_score'] *= boost_factor
        if 'semantic_score' in result:
            result['semantic_score'] *= boost_factor
        
        # Filter out very low relevance results
        min_score = 0.1
        if result.get('hybrid_score', result.get('semantic_score', 0)) >= min_score:
            filtered_results.append(result)
    
    # Re-sort by boosted scores
    filtered_results.sort(
        key=lambda x: x.get('hybrid_score', x.get('semantic_score', 0)), 
        reverse=True
    )
    return filtered_results


def iso42001_retriever(state: ISO42001ChatState) -> ISO42001ChatState:
    """Enhanced retrieval with hybrid search, query expansion, and metadata filtering.
    
    Parameters
    ----------
    state : ISO42001ChatState
        Current state containing queries.
        
    Returns
    -------
    ISO42001ChatState
        Updated state with retrieved documents.
    """
    if not hybrid_search:
        state.docs = []
        return state
    
    results: List[Dict[str, Any]] = []
    queries = state.subqueries or [state.user_question]
    
    # Track previous docs for context enhancement
    previous_docs = getattr(state, 'docs', [])
    
    for q in queries:
        try:
            # 1. Query enhancement and expansion
            enhanced_query = query_enhancer.enhance_query_with_context(
                q, previous_docs
            )
            expanded_queries = query_enhancer.expand_query(enhanced_query)
            
            all_query_results = []
            
            # 2. Search with each query variant
            for query_variant in expanded_queries:
                # Hybrid search (semantic + keyword)
                search_results = hybrid_search.hybrid_search(
                    query_variant, k=TOP_K
                )
                
                # Add query info to results
                for result in search_results:
                    result['query'] = q
                    result['query_variant'] = query_variant
                    all_query_results.append(result)
            
            # 3. Deduplicate and merge results from query variants
            seen_content = set()
            merged_results = []
            
            for result in all_query_results:
                content_key = result['content'][:100]
                if content_key not in seen_content:
                    seen_content.add(content_key)
                    merged_results.append(result)
            
            # 4. Apply metadata filtering for better relevance
            filtered_results = iso42001_metadata_filter(merged_results, q)
            
            results.extend(filtered_results)
            
        except Exception:
            # Fallback to basic FAISS search
            try:
                hits = db.similarity_search(q, k=TOP_K)
                for hit in hits:
                    results.append({
                        "content": hit.page_content,
                        "metadata": hit.metadata,
                        "query": q,
                        "hybrid_score": 0.5,  # Default score
                    })
            except Exception:
                pass  # Silent fallback failure
    
    # 5. Apply MMR for diversity
    if results:
        results = mmr_diversify(
            results, None, lambda_param=MMR_DIVERSITY, top_k=TOP_K
        )
    
    state.docs = results
    return state


def iso42001_reranker(state: ISO42001ChatState) -> ISO42001ChatState:
    """Enhanced reranking with multiple signals.
    
    Parameters
    ----------
    state : ISO42001ChatState
        Current state containing retrieved documents.
        
    Returns
    -------
    ISO42001ChatState
        Updated state with reranked documents.
    """
    if not state.docs:
        return state
    
    try:
        # 1. Cross-encoder reranking (if available)
        current_reranker = get_reranker()
        if current_reranker:
            pairs = [(state.user_question, d["content"]) for d in state.docs]
            rerank_scores = current_reranker.predict(pairs)
            
            # Combine with hybrid scores
            for i, doc in enumerate(state.docs):
                hybrid_score = doc.get(
                    'hybrid_score', doc.get('semantic_score', 0)
                )
                rerank_score = float(rerank_scores[i])
                
                # Weighted combination of scores
                doc['final_score'] = 0.6 * rerank_score + 0.4 * hybrid_score
        else:
            # Use hybrid scores as final scores
            for doc in state.docs:
                doc['final_score'] = doc.get(
                    'hybrid_score', doc.get('semantic_score', 0)
                )
        
        # 2. Additional ranking signals
        for doc in state.docs:
            metadata = doc.get('metadata', {})
            bonus = 0
            
            # Source quality bonus
            source = metadata.get('source', '').lower()
            if any(term in source for term in ['standard', 'iso', 'iec']):
                bonus += 0.1
            
            # Content quality bonus
            if metadata.get('has_requirements'):
                bonus += 0.05
            if metadata.get('clause'):
                bonus += 0.05
            
            doc['final_score'] += bonus
        
        # 3. Final ranking and selection
        ranked = sorted(
            state.docs, key=lambda x: x.get('final_score', 0), reverse=True
        )
        state.docs = ranked[:RERANK_KEEP]
        
    except Exception:
        # Fallback to simple scoring
        for doc in state.docs:
            doc['final_score'] = doc.get(
                'hybrid_score', doc.get('semantic_score', 0.5)
            )
        state.docs = sorted(
            state.docs, key=lambda x: x.get('final_score', 0), reverse=True
        )[:RERANK_KEEP]
    
    return state


def iso42001_sufficiency(state: ISO42001ChatState) -> ISO42001ChatState:
    """Assess evidence sufficiency and optionally refine the query.
    
    Parameters
    ----------
    state : ISO42001ChatState
        Current state containing documents and loop count.
        
    Returns
    -------
    ISO42001ChatState
        Updated state with sufficiency decision.
    """
    if state.loops >= MAX_LOOPS:
        state.final["_decision"] = "go_write"
        return state
        
    # Use smart truncation for evidence
    sample = smart_truncate_evidence(state.docs[:3], max_chars=800)
    prompt = SUFFICIENCY_PROMPT.format(q=state.user_question, snips=sample)
    text = call_llm_with_tracking(prompt, llm)
    
    if text.startswith("ERROR"):
        state.final["_decision"] = "go_write"
        return state
        
    sufficient = "true" in text.lower()
    if sufficient:
        state.final["_decision"] = "go_write"
        return state
        
    try:
        # Extract refined query
        if "refined_query" in text:
            rq = text.split("refined_query")[1].split(":", 1)[1]
            rq = rq.replace("}", "").replace('"', "").strip()
            state.subqueries = [rq] if rq else [state.user_question]
    except Exception:
        pass
        
    state.loops += 1
    state.final["_decision"] = "loop"
    return state


def sufficiency_router(state: ISO42001ChatState) -> str:
    """Route based on sufficiency decision.
    
    Parameters
    ----------
    state : ISO42001ChatState
        Current state with decision.
        
    Returns
    -------
    str
        Routing decision.
    """
    return state.final.get("_decision", "go_write")


def iso42001_writer(state: ISO42001ChatState) -> ISO42001ChatState:
    """Draft a structured answer bound to the retrieved evidence.
    
    Parameters
    ----------
    state : ISO42001ChatState
        Current state containing documents and question.
        
    Returns
    -------
    ISO42001ChatState
        Updated state with drafted answer.
    """
    ev = smart_truncate_evidence(state.docs[:4], max_chars=MAX_EVIDENCE_CHARS)
    prompt = WRITER_PROMPT.format(q=state.user_question, ev=ev)
    raw = call_llm_with_tracking(prompt, llm)
    
    if raw.startswith("ERROR"):
        state.final = {"answer_md": raw, "citations": [], "limits": ""}
    else:
        try:
            # Try to parse JSON response
            if "{" in raw and "}" in raw:
                json_part = raw[raw.find("{"):raw.rfind("}")+1]
                parsed = json.loads(json_part)
                state.final = parsed
            else:
                state.final = {"answer_md": raw, "citations": [], "limits": ""}
        except Exception:
            state.final = {"answer_md": raw, "citations": [], "limits": ""}
    
    return state


def iso42001_verifier(state: ISO42001ChatState) -> ISO42001ChatState:
    """Verify fidelity to evidence and attach caveats/limits if needed.
    
    Parameters
    ----------
    state : ISO42001ChatState
        Current state containing draft answer and documents.
        
    Returns
    -------
    ISO42001ChatState
        Updated state with verified answer.
    """
    ev = smart_truncate_evidence(state.docs, max_chars=1000)
    prompt = VERIFIER_PROMPT.format(
        draft=json.dumps(state.final), ev=ev
    )
    fixed = call_llm_with_tracking(prompt, llm)
    
    if not fixed.startswith("ERROR"):
        try:
            if "{" in fixed and "}" in fixed:
                json_part = fixed[fixed.find("{"):fixed.rfind("}")+1]
                parsed = json.loads(json_part)
                state.final.update(parsed)
        except Exception:
            pass
    
    return state


def iso42001_policy_guard(state: ISO42001ChatState) -> ISO42001ChatState:
    """Enforce scope, quoting, and privacy guardrails on the final answer.
    
    Parameters
    ----------
    state : ISO42001ChatState
        Current state containing final answer.
        
    Returns
    -------
    ISO42001ChatState
        Updated state with policy guardrails applied.
    """
    q = state.user_question.lower()
    out = state.final if state.final else {
        "answer_md": "", "citations": [], "limits": ""
    }
    
    forbidden = any(kw in q for kw in ["full text", "verbatim", "copy all"])
    offscope = any(kw in q for kw in ["tax law", "hipaa", "finance", "criminal"])
    
    if forbidden:
        out["limits"] = (
            out.get("limits", "") + 
            "\nRefusal: Cannot provide verbatim excerpts."
        ).strip()
    if offscope:
        out["limits"] = (
            out.get("limits", "") + 
            "\nRefusal: Outside ISO/IEC 42001 scope."
        ).strip()
    
    disclaimer = (
        "**Note**: This explains ISO/IEC 42001 (AIMS) but is not "
        "legal/certification advice."
    )
    answer = out.get("answer_md", "")
    out["answer_md"] = f"{disclaimer}\n\n" + answer
    out["answer_md"] = re.sub(
        r"[A-Za-z]:\\\\[^\s]+", "[REDACTED_PATH]", out["answer_md"]
    )
    
    state.final = out
    return state


# -------------------- Graph Construction --------------------
graph = StateGraph(ISO42001ChatState)

graph.add_node("planner", iso42001_planner)
graph.add_node("retriever", iso42001_retriever)
graph.add_node("reranker", iso42001_reranker)
graph.add_node("sufficiency", iso42001_sufficiency)
graph.add_node("writer", iso42001_writer)
graph.add_node("verifier", iso42001_verifier)
graph.add_node("policy_guard", iso42001_policy_guard)

graph.set_entry_point("planner")
graph.add_edge("planner", "retriever")
graph.add_edge("retriever", "reranker")
graph.add_edge("reranker", "sufficiency")
graph.add_conditional_edges(
    "sufficiency", 
    sufficiency_router,
    {"loop": "retriever", "go_write": "writer"},
)
graph.add_edge("writer", "verifier")
graph.add_edge("verifier", "policy_guard")
graph.add_edge("policy_guard", END)

app = graph.compile()


# -------------------- CLI Functions --------------------
def iso42001_maybe_ingest(force: bool = False) -> None:
    """Build the enhanced FAISS index with hybrid search capabilities.
    
    Parameters
    ----------
    force : bool, optional
        If True, rebuild the index even if it exists.
    """
    global db, documents, hybrid_search
    if force or db is None:
        print(f"Building knowledge base from: {KB_FOLDER}")
        raw = iso42001_load_docs(KB_FOLDER)
        if not raw:
            print("No documents found.")
            return
            
        chunks = iso42001_chunk_docs(raw)
        print(f"Processing {len(raw)} files → {len(chunks)} chunks...")
        
        # Build enhanced index with metadata
        db_local, docs_local = iso42001_ingest_faiss(chunks, emb, FAISS_DIR)
        db = db_local
        documents = docs_local
        
        # Initialize hybrid search
        hybrid_search = HybridSearchEngine(db, documents)
        
        search_type = "Hybrid search" if hybrid_search.bm25 else "Semantic search"
        print(f"✓ Knowledge base ready - {search_type} enabled")
    else:
        if not hybrid_search and db and documents:
            hybrid_search = HybridSearchEngine(db, documents)


def iso42001_chat_cli() -> None:
    """Run the interactive CLI loop for the ISO/IEC 42001 AIMS chatbot."""
    parser = argparse.ArgumentParser(
        description="ISO/IEC 42001 Agentic AIMS Chatbot - Token Optimized"
    )
    parser.add_argument(
        "--reingest", action="store_true", help="Force re-ingestion"
    )
    parser.add_argument(
        "--token-budget", 
        type=int, 
        default=TOKEN_BUDGET, 
        help="Token budget for session"
    )
    args = parser.parse_args()

    # Set token budget
    global token_tracker
    token_tracker = TokenTracker(args.token_budget)

    iso42001_maybe_ingest(force=args.reingest)

    search_type = "Hybrid" if hybrid_search and hybrid_search.bm25 else "Semantic"
    print(f"\n{'='*60}")
    print(f"🤖 AIMS Chatbot")
    print(f"🔍 {search_type} Search | 📊 Budget: {token_tracker.budget:,} tokens")
    print(f"{'='*60}")
    
    while True:
        try:
            q = input("❓ Ask: ")
            if not q.strip():
                continue
                
            if q.lower() in ['exit', 'quit', 'bye']:
                print(f"\n{'='*60}")
                print(f"👋 Session Complete!")
                print(f"📊 Total tokens used: {token_tracker.session_usage.total_tokens:,}")
                print(f"{'='*60}")
                break
                
            if q.lower() == 'status':
                print(
                    f"📊 Used: {token_tracker.session_usage.total_tokens:,} | "
                    f"Remaining: {token_tracker.budget_remaining():,}"
                )
                continue
                
            if not token_tracker.check_budget(1000):
                print(f"⚠️  Budget exceeded! Start a new session.")
                break
            
            # Reset query tracking
            token_tracker.reset_query()
            
            print("\n🔍 Processing your question...")
            state = ISO42001ChatState(user_question=q)
            result = app.invoke(state)
            
            # Handle both dict and state object returns
            if isinstance(result, dict):
                final_result = result.get('final', {})
            else:
                final_result = result.final
            
            print(f"\n{'─'*60}")
            print("📋 ANSWER:")
            print(f"{'─'*60}")
            answer = final_result.get("answer_md", "No answer generated.")
            print(answer)
            
            if final_result.get("limits"):
                print(f"\n⚠️  NOTICE: {final_result['limits']}")
            
            # Token usage summary
            used = token_tracker.query_usage.total_tokens
            remaining = token_tracker.budget_remaining()
            print(f"\n{'─'*60}")
            print(f"📊 Tokens: {used} used | {remaining:,} remaining")
            print(f"{'─'*60}\n")
            
        except KeyboardInterrupt:
            print(f"\n\n{'='*60}")
            print(f"👋 Session Complete!")
            print(f"📊 Total tokens used: {token_tracker.session_usage.total_tokens:,}")
            print(f"{'='*60}")
            break
        except Exception as e:
            print(f"\n❌ ERROR: {str(e)}")
            print(f"{'─'*60}\n")
            continue


if __name__ == "__main__":
    if not OPENAI_API_KEY:
        print("❌ OPENAI_API_KEY not set. Please configure your API key.")
    else:
        iso42001_chat_cli()