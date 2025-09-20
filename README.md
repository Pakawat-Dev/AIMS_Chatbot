# AIMS_Chatbot
An intelligent chatbot system for ISO/IEC 42001 (AI Management System) that uses advanced RAG (Retrieval-Augmented Generation) with agentic workflow patterns to provide accurate, context-aware responses about AI management standards.

## What This Chatbot Does

This chatbot helps you understand and navigate ISO/IEC 42001 standards by:
- Searching through your ISO documents using hybrid semantic + keyword search
- Providing accurate, source-backed answers about AI management requirements
- Tracking token usage to stay within budget limits
- Using an agentic workflow to iteratively improve responses

## Prerequisites

Before you start, make sure you have:
- Python 3.8 or higher installed
- An OpenAI API key
- ISO/IEC 42001 documents (PDF, DOCX, HTML, or TXT format)

## Step-by-Step Setup Guide

### Step 1: Clone or Download the Project
```bash
# If using git
git clone <your-repo-url>
cd <project-folder>

# Or simply download and extract the files to a folder
```

### Step 2: Create a Virtual Environment
```bash
# Create virtual environment
python -m venv .venv

# Activate it (Windows)
.venv\Scripts\activate

# Activate it (Mac/Linux)
source .venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Set Up Environment Variables
Create a `.env` file in your project folder with these settings:

```env
# Required: Your OpenAI API key
OPENAI_API_KEY=your_openai_api_key_here

# Optional: Customize these settings
EMBED_MODEL=text-embedding-3-large
CHAT_MODEL=gpt-4o-mini
KB_FOLDER=D:\AIMS Standard
FAISS_DIR=./faiss_iso42001
TOP_K=12
RERANK_KEEP=6
MAX_LOOPS=2
TOKEN_BUDGET=15000
MAX_EVIDENCE_CHARS=2000
SEMANTIC_WEIGHT=0.7
BM25_WEIGHT=0.3
MMR_DIVERSITY=0.3
QUERY_EXPANSION=true
```

### Step 5: Prepare Your Documents
1. Create a folder for your ISO/IEC 42001 documents
2. Place your documents in supported formats:
   - PDF files (.pdf)
   - Word documents (.docx, .doc)
   - HTML files (.html, .htm)
   - Text files (.txt, .md)
3. Update the `KB_FOLDER` path in your `.env` file

### Step 6: Build the Knowledge Base
```bash
python chatbot.py --build
```

This will:
- Scan your document folder
- Split documents into chunks
- Create embeddings
- Build a searchable index
- Save everything for future use

### Step 7: Start Chatting
```bash
python chatbot.py --chat
```

## Understanding the Agent Workflow Pattern

This chatbot uses an **agentic workflow** that follows these steps:

### Agent Workflow Steps

#### 1. **Query Analysis Agent**
- **What it does**: Analyzes your question and breaks it down
- **How it works**: 
  - Receives your question
  - Identifies key concepts and terms
  - Expands the query with ISO-specific terminology
  - Creates multiple search variations

#### 2. **Search Agent** 
- **What it does**: Finds relevant documents using hybrid search
- **How it works**:
  - **Semantic Search**: Uses AI embeddings to find conceptually similar content
  - **Keyword Search**: Uses BM25 algorithm for exact term matching
  - **Hybrid Scoring**: Combines both approaches for better results
  - **Diversity**: Ensures results cover different aspects of your question

#### 3. **Reranking Agent**
- **What it does**: Improves search results quality
- **How it works**:
  - Uses a specialized reranking model
  - Scores document relevance more accurately
  - Keeps only the most relevant documents
  - Applies diversity filtering to avoid redundancy

#### 4. **Response Generation Agent**
- **What it does**: Creates the final answer
- **How it works**:
  - Takes the best documents from search
  - Constructs a detailed prompt with evidence
  - Generates a comprehensive answer
  - Includes source citations

#### 5. **Quality Control Agent**
- **What it does**: Ensures response quality
- **How it works**:
  - Checks if the answer addresses your question
  - Determines if more information is needed
  - Can trigger additional search loops
  - Manages token budget to prevent overuse

### Agent State Management

The system maintains state throughout the workflow:

```python
class ISO42001ChatState:
    user_question: str          # Your original question
    subqueries: List[str]       # Generated search variations  
    docs: List[Dict]            # Found documents
    loops: int                  # Number of search iterations
    final: Dict                 # Final response data
```

## Usage Examples

### Basic Questions
```
User: "What are the requirements for AI risk management?"
Agent: Searches → Finds relevant clauses → Provides detailed answer with sources
```

### Complex Queries
```
User: "How should we document our AI model validation process?"
Agent: Breaks down query → Searches multiple aspects → Combines information → Comprehensive response
```

### Follow-up Questions
```
User: "Can you provide more details about clause 8.2?"
Agent: Uses previous context → Enhanced search → Focused response
```

## Configuration Options

### Search Behavior
- `TOP_K`: Number of documents to retrieve initially
- `RERANK_KEEP`: Number of documents after reranking
- `SEMANTIC_WEIGHT`: How much to weight semantic search (0.0-1.0)
- `BM25_WEIGHT`: How much to weight keyword search (0.0-1.0)

### Response Quality
- `MAX_LOOPS`: Maximum search iterations per question
- `TOKEN_BUDGET`: Maximum tokens per session
- `MAX_EVIDENCE_CHARS`: Maximum characters in evidence section
- `QUERY_EXPANSION`: Whether to expand queries with related terms

### Model Selection
- `EMBED_MODEL`: Embedding model for semantic search
- `CHAT_MODEL`: Language model for response generation

## Troubleshooting

### Common Issues

**"No FAISS index found"**
- Run `python chatbot.py --build` first
- Check that your `KB_FOLDER` path is correct
- Ensure you have documents in supported formats

**"Token budget exceeded"**
- Increase `TOKEN_BUDGET` in .env
- Use shorter questions
- Restart the chat session

**"Import errors"**
- Make sure virtual environment is activated
- Run `pip install -r requirements.txt` again
- Check Python version (3.8+ required)

**"No relevant documents found"**
- Try rephrasing your question
- Check if your documents contain the information
- Verify document indexing was successful

## Advanced Features

### Token Tracking
The system tracks token usage to help manage costs:
- Shows tokens used per query
- Maintains session budget
- Prevents overuse with automatic limits

### Hybrid Search
Combines multiple search strategies:
- **Semantic**: Understands meaning and context
- **Keyword**: Finds exact terms and phrases
- **Reranking**: Improves result quality
- **Diversity**: Ensures comprehensive coverage

### Query Enhancement
Automatically improves your questions:
- Adds ISO-specific terminology
- Creates multiple search variations
- Uses context from previous results
- Expands abbreviations and technical terms

## File Structure

```
project/
├── chatbot.py              # Main chatbot application
├── requirements.txt        # Python dependencies
├── .env                   # Environment configuration
├── README.md              # This file
├── .venv/                 # Virtual environment
└── faiss_iso42001/        # Generated knowledge base
    ├── index.faiss        # Vector search index
    ├── index.pkl          # Index metadata
    ├── faiss_store.pkl    # FAISS store
    └── documents.pkl      # Document metadata
```

## Getting Help

If you encounter issues:
1. Check the troubleshooting section above
2. Verify your .env configuration
3. Ensure all dependencies are installed
4. Make sure your OpenAI API key is valid
5. Check that your documents are in the correct folder

## Security Notes

- Keep your OpenAI API key secure
- Don't commit .env files to version control
- Be mindful of token usage costs
- Review generated responses for accuracy

---

**Ready to start?** Follow the setup steps above, then run `python chatbot.py --build` followed by `python chatbot.py --chat` to begin exploring ISO/IEC 42001 with your AI assistant!
