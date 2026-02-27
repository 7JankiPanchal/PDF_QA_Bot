from utils import (
    RELEVANCE_THRESHOLD,
    faiss_score_to_cosine_sim,
    compute_confidence,
    normalize_answer,
    normalize_spaced_text,
)
from fastapi import FastAPI, HTTPException, Request
from fastapi.routing import APIRouter
from pydantic import BaseModel, Field
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from slowapi import Limiter
from slowapi.util import get_remote_address
import os
import re
import time
import uvicorn

# ------------------------------
# Load environment
# ------------------------------
load_dotenv()
BASE_DIR = Path(__file__).resolve().parent.parent
UPLOAD_DIR = (BASE_DIR / "uploads").resolve()

# ------------------------------
# FastAPI app and rate limiter
# ------------------------------
app = FastAPI()
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter

# ------------------------------
# Session storage
# ------------------------------
sessions = {}
SESSION_TIMEOUT = 3600  # 1 hour

# ------------------------------
# Embedding model
# ------------------------------
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# ------------------------------
# PDF text splitter
# ------------------------------
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

# ------------------------------
# Helper functions
# ------------------------------
def normalize_spaced_text(text: str) -> str:
    def fix_spaced_word(match):
        return match.group(0).replace(" ", "")
    pattern = r"\b(?:[A-Za-z] ){2,}[A-Za-z]\b"
    return re.sub(pattern, fix_spaced_word, text)

def normalize_answer(text: str) -> str:
    text = normalize_spaced_text(text)
    text = re.sub(r'^(Final Answer:|Context:|Question:)\s*', '', text, flags=re.IGNORECASE)
    text = re.sub(r'[ \t]{2,}', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()

def cleanup_expired_sessions():
    now = time.time()
    expired = [sid for sid, s in sessions.items() if now - s["last_accessed"] > SESSION_TIMEOUT]
    for sid in expired:
        del sessions[sid]

def load_document(file_path: str):
    loader = PyPDFLoader(file_path)
    return loader.load()

# ------------------------------
# Request models
# ------------------------------
class DocumentPath(BaseModel):
    filePath: str
    session_id: str

class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    session_id: str
    history: list = []

class SummarizeRequest(BaseModel):
    session_id: str
    pdf: str | None = None

# ------------------------------
# Endpoints
# ------------------------------
SUPPORTED_EXTENSIONS = [".pdf"]

@app.post("/process-pdf")
@limiter.limit("15/15 minutes")
def process_pdf(request: Request, data: DocumentPath):
    cleanup_expired_sessions()
    file_path = Path(data.filePath).resolve()
    if not str(file_path).startswith(str(UPLOAD_DIR)):
        raise HTTPException(status_code=400, detail="Invalid file path")
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Document not found")
    ext = file_path.suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise HTTPException(status_code=400, detail=f"Unsupported file format: {ext}. Supported: {', '.join(SUPPORTED_EXTENSIONS)}")
    try:
        raw_docs = load_document(str(file_path))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to load document: {str(e)}")
    
    cleaned_docs = [Document(page_content=normalize_spaced_text(doc.page_content), metadata=doc.metadata) for doc in raw_docs]
    chunks = splitter.split_documents(cleaned_docs)
    if not chunks:
        raise HTTPException(status_code=400, detail="No text extracted from the document.")
    
    sessions[data.session_id] = {
        "vectorstore": FAISS.from_documents(chunks, embedding_model),
        "last_accessed": time.time(),
        "last_docs": []  # Initialize for retrieval metadata
    }

    return {"message": "Document processed successfully"}

# ------------------------------
# Ask question
# ------------------------------
# Dummy CCC system prompt and template
_CCC_SYSTEM = "You are a PDF QA Assistant."
_CCC_USER_TEMPLATE = "Context:\n{context}\nQuestion:\n{question}\nAnswer:"
CCC_PROMPT = PromptTemplate(input_variables=["context", "question"], template=_CCC_USER_TEMPLATE)

def generate_response(system_prompt: str, user_prompt: str, max_tokens: int = 600) -> str:
    # Mocked LLM response for example; replace with actual LLM call
    return f"Answer for: {user_prompt}"

@app.post("/ask")
@limiter.limit("60/15 minutes")
def ask_question(request: Request, data: AskRequest):
    cleanup_expired_sessions()
    session_data = sessions.get(data.session_id)
    if not session_data:
        return {"answer": "Session expired or no PDF uploaded!", "confidence_score": 0}

    session_data["last_accessed"] = time.time()
    vectorstore = session_data["vectorstore"]

    # Prepare conversation context
    conversation_context = ""
    for msg in data.history[-5:]:
        role = msg.get("role", "")
        content = msg.get("content", "")
        conversation_context += f"{role}: {content}\n"

    docs = vectorstore.similarity_search(data.question, k=4)
    if not docs:
        return {"answer": "No relevant context found in the uploaded document.", "confidence_score": 0}

    # -------------------------------
    # Store retrieved docs for /retrieval-info
    session_data["last_docs"] = docs
    # -------------------------------

    context = "\n\n".join([doc.page_content for doc in docs])
    question_with_history = f"Conversation so far:\n{conversation_context.strip()}\n\nCurrent Question: {data.question}" if conversation_context.strip() else data.question

    user_prompt = CCC_PROMPT.format(context=context, question=question_with_history)
    raw_answer = generate_response(system_prompt=_CCC_SYSTEM, user_prompt=user_prompt)
    answer = normalize_answer(raw_answer)
    
    return {"answer": answer, "confidence_score": 95}  # dummy confidence

# ------------------------------
# Retrieval metadata endpoint
# ------------------------------
retrieval_router = APIRouter(prefix="/retrieval-info")

@retrieval_router.get("/{session_id}")
def get_retrieval_info(session_id: str):
    session_data = sessions.get(session_id)
    if not session_data or not session_data.get("last_docs"):
        raise HTTPException(status_code=404, detail="No retrieval data found for this session")
    
    results = []
    for doc in session_data["last_docs"]:
        results.append({
            "page_content": doc.page_content,
            "metadata": doc.metadata,
        })
    return {"retrieved_chunks": results}

app.include_router(retrieval_router)

# ------------------------------
# Summarize PDF
# ------------------------------
@app.post("/summarize")
@limiter.limit("15/15 minutes")
def summarize_pdf(request: Request, data: SummarizeRequest):
    cleanup_expired_sessions()
    session = sessions.get(data.session_id)
    if not session:
        return {"summary": "Session expired or PDF not uploaded"}

    session["last_accessed"] = time.time()
    vectorstore = session["vectorstore"]
    docs = vectorstore.similarity_search("Summarize the document.", k=6)
    if not docs:
        return {"summary": "No content available"}

    context = "\n\n".join([doc.page_content for doc in docs])
    raw_summary = f"Summary for context: {context}"  # Mocked
    summary = normalize_answer(raw_summary)
    return {"summary": summary}

# ------------------------------
# Start server
# ------------------------------
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=5000, reload=True)