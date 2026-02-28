
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

from fastapi import FastAPI, Request, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document

from langchain_core.prompts import PromptTemplate

from dotenv import load_dotenv
from transformers import AutoConfig, AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from uuid import uuid4
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

import time
import uuid
import torch
import uvicorn
import pdf2image
import pytesseract
from PIL import Image

# Post-processing helpers: strip prompt echoes / context leakage from LLM output
# so that the API always returns only the clean, user-facing answer/summary/comparison.
from utils.postprocess import extract_final_answer, extract_final_summary, extract_comparison

# Centralised minimal prompt builders (short prompts → less instruction echoing).
from utils.prompt_templates import build_ask_prompt, build_summarize_prompt, build_compare_prompt

load_dotenv()

app = FastAPI(
    title="PDF QA Bot API",
    description="PDF Question-Answering Bot (Session-based, No Auth)",
    version="2.1.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Rate Limiter
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# ===============================
# SESSION STORAGE (REQUIRED: keep sessionId)
# ===============================
# Format: { session_id: { "vectorstores": [FAISS], "last_accessed": float } }
sessions = {}
SESSION_TIMEOUT = 3600  # 1 hour


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

# ===============================
# LOAD GENERATION MODEL ONCE
# ===============================
HF_GENERATION_MODEL = os.getenv("HF_GENERATION_MODEL", "google/flan-t5-small")

config = AutoConfig.from_pretrained(HF_GENERATION_MODEL)
is_encoder_decoder = bool(getattr(config, "is_encoder_decoder", False))
tokenizer = AutoTokenizer.from_pretrained(HF_GENERATION_MODEL)

if is_encoder_decoder:
    model = AutoModelForSeq2SeqLM.from_pretrained(HF_GENERATION_MODEL)
else:
    model = AutoModelForCausalLM.from_pretrained(HF_GENERATION_MODEL)

if torch.cuda.is_available():
    model = model.to("cuda")

model.eval()

# ===============================
# REQUEST MODELS
# ===============================
class AskRequest(BaseModel):
    question: str = Field(..., min_length=1)
    session_ids: list = []


class SummarizeRequest(BaseModel):
    session_ids: list = []



class CompareRequest(BaseModel):
    session_ids: list = []


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


# ===============================
# UTILITIES
# ===============================
def cleanup_expired_sessions():
    current_time = time.time()
    expired = [
        sid for sid, data in sessions.items()
        if current_time - data["last_accessed"] > SESSION_TIMEOUT
    ]
    for sid in expired:
        del sessions[sid]


def generate_response(prompt: str, max_new_tokens: int = 200) -> str:
    device = next(model.parameters()).device
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    output = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
    )

    if is_encoder_decoder:
        return tokenizer.decode(output[0], skip_special_tokens=True)

    return tokenizer.decode(
        output[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True,
    )


# ===============================
# HEALTH ENDPOINTS (kept from enhancement branch)
# ===============================
@app.get("/healthz")
def health_check():
    return {"status": "healthy"}


@app.get("/readyz")
def readiness_check():
    return {"status": "ready"}




# ------------------------------
# Ask question
# ------------------------------
# Dummy CCC system prompt and template
_CCC_SYSTEM = "You are a PDF QA Assistant."
_CCC_USER_TEMPLATE = "Context:\n{context}\nQuestion:\n{question}\nAnswer:"
CCC_PROMPT = PromptTemplate(input_variables=["context", "question"], template=_CCC_USER_TEMPLATE)

# ===============================
# UPLOAD (NO AUTH, RETURNS session_id)
# ===============================
@app.post("/upload")
@limiter.limit("10/15 minutes")
async def upload_file(request: Request, file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        return {"error": "Only PDF files are supported"}

    session_id = str(uuid4())
    upload_dir = "uploads"
    os.makedirs(upload_dir, exist_ok=True)
    # SECURITY: Use only uuid4().hex to prevent path traversal from client filename
    file_path = os.path.join(upload_dir, f"{uuid4().hex}.pdf")
    upload_dir_resolved = os.path.abspath(upload_dir)
    file_path_resolved = os.path.abspath(file_path)
    
    # SECURITY: Validate that file_path is within upload_dir (prevent path traversal)
    if not file_path_resolved.startswith(upload_dir_resolved + os.sep):
        return {"error": "Upload failed: Invalid file path detected."}

    try:
        with open(file_path, "wb") as buffer:
            buffer.write(await file.read())

        loader = PyPDFLoader(file_path)
        docs = loader.load()

        # Check if each page has extractable text
        final_docs = []
        images = None
        
        for i, doc in enumerate(docs):
            if len(doc.page_content.strip()) < 50:
                # Fallback to OCR for this specific page
                if images is None:
                    print("Low text content detected on one or more pages. Falling back to OCR...")
                    images = pdf2image.convert_from_path(file_path)
                
                if i < len(images):
                    ocr_text = pytesseract.image_to_string(images[i])
                    final_docs.append(Document(
                        page_content=ocr_text,
                        metadata={"source": file_path, "page": i}
                    ))
                else:
                    final_docs.append(doc)
            else:
                final_docs.append(doc)

        docs = final_docs

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100
        )
        chunks = splitter.split_documents(docs)

        if not chunks:
            return {"error": "Upload failed: No extractable text found in the document (OCR yielded nothing)."}

        vectorstore = FAISS.from_documents(chunks, embedding_model)

        sessions[session_id] = {
            "vectorstores": [vectorstore],
            "filename": file.filename,
            "last_accessed": time.time()
        }

        return {
            "message": "PDF uploaded and processed",
            "session_id": session_id,
            "page_count": len(docs)
        }

    except Exception as e:
        return {"error": f"Upload failed: {str(e)}"}
    
    finally:
        # FIX: Delete PDF file after processing to prevent disk space exhaustion (Issue #110)
        # This ensures the physical file is deleted even if OCR or embedding fails
        try:
            os.remove(file_path)
        except FileNotFoundError:
            # File already deleted or never created; nothing to clean up
            pass
        except OSError as delete_err:
            # Log other errors but don't crash
            print(f"[/upload] Warning: Failed to delete file: {str(delete_err)}")



def generate_response(system_prompt: str, user_prompt: str, max_tokens: int = 600) -> str:
    # Mocked LLM response for example; replace with actual LLM call
    return f"Answer for: {user_prompt}"

# ===============================
# ASK (USES session_ids — matches fixed App.js)
# ===============================
@app.post("/ask")
@limiter.limit("60/15 minutes")
def ask_question(request: Request, data: AskRequest):
    cleanup_expired_sessions()

    session_data = sessions.get(data.session_id)
    if not session_data:
        return {"answer": "Session expired or no PDF uploaded!", "confidence_score": 0}


    if not data.session_ids:
        return {"answer": "No session selected.", "citations": []}



    vectorstores = []

    # Update last_accessed for all sessions

    for sid in data.session_ids:
        session = sessions.get(sid)
        if session:
            session["last_accessed"] = time.time()

            vectorstores.extend(session["vectorstores"])


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

    if not vectorstores:
        return {"answer": "No documents found for selected sessions."}

    docs = []
    for vs in vectorstores:
        docs.extend(vs.similarity_search(data.question, k=4))

    if not docs:
        return {"answer": "No relevant context found."}

    # ── Build minimal prompt via prompt_templates (reduces instruction echoing) ──
    prompt = build_ask_prompt(
        context=context,
        question=question,
        conversation_context=conversation_context,
    )



    # Gather retrieved docs with their session filenames
    docs_with_meta = []
    for sid in data.session_ids:
        session = sessions.get(sid)
        if session:
            vs = session["vectorstores"][0]
            filename = session.get("filename", "unknown")
            retrieved = vs.similarity_search(data.question, k=4)
            for doc in retrieved:
                docs_with_meta.append({
                    "doc": doc,
                    "filename": filename,
                    "sid": sid
                })

    if not docs_with_meta:
        return {"answer": "No relevant context found.", "citations": []}

    # Build context with page annotations for the prompt
    context_parts = []
    for item in docs_with_meta:
        # PyPDFLoader sets metadata["page"] as 0-indexed
        raw_page = item["doc"].metadata.get("page", 0)
        page_num = int(raw_page) + 1  # Convert to 1-indexed
        context_parts.append(f"[Page {page_num}] {item['doc'].page_content}")

    context = "\n\n".join(context_parts)

    # Use minimal prompt builder to reduce instruction echoing (upstream fix)
    prompt = build_ask_prompt(context=context, question=data.question)
    raw_answer = generate_response(prompt, max_new_tokens=150)
    # Strip any leaked prompt/context text from the raw output
    clean_answer = extract_final_answer(raw_answer)

    # Build deduplicated, sorted citations
    seen = set()
    citations = []
    for item in docs_with_meta:
        raw_page = item["doc"].metadata.get("page", 0)
        page_num = int(raw_page) + 1
        key = (item["filename"], page_num)
        if key not in seen:
            seen.add(key)
            citations.append({
                "page": page_num,
                "source": item["filename"]
            })

    citations.sort(key=lambda c: (c["source"], c["page"]))

    return {"answer": clean_answer, "citations": citations}


# ===============================
# SUMMARIZE
# ===============================

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


    if not data.session_ids:
        return {"summary": "No session selected."}

    vectorstores = []
    for sid in data.session_ids:
        session = sessions.get(sid)
        if session:
            vectorstores.extend(session["vectorstores"])

    if not vectorstores:
        return {"summary": "No documents found."}

    docs = []
    for vs in vectorstores:
        docs.extend(vs.similarity_search("Summarize the document", k=6))

    context = "\n\n".join([d.page_content for d in docs])

    # ── Build minimal summarization prompt ───────────────────────────────────
    prompt = build_summarize_prompt(context=context)

    raw_summary = generate_response(prompt, max_new_tokens=300)
    # Post-process: strip any leaked prompt/context text from the summary.
    summary = extract_final_summary(raw_summary)
    return {"summary": summary}


# ===============================
# COMPARE
# ===============================
@app.post("/compare")
@limiter.limit("10/15 minutes")
def compare_documents(request: Request, data: CompareRequest):
    cleanup_expired_sessions()

    if len(data.session_ids) < 2:
        return {"comparison": "Select at least 2 documents."}

    contexts = []
    for sid in data.session_ids:
        session = sessions.get(sid)
        if session:
            vs = session["vectorstores"][0]
            chunks = vs.similarity_search("main topics", k=4)
            text = "\n".join([c.page_content for c in chunks])
            contexts.append(text)

    # Retrieve top chunks from each document separately for fair comparison
    query = "summarize the main topic, purpose, and key details of this document"
    per_doc_contexts = []
    for i, vs in enumerate(vectorstores):
        chunks = vs.similarity_search(query, k=4)
        text = "\n".join([c.page_content for c in chunks])
        per_doc_contexts.append(text)

    # ── Build minimal comparison prompt ───────────────────────────────────────
    prompt = build_compare_prompt(per_doc_contexts=per_doc_contexts)

    raw = generate_response(prompt, max_new_tokens=400)
    # Post-process: strip any leaked prompt/context text from the comparison.
    comparison = extract_comparison(raw)
    return {"comparison": comparison}


@app.get("/health")
def health():
    return {"status": "ok"}


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=5000)