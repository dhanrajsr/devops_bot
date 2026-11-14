import os
import glob
import httpx
import operator
import hashlib
import secrets
import httpx
import psycopg
import json
from fastapi.responses import StreamingResponse, Response
from typing import TypedDict, Annotated, List
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
from dotenv import load_dotenv
from collections import defaultdict
from datetime import datetime, timezone
import csv
import io
from urllib.parse import quote_plus

from openai import AsyncOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langgraph.graph import StateGraph, START, END

from psycopg_pool import AsyncConnectionPool
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

# Load environment variables from .env file
load_dotenv()

# 1. Organization Chat & Embedding LLM Initialization
OMNI_API_KEY = os.getenv("OMNI_API_KEY")
OMNI_BASE_URL = os.getenv("OMNI_BASE_URL", "https://omnisphere.ai")
OMNI_MODEL = os.getenv("OMNI_MODEL", "deepseek-ai/DeepSeek-V4.1-Flash")

OMNI_EMBEDDING_API_KEY = os.getenv("OMNI_EMBEDDING_API_KEY")
OMNI_EMBEDDING_BASE_URL = os.getenv("OMNI_EMBEDDING_BASE_URL", OMNI_BASE_URL)
OMNI_EMBEDDING_MODEL = os.getenv("OMNI_EMBEDDING_MODEL", "google/gemma-2")

DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME", "rag_sessions")

if not OMNI_API_KEY or not OMNI_EMBEDDING_API_KEY:
    raise ValueError("WARNING: OMNI_API_KEY or OMNI_EMBEDDING_API_KEY is missing!")

# HTTP clients configured for internal enterprise SSL certificates
sync_http_client = httpx.Client(verify=False)
async_http_client = httpx.AsyncClient(verify=False)

omni_client = AsyncOpenAI(
    api_key=OMNI_API_KEY,
    base_url=OMNI_BASE_URL,
    http_client=async_http_client,
)

embeddings = OpenAIEmbeddings(
    model=OMNI_EMBEDDING_MODEL,
    api_key=OMNI_EMBEDDING_API_KEY,
    base_url=OMNI_EMBEDDING_BASE_URL,
    check_embedding_ctx_length=False,
    chunk_size=50,
    http_client=sync_http_client,
    http_async_client=async_http_client,
    extra_body={
        "encoding_format": "float",
        "user": "user-1234"
    }
)


async def call_omni_llm(prompt: str) -> str:
    """Calls Omnisphere Responses API with a plain string input."""
    response = await omni_client.responses.create(
        model=OMNI_MODEL,
        input=prompt,
        temperature=0.0,
    )
    return response.output_text

# 2. Password Hashing Helpers for Local Authentication
def hash_password(password: str) -> str:
    salt = secrets.token_hex(16)
    pwd_hash = hashlib.pbkdf2_hmac("sha256", password.encode(), salt.encode(), 100_000).hex()
    return f"{salt}${pwd_hash}"


def verify_password(password: str, stored_hash: str) -> bool:
    try:
        salt, pwd_hash = stored_hash.split("$")
        check_hash = hashlib.pbkdf2_hmac("sha256", password.encode(), salt.encode(), 100_000).hex()
        return secrets.compare_digest(pwd_hash, check_hash)
    except Exception:
        return False

# 3. Safe Document Loader & FAISS Vector DB Setup
pdf_folder_path = "./runbooks_pdf"
faiss_index_path = "./faiss_index"
os.makedirs(pdf_folder_path, exist_ok=True)
os.makedirs(faiss_index_path, exist_ok=True)


def load_runbooks(folder_path: str):
    """Safely loads genuine PDFs and falls back to TextLoader for plain-text files."""
    documents = []
    for file_path in glob.glob(os.path.join(folder_path, "*")):
        if os.path.isdir(file_path):
            continue
        try:
            with open(file_path, "rb") as f:
                header = f.read(5)

            if header == b"%PDF-":
                print(f"INFO: Loading genuine PDF -> {file_path}")
                loader = PyPDFLoader(file_path)
                documents.extend(loader.load())
            else:
                print(f"WARNING: File lacks %PDF- header, loading as plain text -> {file_path}")
                loader = TextLoader(file_path, encoding="utf-8")
                documents.extend(loader.load())
        except Exception as e:
            print(f"ERROR: Skipping {file_path} due to error: {e}")

    return documents

all_chunks = []

def build_or_load_vectorstore(force_rebuild: bool = False):
    global all_chunks
    raw_documents = load_runbooks(pdf_folder_path)

    if len(raw_documents) > 0:
        # Larger chunk size keeps headings attached to multi-line NRQL/SQL code blocks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=250)
        all_chunks = text_splitter.split_documents(raw_documents)
        print(f"INFO: Loaded {len(raw_documents)} document(s) and split into {len(all_chunks)} chunk(s).")
    else:
        print("WARNING: No readable runbooks found. Creating empty placeholder index.")
        all_chunks = [Document(page_content="No PDF runbooks found.", metadata={"source": "empty"})]

    index_file = os.path.join(faiss_index_path, "index.faiss")
    if os.path.exists(index_file) and not force_rebuild:
        print("INFO: Loading saved FAISS index from disk...")
        return FAISS.load_local(faiss_index_path, embeddings, allow_dangerous_deserialization=True)

    print("INFO: Creating new FAISS index...")
    vs = FAISS.from_documents(all_chunks, embeddings)
    vs.save_local(faiss_index_path)
    print("INFO: FAISS index successfully saved!")
    return vs


vectorstore = build_or_load_vectorstore()


# 3. Simplified State Definition
class RAGAgentState(TypedDict):
    query: str
    chat_history: Annotated[List[str], operator.add]
    retrieved_docs: str
    sources: List[str]
    solution: str
    review_comment: str
    iteration_count: int


def format_source_label(doc: Document) -> str:
    raw_source = doc.metadata.get("source", "Unknown")
    filename = os.path.basename(raw_source)
    page = doc.metadata.get("page")
    if page is not None:
        return f"`{filename}` (Page {int(page) + 1})"
    return f"`{filename}`"


def format_source_label(doc: Document) -> str:
    raw_source = doc.metadata.get("source", "Unknown")
    filename = os.path.basename(raw_source)
    page = doc.metadata.get("page")
    if page is not None:
        return f"`{filename}` (Page {int(page) + 1})"
    return f"`{filename}`"

def group_source_citations(docs: List[Document]) -> List[str]:
    """Groups page numbers by PDF filename into clean, sorted citation labels."""
    file_pages = defaultdict(set)
    file_order = []

    for doc in docs:
        raw_source = doc.metadata.get("source", "Unknown")
        filename = os.path.basename(raw_source)
        if filename not in file_order:
            file_order.append(filename)
        page = doc.metadata.get("page")
        if page is not None:
            file_pages[filename].add(int(page) + 1)

    formatted = []
    for filename in file_order:
        pages = sorted(file_pages[filename])
        if len(pages) == 1:
            formatted.append(f"`{filename}` (Page {pages[0]})")
        elif len(pages) > 1:
            page_str = ", ".join(str(p) for p in pages)
            formatted.append(f"`{filename}` (Pages {page_str})")
        else:
            formatted.append(f"`{filename}`")

    return formatted

def rag_search_node(state: RAGAgentState):
    query = state["query"]
    history = state.get("chat_history", [])

    recent_context = " ".join(history[-2:]) if history else ""
    search_query = f"{recent_context} {query}".strip()

    # 1. Vector search
    vector_results = vectorstore.similarity_search_with_score(search_query, k=6)
    matched_docs = []
    seen = set()

    for doc, score in vector_results:
        preview = doc.page_content[:60].replace("\n", " ")
        print(f"DEBUG VECTOR | Score: {score:.4f} | Text: {preview}...")
        if doc.page_content not in seen:
            matched_docs.append(doc)
            seen.add(doc.page_content)

    # 2. Keyword fallback (includes 3-letter codes like 4xx, 5xx, cpu, pod)
    stop_words = {
        "provide", "give", "what", "steps", "queries", "query",
        "around", "check", "for", "the", "and", "nrql",
        "related", "issue", "facing", "solution", "user", "assistant"
    }
    keywords = [
        w.lower().strip("?.,\"':")
        for w in search_query.split()
        if len(w.strip("?.,\"':")) >= 3 and w.lower().strip("?.,\"':") not in stop_words
    ]

    if "nrql" in query.lower() or "quer" in query.lower():
        keywords.extend(["select", "from"])

    keyword_matches = []
    for doc in all_chunks:
        if doc.page_content in seen:
            continue
        text_lower = doc.page_content.lower()
        match_count = sum(1 for kw in set(keywords) if kw in text_lower)
        if match_count > 0:
            keyword_matches.append((match_count, doc))

    keyword_matches.sort(key=lambda x: x[0], reverse=True)
    for _, doc in keyword_matches[:4]:
        matched_docs.append(doc)
        seen.add(doc.page_content)

    if not matched_docs:
        return {"retrieved_docs": "NO_RELEVANT_DOCS", "sources": []}

    top_docs = matched_docs[:8]
    context = "\n\n---\n\n".join(doc.page_content for doc in top_docs)

    # Collect unique source citations in order of appearance
    # unique_sources = list(dict.fromkeys(format_source_label(doc) for doc in top_docs))
    # return {"retrieved_docs": context, "sources": unique_sources}
    unique_sources = group_source_citations(top_docs)
    return {"retrieved_docs": context, "sources": unique_sources}

async def draft_solution_node(state: RAGAgentState):
    query = state["query"]
    docs = state["retrieved_docs"]
    sources = state.get("sources", [])
    history = state.get("chat_history", [])
    count = state.get("iteration_count", 0)

    if docs == "NO_RELEVANT_DOCS" or "No PDF runbooks found." in docs:
        fallback = "I don't have an answer for that as it is not covered in the provided runbooks."
        return {
            "solution": fallback,
            "iteration_count": count + 1,
            "chat_history": [f"User: {query}", f"Assistant: {fallback}"]
        }

    history_text = "\n".join(history[-6:]) if history else "No previous conversation."

    full_prompt = (
        "SYSTEM INSTRUCTIONS:\n"
        "You are a strict IT runbook assistant. Answer the user's question using ONLY the provided Runbook Context. "
        "Use the Conversation History to understand follow-up references (such as 'it', 'that issue', or 'NRQL queries for this'). "
        "Format any NRQL queries inside ```sql code blocks and shell commands inside ```bash code blocks. "
        "Do NOT invent commands, queries, or steps that are not present in the Runbook Context. "
        "Note: Queries written with 'SELECT ... FROM ...' in the context are NRQL queries.\n"
        "If the requested topic or query is completely absent from the Runbook Context, reply ONLY with: "
        "'I don't have an answer for that as it is not covered in the provided runbooks.'\n\n"
        f"CONVERSATION HISTORY:\n{history_text}\n\n"
        f"RUNBOOK CONTEXT:\n{docs}\n\n"
        f"CURRENT QUESTION:\n{query}\n\n"
        "SOLUTION:"
    )

    content = await call_omni_llm(full_prompt)

    # Append source citations only if the LLM found an actual answer
    if "I don't have an answer" not in content and sources:
        citation_footer = "\n\n---\n**Sources:** " + ", ".join(sources)
        final_output = content + citation_footer
    else:
        final_output = content

    return {
        "solution": final_output,
        "iteration_count": count + 1,
        "chat_history": [f"User: {query}", f"Assistant: {content}"]
    }


async def security_review_node(state: RAGAgentState):
    solution = state["solution"]

    if "I don't have an answer" in solution:
        return {"review_comment": "APPROVED"}

    prompt = f"Check the following solution for safety. If it is safe, reply with 'APPROVED':\n\n{solution}"
    content = await call_omni_llm(prompt)
    return {"review_comment": content}


# 5. Conditional Edge Logic
def evaluate_review(state: RAGAgentState):
    review = state["review_comment"]
    count = state["iteration_count"]
    if "APPROVED" in review.upper() or count >= 2:
        return "complete"
    return "rewrite"


# 6. Workflow Definition (Direct Q&A Flow)
workflow = StateGraph(RAGAgentState)
workflow.add_node("rag_search", rag_search_node)
workflow.add_node("draft_solution", draft_solution_node)
workflow.add_node("security_review", security_review_node)

workflow.add_edge(START, "rag_search")
workflow.add_edge("rag_search", "draft_solution")
workflow.add_edge("draft_solution", "security_review")

workflow.add_conditional_edges(
    "security_review",
    evaluate_review,
    {"rewrite": "draft_solution", "complete": END}
)


# 7. FastAPI & PostgreSQL Lifespan Setup
if not DB_USER or not DB_PASSWORD:
    raise ValueError("CRITICAL: DB_USER and DB_PASSWORD (or DATABASE_URL) must be set in .env!")

DATABASE_URL = f"postgresql://{DB_USER}:{quote_plus(DB_PASSWORD)}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

rag_multi_agent_app = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global rag_multi_agent_app
    async with AsyncConnectionPool(
        conninfo=DATABASE_URL,
        max_size=20,
        open=False,
        kwargs={"autocommit": True, "prepare_threshold": 0},
    ) as pool:
        await pool.open()
        checkpointer = AsyncPostgresSaver(pool)
        await checkpointer.setup()
        print("INFO: PostgreSQL checkpointer tables initialized!")

        # Ensure users and feedback tables exist, and seed initial admin from .env if empty
        async with pool.connection() as conn:
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    id SERIAL PRIMARY KEY,
                    username VARCHAR(50) UNIQUE NOT NULL,
                    password_hash TEXT NOT NULL,
                    role VARCHAR(20) DEFAULT 'user',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)
            cur = await conn.execute("SELECT COUNT(*) FROM users;")
            row = await cur.fetchone()
            if row and row == 0:
                admin_user = os.getenv("ADMIN_USER")
                admin_pass = os.getenv("ADMIN_PASSWORD")
                if not admin_user or not admin_pass:
                    raise ValueError(
                        "CRITICAL: 'users' table is empty. Set ADMIN_USER and ADMIN_PASSWORD in .env!"
                    )
                default_hash = hash_password(admin_pass)
                await conn.execute(
                    "INSERT INTO users (username, password_hash, role) VALUES (%s, %s, %s)",
                    (admin_user.strip(), default_hash, "admin")
                )
                print(f"INFO: Created initial admin user '{admin_user.strip()}' from .env.")

            await conn.execute("""
                CREATE TABLE IF NOT EXISTS feedback (
                    id SERIAL PRIMARY KEY,
                    username VARCHAR(50) REFERENCES users(username) ON DELETE SET NULL,
                    thread_id VARCHAR(100),
                    query TEXT,
                    response TEXT,
                    rating VARCHAR(10),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)
            print("INFO: PostgreSQL feedback table initialized!")

        rag_multi_agent_app = workflow.compile(checkpointer=checkpointer)
        yield


class FeedbackRequest(BaseModel):
    username: str
    thread_id: str
    query: str
    response: str
    rating: str


app = FastAPI(
    title="Production-Grade Multi-Agent RAG API",
    description="IT Runbook Assistant with Auth, Omnisphere LLM, FAISS & PostgreSQL Sessions",
    version="8.0",
    lifespan=lifespan
)


class LoginRequest(BaseModel):
    username: str
    password: str


class RegisterRequest(BaseModel):
    username: str
    password: str
    role: str = "user"


class ChatRequest(BaseModel):
    query: str
    thread_id: str = "prod_session_1"


@app.post("/login")
async def login(request: LoginRequest):
    async with await psycopg.AsyncConnection.connect(DATABASE_URL) as conn:
        cur = await conn.execute(
            "SELECT username, password_hash, role FROM users WHERE username = %s",
            (request.username.strip(),)
        )
        user = await cur.fetchone()

    if not user or not verify_password(request.password, user[1]):
        raise HTTPException(status_code=401, detail="Invalid username or password")

    return {
        "status": "success",
        "username": user[0],
        "role": user[2]
    }


@app.post("/register")
async def register(request: RegisterRequest):
    username = request.username.strip()
    if len(username) < 3 or len(request.password) < 4:
        raise HTTPException(status_code=400, detail="Username must be >= 3 chars and password >= 4 chars.")

    pwd_hash = hash_password(request.password)
    try:
        async with await psycopg.AsyncConnection.connect(DATABASE_URL) as conn:
            await conn.execute(
                "INSERT INTO users (username, password_hash, role) VALUES (%s, %s, %s)",
                (username, pwd_hash, request.role)
            )
        return {"status": "success", "message": f"User '{username}' created successfully."}
    except psycopg.errors.UniqueViolation:
        raise HTTPException(status_code=409, detail="Username already exists.")


@app.post("/chat")
async def chat(request: ChatRequest):
    if rag_multi_agent_app is None:
        raise HTTPException(status_code=503, detail="Agent workflow not initialized yet.")

    cleaned_query = request.query.strip().lower().rstrip("!.,?")
    greetings = {"hi", "hello", "hey", "hi there", "hello there", "good morning", "good afternoon", "good evening"}

    if cleaned_query in greetings:
        return {
            "solution": "Welcome! How can I help you?"
        }

    initial_input = {
        "query": request.query,
        "retrieved_docs": "",
        "solution": "",
        "review_comment": "",
        "iteration_count": 0
    }

    config = {"configurable": {"thread_id": request.thread_id}}
    result = await rag_multi_agent_app.ainvoke(initial_input, config=config)

    return {
        "solution": result.get("solution", "No solution generated")
    }


@app.get("/sessions/{username}")
async def get_user_sessions(username: str):
    """Fetches past sessions with ChatGPT-style titles ordered by most recent."""
    if rag_multi_agent_app is None:
        return {"sessions": []}

    prefix = f"{username.strip()}_%"
    async with await psycopg.AsyncConnection.connect(DATABASE_URL) as conn:
        cur = await conn.execute(
            """
            SELECT thread_id, MAX(checkpoint_id) as latest_cp
            FROM checkpoints
            WHERE thread_id LIKE %s
            GROUP BY thread_id
            ORDER BY latest_cp DESC
            LIMIT 30;
            """,
            (prefix,)
        )
        rows = await cur.fetchall()

    session_list = []
    for row in rows:
        tid = row[0]
        config = {"configurable": {"thread_id": tid}}
        state_snapshot = await rag_multi_agent_app.aget_state(config)
        title = "New Conversation"

        if state_snapshot and state_snapshot.values:
            raw_history = state_snapshot.values.get("chat_history", [])
            for entry in raw_history:
                if entry.startswith("User: "):
                    first_q = entry[len("User: "):].strip()
                    title = first_q[:42]
                    break

        session_list.append({"thread_id": tid, "title": title})

    return {"sessions": session_list}


@app.get("/history/{thread_id}")
async def get_session_history(thread_id: str):
    """Loads saved messages from LangGraph's PostgreSQL state for a thread_id."""
    if rag_multi_agent_app is None:
        raise HTTPException(status_code=503, detail="Agent workflow not initialized yet.")

    config = {"configurable": {"thread_id": thread_id}}
    state_snapshot = await rag_multi_agent_app.aget_state(config)

    if not state_snapshot or not state_snapshot.values:
        return {"messages": [{"role": "assistant", "content": "Welcome! How can I help you?"}]}

    raw_history = state_snapshot.values.get("chat_history", [])
    messages = [{"role": "assistant", "content": "Welcome! How can I help you?"}]

    for entry in raw_history:
        if entry.startswith("User: "):
            messages.append({"role": "user", "content": entry[len("User: "):]})
        elif entry.startswith("Assistant: "):
            messages.append({"role": "assistant", "content": entry[len("Assistant: "):]})

    return {"messages": messages}


@app.get("/runbooks")
async def list_runbooks():
    """Returns the list of active runbook files and total indexed chunks."""
    files = sorted([
        os.path.basename(f)
        for f in glob.glob(os.path.join(pdf_folder_path, "*"))
        if os.path.isfile(f)
    ])
    return {
        "files": files,
        "total_chunks": len(all_chunks)
    }

@app.delete("/runbooks/{filename}")
async def delete_runbook(filename: str):
    """Deletes a runbook file from ./runbooks_pdf and rebuilds the FAISS index."""
    global vectorstore
    safe_filename = os.path.basename(filename)
    file_path = os.path.join(pdf_folder_path, safe_filename)

    if not os.path.isfile(file_path):
        raise HTTPException(status_code=404, detail=f"Runbook '{safe_filename}' not found.")

    try:
        os.remove(file_path)
        vectorstore = build_or_load_vectorstore(force_rebuild=True)
        return {
            "status": "success",
            "message": f"Deleted '{safe_filename}' and rebuilt index ({len(all_chunks)} chunks)."
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/upload-runbook")
async def upload_runbook(file: UploadFile = File(...)):
    """Saves an uploaded PDF/TXT runbook and rebuilds the FAISS index."""
    global vectorstore
    if not file.filename.lower().endswith((".pdf", ".txt", ".md")):
        raise HTTPException(status_code=400, detail="Only .pdf, .txt, or .md files are supported.")

    safe_filename = os.path.basename(file.filename)
    destination = os.path.join(pdf_folder_path, safe_filename)

    try:
        content = await file.read()
        with open(destination, "wb") as f:
            f.write(content)

        vectorstore = build_or_load_vectorstore(force_rebuild=True)
        return {
            "status": "success",
            "message": f"Uploaded '{safe_filename}' and rebuilt index ({len(all_chunks)} chunks)."
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    

@app.post("/reload")
async def reload_runbooks():
    """Rebuilds the FAISS index from ./runbooks_pdf without restarting the server."""
    global vectorstore
    try:
        vectorstore = build_or_load_vectorstore(force_rebuild=True)
        return {"status": "success", "message": "FAISS index rebuilt from runbooks_pdf."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/feedback")
async def save_feedback(request: FeedbackRequest):
    """Stores user thumbs up/down ratings in PostgreSQL."""
    async with await psycopg.AsyncConnection.connect(DATABASE_URL) as conn:
        await conn.execute(
            """
            INSERT INTO feedback (username, thread_id, query, response, rating)
            VALUES (%s, %s, %s, %s, %s)
            """,
            (request.username, request.thread_id, request.query, request.response, request.rating)
        )
    return {"status": "success", "message": "Feedback recorded."}


@app.get("/feedback")
async def get_feedback_summary():
    """Returns total positive/negative counts and recent feedback entries for admins."""
    async with await psycopg.AsyncConnection.connect(DATABASE_URL) as conn:
        # 1. Fetch counts grouped by rating ('up' / 'down')
        count_cur = await conn.execute(
            "SELECT rating, COUNT(*) FROM feedback GROUP BY rating;"
        )
        counts = dict(await count_cur.fetchall())

        # 2. Fetch the 15 most recent feedback logs
        logs_cur = await conn.execute(
            """
            SELECT username, query, rating, created_at
            FROM feedback
            ORDER BY created_at DESC
            LIMIT 15;
            """
        )
        rows = await logs_cur.fetchall()

    recent_logs = [
        {
            "username": r[0],
            "query": r[1],
            "rating": r[2],
            "created_at": r[3].strftime("%Y-%m-%d %H:%M") if r[3] else ""
        }
        for r in rows
    ]


    return {
        "positive": counts.get("up", 0),
        "negative": counts.get("down", 0),
        "recent": recent_logs
    }


@app.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    """Streams LLM tokens word-by-word and persists the turn to PostgreSQL."""
    if rag_multi_agent_app is None:
        raise HTTPException(status_code=503, detail="Agent workflow not initialized yet.")

    cleaned_query = request.query.strip().lower().rstrip("!.,?")
    greetings = {"hi", "hello", "hey", "hi there", "hello there", "good morning", "good afternoon", "good evening"}

    async def event_generator():
        try:
            if cleaned_query in greetings:
                yield f"data: {json.dumps({'token': 'Welcome! How can I help you?'})}\n\n"
                yield "data: [DONE]\n\n"
                return

            config = {"configurable": {"thread_id": request.thread_id}}
            state_snapshot = await rag_multi_agent_app.aget_state(config)
            history = state_snapshot.values.get("chat_history", []) if state_snapshot and state_snapshot.values else []

            # 1. Run Hybrid RAG Search
            search_res = rag_search_node({"query": request.query, "chat_history": history})
            docs = search_res.get("retrieved_docs", "NO_RELEVANT_DOCS")
            sources = search_res.get("sources", [])

            if docs == "NO_RELEVANT_DOCS" or "No PDF runbooks found." in docs:
                fallback = "I don't have an answer for that as it is not covered in the provided runbooks."
                yield f"data: {json.dumps({'token': fallback})}\n\n"
                try:
                    await rag_multi_agent_app.aupdate_state(
                        config,
                        {
                            "query": request.query,
                            "chat_history": [f"User: {request.query}", f"Assistant: {fallback}"]
                        },
                        as_node="draft_solution"
                    )
                except Exception as state_err:
                    print(f"WARNING: Failed to save fallback state: {state_err}")
                yield "data: [DONE]\n\n"
                return

            history_text = "\n".join(history[-6:]) if history else "No previous conversation."
            full_prompt = (
                "SYSTEM INSTRUCTIONS:\n"
                "You are a strict IT runbook assistant. Answer the user's question using ONLY the provided Runbook Context. "
                "Use the Conversation History to understand follow-up references (such as 'it', 'that issue', or 'NRQL queries for this'). "
                "Format any NRQL queries inside ```sql code blocks and shell commands inside ```bash code blocks. "
                "Do NOT invent commands, queries, or steps that are not present in the Runbook Context. "
                "Note: Queries written with 'SELECT ... FROM ...' in the context are NRQL queries.\n"
                "If the requested topic or query is completely absent from the Runbook Context, reply ONLY with: "
                "'I don't have an answer for that as it is not covered in the provided runbooks.'\n\n"
                f"CONVERSATION HISTORY:\n{history_text}\n\n"
                f"RUNBOOK CONTEXT:\n{docs}\n\n"
                f"CURRENT QUESTION:\n{request.query}\n\n"
                "SOLUTION:"
            )

            # 2. Stream tokens from Omnisphere Responses API
            full_answer = ""
            try:
                stream = await omni_client.responses.create(
                    model=OMNI_MODEL,
                    input=full_prompt,
                    temperature=0.0,
                    stream=True,
                )
                async for event in stream:
                    if getattr(event, "type", "") == "response.output_text.delta":
                        delta = getattr(event, "delta", "")
                        if delta:
                            full_answer += delta
                            yield f"data: {json.dumps({'token': delta})}\n\n"
            except Exception:
                full_answer = await call_omni_llm(full_prompt)
                yield f"data: {json.dumps({'token': full_answer})}\n\n"

            # 3. Append grouped source citations at the end of the stream
            if "I don't have an answer" not in full_answer and sources:
                citation_footer = "\n\n---\n**Sources:** " + ", ".join(sources)
                yield f"data: {json.dumps({'token': citation_footer})}\n\n"

            # 4. Safely save completed turn to PostgreSQL checkpointer
            try:
                await rag_multi_agent_app.aupdate_state(
                    config,
                    {
                        "query": request.query,
                        "chat_history": [f"User: {request.query}", f"Assistant: {full_answer}"]
                    },
                    as_node="draft_solution"
                )
            except Exception as state_err:
                print(f"WARNING: Failed to persist stream checkpoint: {state_err}")

            yield "data: [DONE]\n\n"

        except Exception as e:
            print(f"ERROR in chat_stream: {e}")
            yield f"data: {json.dumps({'token': f'Stream error: {str(e)}'})}\n\n"
            yield "data: [DONE]\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")

@app.get("/feedback/export")
async def export_negative_feedback():
    """Exports all negative ('down') feedback entries as a downloadable CSV."""
    async with await psycopg.AsyncConnection.connect(DATABASE_URL) as conn:
        cur = await conn.execute(
            """
            SELECT id, username, thread_id, query, response, created_at
            FROM feedback
            WHERE rating = 'down'
            ORDER BY created_at DESC;
            """
        )
        rows = await cur.fetchall()

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["ID", "Username", "Thread ID", "User Query", "Bot Response", "Timestamp"])

    for r in rows:
        timestamp_str = r[5].strftime("%Y-%m-%d %H:%M:%S") if r[5] else ""
        writer.writerow([r[0], r[1], r[2], r[3], r[4], timestamp_str])

    return Response(
        content=output.getvalue(),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=negative_feedback_report.csv"}
    )


@app.delete("/sessions/{thread_id}")
async def delete_user_session(thread_id: str):
    """Deletes all LangGraph checkpoint records for a given thread_id from PostgreSQL."""
    async with await psycopg.AsyncConnection.connect(DATABASE_URL) as conn:
        await conn.execute("DELETE FROM checkpoint_writes WHERE thread_id = %s;", (thread_id,))
        await conn.execute("DELETE FROM checkpoint_blobs WHERE thread_id = %s;", (thread_id,))
        await conn.execute("DELETE FROM checkpoints WHERE thread_id = %s;", (thread_id,))

    return {"status": "success", "message": f"Session '{thread_id}' deleted."}
