import os
import tempfile
import streamlit as st
import time
import hashlib
import io
from typing import List, Optional, Tuple

# Workaround for triton compatibility issue on Windows
os.environ.setdefault('TRITON_INTERPRET', '1')
os.environ.setdefault('TRITON_DISABLE_LINE_INFO', '1')

st.set_page_config(page_title="Vaidya Ai - Healthcare assistant", layout="wide")

from brain_of_the_doctor import (
    encode_image,
    analyze_image_with_query,
    generate_prescription,
    analyze_text_query,
)

try:
    from voice_of_the_patient import transcribe_with_groq
except ImportError as e:
    st.error(f"Error importing voice module: {e}")

    def transcribe_with_groq(stt_model, audio_filepath, GROQ_API_KEY):
        st.error("Voice transcription not available due to import error")
        return None

# gTTS is optional
try:
    from gtts import gTTS
except Exception:
    gTTS = None  # type: ignore

# --- Offline RAG imports ---
import pdfplumber
import chromadb
from chromadb.config import Settings

# Ollama is optional (not available on Streamlit Cloud by default)
try:
    import ollama  # type: ignore
except Exception:
    ollama = None  # type: ignore

# Load .env if present (local development)
try:
    from dotenv import load_dotenv  # type: ignore
    load_dotenv()
except Exception:
    pass

# --- Config ---
BOOKS_DIR = "Books"
CHROMA_DIR = "chroma_db"
COLLECTION_NAME = "medical_books"
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
OLLAMA_MODEL = "gemma3:4b"

DEFAULT_CHUNK_SIZE_CHARS = 400
DEFAULT_CHUNK_OVERLAP_CHARS = 30
DEFAULT_TOP_K = 2

REQUIRED_IDK = "I don’t know based on the provided book"
HEALTHCARE_DISCLAIMER = (
    "Healthcare disclaimer: This tool is for informational/educational use only and is NOT medical advice. "
    "It may be incomplete or incorrect. For diagnosis or treatment, consult a licensed healthcare professional. "
    "If this is an emergency, seek urgent care immediately."
)

MAX_RETRIEVAL_QUERY_CHARS = 600
INDEXING_STATUS_PREFIX = "📦 Indexing book (building vector DB)..."

# --- Secrets / env config (Streamlit Cloud friendly) ---

def _get_secret(name: str, default: Optional[str] = None) -> Optional[str]:
    """Fetch from st.secrets first (Streamlit Cloud), then env vars."""
    try:
        val = st.secrets.get(name)  # type: ignore[attr-defined]
        if val is not None:
            return str(val)
    except Exception:
        pass
    val = os.environ.get(name)
    return val if val is not None else default

# Get API key from Streamlit secrets or env
GROQ_API_KEY = _get_secret("GROQ_API_KEY")

# Allow overriding Groq generation model via secrets/env var (and sidebar)
DEFAULT_GROQ_CHAT_MODEL = _get_secret("GROQ_CHAT_MODEL", "llama-3.3-70b-versatile") or "llama-3.3-70b-versatile"

# --- Sidebar: RAG tuning ---
with st.sidebar:
    st.markdown("## RAG Settings")
    st.caption("Smaller chunks are faster but may miss context. Larger chunks are slower but more complete.")

    chunk_size = st.slider("Chunk size (characters)", 200, 2000, DEFAULT_CHUNK_SIZE_CHARS, 100)
    chunk_overlap = st.slider(
        "Chunk overlap (characters)",
        0,
        min(500, chunk_size // 2),
        DEFAULT_CHUNK_OVERLAP_CHARS,
        10,
    )
    top_k = st.slider("Top K chunks to retrieve", 1, 8, DEFAULT_TOP_K, 1)

    st.markdown("## Answer Mode")
    answer_mode = st.radio(
        "Choose backend",
        options=[
            "Book + Ollama (offline)",
            "Book + Groq (internet, faster)",
        ],
        index=1 if GROQ_API_KEY else 0,
        help="Retrieval always uses your book. This chooses which model generates the final answer/prescription.",
    )

    generate_prescription_mode = st.checkbox(
        "Generate prescription too (uses same backend)",
        value=True,
        help="Creates a prescription-style output constrained to BOOK_CONTEXT.",
    )

    if answer_mode == "Book + Groq (internet, faster)":
        groq_chat_model = st.text_input(
            "Groq chat model",
            value=DEFAULT_GROQ_CHAT_MODEL,
            help="Model used for Groq generation. Common: llama-3.3-70b-versatile or llama-3.1-8b-instant",
        )
    else:
        groq_chat_model = DEFAULT_GROQ_CHAT_MODEL

    st.markdown("## Answer Source")
    answer_source = st.radio(
        "Choose source",
        options=[
            "Book (RAG)",
            "Groq API only (no book)",
        ],
        index=0,
        help="Book (RAG) answers are constrained to Medical_book.pdf. Groq API only ignores the book and uses the online model.",
    )

    st.markdown("## Combination Mode")
    allow_api_augmentation = st.checkbox(
        "Book + Groq augmentation (use API if book is missing)",
        value=False,
        help=(
            "If enabled, the app first tries Book (RAG). If the book cannot answer, it will also ask Groq API. "
            "Output will be split into 'From Book' and 'From Groq API'."
        ),
    )

    rebuild_index = st.button("Rebuild index with these settings")

# Persist in session state so helper functions can access
st.session_state["groq_chat_model"] = groq_chat_model
st.session_state["answer_source"] = answer_source
st.session_state["allow_api_augmentation"] = bool(allow_api_augmentation)

# Map sidebar selection to booleans
USE_GROQ_GENERATION = (answer_mode == "Book + Groq (internet, faster)") and bool(GROQ_API_KEY)

CHUNK_SIZE_CHARS = int(chunk_size)
CHUNK_OVERLAP_CHARS = int(chunk_overlap)
TOP_K = int(top_k)

# --- Embeddings ---
@st.cache_resource
def _get_embedder():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(EMBED_MODEL_NAME)


@st.cache_data(show_spinner=False)
def _embed_query_cached(q: str):
    q = (q or "").strip()
    if len(q) > MAX_RETRIEVAL_QUERY_CHARS:
        q = q[:MAX_RETRIEVAL_QUERY_CHARS]
    emb = _get_embedder().encode([q], normalize_embeddings=True).tolist()[0]
    return emb


@st.cache_resource
def _get_chroma_client() -> chromadb.api.client.ClientAPI:
    os.makedirs(CHROMA_DIR, exist_ok=True)
    try:
        return chromadb.PersistentClient(
            path=CHROMA_DIR,
            settings=Settings(anonymized_telemetry=False),
        )
    except Exception as e:
        st.warning(
            "Chroma persistent storage failed to initialize. "
            "Falling back to in-memory vector DB for this session. "
            f"Details: {e}"
        )
        return chromadb.EphemeralClient(settings=Settings(anonymized_telemetry=False))


def _list_pdf_books() -> List[str]:
    if not os.path.isdir(BOOKS_DIR):
        os.makedirs(BOOKS_DIR, exist_ok=True)
        return []
    return sorted(
        os.path.join(BOOKS_DIR, f)
        for f in os.listdir(BOOKS_DIR)
        if f.lower().endswith(".pdf")
    )


def _ensure_books_exist() -> bool:
    pdfs = _list_pdf_books()
    if not pdfs:
        st.error(f"No PDF books found in folder: {BOOKS_DIR}")
        st.info(f"Add one or more PDFs here: {os.path.abspath(BOOKS_DIR)}")
        return False
    return True


def _file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _books_fingerprint(paths: List[str]) -> str:
    h = hashlib.sha256()
    for p in paths:
        h.update(os.path.basename(p).encode("utf-8"))
        h.update(_file_sha256(p).encode("utf-8"))
    return h.hexdigest()


def _read_pdf_pages(pdf_path: str) -> List[Tuple[int, str]]:
    pages: List[Tuple[int, str]] = []
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            page_text = (page.extract_text() or "").strip()
            if page_text:
                pages.append((i, page_text))
    return pages


def _chunk_text(text: str, chunk_size: int, overlap: int) -> List[str]:
    if not text:
        return []
    chunks: List[str] = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + chunk_size, n)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == n:
            break
        start = max(0, end - overlap)
    return chunks


def _chunk_pages(pages: List[Tuple[int, str]], chunk_size: int, overlap: int) -> List[Tuple[str, int]]:
    out: List[Tuple[str, int]] = []
    for page_no, page_text in pages:
        for ch in _chunk_text(page_text, chunk_size, overlap):
            out.append((ch, page_no))
    return out


def _build_retrieval_query(symptoms: str, earlier: str, duration_days: int) -> str:
    parts: List[str] = []
    if symptoms:
        parts.append(f"Symptoms: {symptoms.strip()}")
    if earlier:
        parts.append(f"History: {earlier.strip()}")
    parts.append(f"Duration days: {int(duration_days) if duration_days is not None else 0}")
    q = "\n".join(parts).strip()
    return q[:MAX_RETRIEVAL_QUERY_CHARS]


def _rebuild_index_now() -> None:
    import shutil

    try:
        if os.path.isdir(CHROMA_DIR):
            shutil.rmtree(CHROMA_DIR)
    except Exception:
        pass

    try:
        _get_chroma_client.clear()
    except Exception:
        pass
    try:
        _get_collection_ready.clear()
    except Exception:
        pass

    st.success("Deleted chroma_db. It will be rebuilt (re-indexed) on the next question.")


if rebuild_index:
    _rebuild_index_now()


def _get_or_build_collection() -> Optional[chromadb.api.models.Collection.Collection]:
    if not _ensure_books_exist():
        return None

    client = _get_chroma_client()

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    pdfs = _list_pdf_books()
    current_fp = _books_fingerprint(pdfs)

    existing_count = collection.count()
    existing_fp = None
    try:
        sample = collection.get(include=["metadatas"], limit=1)
        if sample and sample.get("metadatas"):
            existing_fp = sample["metadatas"][0].get("library_fingerprint")
    except Exception:
        existing_fp = None

    needs_rebuild = (existing_count == 0) or (existing_fp != current_fp)

    if needs_rebuild:
        st.info(f"{INDEXING_STATUS_PREFIX} First run or book changed.")
        try:
            client.delete_collection(COLLECTION_NAME)
        except Exception:
            pass

        collection = client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )

        ids: List[str] = []
        documents: List[str] = []
        metadatas: List[dict] = []

        global_chunk_i = 0
        for pdf_path in pdfs:
            book_file = os.path.basename(pdf_path)
            st.info(f"{INDEXING_STATUS_PREFIX} Reading: {book_file} ...")
            pages = _read_pdf_pages(pdf_path)
            if not pages:
                continue

            chunks_with_pages = _chunk_pages(pages, CHUNK_SIZE_CHARS, CHUNK_OVERLAP_CHARS)
            st.info(f"{INDEXING_STATUS_PREFIX} Chunking: {book_file} ({len(chunks_with_pages)} chunks)...")

            for chunk_text, page_no in chunks_with_pages:
                ids.append(f"{book_file}-chunk-{global_chunk_i}")
                documents.append(chunk_text)
                metadatas.append(
                    {
                        "chunk_index": global_chunk_i,
                        "book_file": book_file,
                        "book_path": os.path.abspath(pdf_path),
                        "page": page_no,
                        "library_fingerprint": current_fp,
                    }
                )
                global_chunk_i += 1

        if not documents:
            st.error("No text could be indexed from the PDFs (scanned PDFs need OCR; not included).")
            return None

        batch_size = 128
        total = len(documents)
        prog = st.progress(0, text=f"{INDEXING_STATUS_PREFIX} Embedding chunks... (0/{total})")

        embedder = _get_embedder()
        for i in range(0, total, batch_size):
            batch_docs = documents[i : i + batch_size]
            batch_ids = ids[i : i + batch_size]
            batch_metas = metadatas[i : i + batch_size]
            batch_emb = embedder.encode(batch_docs, normalize_embeddings=True).tolist()
            collection.add(
                ids=batch_ids,
                documents=batch_docs,
                metadatas=batch_metas,
                embeddings=batch_emb,
            )
            done = min(i + batch_size, total)
            prog.progress(int(done * 100 / total), text=f"{INDEXING_STATUS_PREFIX} Embedding chunks... ({done}/{total})")

        st.success(f"✅ Indexed {len(documents)} chunks from {len(pdfs)} PDF(s).")

    return collection


@st.cache_resource
def _get_collection_ready() -> Optional[chromadb.api.models.Collection.Collection]:
    return _get_or_build_collection()


def _retrieve_context(collection, question: str, top_k: int = TOP_K) -> Tuple[str, List[str]]:
    q_emb = _embed_query_cached(question)

    results = collection.query(
        query_embeddings=[q_emb],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    docs = (results.get("documents") or [[]])[0]
    metas = (results.get("metadatas") or [[]])[0]

    context_parts: List[str] = []
    citations: List[str] = []
    for d, m in zip(docs, metas):
        if not d:
            continue
        book = m.get("book_file", "unknown.pdf")
        page = m.get("page", "?")
        idx = m.get("chunk_index", "?")
        context_parts.append(f"[{book} | Page {page} | Chunk {idx}]\n{d}")
        citations.append(f"{book} p.{page} (chunk {idx})")

    return "\n\n".join(context_parts).strip(), citations


def _ask_ollama_book_only(question: str, context: str) -> str:
    if ollama is None:
        return ""
    context = context[:800] if len(context) > 800 else context

    system = (
        "You are a healthcare assistant that MUST answer ONLY using the provided BOOK_CONTEXT. "
        "Do not use any outside knowledge. Do not guess. "
        f"If the answer is not explicitly in the BOOK_CONTEXT, reply exactly: {REQUIRED_IDK}. "
        "Keep answers concise (2-3 sentences max)."
    )

    user = (
        "BOOK_CONTEXT:\n"
        f"{context}\n\n"
        "QUESTION:\n"
        f"{question}\n\n"
        "INSTRUCTIONS:\n"
        "- Use ONLY BOOK_CONTEXT.\n"
        f"- If not in BOOK_CONTEXT, reply exactly: {REQUIRED_IDK}\n"
        "- Do not add extra facts.\n"
    )

    resp = ollama.chat(
        model=OLLAMA_MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        options={
            "temperature": 0.2,
            "num_predict": 150,
            "num_ctx": 1024,
        },
    )
    return (resp.get("message") or {}).get("content", "").strip()


def _ask_groq_book_only(question: str, context: str) -> str:
    if not GROQ_API_KEY:
        return ""
    try:
        from groq import Groq  # type: ignore

        client = Groq(api_key=GROQ_API_KEY)
        context = context[:1600]
        system = (
            "You are a healthcare assistant that MUST answer ONLY using the provided BOOK_CONTEXT. "
            "Do not use any outside knowledge. Do not guess. "
            f"If the answer is not explicitly in the BOOK_CONTEXT, reply exactly: {REQUIRED_IDK}. "
            "Keep answers short, direct, and well-structured."
        )
        user = f"BOOK_CONTEXT:\n{context}\n\nQUESTION:\n{question}"
        model_name = (st.session_state.get("groq_chat_model") or DEFAULT_GROQ_CHAT_MODEL).strip()
        resp = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            temperature=0.2,
            max_tokens=300,
        )
        return (resp.choices[0].message.content or "").strip()
    except Exception as e:
        st.warning(f"Groq generation failed, falling back to Ollama. Details: {e}")
        return ""


def _ask_groq_api_only(prompt: str) -> str:
    """Unrestricted Groq answer (NOT book-limited)."""
    if not GROQ_API_KEY:
        return ""
    try:
        from groq import Groq  # type: ignore

        client = Groq(api_key=GROQ_API_KEY)
        model_name = (st.session_state.get("groq_chat_model") or DEFAULT_GROQ_CHAT_MODEL).strip()
        resp = client.chat.completions.create(
            model=model_name,
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful medical assistant. Provide clear, safe, concise information. ",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,
            max_tokens=450,
        )
        return (resp.choices[0].message.content or "").strip()
    except Exception as e:
        st.warning(f"Groq API request failed: {e}")
        return ""


def _ask_groq_api_only_with_safety(prompt: str) -> str:
    """Groq API answer (not book-limited) with a stronger safety prompt and short output."""
    if not GROQ_API_KEY:
        return ""
    try:
        from groq import Groq  # type: ignore

        client = Groq(api_key=GROQ_API_KEY)
        model_name = (st.session_state.get("groq_chat_model") or DEFAULT_GROQ_CHAT_MODEL).strip()
        resp = client.chat.completions.create(
            model=model_name,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a cautious healthcare assistant. Provide general health information (not medical advice). "
                        "Be concise. Include red flags and when to see a clinician. "
                        "Avoid prescribing exact dosages for prescription-only medicines."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,
            max_tokens=450,
        )
        return (resp.choices[0].message.content or "").strip()
    except Exception as e:
        st.warning(f"Groq API request failed: {e}")
        return ""


def _augment_with_api_if_needed(symptoms_text: str, earlier_text: str, duration_days: int, book_answer: str) -> str:

    """Optionally add a Groq API section when the book is missing or insufficient.

    Trigger conditions:
    - Combination mode enabled
    - Groq API key available
    - Book answer is REQUIRED_IDK OR looks non-actionable (e.g., says not provided / not explicitly)
    """
    if not st.session_state.get("allow_api_augmentation", False):
        return ""
    if not GROQ_API_KEY:
        return ""

    ans = (book_answer or "").strip()
    if not ans:
        return ""

    # Book definitely missing
    should = (ans == REQUIRED_IDK)

    # Heuristic: book answer exists but says it cannot provide details/guidance
    lowered = ans.lower()
    weak_phrases = [
        "not explicitly",
        "does not provide",
        "not provide",
        "not mentioned",
        "not available",
        "insufficient",
        "cannot be determined",
        "no specific",
        "no guidance",
    ]
    if any(p in lowered for p in weak_phrases):
        should = True

    if not should:
        return ""

    prompt = (
        "The medical book information is missing or insufficient. Using general knowledge, help the user.\n\n"
        f"Symptoms: {symptoms_text or ''}\n"
        f"History: {earlier_text or ''}\n"
        f"Duration (days): {duration_days}\n\n"
        "Provide: likely causes (differential), self-care, and red flags."
    )
    return _ask_groq_api_only_with_safety(prompt)


def _ask_book_only(question: str, context: str) -> str:
    """Use selected backend to answer from BOOK_CONTEXT."""
    if USE_GROQ_GENERATION and GROQ_API_KEY:
        out = _ask_groq_book_only(question, context)
        return out or _ask_ollama_book_only(question, context)
    return _ask_ollama_book_only(question, context)


def _generate_prescription_book_only(diagnosis: str, context: str) -> str:
    """Generate a prescription-style result constrained to BOOK_CONTEXT."""
    system = (
        "You are a medical assistant. Generate a prescription-style recommendation ONLY from BOOK_CONTEXT. "
        "Do not use outside knowledge. If BOOK_CONTEXT has no medication/treatment for this case, "
        "reply: Consult healthcare professional for specific medication."
    )
    user = (
        "BOOK_CONTEXT:\n"
        f"{context[:1200]}\n\n"
        "DIAGNOSIS:\n"
        f"{diagnosis[:600]}\n\n"
        "OUTPUT FORMAT:\n"
        "- Medication/Treatment: Dose, Frequency, Duration, Instructions\n"
    )

    if USE_GROQ_GENERATION and GROQ_API_KEY:
        try:
            from groq import Groq  # type: ignore

            client = Groq(api_key=GROQ_API_KEY)
            model_name = (st.session_state.get("groq_chat_model") or DEFAULT_GROQ_CHAT_MODEL).strip()
            resp = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                temperature=0.2,
                max_tokens=320,
            )
            return (resp.choices[0].message.content or "").strip()
        except Exception:
            pass

    if ollama is None:
        return "Consult healthcare professional for specific medication."

    resp = ollama.chat(
        model=OLLAMA_MODEL,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        options={"temperature": 0.2, "num_predict": 220, "num_ctx": 1024},
    )
    return (resp.get("message") or {}).get("content", "").strip()


def _generate_prescription_groq_api_only(diagnosis: str) -> str:
    """Prescription-style output using Groq only (NOT book-limited)."""
    prompt = (
        "Generate a prescription-style recommendation based on this diagnosis. "
        "Include: probable condition, OTC care, red flags, and when to see a doctor. "
        "Do NOT invent exact drug dosages for prescription-only medicines.\n\n"
        f"DIAGNOSIS:\n{diagnosis}\n"
    )
    return _ask_groq_api_only(prompt)


def rag_answer_from_book(question: str, retrieval_query: Optional[str] = None) -> Tuple[str, List[str], str]:
    collection = _get_collection_ready()
    if collection is None:
        return REQUIRED_IDK, [], ""

    q_for_retrieval = retrieval_query if retrieval_query else question
    context, citations = _retrieve_context(collection, q_for_retrieval, TOP_K)
    if not context.strip():
        return REQUIRED_IDK, citations, context

    if USE_GROQ_GENERATION and GROQ_API_KEY:
        answer = _ask_groq_book_only(question, context)
        if not answer:
            answer = _ask_ollama_book_only(question, context)
    else:
        answer = _ask_ollama_book_only(question, context)

    lowered = answer.lower().strip()
    if (not answer) or (REQUIRED_IDK.lower() in lowered):
        return REQUIRED_IDK, citations, context

    return answer, citations, context


@st.cache_data
def generate_audio_from_text(text, lang):
    if gTTS is None:
        return None
    try:
        tts = gTTS(text=text, lang=lang)
        audio_bytes_io = io.BytesIO()
        tts.write_to_fp(audio_bytes_io)
        audio_bytes_io.seek(0)
        return audio_bytes_io.getvalue()
    except Exception:
        return None

# ----- UI: the rest of the app remains same pattern as before -----

if GROQ_API_KEY:
    st.success("🔑 GROQ_API_KEY: configured")
else:
    st.warning("ℹ️ GROQ_API_KEY not set. Groq features (voice + online answers) will be disabled.")

LANGUAGE_CODES = {
    "English": "en",
    "Hindi": "hi",
    "Marathi": "mr",
}

TRANSLATIONS = {
    "English": {
        "title": "🩺 Vaidya Ai - Healthcare assistant",
        "subtitle": "Professional medical diagnosis powered by AI",
        "input": "Input",
        "voice_tab": "🎤 Voice Input",
        "text_tab": "✍️ Text Input",
        "describe_symptoms": "Describe your symptoms",
        "earlier_symptoms": "Earlier symptoms / what problem are you facing?",
        "days_suffering": "Days suffering",
        "days_help": "From how many days are you suffering?",
        "upload_image": "Upload Medical Image (Optional)",
        "doctor_panel": "Your Doctor",
        "get_diagnosis": "🔍 Get Diagnosis",
        "language": "Language",
    },
    "Hindi": {
        "title": "🩺 वैद्य AI - एक हेल्थकेयर असिस्टेंट",
        "subtitle": "एआई द्वारा संचालित पेशेवर चिकित्सा निदान",
        "input": "इनपुट",
        "voice_tab": "🎤 वॉइस इनपुट",
        "text_tab": "✍️ टेक्स्ट इनपुट",
        "describe_symptoms": "अपने लक्षणों का वर्णन करें",
        "earlier_symptoms": "पहले के लक्षण / आप किस प्रकार की समस्या का सामना कर रहे हैं?",
        "days_suffering": "कितने दिनों से",
        "days_help": "आप कितने दिनों से पीड़ित हैं?",
        "upload_image": "मेडिकल इमेज अपलोड करें (वैकल्पिक)",
        "doctor_panel": "आपके डॉक्टर",
        "get_diagnosis": "🔍 निदान प्राप्त करें",
        "language": "भाषा",
    },
    "Marathi": {
        "title": "🩺 वैद्य AI - आरोग्य सहाय्यक",
        "subtitle": "एआय द्वारा समर्थित व्यावसायिक वैद्यकीय निदान",
        "input": "इनपुट",
        "voice_tab": "🎤 आवाज इनपुट",
        "text_tab": "✍️ मजकूर इनपुट",
        "describe_symptoms": "आपली लक्षणे वर्णन करा",
        "earlier_symptoms": "पूर्वीची लक्षणे / तुम्ही कोणत्या प्रकारची समस्या अनुभवत आहात?",
        "days_suffering": "किती दिवसांपासून",
        "days_help": "आपण किती दिवसांपासून त्रस्त आहात?",
        "upload_image": "वैद्यकीय प्रतिमा अपलोड करा (ऐच्छिक)",
        "doctor_panel": "आपले डॉक्टर",
        "get_diagnosis": "🔍 निदान मिळवा",
        "language": "भाषा",
    },
}


def tr(key: str) -> str:
    lang = st.session_state.get("language", "English")
    return TRANSLATIONS.get(lang, TRANSLATIONS["English"]).get(key, key)

# --- Style and layout ---
st.markdown(
    """
<style>
    .block-container {padding-top: 0.5rem; padding-bottom: 1rem;}
    .stButton>button {width: 100%;}
    .stTextArea textarea {font-size: 1rem;}
    .diagnosis-card, .prescription-card {
        background: #22232b;
        border-radius: 10px;
        padding: 1rem;
        margin-bottom: 1rem;
        color: #fff;
        border: 1px solid #444;
    }
    .section-title {color: #ff9800; font-weight: bold; margin-bottom: 0.5rem;}
    .title-nowrap {white-space: nowrap; font-size: clamp(1.5rem, 2.6vw + 0.5rem, 3rem);}
</style>
""",
    unsafe_allow_html=True,
)

header_left, header_spacer, header_right = st.columns([8, 2, 2], gap="small")
with header_left:
    st.markdown(f"<h1 class='title-nowrap'>{tr('title')}</h1>", unsafe_allow_html=True)
    st.markdown(f"*{tr('subtitle')}*")
with header_right:
    st.selectbox(tr("language"), list(LANGUAGE_CODES.keys()), key="language")

col1, col2 = st.columns([2, 1], gap="large")

with col1:
    st.markdown(f"### {tr('input')}")
    tab1, tab2 = st.tabs([tr("voice_tab"), tr("text_tab")])
    with tab1:
        st.caption("Upload an audio file for transcription")
        audio_input = st.file_uploader("Record your symptoms (upload .wav/.mp3)", type=["wav", "mp3"])
        if audio_input is None:
            st.info("Please upload an audio file or use text input")
        st.number_input(tr("days_suffering"), min_value=0, step=1, help=tr("days_help"), key="duration_days_voice", value=st.session_state.get("duration_days_general", 0))
        if "duration_days_voice" in st.session_state:
            st.session_state["duration_days_general"] = st.session_state.get("duration_days_voice", 0)
    with tab2:
        c1, c2 = st.columns([3, 1])
        with c1:
            text_input = st.text_area(tr("describe_symptoms"), value=st.session_state.get("prefill_text", ""), placeholder="Type your symptoms here...", height=120)
        with c2:
            st.number_input(tr("days_suffering"), min_value=0, step=1, help=tr("days_help"), key="duration_days_general", value=st.session_state.get("duration_days_general", 0))

    earlier_symptoms = st.text_area(tr("earlier_symptoms"), placeholder="List early signs or describe the specific problem type...", height=100)
    image_input = st.file_uploader(tr("upload_image"), type=["jpg", "jpeg", "png", "webp"])

    enable_audio = st.checkbox("Generate audio output (can be slow / requires internet)", value=False)

    response_language = st.session_state.get("language", "English")
    submit_btn = st.button(tr("get_diagnosis"))

with col2:
    st.markdown(f"### {tr('doctor_panel')}")
    st.image("portrait-3d-female-doctor[1].jpg", caption="Your Doctor")

if submit_btn:
    with st.status("Initializing AI Doctor...", expanded=True) as status:
        try:
            t0 = time.time()

            # Determine source
            use_api_only = st.session_state.get("answer_source") == "Groq API only (no book)"

            # Ensure these exist for the footer/caption regardless of branch
            pdfs: List[str] = []
            pdf_names: List[str] = []

            if use_api_only:
                status.write("🌐 Using Groq API only (no book/RAG)...")
                if not GROQ_API_KEY:
                    st.error("GROQ_API_KEY not set. Add it to .env and restart.")
                    st.stop()

                duration_val = st.session_state.get("duration_days_general", 0)
                api_prompt = (
                    f"Symptoms: {text_input or ''}\n"
                    f"History: {earlier_symptoms or ''}\n"
                    f"Duration (days): {duration_val}\n\n"
                    "Task: Provide a medical explanation/differential, self-care guidance, and red flags."
                )

                diagnosis_raw = _ask_groq_api_only(api_prompt)
                diagnosis = (
                    "Answer based on: Groq API\n\n"
                    f"{diagnosis_raw}\n\n{HEALTHCARE_DISCLAIMER}"
                )

                # Always produce something for prescription box
                if generate_prescription_mode and diagnosis_raw:
                    status.write("💊 Generating prescription (Groq API only)...")
                    presc = _generate_prescription_groq_api_only(diagnosis_raw)
                    prescription = (
                        "PRESCRIPTION\n\n"
                        "Answer based on: Groq API\n\n"
                        f"{presc}\n\n{HEALTHCARE_DISCLAIMER}"
                    )
                else:
                    prescription = (
                        "PRESCRIPTION\n\n"
                        "Answer based on: Groq API\n\n"
                        "Consult healthcare professional for specific medication.\n\n"
                        f"{HEALTHCARE_DISCLAIMER}"
                    )

                status.write(f"⏱️ Timing: total={(time.time()-t0):.2f}s")

            else:
                # Existing Book (RAG) flow
                status.write("📚 Verifying medical library...")
                pdfs = _list_pdf_books()
                pdf_names = [os.path.basename(p) for p in pdfs] if pdfs else []
                if pdfs:
                    st.write(f"**Loaded Books ({len(pdfs)}):** {', '.join(pdf_names)}")
                else:
                    st.warning("⚠️ No Books found in 'Books/' folder.")

                # Audio
                if audio_input is not None:
                    status.write("🎤 Transcribing audio...")
                    if not GROQ_API_KEY:
                        st.warning("Voice transcription requires GROQ_API_KEY. Using offline mode.")
                    else:
                        temp_audio = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(audio_input.name)[-1])
                        temp_audio.write(audio_input.read())
                        temp_audio.close()
                        audio_path = temp_audio.name
                        text_input = transcribe_with_groq(
                            stt_model="whisper-large-v3",
                            audio_filepath=audio_path,
                            GROQ_API_KEY=GROQ_API_KEY,
                        )
                        os.remove(audio_path)

                # Image
                image_base64 = None
                if image_input is not None:
                    status.write("🖼️ Processing image...")
                    temp_image = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(image_input.name)[-1])
                    temp_image.write(image_input.read())
                    temp_image.close()
                    image_base64 = encode_image(temp_image.name)
                    os.remove(temp_image.name)

                diagnosis = None
                prescription = None
                language_code = LANGUAGE_CODES.get(response_language, "en")
                duration_val = st.session_state.get("duration_days_general", 0)

                if text_input and not image_base64:
                    status.write("📚 Searching medical books (RAG)...")
                    status.write(f"🧠 Mode selected: {'Groq' if USE_GROQ_GENERATION else 'Ollama'}")
                    status.write(f"🔑 GROQ_API_KEY detected: {'Yes' if GROQ_API_KEY else 'No'}")

                    retrieval_query = _build_retrieval_query(text_input, earlier_symptoms, duration_val)
                    question = (
                        f"{retrieval_query}\n\n"
                        "Question: Based on the book, provide (1) likely diagnosis/explanation and (2) brief guidance."
                    )

                    # Timing
                    t_embed0 = time.time()
                    _ = _embed_query_cached(retrieval_query)
                    t_embed = time.time() - t_embed0

                    t_rag0 = time.time()
                    answer, citations, _ctx = rag_answer_from_book(question, retrieval_query=retrieval_query)
                    t_rag = time.time() - t_rag0

                    # If book can't answer OR is insufficient and augmentation is enabled, ask Groq API as a second section
                    api_aug = _augment_with_api_if_needed(text_input, earlier_symptoms, duration_val, answer)

                    status.write(
                        "⏱️ Timing: "
                        f"embed={t_embed:.2f}s | rag_total={t_rag:.2f}s | total={(time.time()-t0):.2f}s"
                    )

                    if answer == REQUIRED_IDK:
                        status.warning("Could not find specific info in the provided books.")
                    else:
                        status.write("✅ Found relevant information in books.")

                    sources_line = f"Sources: {', '.join(citations) if citations else 'N/A'}"
                    book_block = (
                        f"From Book: {', '.join(pdf_names) if pdfs else 'your medical books'}\n"
                        f"{sources_line}\n\n"
                        f"{answer}"
                    )

                    if api_aug:
                        diagnosis_body = (
                            f"{book_block}\n\n"
                            "---\n"
                            "From Groq API (general info, not from the book):\n"
                            f"{api_aug}"
                        )
                    else:
                        diagnosis_body = book_block

                    diagnosis = f"{diagnosis_body}\n\n{HEALTHCARE_DISCLAIMER}"

                    # Prescription: if augmentation provided, generate from API; else from book (if possible)
                    if generate_prescription_mode:
                        if api_aug:
                            status.write("💊 Generating prescription (Groq API augmentation)...")
                            presc = _generate_prescription_groq_api_only(api_aug)
                            prescription = (
                                "PRESCRIPTION\n\n"
                                "Answer based on: Groq API (augmentation)\n\n"
                                f"{presc}\n\n{HEALTHCARE_DISCLAIMER}"
                            )
                        elif _ctx.strip() and answer != REQUIRED_IDK:
                            status.write("💊 Generating prescription (book-only)...")
                            presc_body = _generate_prescription_book_only(answer, _ctx)
                            prescription = (
                                "PRESCRIPTION\n\n"
                                f"Answer based on: {', '.join(pdf_names) if pdfs else 'your medical books'}\n"
                                f"{sources_line}\n\n"
                                f"{presc_body}\n\n{HEALTHCARE_DISCLAIMER}"
                            )
                        else:
                            prescription = (
                                "PRESCRIPTION\n\n"
                                "Consult healthcare professional for specific medication.\n\n"
                                f"{HEALTHCARE_DISCLAIMER}"
                            )

                elif image_base64:
                    status.write("📚 Searching medical books for image condition...")
                    status.write(f"🧠 Mode: {'Groq' if USE_GROQ_GENERATION else 'Ollama'} (book-only)")

                    retrieval_query = _build_retrieval_query(text_input or "visible symptoms", earlier_symptoms or "", duration_val)
                    image_question = (
                        f"{retrieval_query}\n\n"
                        "Question: Based on the book, what condition does this match? What is the diagnosis, symptoms, treatment, and medication recommendations?"
                    )

                    t_embed0 = time.time()
                    _ = _embed_query_cached(retrieval_query)
                    t_embed = time.time() - t_embed0

                    t_rag0 = time.time()
                    answer, citations, _ctx = rag_answer_from_book(image_question, retrieval_query=retrieval_query)
                    t_rag = time.time() - t_rag0

                    status.write(
                        "⏱️ Timing: "
                        f"embed={t_embed:.2f}s | rag_total={t_rag:.2f}s | total={(time.time()-t0):.2f}s"
                    )

                    if answer == REQUIRED_IDK:
                        status.warning("Could not find specific info in the provided books.")
                        diagnosis = f"{answer}\n\n{HEALTHCARE_DISCLAIMER}"
                        prescription = f"Prescription: {REQUIRED_IDK}\n\nConsult a healthcare professional for medication recommendations."
                    else:
                        status.write("✅ Found relevant information in books.")
                        diagnosis = f"{answer}\n\nSources: {', '.join(citations) if citations else 'N/A'}\n\n{HEALTHCARE_DISCLAIMER}"
                        prescription = (
                            f"PRESCRIPTION\n\nBased on diagnosis: {answer[:150]}...\n\n"
                            "Consult healthcare professional for specific medication.\n\n"
                            f"{HEALTHCARE_DISCLAIMER}"
                        )

                        if generate_prescription_mode and _ctx.strip() and answer != REQUIRED_IDK:
                            status.write("💊 Generating prescription (book-only)...")
                            presc_body = _generate_prescription_book_only(answer, _ctx)
                            prescription = f"PRESCRIPTION\n\n{presc_body}\n\n{HEALTHCARE_DISCLAIMER}"
                        else:
                            prescription = (
                                f"PRESCRIPTION\n\nBased on diagnosis: {answer[:150]}...\n\n"
                                "Consult healthcare professional for specific medication.\n\n"
                                f"{HEALTHCARE_DISCLAIMER}"
                            )

            st.markdown("---")
            st.markdown("## 📋 Diagnosis Results")
            st.text_area(
                "Input Summary",
                value=str(text_input) if text_input else "Image analysis",
                height=80,
                disabled=True,
                label_visibility="collapsed",
            )
            st.markdown("<div class='section-title'>🩺 Detailed Diagnosis</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='diagnosis-card'>{diagnosis or ''}</div>", unsafe_allow_html=True)

            # Footer/caption: use correct wording for API-only mode
            if use_api_only:
                st.caption("ℹ️ Answer based on: Groq API | Powered by Groq")
            else:
                powered = "Groq" if USE_GROQ_GENERATION else "Ollama"
                st.caption(
                    f"ℹ️ Answer based on your medical books: {', '.join(pdf_names) if pdfs else 'No books found'} | Powered by {powered}"
                )

            st.markdown("<div class='section-title'>💊 Prescription</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='prescription-card'>{prescription or ''}</div>", unsafe_allow_html=True)

            status.update(label="Diagnosis Complete!", state="complete", expanded=False)

            audio_bytes = None
            if enable_audio and gTTS is None:
                st.warning("Audio is enabled but gTTS is not installed. Install with: pip install gTTS")
            if enable_audio and gTTS is not None and diagnosis and prescription:
                with st.spinner("Generating audio..."):
                    full_text_for_audio = f"Diagnosis: {diagnosis}. Prescription: {prescription}"
                    audio_bytes = generate_audio_from_text(full_text_for_audio, language_code)

            st.markdown("<div class='section-title'>🎧 Audio Diagnosis</div>", unsafe_allow_html=True)
            if enable_audio and audio_bytes:
                st.audio(audio_bytes, format="audio/mp3")
            elif enable_audio and not audio_bytes:
                st.info("Audio enabled but not available (gTTS may be offline).")
            else:
                st.info("Audio generation is disabled.")

        except Exception as e:
            st.error(f"❌ Error during processing: {str(e)}")
            st.error(f"If this is your first run, add PDFs under {BOOKS_DIR} and make sure Ollama is running.")
