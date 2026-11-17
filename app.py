"""Streamlit UI for the AI & Machine Learning Learning Assistant (RAG + free LLM)."""

from dataclasses import asdict
from pathlib import Path

import streamlit as st

from rag import config
from rag.llm import get_llm
from rag.loader import SUPPORTED_EXTENSIONS
from rag.pipeline import RAGPipeline
from rag.prompts import NOT_FOUND
from rag.vector_store import build_session_index, get_embeddings

st.set_page_config(page_title="AI & ML Learning Assistant", page_icon="🎓", layout="wide")
CSS = """
:root { --card-bg: rgba(128,128,128,0.08); --card-border: rgba(128,128,128,0.25); }

/* soft color glow behind the page */
[data-testid="stAppViewContainer"] {
  background-image:
    radial-gradient(circle at 0% 0%, rgba(108,92,231,0.38), transparent 45%),
    radial-gradient(circle at 100% 8%, rgba(0,184,217,0.32), transparent 42%),
    radial-gradient(circle at 85% 100%, rgba(255,121,198,0.28), transparent 45%),
    radial-gradient(circle at 5% 95%, rgba(253,203,110,0.25), transparent 40%);
  background-attachment: fixed;
}
[data-testid="stHeader"], [data-testid="stBottom"], [data-testid="stBottom"] > div {
  background: transparent !important;
}

[data-testid="stDeployButton"], footer { display: none !important; }

/* hero banner */
.hero { padding: 1.6rem 1.9rem; border-radius: 20px; margin-bottom: 1.2rem;
  background: linear-gradient(120deg, #6C5CE7 0%, #8E6CF0 45%, #00B8D9 100%);
  box-shadow: 0 12px 32px rgba(108,92,231,0.30); }
.hero h1 { margin: 0; padding: 0; font-size: 2rem; color: #fff !important; }
.hero p { margin: .5rem 0 0; color: #fff !important; opacity: .93; }
.hero .chips span { display: inline-block; margin: .8rem .5rem 0 0; padding: .25rem .8rem;
  border-radius: 999px; background: rgba(255,255,255,0.2); color: #fff; font-size: .8rem; }
.welcome { padding: 1rem 1.3rem; border-radius: 16px; border: 1px dashed var(--card-border);
  background: var(--card-bg); margin-bottom: 1rem; }

/* chat bubbles */
[data-testid="stChatMessage"], .welcome { border: 1px solid var(--card-border);
  background: rgba(128,128,128,0.12); backdrop-filter: blur(10px);
  box-shadow: 0 4px 16px rgba(0,0,0,0.08);
  border-radius: 16px; padding: 1rem 1.2rem; margin-bottom: .8rem; }
  
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
  align-self: flex-start; width: fit-content; max-width: 60%; min-width: 220px;
  background: linear-gradient(120deg, #4338CA, #7C3AED); border: none;
  border-radius: 18px; box-shadow: 0 6px 18px rgba(67,56,202,0.40); }
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) p,
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) span,
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) [data-testid="stMarkdownContainer"] {
  color: #ffffff !important; font-weight: 500; font-size: 1.05rem; }
  
[data-testid="stChatInput"] { border-radius: 16px; box-shadow: 0 4px 18px rgba(108,92,231,0.20); }

/* sidebar */
[data-testid="stSidebar"] { border-right: 1px solid var(--card-border);
    background-image: linear-gradient(180deg, rgba(108,92,231,0.34), rgba(0,184,217,0.22)); }

/* buttons, alerts, expanders */
.stButton > button { border-radius: 12px; border: 1px solid var(--card-border); transition: all .15s ease; }
.stButton > button:hover { transform: translateY(-1px); border-color: #6C5CE7;
  box-shadow: 0 4px 14px rgba(108,92,231,0.28); }
button[kind="primary"], [data-testid="stBaseButton-primary"] {
  background-image: linear-gradient(120deg, #6C5CE7, #00B8D9); border: none; color: #fff; }
[data-testid="stAlert"], [data-testid="stExpander"] { border-radius: 12px; }
"""
st.markdown("<style>" + CSS + "</style>", unsafe_allow_html=True)


SUGGESTED_QUESTIONS = [
    "What is Artificial Intelligence?",
    "What is Machine Learning?",
    "Explain supervised learning with an example.",
    "What is the difference between classification and regression?",
    "Explain overfitting and underfitting.",
    "What is Random Forest?",
    "Explain model training and testing.",
    "What is accuracy, precision, recall, and F1-score?",
]


# ---------- cached resources (loaded once, reused across reruns) ----------
@st.cache_resource(show_spinner="Loading embedding model…")
def load_embeddings():
    return get_embeddings()



@st.cache_resource(show_spinner="Loading language model (first time can take a few minutes)…")
def load_llm(backend: str, model: str):
    return get_llm(backend, model)


# ---------- helpers ----------
def build_my_knowledge_base(files) -> None:
    """Index the files this visitor uploaded. Stored only in THIS browser session's memory."""
    payload = [(f.name, f.getvalue()) for f in files]
    with st.spinner("Reading your documents and building your private index…"):
        try:
            db, stats = build_session_index(load_embeddings(), payload)
        except ValueError as exc:
            st.error(str(exc))
            return
    st.session_state.vector_db = db
    st.session_state.kb_stats = stats
    for err in stats["errors"]:
        st.warning(f"Skipped {err}")


def render_sources(sources: list[dict]) -> None:
    """Show a bulleted list of unique file + page/slide, no snippets or scores."""
    labels = []
    for s in sources:
        if s["location"].startswith("Whole"):  # txt/docx have no page numbers
            label = s["file"]
        else:
            label = f"{s['file']} ({s['location']})"
        if label not in labels:
            labels.append(label)
    st.markdown("**Sources:**\n" + "\n".join(f"- {label}" for label in labels))


# ---------- state ----------
if "messages" not in st.session_state:
    st.session_state.messages = []


# ---------- sidebar ----------
with st.sidebar:
    st.header("⚙️ Settings")

    backend = config.LLM_BACKEND  # "groq"
    
    labels = list(config.GROQ_MODELS)
    default_index = next(
        (i for i, name in enumerate(labels) if config.GROQ_MODELS[name] == config.GROQ_MODEL), 0
    )
    choice = st.selectbox("Groq model (free tier)", labels, index=default_index)
    model = config.GROQ_MODELS[choice]
    st.caption(f"Model ID: `{model}`")
    
    temperature = st.slider("Temperature", 0.0, 1.0, config.TEMPERATURE, 0.05)
    top_k = st.slider("Chunks retrieved (top-k)", 1, 8, config.TOP_K)

    ready, status = load_llm(backend, model).check()
    (st.success if ready else st.error)(status)
    
    st.divider()
    st.header("📁 Your knowledge base")
    st.caption("🔒 Private: only you can see these files. They are deleted when you refresh or close this page.")
    uploaded = st.file_uploader(
        "Upload your learning materials",
        type=[ext.strip(".") for ext in sorted(SUPPORTED_EXTENSIONS)],
        accept_multiple_files=True,
    )
    if st.button("Build my knowledge base", type="primary", use_container_width=True):
        if uploaded:
            build_my_knowledge_base(uploaded)
        else:
            st.warning("Please choose at least one file first.")

    stats = st.session_state.get("kb_stats")
    if stats:
        st.caption(f"{len(stats['files'])} files · {stats['chunks']} chunks indexed")
        with st.expander("Your indexed files"):
            for name in stats["files"]:
                st.write(f"• {name}")
        if st.button("🗑️ Delete my documents", use_container_width=True):
            st.session_state.pop("vector_db", None)
            st.session_state.pop("kb_stats", None)
            st.rerun()

    st.divider()
    with st.expander("💡 Suggested questions", expanded=True):
        for i, q in enumerate(SUGGESTED_QUESTIONS):
            if st.button(q, key=f"suggested_{i}", use_container_width=True):
                st.session_state.pending_question = q

    if st.button("🗑️ Clear chat", use_container_width=True):
        st.session_state.messages = []

# ---------- main page ----------
st.markdown(
    '<div class="hero"><h1>🎓 AI &amp; Machine Learning Learning Assistant</h1>'
    '<p>Ask questions about AI, Machine Learning, Data Science and Analytics. '
    'Answers come only from your uploaded learning materials.</p>'
    '<div class="chips"><span>📄 Your own materials</span><span>🔎 Source-backed answers</span>'
    '<span>⚡ Free Groq LLM</span></div></div>',
    unsafe_allow_html=True,
)

vector_db = st.session_state.get("vector_db")
if vector_db is None:
    st.info("Upload your own PDF, DOCX, PPTX or TXT files in the sidebar and click "
            "**Build my knowledge base**. Your files stay private to you.") 
if not st.session_state.messages:
    st.markdown('<div class="welcome">👋 <b>Welcome!</b> Type a question below, or pick a '
                'suggested question from the sidebar to get started.</div>', unsafe_allow_html=True)

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            render_sources(msg["sources"])

typed = st.chat_input("Type your question here…")
prompt = typed or st.session_state.pop("pending_question", None)

if prompt and prompt.strip():
    prompt = prompt.strip()
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        sources: list[dict] = []
        if vector_db is None:
            answer = "Please build the knowledge base first (see the sidebar)."
            st.markdown(answer)
        else:
            pipeline = RAGPipeline(vector_db, load_llm(backend, model))
            response = pipeline.ask(prompt, top_k=top_k, temperature=temperature)
            if response.stream is not None:
                answer = str(st.write_stream(response.stream))
                failed = "The language model failed" in answer
                if NOT_FOUND not in answer and not failed:
                    sources = [{**asdict(s), "snippet": s.snippet} for s in response.sources]
            else:
                answer = response.text
                st.markdown(answer)
            if sources:
                render_sources(sources)

    st.session_state.messages.append({"role": "assistant", "content": answer, "sources": sources})
