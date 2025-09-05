from flask import Flask, render_template, request, session, redirect, url_for
from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    Settings,
    StorageContext,
    load_index_from_storage,
)
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.llms import (
    LLM,
    ChatMessage,
    ChatResponse,
    CompletionResponse,
    LLMMetadata,
)
import requests
import json
import os
from typing import Optional, List, Iterator, AsyncIterator

# ------------------ Config ------------------
DATA_DIR = "data"                 # folder with your PDFs/TXT/MD/etc.
PERSIST_DIR = "storage"           # where the index will be saved
CHECK_FILE = os.path.join(PERSIST_DIR, ".last_built")  # timestamp cache

app = Flask(__name__)
app.secret_key = "supersecretkey"  # needed for sessions

# ✅ Your OpenRouter API key
OPENROUTER_API_KEY = os.getenv(
    "OPENROUTER_API_KEY",
    "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",
)
API_URL = "https://openrouter.ai/api/v1/chat/completions"

# ✅ Model you want to use
MODEL = "openai/gpt-oss-120b:free"

# ------------------ Custom LLM backed by OpenRouter ------------------
class OpenRouterLLM(LLM):
    """Custom LLM wrapper for OpenRouter API."""

    model: str = MODEL
    api_key: str = OPENROUTER_API_KEY

    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(
            context_window=4000,
            num_output=512,
            is_chat_model=True,
            model_name=self.model,
        )

    # ---- helpers ----
    def _post(self, messages: List[ChatMessage]) -> dict:
        payload = {
            "model": self.model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "max_tokens": 512
        }
        resp = requests.post(
            API_URL,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "http://localhost:5000",
                "X-Title": "Flask RAG App",
            },
            data=json.dumps(payload),
            timeout=60,
        )
        if resp.status_code != 200:
            raise RuntimeError(f"OpenRouter {resp.status_code}: {resp.text}")
        return resp.json()

    # ---- required sync APIs ----
    def chat(self, messages: List[ChatMessage], **kwargs) -> ChatResponse:
        data = self._post(messages)
        content = data["choices"][0]["message"]["content"]
        msg = ChatMessage(role="assistant", content=content)
        return ChatResponse(message=msg, raw=data)

    def complete(self, prompt: str, **kwargs) -> CompletionResponse:
        data = self._post([ChatMessage(role="user", content=prompt)])
        content = data["choices"][0]["message"]["content"]
        return CompletionResponse(text=content, raw=data)

    def stream_chat(self, messages: List[ChatMessage], **kwargs) -> Iterator[ChatResponse]:
        yield self.chat(messages, **kwargs)

    def stream_complete(self, prompt: str, **kwargs) -> Iterator[CompletionResponse]:
        yield self.complete(prompt, **kwargs)

    # ---- required async APIs ----
    async def achat(self, messages: List[ChatMessage], **kwargs) -> ChatResponse:
        return self.chat(messages, **kwargs)

    async def acomplete(self, prompt: str, **kwargs) -> CompletionResponse:
        return self.complete(prompt, **kwargs)

    async def astream_chat(self, messages: List[ChatMessage], **kwargs) -> AsyncIterator[ChatResponse]:
        yield self.chat(messages, **kwargs)

    async def astream_complete(self, prompt: str, **kwargs) -> AsyncIterator[CompletionResponse]:
        yield self.complete(prompt, **kwargs)

# ------------------ LLM + Embeddings ------------------
Settings.embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
Settings.llm = OpenRouterLLM()

# ------------------ Globals ------------------
_index: Optional[VectorStoreIndex] = None
_query_engine = None

# ------------------ Helpers ------------------
def _latest_mtime(folder: str) -> float:
    if not os.path.isdir(folder):
        return 0.0
    files = [os.path.join(folder, f) for f in os.listdir(folder)]
    files = [p for p in files if os.path.isfile(p)]
    if not files:
        return 0.0
    return max(os.path.getmtime(p) for p in files)

def _read_cached_build_time() -> float:
    if not os.path.exists(CHECK_FILE):
        return 0.0
    try:
        with open(CHECK_FILE, "r") as f:
            return float(f.read().strip())
    except Exception:
        return 0.0

def _write_cached_build_time(ts: float) -> None:
    os.makedirs(PERSIST_DIR, exist_ok=True)
    with open(CHECK_FILE, "w") as f:
        f.write(str(ts))

def _build_and_persist_index() -> VectorStoreIndex:
    print("🔄 Rebuilding index (embedding documents)…")
    documents = SimpleDirectoryReader(DATA_DIR).load_data()
    index = VectorStoreIndex.from_documents(documents)
    index.storage_context.persist(persist_dir=PERSIST_DIR)
    _write_cached_build_time(_latest_mtime(DATA_DIR))
    print("✅ Index built & persisted.")
    return index

def _load_index_from_disk() -> VectorStoreIndex:
    storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
    return load_index_from_storage(storage_context)

def get_index_and_engine():
    global _index, _query_engine
    data_mtime = _latest_mtime(DATA_DIR)
    cached_mtime = _read_cached_build_time()

    if _index is not None and data_mtime <= cached_mtime:
        return _index, _query_engine

    if os.path.isdir(PERSIST_DIR) and os.path.exists(CHECK_FILE) and data_mtime <= cached_mtime:
        print("📦 Loading index from disk cache…")
        _index = _load_index_from_disk()
    else:
        if not os.path.isdir(DATA_DIR) or _latest_mtime(DATA_DIR) == 0.0:
            print("⚠️ No files in /data.")
            _index = None
            _query_engine = None
            return _index, _query_engine
        _index = _build_and_persist_index()

    _query_engine = _index.as_query_engine(response_mode="compact")
    return _index, _query_engine

# ------------------ Response cleaning ------------------
def clean_answer(text: str) -> str:
    """Strip common model control tokens and whitespace."""
    if not isinstance(text, str):
        text = str(text)
    # tokens/models may use different markers — this covers common ones
    bad_tokens = [
        "<|start|>", "<|end|>", "<|assistant|>", "<|channel|>", "<|message|>",
        "|start|", "|end|", "final"
    ]
    for token in bad_tokens:
        text = text.replace(token, "")
    return text.strip()

# ------------------ Routes ------------------
@app.route("/", methods=["GET", "POST"])
def home():
    index, query_engine = get_index_and_engine()

    if "chat_history" not in session:
        session["chat_history"] = []

    if request.method == "POST":
        question = request.form.get("question", "").strip()
        if not question:
            return redirect(url_for("home"))

        if query_engine is None:
            answer = "No data found in /data folder."
        else:
            try:
                response = query_engine.query(question)
                raw = response.response if hasattr(response, "response") else response
                answer = clean_answer(raw)
            except Exception as e:
                answer = f"Query error: {e}"

        session["chat_history"].append({"q": question, "a": answer})
        session.modified = True
        return redirect(url_for("home"))

    return render_template("index.html", chat_history=session["chat_history"])



if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # Render sets PORT
    app.run(host="0.0.0.0", port=port)

