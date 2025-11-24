# scripts/run_agent.py
"""
RAG agent (LangChain + Chroma + Groq).
- Updated for Groq's current model names (llama-3.1-*).
- Uses .invoke() to avoid LangChain deprecation warnings.
"""
import os
from typing import List
from dotenv import load_dotenv

from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from utils import get_chroma_store

try:
    from langchain_groq import ChatGroq  # preferred LLM
except Exception:
    ChatGroq = None

# ---- Config ----
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "...")

# Default to a supported Groq model; override with GROQ_MODEL env var
MODEL_NAME = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")  # alt: "llama-3.1-70b-versatile"

QA_TEMPLATE = """You are a precise assistant. Use ONLY the provided context to answer.
If the answer is not in the context, say you don't know.

Context:
{context}

Question:
{question}

Answer in the user's language. Cite inline like [title #chunk_id] when applicable."""
qa_prompt = PromptTemplate(template=QA_TEMPLATE, input_variables=["context", "question"])

DOCUMENT_PROMPT = PromptTemplate(
    input_variables=["page_content", "title", "chunk_id", "source"],
    template="[Title: {title} | Chunk: {chunk_id} | Source: {source}]\n{page_content}\n",
)

def print_sources(sources: List) -> None:
    if not sources:
        print("📚 Sources: (none)")
        return
    print("📚 Sources:")
    for i, doc in enumerate(sources, 1):
        m = doc.metadata or {}
        print(f"  {i}. {m.get('title','?')}  (# {m.get('chunk_id','?')})  —  {m.get('source','')}")

def build_llm():
    # WHY: Groq's current client expects `model=...`
    if ChatGroq is None:
        # OpenAI-compatible fallback via Groq endpoint
        from langchain_community.chat_models import ChatOpenAI
        os.environ["OPENAI_API_KEY"] = GROQ_API_KEY
        os.environ["OPENAI_API_BASE"] = "https://api.groq.com/openai/v1"
        return ChatOpenAI(temperature=0, model=MODEL_NAME)
    return ChatGroq(temperature=0, model=MODEL_NAME, api_key=GROQ_API_KEY)

def run_agent():
    print("\n" + "=" * 80)
    print(f"🚀 RAG Agent: Chroma + {MODEL_NAME}")
    print("=" * 80 + "\n")

    vectorstore = get_chroma_store()
    retriever = vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 4})

    llm = build_llm()
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        chain_type_kwargs={
            "prompt": qa_prompt,
            "document_prompt": DOCUMENT_PROMPT,
            "document_variable_name": "context",
        },
        return_source_documents=True,
    )

    print("✅ Agent ready. Type your questions (`exit` to quit).")
    while True:
        q = input("\n Question: ").strip()
        if q.lower() in {"exit", "quit", "q"}:
            print(" Bye")
            break
        if not q:
            continue
        try:
            # use .invoke to avoid deprecation warning
            out = qa_chain.invoke({"query": q})
            answer = out.get("result") or out.get("answer") or " No answer."
            print("\n Answer:\n")
            print(answer)
            print()
            print_sources(out.get("source_documents", []))
            print("\n" + "=" * 80)
        except Exception as e:
            print(f"⚠️ Error: {e}")
            print("=" * 80)

if __name__ == "__main__":
    run_agent()
