from langchain_core.documents import Document
from langchain.chat_models import init_chat_model
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.prompts import ChatPromptTemplate
from src.rag_agent.config import vectorstore as myvectorstore
import os

# --- Context Holder ---
class RAGContextHolder:
    def __init__(self):
        self.last_retrieved_docs = []
        self.last_web_results = []

    def set_docs(self, docs: list[Document]):
        self.last_retrieved_docs = docs

    def set_web(self, docs: list[Document]):
        self.last_web_results = docs

    def get_all(self):
        return self.last_retrieved_docs + self.last_web_results

context_holder = RAGContextHolder()

# --- LLM init ---
os.environ["GOOGLE_API_KEY"] = "AIzaSyBcz8WPGydT8R9Tj7M8BJ7_Ge9djK9sBzc"
model = init_chat_model("google_genai:gemini-2.5-flash")

# --- Web Search Tool ---
tavily = TavilySearchResults(max_results=3)   # 검색 결과 3개 가져오기

def web_search(query: str) -> list[Document]:
    """Tavily 검색 결과를 Document 리스트로 변환"""
    try:
        results = tavily.invoke({"query": query})
    except Exception as e:
        print("Tavily Error:", e)
        return []

    docs = []
    for r in results:
        content = f"Title: {r.get('title')}\nURL: {r.get('url')}\n\nSnippet: {r.get('content')}"
        docs.append(Document(page_content=content))
    return docs


# --- Hybrid RAG Query ---
def rag_query(query: str, top_k: int = 3, use_web=True):
    
    # 1) Rewrite query (optional)
    rewrite_system_msg = "Rewrite the user's question for vector search."
    rewrite_template = ChatPromptTemplate(
        [("system", rewrite_system_msg), ("human", "{user_input}")]
    )
    rewrite_prompt = rewrite_template.invoke({"user_input": query})
    rewritten_query = model.invoke(rewrite_prompt.messages).content

    # 2) Vector search
    vec_docs = myvectorstore.similarity_search(rewritten_query, k=top_k)
    context_holder.set_docs(vec_docs)

    # 3) Web search (optional)
    web_docs = []
    if use_web:
        web_docs = web_search(query)
        context_holder.set_web(web_docs)

    # 4) Combine all context
    all_docs = vec_docs + web_docs
    docs_text = "\n\n".join(doc.page_content for doc in all_docs)

    # 5) LLM answer
    final_prompt = (
        "You are a helpful AI assistant.\n\n"
        "Use the following information from documents and the web.\n\n"
        f"{docs_text}\n\n"
        f"Question: {query}\n"
        "Answer:"
    )

    answer = model.invoke(final_prompt)
    return answer, vec_docs, web_docs


# --- 실행 ---
if __name__ == "__main__":
    query = "How do I set up a musculoskeletal model in OpenSim?"
    answer, vec_docs, web_docs = rag_query(query)

    print("=== Answer ===")
    print(answer)

    print("\n=== Vector Docs ===")
    for d in vec_docs:
        print(d.page_content[:200], "...")

    print("\n=== Web Search Results ===")
    for d in web_docs:
        print(d.page_content[:200], "...")
