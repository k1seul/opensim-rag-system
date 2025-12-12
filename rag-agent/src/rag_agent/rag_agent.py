from langchain_core.documents import Document
from langchain.chat_models import init_chat_model
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.prompts import ChatPromptTemplate
from src.rag_agent.config import vectorstore as myvectorstore
import os

class RAGContextHolder:
    def __init__(self):
        self.last_retrieved_docs = []
        self.last_web_results = []

    def set_docs(self, docs):
        self.last_retrieved_docs = docs

    def set_web(self, docs):
        self.last_web_results = docs

    def get_all(self):
        return self.last_retrieved_docs + self.last_web_results


class RAGAgent:
    def __init__(self, api_key: str = None):
        # --- Context Holder ---
        self.context_holder = RAGContextHolder()

        # --- API Keys ---
        if api_key:
            os.environ["GOOGLE_API_KEY"] = api_key

        # --- LLM ---
        self.model = init_chat_model("google_genai:gemini-2.5-pro")

        # --- Search Tools ---
        self.vectorstore = myvectorstore
        self.tavily = TavilySearchResults(max_results=3)

    # -------------------------------
    # Utility Function: Lang Detect
    # -------------------------------
    def _is_korean(self, text: str):
        # 간단한 한국어 포함 여부 체크 (실제 환경에서는 langid 등 사용 권장)
        return any('\uac00' <= char <= '\ud7a3' for char in text)

    # -------------------------------
    # Tavily Web Search
    # -------------------------------
    def web_search(self, query: str):
        try:
            results = self.tavily.invoke({"query": query})
        except Exception as e:
            print("Tavily Error:", e)
            return []

        docs = []
        for r in results:
            content = f"Title: {r.get('title')}\nURL: {r.get('url')}\n\nSnippet: {r.get('content')}"
            docs.append(Document(page_content=content))
        return docs

    # -------------------------------
    # Main RAG Pipeline
    # -------------------------------
    def query(self, query: str, top_k: int = 3, use_web: bool = True):
        original_query = query # 원본 쿼리를 저장

        # 1-1) Query Translation (한국어 -> 영어)
        if self._is_korean(query):
            translation_prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", "Translate the user's Korean question into a concise English query that is optimized for vector search."),
                    ("human", "{user_input}")
                ]
            )
            tp = translation_prompt.invoke({"user_input": original_query})
            query_for_search = self.model.invoke(tp.messages).content
            print(f"DEBUG: Translated Query: {query_for_search}")
        else:
            query_for_search = original_query

        # 1-2) Rewrite query (영문 쿼리에 대한 재작성)
        rewrite_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", "Rewrite the user's question for vector search. Only output the rewritten query."),
                ("human", "{user_input}")
            ]
        )
        rp = rewrite_prompt.invoke({"user_input": query_for_search})
        rewritten_query = self.model.invoke(rp.messages).content

        # 2) Vector search
        vec_docs = self.vectorstore.similarity_search(rewritten_query, k=top_k)
        self.context_holder.set_docs(vec_docs)

        # 3) Web search
        web_docs = []
        if use_web:
            # 웹 검색은 원본 쿼리(original_query) 또는 번역된 쿼리(query_for_search) 사용 가능
            web_docs = self.web_search(query_for_search) 
            self.context_holder.set_web(web_docs)

        # 4) Combine all docs & Add Citation Index
        all_docs = vec_docs + web_docs
        
        # 인덱스 번호를 붙여 프롬프트 문맥 생성
        docs_text_with_index = []
        for i, doc in enumerate(all_docs):
            # [1], [2], ... 와 같이 인덱스 번호를 붙입니다.
            docs_text_with_index.append(f"[{i+1}]: {doc.page_content}")
            
        docs_text = "\n\n---\n\n".join(docs_text_with_index)

        # 5) Answer with Citations
        final_prompt = (
            "You are a highly specialized OpenSim expert and technical guide. Your primary goal is to provide precise, actionable, and comprehensive answers to user queries related to OpenSim modeling, simulation, analysis, and data processing.\n"
            "Answer in the same language as the user's question.\n\n"
            "--- GUIDANCE ON ANSWERING ---\n"
            "1. **Technical Precision:** Use correct OpenSim terminology (e.g., Inverse Kinematics, Coordinates, Actuator, .osim, .trc) throughout your response.\n"
            "2. **Structure & Clarity:** For procedural questions (Setup, Conversion, Modification), present the answer in numbered or bulleted **Step-by-step instructions**.\n"
            "3. **Practicality:** Include practical advice, common pitfalls, or best practices relevant to the OpenSim workflow (e.g., recommended residual limits, error reduction tips).\n"
            "4. **Comparison Format:** For comparison questions (e.g., SO vs. CMC, MocoTrack vs. MocoInverse), use a **Table** format for clear differentiation.\n"
            "5. **Citation Rule (MANDATORY):** ***Crucially, you MUST cite your source by putting the corresponding index number in brackets [1] at the end of the sentence or fact that used the information.***\n\n"
            "----------------------------\n\n"
            f"Sources:\n\n{docs_text}\n\n"
            f"Question: {original_query}\n"
            "Answer:"
        )

        answer = self.model.invoke(final_prompt)

        return answer, vec_docs, web_docs