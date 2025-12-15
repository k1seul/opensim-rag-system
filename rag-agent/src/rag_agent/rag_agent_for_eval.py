import os
import json
from typing import List, Tuple, Optional

from langchain_core.documents import Document
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from src.rag_agent.config import vectorstore as myvectorstore

# ✅ NEW: langchain-tavily
from langchain_tavily import TavilySearch, TavilyExtract


def _is_korean(text: str) -> bool:
    return any("\uac00" <= ch <= "\ud7a3" for ch in text)


def _try_parse_json_doc(text: str) -> Optional[dict]:
    """벡터스토어에 JSON 문자열이 그대로 들어간 경우(url/title/text) 파싱 시도"""
    text = text.strip()
    if not (text.startswith("{") and '"url"' in text and '"text"' in text):
        return None
    try:
        return json.loads(text)
    except Exception:
        return None


def _dedupe_docs(docs: List[Document]) -> List[Document]:
    """page_content 기준 단순 중복 제거"""
    seen = set()
    out = []
    for d in docs:
        key = (d.page_content or "").strip()
        if not key:
            continue
        if key in seen:
            continue
        seen.add(key)
        out.append(d)
    return out


def _pretty_source(d: Document) -> str:
    md = d.metadata or {}
    # web은 url이 있으면 url을 우선 표시
    if md.get("source") == "web" and md.get("url"):
        return md["url"]
    # vector는 metadata에 source/url/title이 없을 수 있으니 안전 fallback
    return md.get("source") or md.get("url") or md.get("title") or "vector"


class RAGAgentForEval:
    def __init__(self, google_api_key: Optional[str] = None):
        # --- API Key (Gemini) ---
        if google_api_key:
            os.environ["GOOGLE_API_KEY"] = google_api_key

        # --- LLM (query translate/rewrite에만 사용) ---
        # 필요하면 flash로 교체:
        # self.model = init_chat_model("google_genai:gemini-2.5-flash")
        self.model = init_chat_model("google_genai:gemini-2.5-pro")

        # --- Vectorstore ---
        self.vectorstore = myvectorstore

        # --- Tavily (Search + Extract) ---
        # Tavily는 TAVILY_API_KEY env 필요
        self.tavily_search = TavilySearch(
            max_results=3,
            search_depth="basic",
            include_answer=False,
            include_raw_content="markdown",  # 가능하면 본문을 같이 받기
        )
        self.tavily_extract = TavilyExtract(
            extract_depth="basic",
            format="markdown",
        )

    def _translate_if_needed(self, query: str) -> str:
        if not _is_korean(query):
            return query

        translation_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", "Translate the user's Korean question into a concise English query that is optimized for vector search."),
                ("human", "{user_input}"),
            ]
        )
        tp = translation_prompt.invoke({"user_input": query})
        out = self.model.invoke(tp.messages).content.strip()
        return out

    def _rewrite_for_vector(self, query_en: str) -> str:
        rewrite_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", "Rewrite the user's question for vector search. Only output the rewritten query."),
                ("human", "{user_input}"),
            ]
        )
        rp = rewrite_prompt.invoke({"user_input": query_en})
        resp = self.model.invoke(rp.messages)

        content = resp.content
        if isinstance(content, list):
            # Gemini / LC 최신 포맷 대응
            content = "".join(
                part.get("text", "") for part in content if isinstance(part, dict)
            )

        out = content.strip()
        return out
    
    def _make_search_query(self, original_query: str) -> str:
        prompt = ChatPromptTemplate.from_messages([
            ("system",
            "Convert the user's question into ONE concise English query optimized for vector search. "
            "If it is already English, rewrite it for vector search. "
            "Only output the final query."),
            ("human", "{q}")
        ])
        msg = prompt.invoke({"q": original_query})
        out = self.model.invoke(msg.messages).content.strip()
        return out

    def web_search(self, query: str, force_extract: bool = True) -> List[Document]:
        """TavilySearch 결과를 Document로 만들고, 필요하면 Extract로 본문을 확실히 채움"""
        try:
            res = self.tavily_search.invoke({"query": query})
        except Exception as e:
            print("Tavily Search Error:", e)
            return []

        if isinstance(res, dict):
            results = res.get("results") or res.get("data") or []
        elif isinstance(res, list):
            results = res
        else:
            results = []

        docs: List[Document] = []

        urls = []
        for r in results:
            if not isinstance(r, dict):
                continue
            url = r.get("url")
            title = r.get("title")
            snippet = r.get("content") or ""
            raw = r.get("raw_content") or ""

            urls.append(url) if url else None

            body = raw if raw else snippet
            docs.append(
                Document(
                    page_content=f"Title: {title}\nURL: {url}\n\n{body}",
                    metadata={
                        "source": "web",
                        "url": url,
                        "title": title,
                        "has_raw_content": bool(raw),
                    },
                )
            )

        # (선택) Extract로 raw_content를 더 확실히 채우기
        if force_extract and urls:
            try:
                ex = self.tavily_extract.invoke({"urls": urls})
                ex_results = ex.get("results", []) if isinstance(ex, dict) else []
                url2raw = {}
                for x in ex_results:
                    raw = x.get("raw_content") or x.get("content") or x.get("text") or x.get("markdown")
                    if isinstance(x, dict) and x.get("url") and raw:
                        url2raw[x["url"]] = raw

                for d in docs:
                    url = (d.metadata or {}).get("url")
                    if url and url in url2raw:
                        title = (d.metadata or {}).get("title")
                        d.page_content = f"Title: {title}\nURL: {url}\n\n{url2raw[url]}"
                        d.metadata["has_raw_content"] = True
            except Exception as e:
                print("Tavily Extract Error:", e)

        return _dedupe_docs(docs)

    def vector_search(self, rewritten_query: str, top_k: int = 3) -> List[Document]:
        raw = self.vectorstore.similarity_search(rewritten_query, k=top_k*5)
        raw = _dedupe_docs(raw)
        docs = raw[:top_k]

        # 벡터 문서에 JSON이 박혀있으면 보기 좋게 정리(선택)
        cleaned = []
        for d in docs:
            parsed = _try_parse_json_doc(d.page_content or "")
            if parsed:
                url = parsed.get("url")
                title = parsed.get("title")
                text = parsed.get("text") or ""
                cleaned.append(
                    Document(
                        page_content=f"Title: {title}\nURL: {url}\n\n{text}",
                        metadata={**(d.metadata or {}), "source": "vector", "url": url, "title": title},
                    )
                )
            else:
                cleaned.append(
                    Document(
                        page_content=d.page_content,
                        metadata={**(d.metadata or {}), "source": (d.metadata or {}).get("source", "vector")},
                    )
                )


        return _dedupe_docs(cleaned)
    
    def _format_sources_with_index(self, vec_docs: List[Document], web_docs: List[Document]) -> str:
        all_docs = list(vec_docs) + list(web_docs)
        blocks = []
        for i, d in enumerate(all_docs, 1):
            blocks.append(f"[{i}]\n{d.page_content}")
        return "\n\n---\n\n".join(blocks)

    def generate_answer(self, original_query: str, vec_docs: List[Document], web_docs: List[Document]) -> str:
        sources_text = self._format_sources_with_index(vec_docs, web_docs)

        prompt = (
            "You are a highly specialized OpenSim expert and technical guide. Your primary goal is to provide precise, actionable, and comprehensive answers to user queries related to OpenSim modeling, simulation, analysis, and data processing.\n"
            "Answer in the same language as the user's question.\n\n"
            "--- GUIDANCE ON ANSWERING ---\n"
            "1. **Technical Precision:** Use correct OpenSim terminology (e.g., Inverse Kinematics, Coordinates, Actuator, .osim, .trc) throughout your response.\n"
            "2. **Structure & Clarity:** For procedural questions (Setup, Conversion, Modification), present the answer in numbered or bulleted **Step-by-step instructions**.\n"
            "3. **Practicality:** Include practical advice, common pitfalls, or best practices relevant to the OpenSim workflow (e.g., recommended residual limits, error reduction tips).\n"
            "4. **Comparison Format:** For comparison questions (e.g., SO vs. CMC, MocoTrack vs. MocoInverse), use a **Table** format for clear differentiation.\n"
            "5. **Citation Rule (MANDATORY):** ***Crucially, you MUST cite your source by putting the corresponding index number in brackets [1] at the end of the sentence or fact that used the information.***\n\n"
            "----------------------------\n\n"
            f"Sources:\n\n{sources_text}\n\n"
            f"Question: {original_query}\n"
            "Answer:"
        )
        return self.model.invoke(prompt).content.strip()


    def retrieve_only(self, query: str, top_k: int = 3, use_web: bool = True, force_extract: bool = True) -> Tuple[List[Document], List[Document]]:
        original_query = query.strip()

        query_for_search = self._translate_if_needed(original_query)
        print(f"DEBUG: Translated Query: {query_for_search}")

        rewritten_query = self._rewrite_for_vector(query_for_search)
        print(f"DEBUG: Rewritten Query: {rewritten_query}")

        vec_docs = self.vector_search(rewritten_query, top_k=top_k)

        web_docs = []
        if use_web:
            web_docs = self.web_search(query_for_search, force_extract=force_extract)

        return vec_docs, web_docs
    
    def query(
        self,
        query: str,
        top_k: int = 3,
        use_web: bool = True,
        force_extract: bool = True,
        return_answer: bool = True,
    ) -> Tuple[List[Document], List[Document], Optional[str]]:

        original_query = query.strip()

        query_for_search = self._translate_if_needed(original_query)
        print(f"DEBUG: Translated Query: {query_for_search}")

        rewritten_query = self._rewrite_for_vector(query_for_search)
        print(f"DEBUG: Rewritten Query: {rewritten_query}")

        vec_docs = self.vector_search(rewritten_query, top_k=top_k)

        web_docs = []
        if use_web:
            web_docs = self.web_search(query_for_search, force_extract=force_extract)

        answer = None
        if return_answer:
            answer = self.generate_answer(original_query, vec_docs, web_docs)

        return answer, vec_docs, web_docs


if __name__ == "__main__":
    # ✅ Tavily 키가 잡혔는지 확인(키 자체는 출력하지 않음)
    print("TAVILY_API_KEY set?", bool(os.getenv("TAVILY_API_KEY")))
    print("GOOGLE_API_KEY set?", bool(os.getenv("GOOGLE_API_KEY")))

    agent = RAGAgentForEval()

    q = input("Question: ").strip()
    vec_docs, web_docs = agent.retrieve_only(q, top_k=3, use_web=True, force_extract=True)

    print("\n📚 Vectorstore")
    for i, d in enumerate(vec_docs, 1):
        print(f"\n[{i}] {_pretty_source(d)}\n{d.page_content}\n")

    print("\n🌐 Web Search")
    if not web_docs:
        print("(no web results)")
    for i, d in enumerate(web_docs, 1):
        print(f"\n[{i}] {_pretty_source(d)}\n{d.page_content}\n")
