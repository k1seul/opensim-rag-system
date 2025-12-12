import gradio as gr
from src.rag_agent.pure_agent import PureAgent
from src.rag_agent.rag_agent import RAGAgent

# 두 개의 agent 완전히 분리
pure_agent = PureAgent()
rag_agent = RAGAgent()

# ============================================
# 1) Pure LLM 답변 (Retrieval 없이)
# ============================================
def chat_pure(messages, query):
    GREETINGS = [
        "안녕", "안녕하세요", "고마워", "감사합니다", "잘 가", "ㅎㅇ",
        "hi", "hello", "thanks", "thank you", "bye", "goodbye", "hey"
    ]
    
    normalized_query = query.strip().lower().replace(" ", "")
    messages.append({"role": "user", "content": query})

    # 인사 처리
    if normalized_query in GREETINGS or any(g in normalized_query for g in GREETINGS):
        answer = "안녕하세요! 저는 OpenSim 관련 전문 챗봇입니다. 어떻게 도와드릴까요?"
        messages.append({"role": "assistant", "content": answer})
        return messages, ""

    # 순수 LLM 답변
    answer = pure_agent.answer(query)
    messages.append({"role": "assistant", "content": answer})
    return messages, ""


# ============================================
# 2) RAG 답변 (Vector + Web Retrieval)
# ============================================
def chat_rag(messages, query):
    GREETINGS = [
        "안녕", "안녕하세요", "고마워", "감사합니다", "잘 가", "ㅎㅇ",
        "hi", "hello", "thanks", "thank you", "bye", "goodbye", "hey"
    ]

    normalized_query = query.strip().lower().replace(" ", "")
    messages.append({"role": "user", "content": query})

    # 인사 처리
    if normalized_query in GREETINGS or any(g in normalized_query for g in GREETINGS):
        answer = "안녕하세요! 저는 OpenSim RAG 챗봇입니다. 무엇을 도와드릴까요?"
        messages.append({"role": "assistant", "content": answer})
        retrieved_md = "## 🔍 Retrieved Documents\n\n_대화형 응답이므로 문서를 검색하지 않았습니다._"
        return messages, query, retrieved_md

    # RAG 실행
    try:
        answer, vec_docs, web_docs = rag_agent.query(query)
        answer_content = answer.content
    except Exception as e:
        answer_content = f"오류 발생: {e}"
        vec_docs, web_docs = [], []

    messages.append({"role": "assistant", "content": answer_content})

    # Retrieval 결과 생성
    retrieved_md = "## 🔍 Retrieved Documents\n"

    if vec_docs:
        retrieved_md += "### 📚 Vectorstore\n"
        for i, d in enumerate(vec_docs):
            retrieved_md += f"**[{i+1}]** {d.metadata.get('source', 'unknown')}\n\n"
            retrieved_md += f"> {d.page_content[:300]}...\n\n---\n"

    if web_docs:
        retrieved_md += "### 🌐 Web Search\n"
        for i, d in enumerate(web_docs):
            retrieved_md += f"**[{i+1}]** web\n\n"
            retrieved_md += f"> {d.page_content[:300]}...\n\n---\n"

    if not vec_docs and not web_docs:
        retrieved_md += "_검색 결과 없음_"

    return messages, query, retrieved_md


# ============================================
# 3) Gradio UI 구성
# ============================================
with gr.Blocks(title="Opensim Chatbot") as demo:
    gr.Markdown("##OpenSim Answer Chatbot\n- 왼쪽: **Pure LLM (Retrieval 없음)**\n- 오른쪽: **RAG (Retrieval 기반)**")

    with gr.Row():
        chatbot_pure = gr.Chatbot(
            label="LLM Only (No Retrieval)",
            height=1400,
        )
        chatbot_rag = gr.Chatbot(
            label="RAG Assistant (With Retrieval)",
            height=1400,
        )

    query_box = gr.Textbox(
        label="Message",
        placeholder="Ask me anything...",
    )

    retrieved_output = gr.Markdown(
        value="### 🔍 Retrieved Documents will appear here...",
    )

    send_btn = gr.Button("Send")

    # 버튼 클릭 → 두 챗봇 각각 실행
    send_btn.click(
        fn=chat_rag,
        inputs=[chatbot_rag, query_box],
        outputs=[chatbot_rag, query_box, retrieved_output],
        queue=False,
    )

    send_btn.click(
        fn=chat_pure,
        inputs=[chatbot_pure, query_box],
        outputs=[chatbot_pure, query_box],
        queue=False,
    )


demo.launch(
    server_name="0.0.0.0",
    server_port=7860,
    theme=gr.themes.Soft()
)
