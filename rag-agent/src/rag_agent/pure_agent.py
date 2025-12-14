from langchain.chat_models import init_chat_model
import os

class PureAgent:
    def __init__(self, api_key: str = None):
        if api_key:
            os.environ["GOOGLE_API_KEY"] = api_key

        self.model = init_chat_model("google_genai:gemini-2.5-pro")

    def _is_korean(self, text: str):
        return any('\uac00' <= c <= '\ud7a3' for c in text)

    def _is_greeting(self, text: str):
        """
        인사, 안부, 감사 등 간단한 문구 감지
        """
        text = text.lower().strip().replace(" ", "")

        GREETINGS = [
            # 한국어
            "안녕", "안녕하세요", "ㅎㅇ", "반가워", "고마워", "감사합니다", "잘가",
            
            # 영어
            "hi", "hello", "hey", "yo", "bye", "goodbye", 
            "thanks", "thankyou"
        ]

        return any(g in text for g in GREETINGS)

    def answer(self, query: str):
        """
        순수 LLM 답변 — 벡터 검색 / 웹 검색 X
        """

        # --------------------------
        # 1) 인사 메시지 예외 처리
        # --------------------------
        if self._is_greeting(query):
            if self._is_korean(query):
                return "안녕하세요! 저는 OpenSim 관련 전문 LLM 어시스턴트입니다. 무엇을 도와드릴까요?"
            else:
                return "Hello! I'm an expert assistant for OpenSim. How can I help you today?"

        # --------------------------
        # 2) 일반 질의 → LLM 호출
        # --------------------------

        answer = self.model.invoke(query)
        return answer.content
