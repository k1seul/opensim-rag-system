# run_langsmith_eval.py
import os
import json
import re
from typing import Dict, Any, Optional

from langsmith import Client
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate


# --------------------------
# Your instructions
# --------------------------
relevance_instructions = """You are an impartial evaluator. Your task is to assess the relevance of a provided ANSWER to a given QUESTION using a 1-5 score.

You will be given a QUESTION and an ANSWER. Here is the grading criteria:
- **1 (Poor):** The ANSWER is completely off-topic, evasive, or does not address the QUESTION at all.
- **2 (Fair):** The ANSWER is tangentially related but does not directly answer the core of the QUESTION.
- **3 (Average):** The ANSWER partially addresses the QUESTION but misses key aspects or includes irrelevant information.
- **4 (Good):** The ANSWER directly addresses the QUESTION and is helpful, but could be slightly more complete or concise.
- **5 (Excellent):** The ANSWER directly, fully, and helpfully addresses the QUESTION's intent.

Explain your reasoning in a step-by-step manner. First, analyze the question's intent. Second, analyze the answer's content. Finally, provide your score from 1 to 5.
"""

retrieval_relevance_instructions = """You are an impartial evaluator. Your task is to assess the relevance of a set of retrieved CONTEXTS to a given QUESTION using a 1-5 score.

You will be given a QUESTION and a set of CONTEXTS. Here are the grading criteria:
- **1 (Poor):** ALL retrieved CONTEXTS are completely irrelevant to the QUESTION.
- **2 (Fair):** Most CONTEXTS are irrelevant, but one or two might be tangentially related.
- **3 (Average):** Some CONTEXTS are relevant to the QUESTION, but many are irrelevant or contain noise.
- **4. (Good):** Most CONTEXTS are relevant and helpful for answering the QUESTION.
- **5 (Excellent):** ALL retrieved CONTEXTS are highly relevant and crucial for answering the QUESTION.

Explain your reasoning in a step-by-step manner. First, analyze the QUESTION's intent. Second, examine each CONTEXT for its relevance. Finally, provide your score from 1 to 5 based on the overall relevance of the set.
"""

groundedness_instructions = """You are an impartial evaluator. Your task is to assess the GROUNDEDNESS of a provided ANSWER in the given CONTEXTS using a 1-5 score.

You will be given an ANSWER and a set of CONTEXTS. Here is the grading criteria:
- **1 (Poor):** The ANSWER is mostly unsupported by the CONTEXTS, with major claims not grounded at all.
- **2 (Fair):** Some parts are supported, but many important claims are not grounded in the CONTEXTS.
- **3 (Average):** The ANSWER is partially grounded; key claims are supported, but there are notable unsupported details.
- **4 (Good):** The ANSWER is largely grounded in the CONTEXTS with only minor unsupported or speculative details.
- **5 (Excellent):** The ANSWER is fully grounded in the CONTEXTS; all key claims are directly supported.

Explain your reasoning step-by-step. Identify the main claims and check whether each is supported by the CONTEXTS. Finally provide your score from 1 to 5.
"""


# --------------------------
# Judge helper
# --------------------------
def _parse_score(text: str) -> int:
    """
    We ask the judge to output JSON: {"score": 1..5, "reasoning": "..."}
    If JSON parsing fails, fallback: find first integer 1-5.
    """
    text = (text or "").strip()
    # try JSON block
    try:
        # find first {...}
        m = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if m:
            obj = json.loads(m.group(0))
            s = int(obj.get("score"))
            return max(1, min(5, s))
    except Exception:
        pass

    m2 = re.search(r"\b([1-5])\b", text)
    if m2:
        return int(m2.group(1))
    return 3


def make_grader(model_name: str, system_instructions: str, mode: str):
    """
    mode:
      - "qa": QUESTION + ANSWER
      - "qc": QUESTION + CONTEXTS
      - "ac": ANSWER + CONTEXTS
    """
    judge = init_chat_model(model_name)

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_instructions + "\n\nReturn STRICT JSON: {{\"score\": <1-5>, \"reasoning\": \"...\"}}"),
            ("human", "{payload}"),
        ]
    )

    def grade(question: str, answer: str, contexts: list[str]) -> Dict[str, Any]:
        if mode == "qa":
            payload = f"QUESTION:\n{question}\n\nANSWER:\n{answer}"
        elif mode == "qc":
            ctx = "\n\n---\n\n".join(contexts or [])
            payload = f"QUESTION:\n{question}\n\nCONTEXTS:\n{ctx}"
        elif mode == "ac":
            ctx = "\n\n---\n\n".join(contexts or [])
            payload = f"ANSWER:\n{answer}\n\nCONTEXTS:\n{ctx}"
        else:
            raise ValueError("Unknown mode")

        msg = prompt.invoke({"payload": payload})
        out = judge.invoke(msg.messages).content
        score = _parse_score(out)
        return {"score": score, "comment": out}

    return grade


def main():
    # ---- LangSmith project ----
    os.environ.setdefault("LANGSMITH_TRACING", "true")
    os.environ.setdefault("LANGSMITH_PROJECT", os.getenv("LANGSMITH_PROJECT", "opensim-rag-eval"))

    dataset_name = os.getenv("EVAL_DATASET_NAME", "opensim-rag-eval-pro")

    # ---- Judge model ----
    judge_model = os.getenv("EVAL_JUDGE_MODEL", "google_genai:gemini-2.5-flash")

    relevance_grader = make_grader(judge_model, relevance_instructions, mode="qa")
    ctxrel_grader = make_grader(judge_model, retrieval_relevance_instructions, mode="qc")
    grounded_grader = make_grader(judge_model, groundedness_instructions, mode="ac")

    client = Client()

    # ✅ Echo target: dataset inputs에 있는 answer/contexts를 그대로 outputs로 반환
    def target(inputs: dict) -> dict:
        return {
            "answer": inputs["answer"],
            "contexts": inputs["contexts"],
        }

    # ---- Evaluators ----
    def relevance_eval(inputs: dict, outputs: dict, reference_outputs: Optional[dict] = None):
        res = relevance_grader(inputs["question"], outputs["answer"], outputs.get("contexts", []))
        return {"key": "relevance", "score": res["score"], "comment": res["comment"]}

    def context_relevance_eval(inputs: dict, outputs: dict, reference_outputs: Optional[dict] = None):
        res = ctxrel_grader(inputs["question"], outputs["answer"], outputs.get("contexts", []))
        return {"key": "context_relevance", "score": res["score"], "comment": res["comment"]}

    def groundedness_eval(inputs: dict, outputs: dict, reference_outputs: Optional[dict] = None):
        res = grounded_grader(inputs["question"], outputs["answer"], outputs.get("contexts", []))
        return {"key": "groundedness", "score": res["score"], "comment": res["comment"]}

    results = client.evaluate(
        target,
        data=dataset_name,
        evaluators=[relevance_eval, context_relevance_eval, groundedness_eval],
        experiment_prefix="opensim-rag-judged-1to5",
        max_concurrency=int(os.getenv("EVAL_MAX_CONCURRENCY", "2")),
    )

    print(results)


if __name__ == "__main__":
    main()
