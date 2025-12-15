# build_eval_bundle_json.py
import os, json
from pathlib import Path
from typing import List, Dict, Any
import re

from src.rag_agent.rag_agent_for_eval import RAGAgentForEval


MAX_CONTEXT_CHARS_PER_DOC = 12000
MAX_DOCS_TOTAL = 12
version = "pro"

# ✅ NEW: 답변 md 저장 폴더 (원하면 env로 바꿀 수 있음)
OUT_MD_DIR = Path(os.getenv("ANSWERS_MD_DIR", f"data/answers/rag_{version}"))

def trunc(s: str, n: int) -> str:
    s = s or ""
    return s if len(s) <= n else s[:n] + "\n\n...[TRUNCATED]"


# ✅ NEW: 문자열 "\\n", "\\t" 를 실제 개행/탭으로 복원해서 md로 저장
def normalize_md(text: str) -> str:
    if text is None:
        return ""

    # 문자열 이스케이프 복원
    text = text.replace("\\r\\n", "\n").replace("\\n", "\n").replace("\\t", "\t").replace("\\r", "\n")
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # ✅ NBSP(유니코드) 제거/치환
    text = text.replace("\u00a0", " ")

    # ✅ "공백만 잔뜩"인 줄 제거
    text = re.sub(r"[ \t]+\n", "\n", text)

    # ✅ 연속 공백(스페이스/탭) 과다 축소 (표 망가지면 2줄 공백만 줄이는 방식으로 바꿔도 됨)
    text = re.sub(r"[ \t]{6,}", "  ", text)  # 6개 이상 연속 공백 → 2개로

    return text.strip() + "\n"


def docs_to_contexts(vec_docs, web_docs) -> List[Dict[str, Any]]:
    out = []
    all_docs = list(vec_docs) + list(web_docs)

    for i, d in enumerate(all_docs[:MAX_DOCS_TOTAL], 1):
        md = d.metadata or {}
        out.append({
            "rank": i,
            "source": md.get("source"),
            "title": md.get("title"),
            "url": md.get("url"),
            "has_raw_content": md.get("has_raw_content"),
            "text": trunc(d.page_content, MAX_CONTEXT_CHARS_PER_DOC),
        })
    return out


def main():
    cases_path = os.getenv("EVAL_CASES_PATH", r"eval/dataset/eval_questions.jsonl")
    out_path = os.getenv("EVAL_BUNDLE_PATH", f"eval/dataset/rag_{version}.json")

    # ✅ NEW: md 저장 폴더 생성
    OUT_MD_DIR.mkdir(parents=True, exist_ok=True)

    cases = [
        json.loads(line)
        for line in Path(cases_path).read_text(encoding="utf-8-sig").splitlines()  # ✅ BOM 방지
        if line.strip()
    ]

    agent = RAGAgentForEval()

    bundle = {
        "meta": {
            "generator": "build_eval_bundle_json.py",
            "max_context_chars_per_doc": MAX_CONTEXT_CHARS_PER_DOC,
            "max_docs_total": MAX_DOCS_TOTAL,
            "answers_md_dir": str(OUT_MD_DIR),  # ✅ NEW: 기록용
        },
        "items": []
    }

    for idx, c in enumerate(cases, 1):
        q = c["question"]
        top_k = int(c.get("top_k", 3))
        use_web = bool(c.get("use_web", True))

        # ✅ retrieval + answer 생성
        answer, vec_docs, web_docs = agent.query(
            q,
            top_k=top_k,
            use_web=use_web,
            force_extract=True,
            return_answer=True,
        )

        # ✅ NEW: 답변 md 파일로 저장 (01.md, 02.md, ...)
        md_path = OUT_MD_DIR / f"{idx:02d}.md"
        md_path.write_text(normalize_md(answer), encoding="utf-8")

        item = {
            "id": f"case_{idx:03d}",
            "question": q,
            "answer": answer, 
            "answer_md_path": str(md_path),  # ✅ NEW: 경로도 같이 저장
            "retrieval": {
                "top_k": top_k,
                "use_web": use_web,
                "contexts": docs_to_contexts(vec_docs, web_docs),
            }
        }
        bundle["items"].append(item)
        print(f"[OK] {idx}/{len(cases)} {item['id']} contexts={len(item['retrieval']['contexts'])} md={md_path}")

    Path(out_path).write_text(json.dumps(bundle, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[DONE] Wrote one JSON bundle: {out_path}")


if __name__ == "__main__":
    main()
