import re

def handle_special_content(doc):
    """
    문서 내 표, 코드, 수식 처리
    - 코드 블록: [[CODE]] ... [[/CODE]]
    - 표: [[TABLE]] ... [[/TABLE]]
    - 수식: $...$ 또는 $$...$$
    """
    content = doc.get("content", "")

    # 코드 블록 처리: 그냥 마크업으로 보존
    content = re.sub(r"<pre><code>(.*?)</code></pre>", r"[[CODE]]\1[[/CODE]]", content, flags=re.DOTALL)

    # 표 처리: 단순 마크업 변환
    content = re.sub(r"<table>(.*?)</table>", r"[[TABLE]]\1[[/TABLE]]", content, flags=re.DOTALL)

    # 인라인 수식 처리
    content = re.sub(r"\$(.*?)\$", r"[[MATH]]\1[[/MATH]]", content)
    content = re.sub(r"\$\$(.*?)\$\$", r"[[MATH]]\1[[/MATH]]", content, flags=re.DOTALL)

    doc["content"] = content
    return doc
