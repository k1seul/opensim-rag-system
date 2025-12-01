# Opensim RAG System for BKMS 2025-FALL Project

RAG implementation for OpenSim documentation

## 🧠 RAG Agent 구축 체크리스트 (간소화)

> 데이터 수집부터 검색 증강 생성(RAG) 에이전트까지

---

## 1️⃣ 데이터 수집 & 전처리

| 체크리스트 | 상태 |
|-----------|------|
| [x] 데이터 출처 정의 및 접근 권한 확인 |
| [x] 원본 데이터 저장 (`/data/raw/`) |
| [x] 텍스트 클린업 (HTML, 특수문자, 코드/표/수식 처리) |
| [x] 중복 제거 및 메타데이터 포함 |
| [x] 전처리 완료 후 `/data/processed/` 저장 |

> **결과:** 깨끗한 텍스트와 메타데이터 확보

---

## 2️⃣ 청킹 (Chunking)

| 체크리스트 | 상태 |
|-----------|------|
| [x] Chunk size & Overlap 결정 (권장: 300~1000 tokens / 20~30%) |
| [x] 문맥 단위 유지 |
| [x] 메타데이터 유지 |
| [x] 청킹된 데이터 `/data/chunks/` 저장 |

---

## 3️⃣ 임베딩 생성

| 체크리스트 | 상태 |
|-----------|------|
| [x] 텍스트 청크 → 벡터 임베딩 변환 |
| [x] 임베딩 차원 확인 |
| [x] 임베딩 + 원문 + 메타데이터 저장 (`/data/embeddings/`) |

---

## 4️⃣ 벡터 스토어 구축

| 체크리스트 | 상태 |
|-----------|------|
| [ ] Vector DB 선택 (FAISS / Pinecone / Milvus 등) |
| [ ] 벡터 삽입 및 검색 설정 (Top-k) |
| [ ] 증분 업데이트 가능하도록 구성 |

> **결과:** 검색 가능한 벡터 DB 구성 완료

---

## 5️⃣ RAG 에이전트 구현 & 테스트

| 체크리스트 | 상태 |
|-----------|------|
| [ ] Rewriter → Retriever → Generator Pipeline 구성 |
| [ ] Context 설정 (max token, score threshold) |
| [ ] 시스템/도메인 프롬프트 작성 |
| [ ] 검색 후 생성 테스트 (샘플 질의) |
| [ ] 성능/정확도 간단 검증 |


