# Opensim RAG System for BKMS 2025-FALL Project

RAG implementation for OpenSim documentation

## Setup

### 1. Environment Variables

OpenSim RAG 시스템 실행을 위해 필요한 환경 변수를 설정합니다:

```bash
export GOOGLE_API_KEY=""
export GOOGLE_APPLICATION_CREDENTIALS=""
export GOOGLE_API_USE_CLIENT_CERTIFICATE="false"
export TAVILY_API_KEY="
```

## Conda Environment Setup

OpenSim RAG 시스템을 실행하기 위해 별도의 conda 환경을 생성해 사용하는 것을 권장합니다.

``` bash
conda env create -f environment.yml
```

## 2. Build Vector Database
데이터 처리 파이프라인(Chunking → Preprocess → Embedding → Vectorstore)을 수행하여 RAG 시스템에 필요한 벡터 데이터베이스를 구축합니다.

실행 단계:

아래 스크립트를 순서대로 실행하십시오.

### 1. Collect Documentation

``` bash
python run_collect.py
```

### 2. Chunking

``` bash
chmod +x run_chunking.sh
./run_chunking.sh
```

### 3. Preprocess Chunks

```bash
python run_preprocess.py
```

### 4. Create Embeddings

```bash
chmod +x run_embedding.sh
./run_embedding.sh
```

### 5. Build Vectorstore

```bash
chmod +x run_vectorstore.sh
./run_vectorstore.sh
```

### 6. (Optional) Analyze Contents


```bash
python analyze_contents.py
```

## Run rag agent
다음 코드로 rag agent 를 실행할 수 있습니다.

``` bash
cd rag-agent
chmod +x ./start_rag.sh
./start_rag.sh
```
