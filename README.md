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

## Run rag agent
다음 코드로 rag agent 를 실행할 수 있습니다.

``` bash
cd rag-agent
chmod +x ./start_rag.sh
./start_rag.sh
```
