# 효천고등학교 2009년 1학년 11반 챗봇

순천효천고등학교 2009년 1학년 11반과 관련된 질문에 답변하는 RAG 기반 AI 챗봇입니다.

## 주요 기능

- PDF 문서를 기반으로 한 질의응답 (RAG)
- 대화 히스토리를 고려한 문맥 인식 검색
- Streamlit 기반 웹 UI
- ChromaDB 벡터 저장소 영구 캐싱

## 기술 스택

| 항목 | 내용 |
|------|------|
| UI | Streamlit |
| LLM | OpenAI GPT-4o-mini |
| 임베딩 | text-embedding-3-small |
| 벡터 DB | ChromaDB |
| RAG 프레임워크 | LangChain |

## 설치

```bash
pip install -r requirements.txt
```

## 실행 전 준비

1. Streamlit secrets에 OpenAI API 키 설정

`.streamlit/secrets.toml` 파일 생성:

```toml
OPENAI_API_KEY = "sk-..."
```

2. `data.pkl` 파일이 프로젝트 루트에 있어야 합니다 (PDF를 pickle로 변환한 파일).

## 실행

```bash
streamlit run hyocheon_app.py
```

## 프로젝트 구조

```
hyocheon_rag/
├── hyocheon_app.py   # 메인 애플리케이션
├── data.pkl          # PDF 로드 데이터 (pickle)
├── chroma_db/        # ChromaDB 벡터 저장소 (자동 생성)
├── requirements.txt
└── README.md
```

## 동작 방식

1. `data.pkl`에서 문서 로드
2. ChromaDB가 없으면 문서를 청크로 분할 후 임베딩하여 저장
3. 사용자 질문 입력 시 대화 히스토리를 고려해 질문을 재구성
4. 재구성된 질문으로 벡터 검색 후 GPT-4o-mini가 답변 생성

## 만든이

김현건 연구원
