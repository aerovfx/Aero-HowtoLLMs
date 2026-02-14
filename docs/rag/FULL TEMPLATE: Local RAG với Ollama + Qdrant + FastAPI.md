**FULL TEMPLATE: Local RAG với Ollama + Qdrant + FastAPI (Chạy 100% Offline)** 🚀
Phù hợp: Privacy cao, Intranet, doanh nghiệp không dùng API cloud.

# 🏗️ LOCAL RAG STACK

```
FastAPI + Ollama (LLM Local) + Qdrant (Vector DB) + Embedding Local
```

Không cần OpenAI – không gửi dữ liệu ra ngoài.

---

## 🔧 TECH STACK

| Layer     | Tool                 |
| --------- | -------------------- |
| LLM       | **Ollama**           |
| Vector DB | **Qdrant**           |
| API       | FastAPI              |
| Embedding | SentenceTransformers |
| Infra     | Docker               |

---

# 📁 PROJECT STRUCTURE

```
local-rag/
│
├── app/
│   ├── main.py
│   ├── ingest.py
│   ├── rag.py
│   ├── vector.py
│   ├── utils.py
│   └── config.py
│
├── data/
├── docker-compose.yml
├── requirements.txt
└── .env
```

---

# 1️⃣ CÀI OLLAMA (LOCAL LLM)

### Mac / Linux

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

### Windows

Tải tại: ollama.com

---

## Pull Model (Khuyến nghị)

```bash
ollama pull llama3
ollama pull mistral
ollama pull qwen2
```

Test:

```bash
ollama run llama3
```

---

# 2️⃣ docker-compose.yml (Qdrant)

```yaml
version: "3.9"

services:
  qdrant:
    image: qdrant/qdrant
    ports:
      - "6333:6333"
    volumes:
      - ./qdrant_data:/qdrant/storage
```

Run:

```bash
docker compose up -d
```

---

# 3️⃣ requirements.txt

```txt
fastapi
uvicorn
qdrant-client
sentence-transformers
python-dotenv
pypdf
tiktoken
requests
```

---

# 4️⃣ .env

```env
QDRANT_URL=http://localhost:6333
COLLECTION_NAME=local_rag
OLLAMA_URL=http://localhost:11434
LLM_MODEL=llama3
```

---

# 5️⃣ app/config.py

```python
import os
from dotenv import load_dotenv

load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL")
COLLECTION = os.getenv("COLLECTION_NAME")

OLLAMA_URL = os.getenv("OLLAMA_URL")
LLM_MODEL = os.getenv("LLM_MODEL")

CHUNK_SIZE = 500
OVERLAP = 80
TOP_K = 5
```

---

# 6️⃣ app/vector.py

```python
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance
from .config import QDRANT_URL, COLLECTION


client = QdrantClient(url=QDRANT_URL)


def init_collection(dim):

    names = [c.name for c in client.get_collections().collections]

    if COLLECTION not in names:
        client.create_collection(
            collection_name=COLLECTION,
            vectors_config=VectorParams(
                size=dim,
                distance=Distance.COSINE
            )
        )


def upsert(vectors, payloads, ids):

    client.upsert(
        collection_name=COLLECTION,
        points=[
            {
                "id": ids[i],
                "vector": vectors[i],
                "payload": payloads[i]
            }
            for i in range(len(vectors))
        ]
    )


def search(qvec, limit):

    return client.search(
        collection_name=COLLECTION,
        query_vector=qvec,
        limit=limit
    )
```

---

# 7️⃣ app/utils.py (Chunking)

```python
import uuid
import tiktoken
from pypdf import PdfReader
from .config import CHUNK_SIZE, OVERLAP


tokenizer = tiktoken.get_encoding("cl100k_base")


def load_pdf(path):

    reader = PdfReader(path)
    text = ""

    for p in reader.pages:
        text += p.extract_text() + "\n"

    return text


def chunk_text(text):

    tokens = tokenizer.encode(text)

    chunks = []

    for i in range(0, len(tokens), CHUNK_SIZE - OVERLAP):
        chunk = tokens[i:i + CHUNK_SIZE]
        chunks.append(tokenizer.decode(chunk))

    return chunks


def gen_ids(n):

    return [str(uuid.uuid4()) for _ in range(n)]
```

---

# 8️⃣ app/ingest.py (Embedding + Index)

```python
from sentence_transformers import SentenceTransformer

from .utils import load_pdf, chunk_text, gen_ids
from .vector import init_collection, upsert


model = SentenceTransformer("all-MiniLM-L6-v2")
EMBED_DIM = 384


def embed(texts):

    return model.encode(texts).tolist()


def ingest_pdf(path, metadata={}):

    text = load_pdf(path)

    chunks = chunk_text(text)

    vectors = embed(chunks)

    payloads = [
        {
            "text": chunks[i],
            **metadata
        }
        for i in range(len(chunks))
    ]

    ids = gen_ids(len(chunks))

    init_collection(EMBED_DIM)

    upsert(vectors, payloads, ids)

    return len(chunks)
```

---

# 9️⃣ app/rag.py (Ollama Integration)

```python
import requests

from .vector import search
from .config import TOP_K, OLLAMA_URL, LLM_MODEL
from sentence_transformers import SentenceTransformer


embed_model = SentenceTransformer("all-MiniLM-L6-v2")


def embed_query(q):

    return embed_model.encode([q])[0].tolist()


def call_ollama(prompt):

    res = requests.post(
        f"{OLLAMA_URL}/api/generate",
        json={
            "model": LLM_MODEL,
            "prompt": prompt,
            "stream": False
        }
    )

    return res.json()["response"]


def ask(question):

    qvec = embed_query(question)

    docs = search(qvec, TOP_K)

    context = "\n".join(
        [d.payload["text"] for d in docs]
    )

    prompt = f"""
You are an internal assistant.
Only use the context below.

Context:
{context}

Question:
{question}

Answer:
"""

    answer = call_ollama(prompt)

    sources = [d.id for d in docs]

    return {
        "answer": answer,
        "sources": sources
    }
```

---

# 🔟 app/main.py (API)

```python
from fastapi import FastAPI, UploadFile, File
import shutil

from .ingest import ingest_pdf
from .rag import ask


app = FastAPI(title="Local RAG System")


@app.post("/upload")
async def upload(file: UploadFile = File(...)):

    path = f"data/{file.filename}"

    with open(path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    n = ingest_pdf(path)

    return {"indexed_chunks": n}


@app.post("/ask")
async def query(q: str):

    return ask(q)
```

---

# 🚀 CHẠY HỆ THỐNG

### 1️⃣ Start Ollama

```bash
ollama serve
```

### 2️⃣ Start Qdrant

```bash
docker compose up -d
```

### 3️⃣ Install Python

```bash
pip install -r requirements.txt
```

### 4️⃣ Run API

```bash
uvicorn app.main:app --reload
```

### 5️⃣ Open Swagger

```
http://localhost:8000/docs
```

---

# 🧪 TEST FLOW

### Upload tài liệu

```
POST /upload
```

### Hỏi AI

```
POST /ask?q=Quy trình hoàn tiền năm 2024?
```

---

# 📊 PERFORMANCE THỰC TẾ (LOCAL)

| Model      | RAM  | Speed | Quality |
| ---------- | ---- | ----- | ------- |
| llama3:8b  | 16GB | ⚡⚡⚡   | ⭐⭐⭐⭐    |
| mistral:7b | 12GB | ⚡⚡    | ⭐⭐⭐     |
| qwen2:7b   | 16GB | ⚡⚡⚡   | ⭐⭐⭐⭐    |

👉 Khuyến nghị: llama3 8B cho production nội bộ

---

# 🔐 ƯU ĐIỂM LOCAL RAG

✅ 100% private
✅ Không tốn API cost
✅ Không lo leak data
✅ Custom model thoải mái

---

# ⚠️ HẠN CHẾ

❌ Cần GPU/RAM
❌ Scale khó hơn cloud
❌ Rerank yếu nếu không thêm module

---

# 🚀 NÂNG CẤP CHO DOANH NGHIỆP

Khi dùng thật:

✅ GPU Server (A10/A100)
✅ Redis Cache
✅ Reranker local (bge-reranker)
✅ RBAC
✅ Audit Log

---

# 💎 ARCHITECTURE PROD

```
User
 ↓
Gateway
 ↓
FastAPI
 ↓
Vector DB → Ollama
 ↓
Answer
```
 **CẤU HÌNH TỐI THIỂU để chạy Local RAG với Ollama (Mac & PC)**

Dùng tốt cho: cá nhân, dev, team nhỏ, hệ thống nội bộ.

---

# 🧠 LOCAL RAG = Ollama + Vector DB + Embedding

LLM local chạy bằng **Ollama**
→ Không gửi dữ liệu ra ngoài → An toàn & tiết kiệm chi phí.

---

# 🍎 CẤU HÌNH TỐI THIỂU – MAC (Apple Silicon & Intel)

Áp dụng cho Mac của **Apple**

---

## ✅ MỨC 1: TỐI THIỂU CHẠY ĐƯỢC (Học – Test – MVP)

| Thành phần | Yêu cầu            |
| ---------- | ------------------ |
| CPU        | M1 / M2 / Intel i5 |
| RAM        | **8 GB** (minimum) |
| Ổ cứng     | 20 GB trống        |
| macOS      | 12+                |
| GPU        | Không bắt buộc     |

### 👉 Chạy được model:

```
mistral:7b (quantized)
qwen2:3b
phi-3
```

⚠️ Tốc độ: chậm – trung bình

---

## ✅ MỨC 2: KHUYẾN NGHỊ (Dùng Thật)

| Thành phần | Yêu cầu             |
| ---------- | ------------------- |
| CPU        | M1 Pro / M2 / M3    |
| RAM        | **16 GB+**          |
| SSD        | 50 GB               |
| GPU        | Apple Neural Engine |

### 👉 Chạy tốt:

```
llama3:8b
qwen2:7b
mistral:7b
```

⚡ Tốc độ: mượt

👉 Đây là mức “ngon – bền – ổn định” nhất cho dev.

---

## ✅ MỨC 3: CAO CẤP (Heavy RAG)

| Thành phần | Yêu cầu            |
| ---------- | ------------------ |
| Chip       | M2 Pro / M3 Max    |
| RAM        | 32 GB+             |
| SSD        | 100 GB             |
| GPU        | Full Apple Silicon |

👉 Chạy được:

```
llama3:13b
mixtral
```

---

# 💻 CẤU HÌNH TỐI THIỂU – PC / WINDOWS / LINUX

Áp dụng cho máy dùng **Microsoft** Windows / Linux PC

---

## ✅ MỨC 1: CPU-ONLY (Rẻ – Phổ thông)

| Thành phần | Yêu cầu              |
| ---------- | -------------------- |
| CPU        | i5 Gen 9 / Ryzen 5   |
| RAM        | **16 GB** (bắt buộc) |
| SSD        | 50 GB                |
| GPU        | Không cần            |

### 👉 Chạy được:

```
mistral:7b
qwen2:3b
phi-3
```

⚠️ Chậm hơn Mac M1

👉 Chỉ nên dùng để test.

---

## ✅ MỨC 2: GPU PHỔ THÔNG (KHUYẾN NGHỊ)

| Thành phần | Yêu cầu              |
| ---------- | -------------------- |
| CPU        | i7 / Ryzen 7         |
| RAM        | 16–32 GB             |
| GPU        | RTX 3060 (12GB VRAM) |
| SSD        | NVMe 100 GB          |

### 👉 Chạy mượt:

```
llama3:8b
qwen2:7b
mistral
```

⚡ Rất ổn cho production nhỏ.

---

## ✅ MỨC 3: GPU MẠNH (ENTERPRISE)

| Thành phần | Yêu cầu         |
| ---------- | --------------- |
| CPU        | Xeon / Ryzen 9  |
| RAM        | 64 GB           |
| GPU        | RTX 4090 / A100 |
| SSD        | 1TB NVMe        |

👉 Chạy:

```
llama3:70b
mixtral
```

(Chỉ cần khi làm doanh nghiệp lớn)

---

# 📊 SO SÁNH NHANH MAC vs PC

| Tiêu chí | Mac M1/M2  | PC RTX 3060 |
| -------- | ---------- | ----------- |
| Setup    | ⭐⭐⭐⭐⭐      | ⭐⭐⭐         |
| Ổn định  | ⭐⭐⭐⭐⭐      | ⭐⭐⭐⭐        |
| Giá      | Cao        | Tốt         |
| Scale    | Trung bình | Rất tốt     |
| GPU      | Integrated | Dedicated   |

👉 Cá nhân/dev → Mac
👉 Server/team → PC GPU

---

# 🧩 CẤU HÌNH RAG TỐI THIỂU (SOFTWARE)

Dù Mac hay PC, bạn cần tối thiểu:

### ✅ Phần mềm

```
Ollama
Python 3.10+
Qdrant (Docker)
FastAPI
```

### ✅ RAM Phân bổ tối thiểu

| Thành phần | RAM    |
| ---------- | ------ |
| Ollama     | 6–10GB |
| Embedding  | 2GB    |
| Qdrant     | 1GB    |
| API        | 1GB    |

➡️ Tổng: ~10–14GB

👉 Vì vậy 8GB RAM chỉ đủ test.

---

# 🚀 CẤU HÌNH “QUỐC DÂN” KHUYẾN NGHỊ 2026

Nếu bạn mua máy mới:

### 🏆 MAC BEST CHOICE

```
Mac M2 / M3
16GB RAM
512GB SSD
```

### 🏆 PC BEST CHOICE

```
Ryzen 7
32GB RAM
RTX 3060 12GB
1TB NVMe
```

→ Chạy RAG mượt 3–5 năm.

---

# ⚠️ LỖI THƯỜNG GẶP DO CẤU HÌNH YẾU

| Lỗi            | Nguyên nhân |
| -------------- | ----------- |
| Treo máy       | Thiếu RAM   |
| Load model lâu | SSD chậm    |
| Trả lời chậm   | Không GPU   |
| Crash          | VRAM thiếu  |

👉 80% lỗi = thiếu RAM.

---

# 💎 GỢI Ý MODEL THEO CẤU HÌNH

| RAM   | Model      |
| ----- | ---------- |
| 8GB   | qwen2:3b   |
| 16GB  | llama3:8b  |
| 32GB  | llama3:13b |
| 64GB+ | llama3:70b |
