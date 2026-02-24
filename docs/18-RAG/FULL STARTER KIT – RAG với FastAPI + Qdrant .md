 **FULL STARTER KIT – RAG với FastAPI + Qdrant (Production-ready Mini Stack)** 🚀
setup → chạy → mở rộng.

Phù hợp: MVP, Startup, Internal AI Tool.

---

# 🧩 RAG STARTER KIT

**FastAPI + Qdrant + OpenAI (hoặc local embedding)**

---

# 📁 CẤU TRÚC PROJECT

```
18-RAG-system/
│
├── app/
│   ├── main.py          # API
│   ├── config.py        # Config
│   ├── ingest.py        # Upload + chunk + embed
│   ├── 18-RAG.py           # Retrieval + Generation
│   ├── vector.py        # Qdrant client
│   └── utils.py
│
├── data/
│   └── docs/
│
├── docker-compose.yml
├── requirements.txt
└── .env
```

---

# 1️⃣ requirements.txt

```txt
fastapi
uvicorn
qdrant-client
openai
python-dotenv
pypdf
tiktoken
langchain
sentence-transformers
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
      - ./qdrant_data:/qdrant/sto18-RAGe
```

Run:

```bash
docker compose up -d
```

---

# 3️⃣ .env

```env
OPENAI_API_KEY=your_key_here
QDRANT_URL=http://localhost:6333
COLLECTION_NAME=18-RAG_docs
```

---

# 4️⃣ app/config.py

```python
import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_KEY = os.getenv("OPENAI_API_KEY")
QDRANT_URL = os.getenv("QDRANT_URL")
COLLECTION = os.getenv("COLLECTION_NAME")

CHUNK_SIZE = 500
OVERLAP = 80
TOP_K = 5
```

---

# 5️⃣ app/vector.py (Qdrant Client)

```python
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance
from .config import QDRANT_URL, COLLECTION


client = QdrantClient(url=QDRANT_URL)


def init_collection(dim: int):

    collections = client.get_collections().collections
    names = [c.name for c in collections]

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


def search(query_vec, limit=5, flt=None):

    return client.search(
        collection_name=COLLECTION,
        query_vector=query_vec,
        limit=limit,
        query_filter=flt
    )
```

---

# 6️⃣ app/utils.py (Chunking + PDF Loader)

```python
import uuid
from pypdf import PdfReader
import tiktoken
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

# 7️⃣ app/ingest.py (Indexing Pipeline)

```python
import openai
from sentence_transformers import SentenceTransformer

from .utils import load_pdf, chunk_text, gen_ids
from .vector import init_collection, upsert
from .config import OPENAI_KEY


openai.api_key = OPENAI_KEY

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

# 8️⃣ app/18-RAG.py (Query Engine)

```python
from sentence_transformers import SentenceTransformer
import openai

from .vector import search
from .config import TOP_K, OPENAI_KEY


openai.api_key = OPENAI_KEY

model = SentenceTransformer("all-MiniLM-L6-v2")


def embed_query(q):

    return model.encode([q])[0].tolist()


def generate_answer(context, question):

    prompt = f"""
Answer only from context.

Context:
{context}

Question:
{question}
"""

    res = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": prompt}
        ]
    )

    return res.choices[0].message.content


def ask(question):

    q_vec = embed_query(question)

    docs = search(q_vec, TOP_K)

    context = "\n".join(
        [d.payload["text"] for d in docs]
    )

    answer = generate_answer(context, question)

    sources = [d.id for d in docs]

    return {
        "answer": answer,
        "sources": sources
    }
```

---

# 9️⃣ app/main.py (FastAPI)

```python
from fastapi import FastAPI, UploadFile, File
import shutil

from .ingest import ingest_pdf
from .18-RAG import ask


app = FastAPI(title="RAG System")


@app.post("/upload")
async def upload(file: UploadFile = File(...)):

    path = f"data/{file.filename}"

    with open(path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    n = ingest_pdf(path)

    return {"chunks_indexed": n}


@app.post("/ask")
async def query(q: str):

    return ask(q)
```

---

# 🚀 RUN PROJECT

### 1️⃣ Start Qdrant

```bash
docker compose up -d
```

### 2️⃣ Install deps

```bash
pip install -r requirements.txt
```

### 3️⃣ Run API

```bash
uvicorn app.main:app --reload
```

### 4️⃣ Open Docs

```
http://localhost:8000/docs
```

---

# 🧪 TEST FLOW

### Upload PDF

```
POST /upload
```

### Ask

```
POST /ask?q=Quy trình hoàn tiền thế nào?
```

---

# 🔥 PRODUCTION UPGRADES (NEXT STEP)

| Feature | How           |
| ------- | ------------- |
| Auth    | JWT           |
| Rerank  | Cohere        |
| Cache   | Redis         |
| Hybrid  | BM25 + Vector |
| Monitor | Prometheus    |
| Eval    | RAGAS         |

---

# 💎 BONUS: Scale Architecture

```
Client
 ↓
API Gateway
 ↓
RAG Service → Vector DB
 ↓
LLM
```
