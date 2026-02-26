# Giới Thiệu FAISS Cho RAG và So Sánh với ChromaDB

## Tổng Quan

Chương này giới thiệu về **FAISS (Facebook AI Similarity Search)** - thư viện tìm kiếm sự tương đồng vector được phát triển bởi Meta, và so sánh với **ChromaDB** - một vector database phổ biến khác.

## 1. Giới Thiệu FAISS

### 1.1 FAISS là gì?

FAISS (Facebook AI Similarity Search) là thư viện mã nguồn mở được phát triển bởi Meta (trước đây là Facebook) cho việc tìm kiếm sự tương đồng trong các tập hợp vector dense:

- **Ngôn ngữ**: C++ với Python bindings
- **Hiệu suất**: Tối ưu cho tìm kiếm nhanh
- **Phần cứng**: Hỗ trợ CPU và GPU
- **Quy mô**: Single-machine operations

### 1.2 Ứng Dụng

FAISS được sử dụng trong:
- Tìm kiếm hình ảnh/video
- Recommendation systems
- NLP và semantic search
- Clustering

## 2. Các Loại Index Trong FAISS

### 2.1 Index Types Overview

| Index | Mô tả | Use Case |
|-------|--------|----------|
| Flat | Brute-force exact search | Small datasets |
| IVF | Inverted File Index | Medium datasets |
| HNSW | Graph-based approximate | Large datasets |
| LSH | Locality-Sensitive Hashing | Approximate search |
| PQ | Product Quantization | Memory constrained |

### 2.2 Flat Index (Exact Search)

```python
import faiss
import numpy as np

# Tạo index với exact search
dimension = 128
index = faiss.IndexFlatL2(dimension)  # L2 distance
# Hoặc
index = faiss.IndexFlatIP(dimension)   # Inner product (cosine)

# Thêm vectors
vectors = np.random.random((10000, dimension)).astype('float32')
index.add(vectors)

# Tìm kiếm
query = np.random.random((5, dimension)).astype('float32')
distances, indices = index.search(query, k=10)
```

**Đặc điểm:**
- Độ chính xác tuyệt đối
- Time: $O(N \cdot D)$ với N = số vectors, D = dimension
- Memory: $O(N \cdot D)$

### 2.3 IVF Index (Inverted File)

```python
# Tạo IVF index
nlist = 100  # Số clusters

quantizer = faiss.IndexFlatL2(dimension)
index = faiss.IndexIVFFlat(quantizer, dimension, nlist)

# Train trước khi add
index.train(vectors)

# Add vectors
index.add(vectors)

# Tìm kiếm
index.nprobe = 10  # Số clusters cần tìm
distances, indices = index.search(query, k=10)
```

**Đặc điểm:**
- Nhanh hơn Flat với large datasets
- Accuracy phụ thuộc vào nprobe
- Time: $O(N/D \cdot nlist + nprobe \cdot k)$

### 2.4 HNSW Index (Hierarchical Navigable Small World)

```python
# Tạo HNSW index
dimension = 128
index = faiss.IndexHNSWFlat(dimension, 32)  # 32 = M parameter

# Cấu hình
index.hnsw.efConstruction = 200  # Xây dựng
index.hnsw.efSearch = 50        # Tìm kiếm

# Add và search
index.add(vectors)
distances, indices = index.search(query, k=10)
```

**Đặc điểm:**
- Graph-based navigation
- Time: $O(\log N)$
- Memory: $O(N \cdot M)$

### 2.5 LSH (Locality-Sensitive Hashing)

```python
# Tạo LSH index
dimension = 128
nbits = 32  # Số bits cho mỗi hash

index = faiss.IndexLSH(dimension, nbits)
index.add(vectors)
distances, indices = index.search(query, k=10)
```

**Đặc điểm:**
- Hash-based approximate search
- Good for high-dimensional data
- Memory efficient

### 2.6 PQ (Product Quantization)

```python
# Tạo PQ index
m = 8           # Số sub-vectors
nbits = 8        # Bits per sub-vector

quantizer = faiss.IndexFlatL2(dimension)
index = faiss.IndexIVFPQ(quantizer, dimension, nlist, m, nbits)

index.train(vectors)
index.add(vectors)
distances, indices = index.search(query, k=10)
```

**Đặc điểm:**
- Compression cao
- Memory efficient
- Good for large-scale search

## 3. So Sánh FAISS vs ChromaDB

### 3.1 Tổng Quan So Sánh

| Tiêu chí | FAISS | ChromaDB |
|----------|-------|----------|
| **Loại** | Library | Database |
| **Architecture** | Single-node | Single + Distributed |
| **Index Types** | Multiple | HNSW only |
| **Metadata** | Không có | Có |
| **Persistence** | Memory-only | Persistent |
| **Query Language** | Python API | SQL-like |
| **Scalability** | Medium | High |
| **Integrations** | LangChain, LlamaIndex | LangChain, LlamaIndex |

### 3.2 FAISS - Ưu và Nhược Điểm

**Ưu điểm:**
- Hiệu suất cao với single machine
- Nhiều thuật toán index
- Hỗ trợ GPU
- Kiểm soát full parameters
- Lightweight

**Nhược điểm:**
- Không có native metadata
- Single-node only
- Cần tự quản lý persistence
- Không có built-in server

### 3.3 ChromaDB - Ưu và Nhược Điểm

**Ưu điểm:**
- Full database với persistence
- Native metadata support
- Filtering
- Dễ sử dụng
- Distributed scaling
- Good LangChain integration

**Nhược điểm:**
- Ít index options (chỉ HNSW)
- Performance thấp hơn FAISS cho một số cases
- Younger project

## 4. Milvus Extension

### 4.1 Giới Thiệu Milvus

Milvus là distributed vector database có thể mở rộng FAISS:

```python
from pymilvus import connections, Collection

# Kết nối Milvus
connections.connect("default", host="localhost", port="19530")

# Tạo collection
collection = Collection("FAISS_Collection")
collection.create_schema(
    fields=[
        {"name": "id", "type": "INT"},
        {"name": "embedding", "type": "FLOAT_VECTOR", "dim": 128}
    ]
)
```

### 4.2 Sử Dụng FAISS với Milvus

```python
# Milvus hỗ trợ nhiều FAISS indexes
index_params = {
    "index_type": "IVF_FLAT",
    "metric_type": "L2",
    "params": {"nlist": 128}
}

collection.create_index(
    field_name="embedding",
    index_params=index_params
)
```

## 5. Khi Nào Sử Dụng

### 5.1 Chọn FAISS Khi:

- Cần hiệu suất cao nhất
- Single-machine deployment
- Kiểm soát full parameters
- Research/prototyping
- Memory constraints (với PQ)

```python
# Use case: Production với hiệu suất cao
index = faiss.IndexHNSWFlat(dimension, 32)
index.hnsw.efSearch = 100
```

### 5.2 Chọn ChromaDB Khi:

- Cần persistence
- Metadata filtering
- Distributed deployment
- Rapid development
- LangChain/LlamaIndex integration

```python
# Use case: Development với LangChain
vectorstore = Chroma.from_documents(
    documents=texts,
    embedding=OpenAIEmbeddings()
)
```

### 5.3 Decision Tree

```
Start
  │
  ├─> Need metadata filtering?
  │     ├─ Yes → ChromaDB
  │     └─ No
  │           │
  │     ├─> Single machine?
  │     │     ├─ Yes
  │     │     │     │
  │     │     │     ├─> Need full control?
  │     │     │     │     ├─ Yes → FAISS
  │     │     │     │     └─ No → ChromaDB
  │     │     └─ No → Milvus/Pinecone
  │     └─ No
```

## 6. Code Examples

### 6.1 FAISS với GPU

```python
import faiss

# Chuyển sang GPU
gpu_index = faiss.index_cpu_to_gpu(
    faiss.StandardGpuResources(),
    0,  # GPU ID
    index  # CPU index
)

# Search trên GPU
distances, indices = gpu_index.search(query, k=10)
```

### 6.2 ChromaDB với Metadata Filtering

```python
import chromadb

client = chromadb.Client()
collection = client.create_collection("documents")

# Add với metadata
collection.add(
    documents=["Doc 1", "Doc 2"],
    metadatas=[{"source": "blog", "year": 2024}, 
               {"source": "news", "year": 2023}],
    ids=["id1", "id2"]
)

# Query với filter
results = collection.query(
    query_texts=["search query"],
    n_results=2,
    where={"source": "blog"}
)
```

## 7. Kết Luận

Việc lựa chọn giữa FAISS và ChromaDB phụ thuộc vào:
- Yêu cầu về hiệu suất
- Nhu cầu metadata
- Quy mô triển khai
- Độ phức tạp của infrastructure

FAISS phù hợp cho ứng dụng cần hiệu suất cao và kiểm soát full, trong khi ChromaDB phù hợp cho rapid development và production với metadata requirements.

## Tài Liệu Tham Khảo

1. Johnson, J., Douze, M., & Jégou, H. (2017). "Billion-scale similarity search with GPUs". *IEEE BigData 2017*.

2. Malkov, Y.A., & Yashunin, D. (2018). "Efficient and robust approximate nearest neighbor search using Hierarchical Navigable Small World graphs". *IEEE TPAMI 2018*.

3. ChromaDB Documentation. (2024). "Chroma: The AI-native embedding database". https://docs.trychroma.com/

4. Milvus Documentation. (2024). "Milvus: A Purpose-Built Vector Database". https://milvus.io/docs

5. G淡水. (2023). "FAISS: Efficient Similarity Search and Clustering of Dense Vectors". *GitHub Repository*.
