# Giới Thiệu Khóa Học: Retrieval-Augmented Generation Nâng Cao với Cơ Sở Dữ Liệu Vector

## Tổng Quan

Chào mừng bạn đến với khóa học về việc sử dụng **cơ sở dữ liệu vector (vector databases)** và **bộ truy xuất (retrievers)** để phát triển các ứng dụng **Retrieval-Augmented Generation (RAG)** nâng cao. Khóa học này sẽ cung cấp cho bạn kiến thức chuyên sâu về các kỹ thuật truy xuất nâng cao và vai trò của chúng trong hệ thống RAG.

## 1. Giới Thiệu về RAG

### 1.1 Retrieval-Augmented Generation là gì?

**Retrieval-Augmented Generation (RAG)** là một kỹ thuật kết hợp khả năng tìm kiếm thông tin từ cơ sở dữ liệu với khả năng sinh văn bản của Large Language Models (LLMs). Thay vì dựa hoàn toàn vào kiến thức được huấn luyện trong mô hình, RAG cho phép:

- Truy xuất thông tin liên quan từ nguồn dữ liệu bên ngoài
- Cung cấp ngữ cảnh chính xác cho LLM
- Giảm hiện tượng "hallucination" (bịa đặt thông tin)
- Cập nhật kiến thức mà không cần retrain mô hình

### 1.2 Công Thức Cơ Bản

$$\text{Response} = \text{LLM}( \text{Query}, \text{Context} )$$

Trong đó:
- **Query**: Câu hỏi của người dùng
- **Context**: Kết quả truy xuất từ cơ sở dữ liệu vector
- **Response**: Câu trả lời được sinh ra

## 2. Các Loại Retriever Nâng Cao

### 2.1 Multi-Query Retriever

Tạo nhiều câu truy vấn từ một câu hỏi gốc để cải thiện độ phủ của kết quả:

```python
from langchain.retrievers import MultiQueryRetriever
from langchain.chat_models import ChatOpenAI

# Tạo multi-query retriever
retriever = MultiQueryRetriever.from_llm(
    vectorstore.as_retriever(),
    llm=ChatOpenAI(temperature=0)
)

# Truy xuất với nhiều query
docs = retriever.get_relevant_documents(
    "What are the main benefits of exercise?"
)
```

**Lợi ích:**
- Tăng độ phủ của tìm kiếm
- Khám phá nhiều khía cạnh của câu hỏi
- Cải thiện recall

### 2.2 Self-Querying Retriever

Cho phép retriever tự động trích xuất bộ lọc từ câu truy vấn:

```python
from langchain.retrievers import SelfQueryRetriever
from langchain.chains.query_constructor.base import AttributeInfo

# Định nghĩa metadata fields
metadata_field_info = [
    AttributeInfo(
        name="source",
        description="The source of the document",
        type="string"
    ),
    AttributeInfo(
        name="date",
        description="The date of the document",
        type="date"
    )
]

# Tạo self-query retriever
retriever = SelfQueryRetriever.from_llm(
    ChatOpenAI(temperature=0),
    vectorstore,
    metadata_field_info,
    document_contents="Academic papers"
)
```

### 2.3 Parent-Document Retriever

Kết hợp kết quả từ nhiều mức độ chi tiết:

```python
from langchain.retrievers import ParentDocumentRetriever
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma

# Tạo text splitters cho parent và child documents
parent_splitter = RecursiveCharacterTextSplitter(chunk_size=2000)
child_splitter = RecursiveCharacterTextSplitter(chunk_size=400)

# Tạo retriever
retriever = ParentDocumentRetriever(
    vectorstores=[Chroma(...)],
    docstore=InMemoryStore(),
    parent_splitter=parent_splitter,
    child_splitter=child_splitter
)
```

## 3. Cơ Sở Dữ Liệu Vector

### 3.1 FAISS (Facebook AI Similarity Search)

FAISS là thư viện của Facebook Research để tìm kiếm sự tương đồng trong các tập hợp vector dense:

```python
import faiss
import numpy as np

# Tạo index
dimension = 128
index = faiss.IndexFlatL2(dimension)

# Thêm vectors
vectors = np.random.random((10000, dimension)).astype('float32')
index.add(vectors)

# Tìm kiếm
query = np.random.random((1, dimension)).astype('float32')
distances, indices = index.search(query, k=10)
```

**Ưu điểm:**
- Tìm kiếm nhanh với nhiều chiều
- Hỗ trợ nhiều thuật toán index
- Memory efficient

### 3.2 ChromaDB

ChromaDB là vector database mã nguồn mở được thiết kế cho AI applications:

```python
import chromadb

# Khởi tạo client
client = chromadb.Client()

# Tạo collection
collection = client.create_collection("documents")

# Thêm documents
collection.add(
    documents=["Doc 1", "Doc 2"],
    ids=["id1", "id2"],
    embeddings=[[1, 2, 3], [4, 5, 6]]
)

# Query
results = collection.query(
    query_texts=["Search query"],
    n_results=2
)
```

### 3.3 So Sánh FAISS vs ChromaDB

| Tiêu chí | FAISS | ChromaDB |
|----------|-------|----------|
| Loại | Library | Database |
| Persistence | Memory-only | Persistent |
| Query Language | Python API | SQL-like |
| Scalability | High | Medium |
| Use Case | Research/Production | Prototyping/Production |

## 4. Indexing Methods

### 4.1 Hierarchical Navigable Small World (HNSW)

HNSW là thuật toán graph-based cho tìm kiếm gần đúng:

$$\text{Time Complexity} = O(\log N)$$

```python
import hnswlib

# Khởi tạo HNSW index
dimension = 128
max_elements = 10000

index = hnswlib.Index(space='l2', dim=dimension)

# Cấu hình
index.init_params(
    max_elements=max_elements,
    ef_construction=200,
    M=16
)

# Thêm elements
index.add_items(vectors, ids)

# Tìm kiếm
labels, distances = index.knn_query(query, k=10)
```

**Đặc điểm:**
- **ef_construction**: Tham số ảnh hưởng đến chất lượng index
- **M**: Số lượng kết nối tối đa
- **ef**: Tham số search time

### 4.2 Inverted Index

```python
from rank_bm25 import BM25Okapi

# Tạo inverted index
corpus = [
    "Doc 1 content",
    "Doc 2 content",
    "Doc 3 content"
]
tokenized_corpus = [doc.split() for doc in corpus]

bm25 = BM25Okapi(tokenized_corpus)

# Query
query = "search query"
results = bm25.get_scores(query.split())
```

## 5. Đánh Giá Retrieval

### 5.1 Các Metrics

| Metric | Công thức | Mô tả |
|--------|----------|--------|
| Precision@K | TP/(TP+FP) | Tỷ lệ relevant trong K kết quả |
| Recall@K | TP/(TP+FN) | Tỷ lệ retrieved relevant |
| MAP | $\frac{1}{m}\sum_{i=1}^{m} \frac{1}{n_i}\sum_{j=1}^{n_i} P(i,j)$ | Mean Average Precision |
| NDCG | $\frac{DCG}{IDCG}$ | Normalized Discounted Cumulative Gain |

### 5.2 Evaluation Framework

```python
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall
)

# Đánh giá RAG
results = evaluate(
    dataset=eval_dataset,
    metrics=[
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall
    ]
)
```

## 6. Ứng Dụng Thực Tế

### 6.1 Xây Dựng RAG Application với LangChain và Gradio

```python
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
import gradio as gr

# Tạo vector store
vectorstore = Chroma.from_documents(
    documents=texts,
    embedding=OpenAIEmbeddings()
)

# Tạo chain
qa_chain = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(),
    chain_type="stuff",
    retriever=vectorstore.as_retriever()
)

# Tạo Gradio interface
def answer_question(query):
    return qa_chain.run(query)

demo = gr.Interface(
    fn=answer_question,
    inputs="text",
    outputs="text"
)

demo.launch()
```

### 6.2 Best Practices

1. **Chunk Size Selection**: Thử nghiệm với các kích thước chunk khác nhau (256, 512, 1024 tokens)
2. **Embedding Model**: Chọn embedding phù hợp với ngôn ngữ và domain
3. **Top-K Tuning**: Điều chỉnh số lượng documents truy xuất
4. **Hybrid Search**: Kết hợp vector search với keyword search

## 7. Kết Luận

Khóa học này sẽ giúp bạn:
- Hiểu sâu về các loại retriever nâng cao
- Triển khai và so sánh các vector databases
- Thiết kế chiến lược retrieval hiệu quả
- Xây dựng ứng dụng RAG hoàn chỉnh

## Tài Liệu Tham Khảo

1. Lewis, P., et al. (2020). "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks". *Advances in Neural Information Processing Systems*, 33, 9459-9474.

2. LangChain Documentation. (2024). "Retrievers". https://python.langchain.com/

3. Johnson, J., et al. (2017). "Billion-scale similarity search with GPUs". *IEEE BigData 2017*.

4. Malkov, Y.A., & Yashunin, D. (2018). "Efficient and robust approximate nearest neighbor search using Hierarchical Navigable Small World graphs". *IEEE TPAMI 2018*.

5. Robertson, S., & Zaragoza, H. (2009). "The probabilistic relevance framework: BM25 and beyond". *Foundations and Trends in Information Retrieval*, 3(4), 333-389.
