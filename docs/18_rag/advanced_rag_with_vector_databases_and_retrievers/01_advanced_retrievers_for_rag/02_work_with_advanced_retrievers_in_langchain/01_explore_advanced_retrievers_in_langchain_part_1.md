
<!-- Aero-Navigation-Start -->
[🏠 Home](../../../../index.md) > [18 rag](../../../index.md) > [advanced rag with vector databases and retrievers](../../index.md) > [01 advanced retrievers for rag](../index.md) > [02 work with advanced retrievers in langchain](index.md)

---
### 🧭 Điều hướng nhanh

- [🏠 Cổng tài liệu](../../../../index.md)
- [📚 Module 01: LLM Course](../../../../01_llm_course/index.md)
- [🔢 Module 02: Tokenization](../../../../02_words_to_tokens_to_numbers/index.md)
- [🏗️ Module 04: Build GPT](../../../../04_buildgpt/index.md)
- [🎯 Module 07: Fine-tuning](../../../../07_fine_tune_pretrained_models/index.md)
- [🔍 Module 19: AI Safety](../../../../19_ai_safety/index.md)
- [🐍 Module 20: Python for AI](../../../../20_python_colab_notebooks/index.md)
---
<!-- Aero-Navigation-End -->
# Khám Phá Advanced Retrievers Trong LangChain - Phần 1

## Tổng Quan

Chương này giới thiệu về **Retrievers trong LangChain** - một thành phần quan trọng trong hệ thống Retrieval-Augmented Generation (RAG). Bạn sẽ hiểu được retriever là gì, cách vector store-based retriever hoạt động, và các kỹ thuật nâng cao như Maximum Marginal Relevance (MMR).

## 1. Giới Thiệu về LangChain Retriever

### 1.1 Khái Niệm Retriever

Trong LangChain, **Retriever** là một interface nhận đầu vào là câu truy vấn dạng chuỗi và trả về danh sách các documents hoặc chunks liên quan:

```python
from langchain.schema import Retriever

# Cấu trúc cơ bản của Retriever
class Retriever(ABC):
    def get_relevant_documents(
        self, 
        query: str
    ) -> List[Document]:
        """Nhận query và trả về documents liên quan"""
        pass

### 1.2 Sự Khác Biệt với Vector Store

| Đặc điểm | Vector Store | Retriever |
|-----------|-------------|----------|
| Mục đích | Lưu trữ vectors | Truy xuất documents |
| Input | Documents/Text | Query string |
| Output | Vectors | Documents |
| Tính tổng quát | Thấp | Cao |

### 1.3 Interface Cơ Bản

```python
# Retriever interface trong LangChain
from langchain_core.documents import Document

# Ví dụ usage
retriever = vectorstore.as_retriever()

# Truy xuất documents
query = "What is machine learning?"
documents = retriever.get_relevant_documents(query)

for doc in documents:
    print(doc.page_content)
    print(doc.metadata)

## 2. Vector Store-Based Retriever

### 2.1 Nguyên Lý Hoạt Động

Vector store-based retriever hoạt động theo các bước sau:

Query Text
    ↓

$$

Embedding Model

$$

↓
Query Vector
    ↓

$$

Vector Similarity Search

$$

↓
Top-K Similar Documents

### 2.2 Quy Trình Chi Tiết

```python
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings

# Khởi tạo vector store
vectorstore = Chroma.from_documents(
    documents=texts,
    embedding=OpenAIEmbeddings()
)

# Chuyển đổi thành retriever
retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 4}
)

# Truy xuất
docs = retriever.get_relevant_documents("What is AI?")

### 2.3 Các Loại Tìm Kiếm

#### 2.3.1 Similarity Search

Tìm kiếm dựa trên độ tương đồng cosine:

$$

\text{similarity}(A, B) = \frac{A \cdot B}{||A|| \cdot ||B||}

$$

```python
retriever = vectorstore.as_retriever(
    search_type="similarity"
)

#### 2.3.2 Maximum Marginal Relevance (MMR)

MMR giúp cân bằng giữa:
- **Relevance**: Độ liên quan đến query
- **Diversity**: Đa dạng trong kết quả

```python
# Cấu hình MMR
retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 4,           # Số lượng documents
        "fetch_k": 20,    # Số lượng lấy trước khi lọc
        "lambda_mult": 0.5  # 0 = relevance, 1 = diversity
    }
)

**Công thức MMR:**

$$

\text{MMR} = \arg\max_{D_i \in R \setminus S} \left[ \lambda \cdot \text{sim}(Q, D_i) - (1-\lambda) \cdot \max_{D_j \in S} \text{sim}(D_i, D_j) \right]

$$

Trong đó:
- $Q$: Query
- $D_i$: Document thứ $i$
- $R$: Tập hợp các documents đã fetch
- $S$: Tập hợp các documents đã chọn
- $\lambda$: Tham số cân bằng (0-1)

## 3. Triển Khai Trong LangChain

### 3.1 Cấu Hình Nâng Cao

```python
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor

# Tạo compressor
compressor = LLMChainExtractor.from_llm(ChatOpenAI(temperature=0))

# Tạo compression retriever
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=vectorstore.as_retriever()
)

# Truy xuất với compression
docs = compression_retriever.get_relevant_documents(query)

### 3.2 Ensemble Retriever

Kết hợp nhiều retrievers:

```python
from langchain.retrievers import EnsembleRetriever

# Khởi tạo các retrievers
retriever1 = vectorstore.as_retriever(search_kwargs={"k": 2})
retriever2 = bm25_retriever  # BM25 retriever

# Ensemble
ensemble = EnsembleRetriever(
    retrievers=[retriever1, retriever2],
    weights=[0.5, 0.5]
)

# Truy xuất
docs = ensemble.get_relevant_documents(query)

## 4. Tối Ưu Hiệu Suất

### 4.1 Chunking Strategies

| Strategy | Kích thước | Use Case |
|----------|-------------|---------|
| Fixed Size | 500-1000 chars | Simple documents |
| Recursive | Variable | Complex documents |
| Semantic | Sentence-based | NLG outputs |
| Markdown | Header-based | Technical docs |

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Cấu hình text splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    separators=["\n\n", "\n", " ", ""]
)

# Split documents
docs = text_splitter.split_documents(raw_documents)

### 4.2 Embedding Optimization

```python
from langchain.embeddings import HuggingFaceEmbeddings

# Sử dụng embedding tối ưu
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={'device': 'cpu'}
)

## 5. Best Practices

### 5.1 Chọn K Value

```python
# Thử nghiệm với các giá trị k khác nhau
for k in [2, 4, 8, 16]:
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})
    docs = retriever.get_relevant_documents(query)
    # Đánh giá kết quả

### 5.2 Filtering

```python
# Lọc theo metadata
retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={
        "k": 4,
        "filter": {
            "source": "blog",
            "date": {"$gte": "2024-01-01"}
        }
    }
)

## 6. Kết Luận

LangChain cung cấp nhiều loại retriever với các tính năng nâng cao:
- Vector store-based retriever cho tìm kiếm similarity
- MMR cho cân bằng relevance và diversity
- Compression retriever cho trích xuất thông tin tinh tế
- Ensemble retriever cho kết hợp nhiều phương pháp

Việc lựa chọn đúng retriever và tham số là yếu tố quan trọng trong việc xây dựng hệ thống RAG hiệu quả.

## Tài Liệu Tham Khảo

1. LangChain Documentation. (2024). "Retrievers". https://python.langchain.com/

2. Carbonell, J., & Goldstein, J. (1998). "The use of MMR, diversity-based reranking for reordering documents and producing summaries". *SIGIR 1998*.

3. Hofstätter, S., et al. (2020). "Neural Re-Ranking for Dense Retrieval". *ECIR 2020*.

4. Karpukhin, V., et al. (2020). "Dense Passage Retrieval for Open-Domain Question Answering". *EMNLP 2020*.
<!-- Aero-Footer-Start -->

## 📄 Tài liệu cùng chuyên mục
| Bài học | Liên kết |
| :--- | :--- |
| 📌 **[Khám Phá Advanced Retrievers Trong LangChain - Phần 1](01_explore_advanced_retrievers_in_langchain_part_1.md)** | [Xem bài viết →](01_explore_advanced_retrievers_in_langchain_part_1.md) |
| [Khám Phá Các Retriever Nâng Cao trong LangChain - Phần 2](02_explore_advanced_retrievers_in_langchain_part_2.md) | [Xem bài viết →](02_explore_advanced_retrievers_in_langchain_part_2.md) |

---
## 🤝 Liên hệ & Đóng góp
Dự án được phát triển bởi **Pixibox**. Mọi đóng góp về nội dung và mã nguồn đều được chào đón.

> *"Kiến thức là để chia sẻ. Hãy cùng nhau xây dựng cộng đồng AI vững mạnh!"* 🚀

*Cập nhật tự động bởi Aero-Indexer - 2026*
<!-- Aero-Footer-End -->
