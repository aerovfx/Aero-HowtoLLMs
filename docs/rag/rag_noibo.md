
# 🚀 CASE STUDY: XÂY DỰNG HỆ RAG CHO HỆ THỐNG TRA CỨU TÀI LIỆU NỘI BỘ

## 🎯 Bối cảnh

Một doanh nghiệp có:

* 📁 50.000+ file: PDF, Word, Email, Policy, Hướng dẫn kỹ thuật
* 📚 Dữ liệu phân tán, tìm kiếm thủ công rất chậm
* 🤖 Muốn xây chatbot AI trả lời câu hỏi nội bộ

Ví dụ câu hỏi:

> “Quy trình hoàn tiền dự án X năm 2024 thế nào?”

---

## ⚠️ Vấn đề ban đầu

Hệ thống RAG version 1 gặp lỗi:

❌ Trả lời sai ngữ cảnh
❌ Hallucination cao
❌ Lấy nhầm tài liệu cũ
❌ Câu trả lời rời rạc

Nguyên nhân chính:
👉 Chunking kém + Metadata yếu + Retrieval đơn giản

---

## 🧩 Giải pháp RAG tối ưu

### 1️⃣ Pipeline tổng thể

```
Document → Cleaning → Chunking → Embedding → Vector DB
                                ↓
User Query → Embedding → Retrieval → Rerank → LLM → Answer
```

---

### 2️⃣ Chiến lược Chunking áp dụng

Team chọn **Hybrid Chunking**:

✅ Section-based → giữ cấu trúc
✅ Semantic → chia theo ngữ nghĩa
✅ Overlap 15–20% → tránh mất context
✅ Size: 400–600 tokens

➡️ Giữ được cả logic + chi tiết

---

### 3️⃣ Metadata chiến lược

Mỗi chunk gắn:

```json
{
  "doc_type": "policy",
  "department": "finance",
  "year": 2024,
  "project": "X",
  "version": "v2.1"
}
```

➡️ Filter trước khi search → giảm nhiễu 40%

---

### 4️⃣ Embedding & Vector DB

| Thành phần | Lựa chọn               |
| ---------- | ---------------------- |
| Embedding  | text-embedding-3-large |
| Vector DB  | Qdrant                 |
| Distance   | Cosine                 |

➡️ Recall tăng ~28%

---

### 5️⃣ Retrieval + Rerank

#### Phase 1: Retrieve

```python
top_k = 20
filter = {year:2024, project:"X"}
```

#### Phase 2: Rerank (Cross-Encoder)

* Dùng Cohere Rerank / BGE-reranker
* Chọn top 5

➡️ Precision tăng mạnh

---

### 6️⃣ Prompt Engineering

Prompt production:

```
You are an internal AI assistant.
Only answer from provided context.
If info missing, say "Not found in documents".
```

➡️ Hallucination giảm ~60%

---

## 📊 Kết quả sau 3 tháng

| Chỉ số        | Trước   | Sau     |
| ------------- | ------- | ------- |
| Accuracy      | 62%     | 89%     |
| Hallucination | Cao     | Thấp    |
| Thời gian tìm | 10 phút | 10 giây |
| User 만족       | 6/10    | 9/10    |

---

## 🔍 Ví dụ truy vấn thực tế

User hỏi:

> “Chi phí marketing Q2/2024 dự án X?”

Hệ thống xử lý:

1️⃣ Filter: year=2024, project=X
2️⃣ Retrieve: 20 chunk
3️⃣ Rerank: top 5
4️⃣ LLM tổng hợp

Output:

> “Theo báo cáo tài chính Q2/2024, chi phí marketing là 2.3 tỷ VNĐ…”

➡️ Có nguồn rõ ràng → tin cậy

---

## 🏗️ Stack Công Nghệ

| Layer    | Tool                 |
| -------- | -------------------- |
| Ingest   | Unstructured / PyPDF |
| Chunking | Custom + LangChain   |
| VectorDB | Qdrant               |
| Rerank   | Cohere               |
| LLM      | GPT-4 / Claude       |
| Backend  | FastAPI              |

---

## 💎 Bài học rút ra

### ❌ Sai lầm thường gặp

* Chunk quá nhỏ (<200 tokens)
* Không overlap
* Không metadata
* Chỉ dùng similarity search

### ✅ Best Practice

🔥 Công thức vàng:

```
Section + Semantic + Overlap
+ Metadata Filter
+ Rerank
+ Strict Prompt
```

= RAG mạnh thật sự

---

## 🧠 Khi nào cần nâng cấp lên Agentic RAG?

Khi hệ thống:

✔️ Có workflow
✔️ Multi-step reasoning
✔️ Query phức tạp
✔️ Cần gọi tool

➡️ Chuyển sang: RAG + Agent + Planner

