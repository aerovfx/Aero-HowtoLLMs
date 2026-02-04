# CS229: Xây Dựng Mô Hình Ngôn Ngữ Lớn (LLMs) 🧠

> **Tổng hợp và biên soạn từ bài giảng CS229 - Machine Learning (Stanford).**
> Tài liệu này tóm tắt các nguyên lý cốt lõi, kiến trúc và quy trình huấn luyện các mô hình ngôn ngữ lớn (Large Language Models) hiện đại.

---

## 📚 Mục Lục

1. [Chương 1: Tổng Quan Về LLMs](#chương-1-tổng-quan-về-llms)
2. [Chương 2: 5 Trụ Cột Của Việc Huấn Luyện](#chương-2-5-trụ-cột-của-việc-huấn-luyện)
3. [Chương 3: Quy Trình: Từ Pre-training Đến Post-training](#chương-3-quy-trình-từ-pre-training-đến-post-training)
4. [Chương 4: Cơ Chế Hoạt Động (Autoregressive & Tokenization)](#chương-4-cơ-chế-hoạt-động-autoregressive--tokenization)
5. [Chương 5: Đánh Giá Mô Hình (Evaluation)](#chương-5-đánh-giá-mô-hình-evaluation)

---

## Chương 1: Tổng Quan Về LLMs

**Định nghĩa:** LLM (Large Language Model) là các mô hình phân phối xác suất trên các chuỗi từ (sequences of tokens). Ngày nay, hầu hết các LLM đều dựa trên kiến trúc **Transformer**.

**Các ví dụ tiêu biểu:**
- **OpenAI:** GPT-3, GPT-4 (ChatGPT)
- **Anthropic:** Claude
- **Google:** Gemini
- **Meta:** Llama

---

## Chương 2: 5 Trụ Cột Của Việc Huấn Luyện

Để xây dựng một LLM thành công, không chỉ cần Code mà cần sự phối hợp của 5 yếu tố:

1.  **Architecture (Kiến trúc):** Thiết kế mạng Neural (ví dụ: Transformer, Attention mechanisms). *Giới hàn lâm thường tập trung vào đây.*
2.  **Training Loss & Algorithm:** Hàm mất mát và thuật toán tối ưu hóa.
3.  **Data (Dữ liệu):** "Nhiên liệu" cho mô hình. *Yếu tố sống còn trong thực tế.*
4.  **Evaluation (Đánh giá):** Thước đo sự thông minh và hiệu quả.
5.  **Systems (Hệ thống):** Tối ưu hóa phần cứng (GPU/TPU) để huấn luyện các mô hình khổng lồ.

> 💡 **Lưu ý:** Trong môi trường công nghiệp (Industry), trọng số thường dồn vào **Data, Evaluation và Systems** nhiều hơn là việc sáng tạo ra kiến trúc mới.

---

## Chương 3: Quy Trình: Từ Pre-training Đến Post-training

Quá trình tạo ra một AI Assistant như ChatGPT trải qua 2 giai đoạn chính:

### Giai đoạn 1: Pre-training (Tiền huấn luyện)
*   **Mục tiêu:** Học cách mô phỏng Internet.
*   **Nhiệm vụ:** Dự đoán từ tiếp theo (Next token prediction).
*   **Kết quả:** Một mô hình có kiến thức rộng nhưng chưa biết cách "phục vụ" con người (Base model).
*   *Ví dụ:* GPT-2, GPT-3.

### Giai đoạn 2: Post-training (Hậu huấn luyện)
*   **Mục tiêu:** Biến mô hình thành trợ lý (Assistant).
*   **Phương pháp:** Instruction tuning, RLHF (Reinforcement Learning from Human Feedback).
*   **Kết quả:** Chatbot biết trả lời câu hỏi, tóm tắt, viết code theo lệnh.
*   *Ví dụ:* ChatGPT, Claude 3.5 Sonnet.

---

## Chương 4: Cơ Chế Hoạt Động (Autoregressive & Tokenization)

### 1. Autoregressive Language Modeling (Mô hình tự hồi quy)
LLM sinh văn bản bằng cách dự đoán từng từ một dựa trên ngữ cảnh (context) phía trước.

$$P(x) = \prod_{i=1}^{L} P(x_i | x_{1}, ..., x_{i-1})$$

*   **Hạn chế:** Tốc độ suy luận (Inference) chậm vì phải chạy vòng lặp (loop) để sinh từng từ một.

### 2. Tokenization (Mã hóa văn bản)
Máy tính không hiểu "từ" (word) hay "câu", chúng hiểu số. Tokenizer là cầu nối chuyển đổi Text $\leftrightarrow$ IDs.

*   **Tại sao cần Tokenizer?**
    *   Xử lý ngôn ngữ không có dấu cách (Tiếng Thái, Tiếng Trung).
    *   Xử lý lỗi chính tả (Typos).
    *   Giảm độ dài chuỗi đầu vào (Sequence length) để tối ưu hiệu năng tính toán.

*   **Thuật toán phổ biến:** BPE (Byte Pair Encoding) - Ghép các cặp ký tự xuất hiện thường xuyên thành một token.

> ⚠️ **Các vấn đề thường gặp với Tokenizer:**
> *   **Toán học:** Các số (ví dụ `327`) có thể bị cắt thành các token rời rạc vô nghĩa, khiến LLM tính toán sai.
> *   **Lập trình:** Trước đây, khoảng trắng (indentation) trong Python bị token hóa kém, gây khó khăn cho việc viết code. (GPT-4 đã cải thiện điều này).

---

## Chương 5: Đánh Giá Mô Hình (Evaluation)

Làm sao biết mô hình A thông minh hơn mô hình B?

1.  **Perplexity (Độ bối rối):**
    *   Đo lường mức độ "chắc chắn" của mô hình khi dự đoán từ tiếp theo.
    *   Chỉ số càng **thấp** càng tốt.
    *   *Lịch sử:* Giảm từ >70 (2017) xuống <10 (2023).

2.  **Benchmarks (Bộ đề thi):**
    *   Sử dụng các bài kiểm tra tiêu chuẩn hóa để chấm điểm.
    *   *Phổ biến:* **HELM** (Holistic Evaluation of Language Models), **Hugging Face Open Leaderboard**.

---
*Biên soạn bởi Pixibot - Dựa trên Stanford CS229.*
