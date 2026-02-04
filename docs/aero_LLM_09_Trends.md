# Lecture 9: Recap & Current Trends 🔮

> **Tóm tắt từ khóa học Stanford CME 295: Transformers & Large Language Models.**
> Bài giảng cuối cùng: Tổng kết lại toàn bộ hành trình và cái nhìn về tương lai của LLM.

---

## 📚 Mục Lục
1. [Hành trình của chúng ta](#1-hành-trình-của-chúng-ta)
2. [Các xu hướng hiện tại (2025)](#2-các-xu-hướng-hiện-tại-2025)
3. [Những thách thức mở (Open Problems)](#3-những-thách-thức-mở-open-problems)

---

## 1. Hành trình của chúng ta
Khóa học đã đi qua một chặng đường dài từ những khái niệm cơ bản nhất đến những kỹ thuật tối tân nhất:
1.  **Kiến trúc:** Transformer, Attention, Encoder-Decoder.
2.  **Training:** Pre-training (Next token prediction), Scaling Laws, Parallelism.
3.  **Tuning:** SFT, RLHF, PEFT (LoRA).
4.  **Reasoning:** Chain-of-Thought, GRPO (DeepSeek-R1).
5.  **Agent:** RAG, Tool Use, ReAct.
6.  **Evaluation:** LLM-as-a-Judge, Benchmarks.

---

## 2. Các xu hướng hiện tại (2025)
Thế giới AI đang dịch chuyển rất nhanh:
*   **Reasoning Models (System 2):** Sự trỗi dậy của các mô hình "biết suy nghĩ" (như o1, DeepSeek-R1) sử dụng Inference-time compute để giải quyết các bài toán khó mà LLM truyền thống bó tay.
*   **Efficient Inference:** Các kỹ thuật như Quantization (4-bit, 1-bit), Speculative Decoding, KV Cache optimization giúp chạy LLM trên thiết bị cá nhân (Edge AI).
*   **Multimodal (Đa phương thức):** LLM không chỉ đọc text mà còn nhìn (Vision), nghe (Audio), nói (Speech) một cách tự nhiên (Native Multimodal như GPT-4o, Gemini 1.5).
*   **Agentic Systems:** Từ Chatbot hỏi-đáp chuyển sang Agent thực thi hành động, tự chủ hoàn thành công việc phức tạp.

---

## 3. Những thách thức mở (Open Problems)
Dù phát triển mạnh, LLM vẫn còn nhiều vấn đề chưa giải quyết được:
*   **Reliability (Độ tin cậy):** Làm sao để loại bỏ hoàn toàn Hallucination? Làm sao để tin tưởng vào code do AI viết trong các hệ thống quan trọng?
*   **Data Wall:** Dữ liệu chất lượng cao trên Internet sắp cạn kiệt. *Giải pháp:* Synthetic Data (Dữ liệu tổng hợp), Self-play.
*   **Energy Consumption:** Chi phí năng lượng cho Training và Inference quá lớn. Cần các kiến trúc xanh hơn.
*   **Safety & Alignment:** Đảm bảo AI siêu thông minh vẫn nằm trong tầm kiểm soát và phục vụ lợi ích con người.

---
*Biên soạn bởi Pixiboss - Dựa trên Stanford CME 295.*
