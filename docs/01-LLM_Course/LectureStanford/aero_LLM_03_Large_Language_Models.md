# Lecture 3: Large Language Models (LLMs) & Inference 🚀

> **Tóm tắt từ khóa học Stanford CME 295: Transformers & Large Language Models.**
> Bài giảng này tập trung vào các mô hình ngôn ngữ lớn (Decoder-only), cách mở rộng quy mô (Scaling), kỹ thuật Prompting và tối ưu hóa suy luận (Inference).

---

## 📚 Mục Lục
1. [Định nghĩa LLM](#1-định-nghĩa-llm)
2. [Mixture of Experts (MoE)](#2-mixture-of-experts-moe)
3. [Kỹ thuật Prompting & In-context Learning](#3-kỹ-thuật-prompting--in-context-learning)
4. [Decoding Strategies (Chiến lược giải mã)](#4-decoding-strategies-chiến-lược-giải-mã)
5. [Tối ưu hóa Inference (KV Cache, Speculative Decoding)](#5-tối-ưu-hóa-inference)

---

## 1. Định nghĩa LLM
**Large Language Model (LLM)** thường ám chỉ các mô hình:
*   Là **Language Model** (mô hình xác suất dự đoán từ tiếp theo).
*   Có kích thước **Lớn** (hàng tỷ tham số, huấn luyện trên hàng nghìn tỷ tokens).
*   Kiến trúc chủ đạo: **Decoder-only Transformer** (bỏ qua phần Encoder và Cross-Attention).

*Ví dụ:* GPT-3, PaLM, Llama, Mistral.

---

## 2. Mixture of Experts (MoE) 🧠
Khi mô hình quá lớn, chi phí tính toán cho mỗi lần suy luận rất cao. **MoE** là giải pháp để "Mở rộng quy mô mà không tăng chi phí suy luận tương ứng".

*   **Ý tưởng:** Thay lớp FFN (Feed Forward Network) dày đặc bằng nhiều "Chuyên gia" nhỏ (Experts).
*   **Router (Gate):** Một mạng con quyết định xem với mỗi token đầu vào, nên gửi nó cho chuyên gia nào xử lý (Ví dụ: Câu hỏi Toán -> Gửi cho chuyên gia Toán).
*   **Sparse Activation (Kích hoạt thưa):** Dù có tổng số tham số khổng lồ (ví dụ 8x7B), nhưng mỗi lần chạy chỉ kích hoạt một phần nhỏ (ví dụ 2 experts/token).
*   **Lợi ích:** Training nhanh hơn, Inference rẻ hơn so với mô hình Dense cùng kích thước.
*   **Thách thức:** Cần cân bằng tải (Load balancing) để tránh việc một vài chuyên gia làm việc quá sức còn số khác thì ngồi chơi (Routing collapse).

---

## 3. Kỹ thuật Prompting & In-context Learning
LLM có khả năng học từ ngữ cảnh (In-context Learning) mà không cần cập nhật trọng số.

*   **Zero-shot:** Ra lệnh trực tiếp (VD: "Dịch câu này sang tiếng Anh").
*   **Few-shot:** Cung cấp vài ví dụ mẫu trước khi hỏi (VD: "Q: Hi A: Chào / Q: Bye A: Tạm biệt / Q: Thanks A: ...").
*   **Chain-of-Thought (CoT):** Yêu cầu mô hình "suy nghĩ từng bước" (Let's think step by step). Giúp cải thiện đáng kể khả năng giải toán và suy luận logic.
*   **Self-Consistency:** Hỏi cùng một câu nhiều lần (sampling) và chọn câu trả lời xuất hiện nhiều nhất (Majority voting).

---

## 4. Decoding Strategies (Chiến lược giải mã)
Làm sao chọn từ tiếp theo từ phân phối xác suất do mô hình dự đoán?

*   **Greedy Decoding:** Luôn chọn từ có xác suất cao nhất. *Nhược điểm:* Dễ bị lặp, văn bản nhàm chán, đôi khi không tối ưu toàn cục.
*   **Beam Search:** Giữ lại K nhánh tiềm năng nhất tại mỗi bước. Tối ưu hơn Greedy nhưng tốn kém và đôi khi vẫn thiếu tự nhiên.
*   **Sampling (Lấy mẫu ngẫu nhiên):** Chọn từ dựa trên xác suất (có tính ngẫu nhiên).
    *   **Temperature (Nhiệt độ):**
        *   $T \to 0$: Trở về Greedy (chính xác, ít sáng tạo).
        *   $T \to \infty$: Phân phối phẳng (rất sáng tạo nhưng dễ nói nhảm).
    *   **Top-k Sampling:** Chỉ chọn trong K từ có xác suất cao nhất.
    *   **Top-p (Nucleus) Sampling:** Chỉ chọn trong nhóm từ có tổng xác suất tích lũy đạt ngưỡng P (ví dụ 0.9). *Phổ biến nhất hiện nay.*

---

## 5. Tối ưu hóa Inference
Chạy LLM tốn kém chủ yếu do băng thông bộ nhớ (Memory Bound).

### KV Cache
*   Trong quá trình sinh từ (Auto-regressive), các token phía trước không đổi.
*   Thay vì tính lại Key và Value cho toàn bộ chuỗi mỗi lần sinh từ mới, ta **lưu lại (Cache)** các Key/Value cũ và chỉ tính thêm cho token mới nhất.
*   Giúp tăng tốc độ suy luận đáng kể nhưng tốn VRAM.

### PagedAttention (vLLM)
*   Quản lý bộ nhớ KV Cache giống như hệ điều hành quản lý RAM (phân trang - paging).
*   Giảm lãng phí bộ nhớ (fragmentation), cho phép batch size lớn hơn -> Tăng throughput.

### Speculative Decoding (Giải mã đầu cơ)
*   Dùng một mô hình nhỏ (Draft model) chạy nhanh để "đoán" trước vài từ.
*   Dùng mô hình lớn (Target model) để kiểm tra lại các từ đó song song.
*   Nếu đoán đúng -> Chấp nhận hàng loạt (Tăng tốc). Nếu sai -> Sửa lại.
*   Tận dụng việc mô hình lớn tính toán song song tốt hơn là chạy tuần tự từng từ.

---
*Biên soạn bởi Pixiboss - Dựa trên Stanford CME 295.*
