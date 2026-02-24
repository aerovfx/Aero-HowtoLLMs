# Lecture 2: Transformer Tricks & BERT 🛠️

> **Tóm tắt từ khóa học Stanford CME 295: Transformers & Large Language Models.**
> Bài giảng này đi sâu vào các cải tiến kỹ thuật giúp Transformer hoạt động tốt hơn và sự ra đời của các mô hình Encoder-only như BERT.

---

## 📚 Mục Lục
1. [Cải tiến Positional Embeddings (RoPE, ALiBi)](#1-cải-tiến-positional-embeddings)
2. [Cải tiến Normalization (LayerNorm vs RMSNorm)](#2-cải-tiến-normalization)
3. [Tối ưu Attention (Sliding Window, GQA)](#3-tối-ưu-attention)
4. [Các họ mô hình Transformer](#4-các-họ-mô-hình-transformer)
5. [BERT & Encoder-only Models](#5-bert--encoder-only-models)

---

## 1. Cải tiến Positional Embeddings
Trong bài báo gốc, Positional Encoding được cộng trực tiếp vào Input Embedding. Tuy nhiên, các mô hình hiện đại sử dụng các phương pháp tiên tiến hơn để xử lý tốt hơn độ dài chuỗi thay đổi.

### Learned Positional Embedding
*   Học một vector riêng cho mỗi vị trí.
*   **Hạn chế:** Không thể mở rộng (extrapolate) cho các chuỗi dài hơn độ dài đã thấy trong khi huấn luyện.

### Rotary Positional Embedding (RoPE) 🔄
*   **Hiện đại nhất:** Được sử dụng trong Llama, Mistral, PaLM.
*   **Cơ chế:** Thay vì cộng vector vị trí, RoPE **xoay** vector Query và Key một góc phụ thuộc vào vị trí của chúng.
*   **Ưu điểm:**
    *   Mô hình học được **khoảng cách tương đối** (relative distance) giữa các từ một cách tự nhiên thông qua tích vô hướng (dot product).
    *   Khả năng mở rộng (extrapolation) tốt hơn cho các chuỗi dài.

### ALiBi (Attention with Linear Biases)
*   Thêm một bias tĩnh vào ma trận Attention score dựa trên khoảng cách giữa hai token.
*   Đơn giản, không cần học tham số, nhưng RoPE hiện nay phổ biến hơn.

---

## 2. Cải tiến Normalization
Chuẩn hóa (Normalization) giúp mô hình hội tụ nhanh và ổn định hơn.

*   **Post-Norm (Gốc):** Norm sau khi cộng nhánh dư (Residual).
*   **Pre-Norm (Hiện đại):** Norm **trước** khi vào Attention/FFN. Giúp huấn luyện ổn định hơn với các mô hình rất sâu.
*   **RMSNorm (Root Mean Square Norm):** Một biến thể của LayerNorm, bỏ qua việc trừ giá trị trung bình (mean), chỉ chia cho căn bậc hai trung bình bình phương.
    *   *Ưu điểm:* Tính toán nhanh hơn, hiệu quả tương đương. Được dùng trong Llama, Gopher.

---

## 3. Tối ưu Attention
Self-Attention có độ phức tạp $O(N^2)$ (với N là độ dài chuỗi), rất tốn kém khi chuỗi dài.

*   **Sliding Window Attention (Cửa sổ trượt):** Mỗi token chỉ nhìn thấy các token lân cận trong một cửa sổ nhất định (ví dụ: Mistral). Giảm chi phí tính toán nhưng vẫn giữ được khả năng hiểu ngữ cảnh nhờ các lớp chồng lên nhau (tương tự Receptive field trong CNN).
*   **Grouped Query Attention (GQA):**
    *   *Multi-Head Attention (MHA):* Mỗi Head có Q, K, V riêng. (Tốn bộ nhớ KV Cache).
    *   *Multi-Query Attention (MQA):* Tất cả Heads chia sẻ chung 1 bộ K, V. (Tiết kiệm nhớ, giảm chất lượng).
    *   *GQA:* Trung hòa. Chia Heads thành các nhóm, mỗi nhóm chia sẻ chung K, V. (Cân bằng tốt nhất giữa tốc độ và chất lượng, dùng trong Llama-2-70b, Llama-3).

---

## 4. Các họ mô hình Transformer
Dựa trên kiến trúc, có 3 nhánh phát triển chính:

1.  **Encoder-Decoder (T5, BART):** Giỏi các tác vụ "Text-to-Text" như dịch thuật, tóm tắt.
2.  **Encoder-only (BERT, RoBERTa):** Chỉ dùng phần Encoder. Giỏi các tác vụ "Hiểu ngôn ngữ" (NLU) như phân loại, tìm kiếm, NER.
3.  **Decoder-only (GPT, Llama):** Chỉ dùng phần Decoder. Giỏi các tác vụ "Sinh ngôn ngữ" (Generative). Đây là nhánh thống trị hiện nay của LLMs.

---

## 5. BERT & Encoder-only Models
**BERT (Bidirectional Encoder Representations from Transformers)** là tượng đài của dòng Encoder-only.

### Đặc điểm:
*   **Bidirectional (Hai chiều):** Mỗi từ nhìn thấy toàn bộ câu (trái và phải) cùng lúc.
*   **Tokens đặc biệt:** `[CLS]` (đại diện cho toàn câu, dùng để phân loại), `[SEP]` (ngăn cách câu).

### Quá trình Huấn luyện (Pre-training)
BERT được huấn luyện với 2 tác vụ tự giám sát (Self-supervised):
1.  **Masked Language Modeling (MLM):** Ẩn đi 15% số từ trong câu, yêu cầu mô hình điền vào chỗ trống. (Giúp mô hình học ngữ cảnh hai chiều).
2.  **Next Sentence Prediction (NSP):** Cho 2 câu A và B, hỏi B có phải là câu tiếp theo của A không? (Giúp mô hình hiểu mối quan hệ giữa các câu).

### Fine-tuning (Tinh chỉnh)
Sau khi Pre-training, BERT tạo ra các vector embedding rất tốt. Ta chỉ cần gắn thêm một lớp Linear nhỏ phía sau để giải quyết các bài toán cụ thể (Sentiment Analysis, Question Answering) với rất ít dữ liệu.

### Biến thể
*   **DistilBERT:** Dùng kỹ thuật *Distillation* (Chưng cất) để tạo mô hình nhỏ hơn, nhanh hơn nhưng giữ được 97% hiệu năng của BERT.
*   **RoBERTa:** Tối ưu hóa BERT (bỏ NSP, train lâu hơn, dữ liệu nhiều hơn) -> Hiệu năng tốt hơn.

---
*Biên soạn bởi Pixiboss - Dựa trên Stanford CME 295.*
