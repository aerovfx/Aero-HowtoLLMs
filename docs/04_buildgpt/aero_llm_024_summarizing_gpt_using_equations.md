
<!-- Aero-Navigation-Start -->
[🏠 Home](../index.md) > [04 buildgpt](index.md)

---
### 🧭 Điều hướng nhanh

- [🏠 Cổng tài liệu](../index.md)
- [📚 Module 01: LLM Course](../01_llm_course/index.md)
- [🔢 Module 02: Tokenization](../02_words_to_tokens_to_numbers/index.md)
- [🏗️ Module 04: Build GPT](../04_buildgpt/index.md)
- [🎯 Module 07: Fine-tuning](../07_fine_tune_pretrained_models/index.md)
- [🔍 Module 19: AI Safety](../19_ai_safety/index.md)
- [🐍 Module 20: Python for AI](../20_python_colab_notebooks/index.md)
---
<!-- Aero-Navigation-End -->
# Kiến Trúc Transformer và Triển Khai GPT-2 trên GPU: Phân Tích Toán Học và Hiệu Năng Tính Toán

## Tóm tắt

Các mô hình ngôn ngữ lớn (Large Language Models – LLMs) dựa trên kiến trúc Transformer đã đạt được nhiều thành tựu trong xử lý ngôn ngữ tự nhiên. Bài viết này trình bày tổng quan về cấu trúc toán học của GPT-2, cơ chế multi-head attention, quy trình huấn luyện và suy luận, cũng như đánh giá hiệu năng khi triển khai trên CPU và GPU. Thông qua phân tích lý thuyết và thực nghiệm, nghiên cứu cho thấy GPU đóng vai trò thiết yếu trong việc vận hành các mô hình ngôn ngữ hiện đại.

---

## 1. Giới thiệu

Transformer là nền tảng của hầu hết các mô hình ngôn ngữ hiện đại. Kiến trúc này cho phép mô hình hóa mối quan hệ dài hạn giữa các token thông qua cơ chế attention. GPT-2 là một trong những mô hình tiêu biểu sử dụng Transformer để sinh ngôn ngữ tự nhiên.

Việc triển khai hiệu quả các mô hình này đòi hỏi sự kết hợp giữa hiểu biết toán học, thiết kế kiến trúc và tối ưu phần cứng.

---

## 2. Biểu diễn Embedding và Dữ liệu Đầu vào

Trong GPT-2, mỗi token được ánh xạ sang một vector embedding thông qua ma trận từ vựng $E \in $\mathbb${R}^{V \times D}$, kết hợp với embedding vị trí $P \in $\mathbb${R}^{L \times D}$. Quá trình này được mô tả bằng one-hot encoding và phép nhân ma trận.

Phép biến đổi từ token sang embedding được thực hiện thông qua:

$$

$$

X = \Delta E + P

$$

$$

trong đó X \in \mathbb{R}^{T \times D} là ma trận biểu diễn chuỗi đầu vào.

$$
Quá trình này được trình bày chi tiết trong tài liệu tổng hợp toán học về GPT. --- ## 3. Cơ Chế Multi-Head Attention ### 3.1. Nguyên lý toán học Multi-head attention chia không gian embedding thành nhiều phần (heads) song song. Với mỗi head h, ta có:
$$

$$
Q_h = XW_Q^h, \quad K_h = XW_K^h, \quad V_h = XW_V^h
$$

$$
Sau đó, attention được tính:
$$

$$
A_h = \text{softmax}\left(\frac{Q_h K_h^T}{\sqrt{D/H}} + M \right)V_h
$$

$$
Các đầu ra được nối lại và chiếu tuyến tính:
$$

$$
A = \text{Concat}(A_1, \dots, A_H)W_0
$$

$$
Việc chia nhỏ attention giúp mô hình học được nhiều kiểu quan hệ ngữ nghĩa khác nhau. --- ### 3.2. Triển khai trong PyTorch Trong thực tế, các ma trận (W_Q, W_K, W_V) thường được gộp thành một ma trận duy nhất để tăng hiệu suất. Quá trình reshape và transpose được sử dụng để tách các head trong forward pass. Việc sử dụng hàm attention tích hợp giúp tối ưu tính toán song song trên GPU. --- ## 4. Khối Transformer và Mạng MLP ### 4.1. Attention Sub-layer Mỗi khối Transformer bắt đầu bằng layer normalization, sau đó là multi-head attention và residual connection:
$$

$$
X' = X + \text{Attention}(\text{LN}(X))
$$

$$
### 4.2. Feed-Forward Network (MLP) Sau attention, dữ liệu được đưa qua mạng MLP gồm hai lớp tuyến tính:
$$

$$
Y = X' + W_2(\text{GELU}(W_1(\text{LN}(X'))))
$$

$$
Mạng MLP giúp mô hình trích xuất đặc trưng phi tuyến trong không gian chiều cao. --- ## 5. Unembedding và Sinh Token Đầu ra cuối cùng được chuẩn hóa và nhân với ma trận embedding ban đầu để tạo logits:
$$

$$
L = \text{LN}(X_{out})E^T
$$
