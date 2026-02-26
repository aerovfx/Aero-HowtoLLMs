
<!-- Aero-Navigation-Start -->
[🏠 Home](../index.md) > [05 embeddings spaces](index.md)

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
# So sánh Biểu Diễn Ngữ Nghĩa của GPT-2 và BERT thông qua Phân Tích Embedding

## Tóm tắt

Các mô hình ngôn ngữ lớn dựa trên Transformer đã thay đổi nền tảng của xử lý ngôn ngữ tự nhiên (NLP). Hai kiến trúc tiêu biểu là [GPT-2](chatgpt://generic-entity?number=0) và [BERT](chatgpt://generic-entity?number=1). Mặc dù cùng dựa trên cơ chế self-attention, hai mô hình có mục tiêu huấn luyện và cấu trúc khác nhau, dẫn đến đặc tính embedding khác biệt. Bài viết này phân tích cơ sở toán học của embedding trong hai mô hình, so sánh cấu trúc không gian biểu diễn và minh họa bằng các công thức định lượng.

---

## 1. Giới thiệu

Trong NLP, một mô hình ngôn ngữ học phân phối xác suất có điều kiện:

$$
P(w_t \mid w_{\lt t})
$$

hoặc trong trường hợp hai chiều:

$$
P(w_i \mid w_{\setminus i})
$$

Tùy vào mục tiêu huấn luyện, embedding thu được sẽ mang đặc trưng khác nhau.

- GPT-2: mô hình tự hồi quy (autoregressive)
- BERT: mô hình hai chiều (bidirectional) với masked language modeling

---

## 2. Cơ sở kiến trúc Transformer

Cả hai mô hình đều dựa trên kiến trúc Transformer (Vaswani et al., 2017), với cơ chế **Scaled Dot-Product Attention**:

\text{Attention}(Q,K,V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V

Trong đó:

- $Q$: Query matrix  
- $K$: Key matrix  
- $V$: Value matrix  
- $d_k$: số chiều của vector key  

Self-attention cho phép mô hình học phụ thuộc dài hạn trong chuỗi.

---

## 3. Embedding trong GPT-2

### 3.1 Mục tiêu huấn luyện

$$
[GPT-2](chatgpt://generic-entity?number=2) được huấn luyện để tối đa hóa log-likelihood: \mathcal{L}_{GPT2} = \sum_{t=1}^{T} \log P(w_t \mid w_{\lt t}) Trong đó: P(w_t \mid w_{\lt t}) = \text{softmax}(W_o h_t) - h_t: hidden state tại vị trí t - W_o: ma trận chiếu đầu ra ### 3.2 Đặc điểm embedding Embedding của GPT-2 mang tính **ngữ cảnh một chiều**: \mathbf{h}_t = f(w_1, w_2, ..., w_t) Do đó, vector tại vị trí t chỉ phụ thuộc vào quá khứ. --- ## 4. Embedding trong BERT ### 4.1 Mục tiêu huấn luyện
$$

[BERT](chatgpt://generic-entity?number=3) sử dụng Masked Language Modeling (MLM):

$\mathcal${L}_{BERT} = $\sum$_{i \in M} $\log$ P($w_i$ \mid w_{\setminus i})

Trong đó:

- $M$: tập các vị trí bị mask
- $w_{\setminus i}$: toàn bộ chuỗi trừ vị trí $i$

### 4.2 Đặc điểm embedding

Embedding của BERT mang tính **hai chiều**:

\mathbf{h}_t = f(w_1, ..., w_T)

Do đó:

- Ngữ cảnh trái và phải đều ảnh hưởng
- Biểu diễn phù hợp cho tác vụ phân loại và suy luận ngữ nghĩa

---

## 5. So sánh hình học không gian embedding

Giả sử:

\mathbf{v}_i^{(GPT2)} \in \mathbb{R}^d

$$
\mathbf{v}_i^{(BERT)} \in \mathbb{R}^d ### 5.1 Độ tương đồng cosine \text{cosine}(\mathbf{v}_i, \mathbf{v}_j) = \frac{\mathbf{v}_i \cdot \mathbf{v}_j} {\|\mathbf{v}_i\|\|\mathbf{v}_j\|} ### 5.2 Khoảng cách Euclid d(\mathbf{v}_i,\mathbf{v}_j) = \|\mathbf{v}_i - \mathbf{v}_j\| = \sqrt{\sum_{k=1}^{d}(v_{ik}-v_{jk})^2} ### 5.3 Phân tích phương sai (PCA) Giả sử ma trận embedding: X \in \mathbb{R}^{n \times d} Ma trận hiệp phương sai: \Sigma = \frac{1}{n} X^T X Giải bài toán trị riêng: \Sigma \mathbf{u} = \lambda \mathbf{u} Các trị riêng lớn phản ánh chiều chiếm ưu thế trong không gian biểu diễn. --- ## 6. Phân tích định lượng Một số khác biệt quan sát được: | Thuộc tính | GPT-2 | BERT | |------------|--------|-------| | Hướng xử lý | Trái → Phải | Hai chiều | | Mục tiêu | Next-token prediction | Masked token prediction | | Embedding | Phù hợp sinh văn bản | Phù hợp phân loại | | Cấu trúc hình học | Mang tính tiến trình | Mang tính ngữ cảnh toàn cục | --- ## 7. Thảo luận ### 7.1 Tính ổn định ngữ nghĩa Nếu xét ma trận tương đồng nội bộ: S_{ij} = \text{cosine}(\mathbf{v}_i,\mathbf{v}_j) Ta có thể sử dụng tương quan Pearson giữa hai ma trận để đánh giá mức độ tương đồng cấu trúc: r = \frac{\sum (x_i - \bar{x})(y_i - \bar{y})} {\sqrt{\sum (x_i - \bar{x})^2} \sqrt{\sum (y_i - \bar{y})^2}} ### 7.2 Tính bất biến quay (Rotation Invariance) Giả sử tồn tại ma trận trực giao R: R^T R = I Khi đó: \mathbf{v}' = R\mathbf{v} Khoảng cách cosine không đổi, nhưng tọa độ thay đổi. --- ## 8. Kết luận - GPT-2 tối ưu hóa mô hình sinh chuỗi → embedding thiên về tiến trình. - BERT tối ưu hóa mô hình suy luận ngữ cảnh → embedding thiên về ngữ nghĩa toàn cục. - Phân tích hình học (cosine, Euclid, PCA, RSA) giúp hiểu cấu trúc biểu diễn. Về mặt toán học:
$$

\text{Objective Function} \Rightarrow \text{Geometry of Embedding Space}