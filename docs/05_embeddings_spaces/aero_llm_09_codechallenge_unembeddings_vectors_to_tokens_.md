
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
# Unembedding trong Mô Hình Ngôn Ngữ Lớn: Từ Vector Ẩn Đến Token

## Tóm tắt

Trong các mô hình ngôn ngữ dựa trên Transformer, quá trình “embedding” ánh xạ token rời rạc sang không gian vector liên tục. Tuy nhiên, bước ngược lại – chuyển từ vector ẩn sang phân phối xác suất trên token – được gọi là **unembedding**. Bài viết này phân tích nền tảng toán học của unembedding trong [GPT-2](chatgpt://generic-entity?number=0), làm rõ vai trò của weight tying, tích vô hướng, softmax và cấu trúc hình học của không gian từ vựng.

---

## 1. Giới thiệu

Quá trình xử lý văn bản trong mô hình ngôn ngữ có thể tóm tắt:

$$
\text{Token} \rightarrow \text{Embedding} \rightarrow \text{Transformer layers} \rightarrow \text{Hidden state} \rightarrow \text{Unembedding} \rightarrow \text{Softmax}
$$

Nếu embedding là ánh xạ:

f: \mathcal{V} \rightarrow \mathbb{R}^d

thì unembedding là ánh xạ ngược:

g: \mathbb{R}^d \rightarrow \mathbb{R}^{|\mathcal{V}|}

---

## 2. Embedding: Từ token đến vector

Giả sử từ vựng có kích thước $|V\mid$, ma trận embedding:

E \in \mathbb{R}^{|V| \times d}

Với token chỉ số $i$:

\mathbf{v}_i = E[i]

Nếu biểu diễn one-hot $\mathbf{x}_i$:

\mathbf{v}_i = \mathbf{x}_i E

---

## 3. Unembedding: Từ vector đến token

Sau khi qua các lớp Transformer, ta thu được hidden state:

\mathbf{h}_t \in \mathbb{R}^d

Để chuyển sang logit:

\mathbf{z} = W_U \mathbf{h}_t

Trong đó:

W_U \in \mathbb{R}^{|V| \times d}

Vector logit:

z_i = \mathbf{w}_i \cdot \mathbf{h}_t

---

## 4. Weight Tying

Trong GPT-2, thường sử dụng weight tying:

W_U = E

hoặc:

W_U = E^T

Khi đó:

z_i = \mathbf{v}_i \cdot \mathbf{h}_t

Điều này có ý nghĩa hình học:

> Logit của token $i$ chính là tích vô hướng giữa embedding của token đó và hidden state.

---

## 5. Softmax và phân phối xác suất

Xác suất dự đoán token tiếp theo:

P(w_i  \mid  h_t) = \frac{e^{z_i}} {\sum_{j=1}^{|V|} e^{z_j}}

Thay z_i = \mathbf{v}_i \cdot \mathbf{h}_t:

$$
P(w_i) = \frac{ \exp(\mathbf{v}_i \cdot \mathbf{h}_t) } { \sum_j \exp(\mathbf{v}_j \cdot \mathbf{h}_t) } Nếu chuẩn hóa: \mathbf{v}_i \cdot \mathbf{h}_t = \|\mathbf{v}_i\| \|\mathbf{h}_t\| \cos \theta_i Suy ra: P(w_i) \propto \exp( \|\mathbf{v}_i\| \|\mathbf{h}_t\| \cos \theta_i ) Góc giữa vector quyết định xác suất. --- ## 6. Diễn giải hình học Hidden state \mathbf{h}_t có thể xem như: - Một “truy vấn ngữ nghĩa” - Một điểm trong không gian embedding Unembedding thực hiện phép chiếu: \mathbf{z} = E \mathbf{h}_t Nghĩa là ta đo mức độ “gần” giữa \mathbf{h}_t và từng vector từ vựng. Nếu hai token có embedding gần nhau: \mathbf{v}_i \approx \mathbf{v}_j thì: z_i \approx z_j Do đó phân phối xác suất sẽ tương tự. --- ## 7. Hàm mất mát và tối ưu hóa Hàm mất mát cross-entropy: \mathcal{L} = - \log P(w_{true}) Gradient theo \mathbf{h}_t: \nabla_{\mathbf{h}_t} \mathcal{L} = \sum_i P(w_i)\mathbf{v}_i - \mathbf{v}_{true} Điều này cho thấy: - Hidden state được điều chỉnh về phía embedding đúng - Và đẩy xa embedding sai --- ## 8. So sánh với phân loại tuyến tính Unembedding tương đương một bộ phân loại tuyến tính: z_i = \mathbf{w}_i^T \mathbf{h}_t Khác biệt là: - Số lớp rất lớn (~50k) - Trọng số gắn trực tiếp với embedding --- ## 9. Quan hệ với Cosine Similarity Nếu chuẩn hóa embedding: \hat{\mathbf{v}}_i = \frac{\mathbf{v}_i}{\|\mathbf{v}_i\|} Khi đó: z_i = \|\mathbf{v}_i\| \|\mathbf{h}_t\| \cos\theta_i Nếu bỏ qua độ lớn: z_i \propto \cos\theta_i Như vậy unembedding về bản chất dựa trên cosine similarity. --- ## 10. Phân tích phổ (Spectral Perspective) Giả sử ma trận embedding: E = U \Sigma V^T (SVD decomposition) Hidden state: \mathbf{h}_t = V \mathbf{c} Logit: \mathbf{z} = U \Sigma \mathbf{c} Các giá trị singular lớn chi phối phân phối xác suất. --- ## 11. Ý nghĩa lý thuyết Unembedding: 1. Chuyển từ không gian liên tục sang rời rạc. 2. Là phép chiếu tuyến tính quy mô lớn. 3. Phụ thuộc trực tiếp vào cấu trúc hình học của embedding. 4. Tạo liên kết chặt chẽ giữa học biểu diễn và dự đoán xác suất. Về mặt toán học: \text{Prediction} = \text{Softmax}(E \mathbf{h}_t)
$$

