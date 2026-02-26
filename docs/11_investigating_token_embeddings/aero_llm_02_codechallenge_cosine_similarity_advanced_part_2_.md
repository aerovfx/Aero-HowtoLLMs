
<!-- Aero-Navigation-Start -->
[🏠 Home](../index.md) > [11 investigating token embeddings](index.md)

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
Cosine Similarity nâng cao (Phần 2):

Phân tích hình học xác suất, anisotropy và tối ưu hoá trong không gian embedding chiều cao

⸻

Tóm tắt

Tiếp nối phần trước về Cosine Similarity, bài viết này mở rộng phân tích sang các vấn đề nâng cao bao gồm: hiện tượng anisotropy trong embedding space, phân phối góc trong không gian chiều cao, ảnh hưởng của chuẩn hóa (normalization), whitening transformation, và vai trò của cosine similarity trong contrastive learning và retrieval hiện đại. Các công thức toán học được trình bày nhằm làm rõ bản chất hình học – xác suất của các embedding được huấn luyện bởi mô hình ngôn ngữ lớn (LLMs).

⸻

1. Giới thiệu

Embedding không còn là vector ngẫu nhiên đơn giản; chúng được huấn luyện thông qua tối ưu hóa gradient, dẫn đến cấu trúc hình học đặc biệt. Các tổ chức như:
	•	OpenAI
	•	Google Research
	•	Meta AI

đã ứng dụng cosine similarity làm lõi cho:
	•	Semantic search
	•	Retrieval-Augmented Generation (RAG)
	•	Vector database indexing

Tuy nhiên, embedding thực tế không phân bố đều trong không gian $\mathbb${R}^d.

⸻

2. Phân phối góc trong không gian chiều cao

Giả sử:

\mathbf{x}, \mathbf{y} \sim $\mathcal${N}(0, I_d)

Sau chuẩn hóa:

\tilde{\mathbf{x}} = \frac{\mathbf{x}}{\|\mathbf{x}\|}

Phân phối của:

\cos \theta = \tilde{\mathbf{x}} \cdot \tilde{\mathbf{y}}

Khi d \to $\infty$:

\cos \theta \xrightarrow{p} 0

Và phương sai:

$$
Var$\cos \theta$ $\approx$ \frac{1}{d}
$$

Điều này giải thích vì sao trong embedding dimension lớn (512–4096), các vector ngẫu nhiên gần như trực giao.

⸻

3. Hiện tượng Anisotropy

3.1 Định nghĩa

Anisotropy xảy ra khi embedding tập trung quanh một hướng ưu thế.

Giả sử trung bình embedding:

$$
\mu = $\mathbb${E}[\mathbf{x}]
$$

Nếu:

\|\mu\| \gg 0

→ embedding lệch hướng.

⸻

3.2 Hệ quả

Cosine similarity giữa hai vector bất kỳ:

\cos$\mathbf{x}, \mathbf{y}$

bị chi phối bởi thành phần chung theo hướng \mu.

⸻

4. Centering và Whitening

4.1 Centering

Loại bỏ trung bình:

\mathbf{x}' = \mathbf{x} - \mu

⸻

4.2 Whitening Transformation

Cho ma trận hiệp phương sai:

$$
\Sigma = $\mathbb${E}[$\mathbf{x}-\mu
$$

