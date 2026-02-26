
<!-- Aero-Navigation-Start -->
[🏠 Home](../../index.md) > [11 Investigating token embeddings](../index.md)

---
### 🧭 Điều hướng nhanh

- [🏠 Cổng tài liệu](../../index.md)
- [📚 Module 01: LLM Course](../../01-LLM_Course/index.md)
- [🔢 Module 02: Tokenization](../../02-Words-to-tokens-to-numbers/index.md)
- [🏗️ Module 04: Build GPT](../../04-buildGPT/index.md)
- [🎯 Module 07: Fine-tuning](../../07-Fine-tune-pretrained-models/index.md)
- [🔍 Module 19: AI Safety](../../19-AI-safety/index.md)
- [🐍 Module 20: Python for AI](../../20-Python-Colab-notebooks/index.md)
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

Tuy nhiên, embedding thực tế không phân bố đều trong không gian \mathbb{R}^d.

⸻

2. Phân phối góc trong không gian chiều cao

Giả sử:

\mathbf{x}, \mathbf{y} \sim \mathcal{N}(0, I_d)

Sau chuẩn hóa:

\tilde{\mathbf{x}} = \frac{\mathbf{x}}{\|\mathbf{x}\|}

Phân phối của:

\cos \theta = \tilde{\mathbf{x}} \cdot \tilde{\mathbf{y}}

Khi d \to \infty:

\cos \theta \xrightarrow{p} 0

Và phương sai:

Var(\cos \theta) \approx \frac{1}{d}

Điều này giải thích vì sao trong embedding dimension lớn (512–4096), các vector ngẫu nhiên gần như trực giao.

⸻

3. Hiện tượng Anisotropy

3.1 Định nghĩa

Anisotropy xảy ra khi embedding tập trung quanh một hướng ưu thế.

Giả sử trung bình embedding:

\mu = \mathbb{E}[\mathbf{x}]

Nếu:

\|\mu\| \gg 0

→ embedding lệch hướng.

⸻

3.2 Hệ quả

Cosine similarity giữa hai vector bất kỳ:

\cos(\mathbf{x}, \mathbf{y})

bị chi phối bởi thành phần chung theo hướng \mu.

⸻

4. Centering và Whitening

4.1 Centering

Loại bỏ trung bình:

\mathbf{x}' = \mathbf{x} - \mu

⸻

4.2 Whitening Transformation

Cho ma trận hiệp phương sai:

\Sigma = \mathbb{E}[(\mathbf{x}-\mu)(\mathbf{x}-\mu)^T]

Whitening:

\mathbf{x}_{white} = \Sigma^{-1/2}(\mathbf{x}-\mu)

Khi đó:

Cov(\mathbf{x}_{white}) = I

Điều này giúp phân phối đồng đều hơn trong không gian.

⸻

5. Cosine Similarity và Contrastive Learning

Trong contrastive loss:

\mathcal{L}_i =
- \log
\frac{\exp(\cos(\mathbf{z}_i,\mathbf{z}_j)/\tau)}
{\sum_k \exp(\cos(\mathbf{z}_i,\mathbf{z}_k)/\tau)}

Trong đó:
	•	\tau: temperature
	•	\mathbf{z}: embedding đã chuẩn hóa

Khi \tau \to 0:

\exp(\cos/\tau)

khuếch đại sự khác biệt góc nhỏ.

⸻

6. Cosine Similarity và Maximum Likelihood

Giả sử embedding query q và document d:

P(d|q) =
\frac{\exp(\alpha \cos(q,d))}
{\sum_j \exp(\alpha \cos(q,d_j))}

Đây chính là softmax over cosine scores.

Hàm log-likelihood:

\mathcal{L} =
\sum_i \log P(d_i|q_i)

⸻

7. Phân tích Gradient trong Không gian Chuẩn hóa

Cho:

S = \frac{\mathbf{x} \cdot \mathbf{y}}{\|\mathbf{x}\|\|\mathbf{y}\|}

Gradient theo \mathbf{x}:

\frac{\partial S}{\partial \mathbf{x}} =
\frac{\mathbf{y}}{\|\mathbf{x}\|\|\mathbf{y}\|}
-
\frac{(\mathbf{x}\cdot\mathbf{y})\mathbf{x}}
{\|\mathbf{x}\|^3\|\mathbf{y}\|}

Gradient này gồm hai thành phần:
	1.	Hướng về phía \mathbf{y}
	2.	Thành phần điều chỉnh độ lớn

Điều này làm embedding tự động chuẩn hóa hướng thay vì độ lớn.

⸻

8. Liên hệ với Information Geometry

Theo Pattern Recognition and Machine Learning:

Khoảng cách Bregman với entropy:

D_\phi(p,q) =
\phi(p) - \phi(q) - \nabla\phi(q)^T(p-q)

Cosine similarity không phải metric Bregman nhưng có thể xem như metric góc trên hypersphere:

S^{d-1} =
\{\mathbf{x} \in \mathbb{R}^d : \|\mathbf{x}\| = 1\}

⸻

9. Ứng dụng trong Vector Database

Các hệ thống như:
	•	FAISS (Meta AI)
	•	ScaNN (Google Research)

sử dụng cosine similarity hoặc inner product.

Nếu vector chuẩn hóa:

\mathbf{x} \cdot \mathbf{y}
=
\cos(\mathbf{x},\mathbf{y})

→ tối ưu tính toán bằng Approximate Nearest Neighbor.

⸻

10. So sánh với các metric khác

Metric	Công thức	Nhạy độ lớn	Phù hợp NLP
Euclidean	\|\mathbf{x}-\mathbf{y}\|	Có	Trung bình
Dot Product	x \cdot y	Có	Cao
Cosine	\frac{x\cdot y}{\|x\|\|y\|}	Không	Rất cao


⸻

11. Thảo luận

Ưu điểm
	•	Bất biến theo scale
	•	Phù hợp cho embedding chuẩn hóa
	•	Tối ưu cho retrieval

Hạn chế
	•	Không xử lý tốt anisotropy
	•	Không đo quan hệ phi tuyến
	•	Dễ bị cluster collapse nếu không regularize

⸻

12. Kết luận

Cosine similarity trong embedding hiện đại không chỉ là phép đo hình học đơn giản mà là:
	•	Metric trên hypersphere
	•	Thành phần cốt lõi của contrastive learning
	•	Cơ sở cho retrieval và vector database

Hiểu rõ phân phối góc, anisotropy và whitening giúp cải thiện đáng kể chất lượng embedding trong LLM.

⸻

Tài liệu tham khảo
	1.	Bishop (2006). Pattern Recognition and Machine Learning.
	2.	Cover & Thomas (2006). Elements of Information Theory.
	3.	Mikolov et al. (2013). Word2Vec.
	4.	Chen et al. (2020). SimCLR.
	5.	Reimers & Gurevych (2019). Sentence-BERT.
<!-- Aero-Footer-Start -->
---
## 🤝 Liên hệ & Đóng góp
Dự án được phát triển bởi **Pixibox**. Mọi đóng góp về nội dung và mã nguồn đều được chào đón.

> *"Kiến thức là để chia sẻ. Hãy cùng nhau xây dựng cộng đồng AI vững mạnh!"* 🚀

*Cập nhật tự động bởi Aero-Indexer - 2026*
<!-- Aero-Footer-End -->
