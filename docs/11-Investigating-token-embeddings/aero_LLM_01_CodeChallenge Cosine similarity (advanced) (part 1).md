
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
Phân tích Cosine Similarity nâng cao trong không gian embedding

Cơ sở toán học, hình học vector và ứng dụng trong mô hình ngôn ngữ lớn

⸻

Tóm tắt

Cosine Similarity là một trong những thước đo cốt lõi trong xử lý ngôn ngữ tự nhiên (NLP), đặc biệt khi làm việc với vector embedding có chiều cao. Bài viết này trình bày nền tảng toán học của Cosine Similarity, mở rộng sang các phân tích hình học trong không gian Hilbert, mối liên hệ với chuẩn hóa vector, phân phối xác suất trong embedding space, và ứng dụng trong retrieval, semantic search và đánh giá mô hình ngôn ngữ lớn (LLMs). Ngoài ra, bài viết bổ sung các công thức minh họa và liên hệ với lý thuyết thông tin.

⸻

1. Giới thiệu

Trong NLP hiện đại, văn bản được ánh xạ sang vector trong không gian \mathbb{R}^d thông qua embedding models. Các tổ chức như:
	•	OpenAI
	•	Google Research
	•	Meta AI

đã phát triển các hệ embedding cho:
	•	Semantic search
	•	Retrieval-augmented generation (RAG)
	•	Clustering
	•	Similarity detection

Trong các hệ này, Cosine Similarity là thước đo chuẩn để so sánh hai vector.

⸻

2. Định nghĩa Cosine Similarity

Cho hai vector \mathbf{x}, \mathbf{y} \in \mathbb{R}^d:

\text{cosine\_sim}(\mathbf{x}, \mathbf{y}) =
\frac{\mathbf{x} \cdot \mathbf{y}}{\|\mathbf{x}\| \|\mathbf{y}\|}

Trong đó:
	•	Tích vô hướng:

\mathbf{x} \cdot \mathbf{y} = \sum_{i=1}^{d} x_i y_i
	•	Chuẩn Euclid:

\|\mathbf{x}\| = \sqrt{\sum_{i=1}^{d} x_i^2}

⸻

3. Diễn giải hình học

Cosine similarity đo cos của góc giữa hai vector:

\cos \theta = \frac{\mathbf{x} \cdot \mathbf{y}}{\|\mathbf{x}\|\|\mathbf{y}\|}

Giá trị:
	•	1 → cùng hướng
	•	0 → trực giao
	•	-1 → ngược hướng

Trong embedding NLP, vector thường được chuẩn hóa:

\tilde{\mathbf{x}} = \frac{\mathbf{x}}{\|\mathbf{x}\|}

Khi đó:

\text{cosine\_sim}(\mathbf{x}, \mathbf{y}) =
\tilde{\mathbf{x}} \cdot \tilde{\mathbf{y}}

⸻

4. Không gian chiều cao và hiện tượng tập trung

Trong không gian chiều cao d \gg 1:
	•	Các vector ngẫu nhiên có xu hướng gần trực giao
	•	Góc giữa hai vector ngẫu nhiên tiệm cận 90^\circ

Theo lý thuyết xác suất:

Nếu x_i, y_i \sim \mathcal{N}(0,1)

\mathbb{E}[\mathbf{x} \cdot \mathbf{y}] = 0

Var(\mathbf{x} \cdot \mathbf{y}) = d

Sau chuẩn hóa:

\mathbb{E}[\cos \theta] \approx 0

Hiện tượng này gọi là concentration of measure.

⸻

5. Quan hệ với khoảng cách Euclid

Khoảng cách Euclid:

\|\mathbf{x} - \mathbf{y}\|^2 =
\|\mathbf{x}\|^2 + \|\mathbf{y}\|^2 - 2\mathbf{x}\cdot\mathbf{y}

Nếu chuẩn hóa:

\|\tilde{\mathbf{x}} - \tilde{\mathbf{y}}\|^2 =
2 - 2\cos \theta

Do đó:

\cos \theta = 1 - \frac{1}{2}\|\tilde{\mathbf{x}} - \tilde{\mathbf{y}}\|^2

→ Cosine similarity tương đương với Euclidean distance trong không gian chuẩn hóa.

⸻

6. Cosine Similarity trong embedding xác suất

Một embedding model ánh xạ văn bản t thành vector:

f_\theta(t) \in \mathbb{R}^d

Xác suất chọn tài liệu d_i trong retrieval:

P(d_i|q) =
\frac{\exp(\alpha \cdot \cos(f(q), f(d_i)))}
{\sum_j \exp(\alpha \cdot \cos(f(q), f(d_j)))}

Trong đó:
	•	\alpha là temperature scaling

⸻

7. Liên hệ với Information Theory

Theo Elements of Information Theory:

Mutual information giữa hai vector embedding:

I(X;Y) =
\mathbb{E}\left[
\log \frac{P(X,Y)}{P(X)P(Y)}
\right]

Cosine similarity có thể xem như xấp xỉ thô của sự phụ thuộc tuyến tính giữa hai biến.

⸻

8. Cosine Similarity và Loss Function

Trong contrastive learning (ví dụ SimCLR):

\mathcal{L} =
- \log
\frac{\exp(\cos(\mathbf{x}_i,\mathbf{x}_j)/\tau)}
{\sum_k \exp(\cos(\mathbf{x}_i,\mathbf{x}_k)/\tau)}

Trong đó:
	•	\tau là temperature
	•	(\mathbf{x}_i,\mathbf{x}_j) là positive pair

⸻

9. Phân tích gradient

Giả sử:

S = \frac{\mathbf{x} \cdot \mathbf{y}}{\|\mathbf{x}\|\|\mathbf{y}\|}

Gradient theo \mathbf{x}:

\frac{\partial S}{\partial \mathbf{x}} =
\frac{\mathbf{y}}{\|\mathbf{x}\|\|\mathbf{y}\|}
-
\frac{(\mathbf{x}\cdot\mathbf{y})\mathbf{x}}
{\|\mathbf{x}\|^3\|\mathbf{y}\|}

Điều này cho thấy quá trình tối ưu sẽ:
	•	Kéo vector cùng hướng lại gần
	•	Đẩy vector khác hướng ra xa

⸻

10. Ứng dụng trong LLM

Các ứng dụng thực tế:
	•	Semantic Search
	•	Retrieval-Augmented Generation
	•	Clustering câu hỏi
	•	Detect duplicate content

Các tổ chức như Stanford University và MIT đã sử dụng cosine similarity trong các hệ thống IR và NLP hiện đại.

⸻

11. Hạn chế
	1.	Không nhạy với độ lớn vector
	2.	Không nắm bắt quan hệ phi tuyến
	3.	Bị ảnh hưởng bởi anisotropy trong embedding space

Một số nghiên cứu đề xuất:
	•	Whitening transformation
	•	Centering embeddings
	•	Angular margin loss

⸻

12. Kết luận

Cosine Similarity là thước đo hình học cơ bản nhưng cực kỳ hiệu quả trong NLP hiện đại. Trong không gian embedding chiều cao, nó:
	•	Ổn định
	•	Dễ tính toán
	•	Phù hợp cho retrieval

Tuy nhiên, cần kết hợp với chuẩn hóa và kỹ thuật regularization để đạt hiệu năng tối ưu.

⸻

Tài liệu tham khảo
	1.	Cover & Thomas (2006). Elements of Information Theory.
	2.	Bishop (2006). Pattern Recognition and Machine Learning.
	3.	Chen et al. (2020). SimCLR: A Simple Framework for Contrastive Learning.
	4.	Mikolov et al. (2013). Word2Vec.
	5.	Reimers & Gurevych (2019). Sentence-BERT.
<!-- Aero-Footer-Start -->
---
## 🤝 Liên hệ & Đóng góp
Dự án được phát triển bởi **Pixibox**. Mọi đóng góp về nội dung và mã nguồn đều được chào đón.

> *"Kiến thức là để chia sẻ. Hãy cùng nhau xây dựng cộng đồng AI vững mạnh!"* 🚀

*Cập nhật tự động bởi Aero-Indexer - 2026*
<!-- Aero-Footer-End -->
