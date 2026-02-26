
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
So sánh Embedding Huấn luyện trên Wikipedia và Twitter

Phân tích Phân bố, Hình học và Khả năng Khái quát hóa Ngữ nghĩa

⸻

Tóm tắt

Dựa trên tài liệu đính kèm “CodeChallenge: Wikipedia vs. Twitter embeddings (part 1)”, bài viết này phân tích sự khác biệt giữa các embedding từ được huấn luyện trên hai miền dữ liệu khác nhau: văn bản bách khoa toàn thư (Wikipedia) và văn bản mạng xã hội (Twitter).

Chúng tôi mở rộng phân tích bằng các cơ sở lý thuyết từ Word2Vec, GloVe và các kết quả về phân bố Zipf của George Kingsley Zipf.

Bài viết cung cấp các mô hình toán học minh họa sự khác biệt về:
	•	Phân bố tần suất từ
	•	Ma trận đồng xuất hiện
	•	Entropy và mutual information
	•	Cấu trúc hình học của không gian embedding
	•	Khả năng khái quát hóa liên miền

⸻

1. Giới thiệu

Embedding từ học được từ corpus phụ thuộc mạnh vào:

$$
\mathcal{D} = \{w_1, w_2, \dots, w_T\}
$$

Hai miền:

$$
•	\mathcal{D}_{wiki}: Wikipedia
$$

$$
•	\mathcal{D}_{twitter}: Twitter
$$

Ta xây dựng ánh xạ:

$$
E_\mathcal{D}: V \rightarrow \mathbb{R}^d
$$

Mục tiêu: so sánh E_{wiki} và E_{twitter}.

⸻

2. Phân bố Tần suất và Luật Zipf

Theo luật Zipf:

$$
fr \propto \frac{1}{r^\alpha}
$$

Trong đó:
	•	r: thứ hạng
	•	f$r$: tần suất

Ta ước lượng:

\alpha_{wiki} \neq \alpha_{twitter}

Twitter có:
	•	Nhiều từ lóng
	•	Hashtag
	•	Viết tắt

Entropy từ vựng:

$$

$$

H = -\sum_i P(w_i)\log P(w_i)

$$

$$

Thường:

H_{twitter} > H_{wiki}

Do phân bố phẳng hơn.

⸻

3. Ma trận Đồng xuất hiện

Với GloVe:

$$

$$

X_{ij} = \text{số lần } w_j \text{ xuất hiện trong ngữ cảnh của } w_i

$$

$$

Ta có:

X^{wiki} \neq X^{twitter}

Sự khác biệt thể hiện ở:
	•	Từ học thuật (wiki)
	•	Biểu tượng cảm xúc, hashtag (twitter)

Log-count:

$$

$$

w_i^\top w_j \approx \log X_{ij}

$$

$$

⸻

4. Không gian Hình học Embedding

Embedding:

$$
Ew \in \mathbb{R}^d
$$

Khoảng cách cosine:

$$
\text{sim}(i,j) =
$$

\frac{E$w_i$^\top E$w_j$}
{\|E$w_i$\\mid  \|E$w_j$\|}

⸻

4.1 Độ lệch miền (Domain Shift)

Giả sử:

$$
\Deltaw = \\mid  E_{wiki}w - E_{twitter}w \|_2
$$

Nếu:

\Delta$w$ \gg 0

→ từ có ngữ nghĩa khác nhau theo miền.

Ví dụ:
	•	“apple” (công ty vs trái cây)
	•	“viral” (sinh học vs mạng xã hội)

⸻

5. Mutual Information giữa từ và miền

Xét biến ngẫu nhiên:
	•	W: từ
	•	D \in \{wiki, twitter\}

Mutual information:

$$

$$

I(W;D) = \sum_{w,d} P(w,d)\log\frac{P(w,d)}{P(w)P(d)}

$$

$$

Nếu:

I(W;D) \text{ cao}

→ từ đặc trưng miền.

⸻

6. Tính Tuyến tính và Quan hệ Ngữ nghĩa

Wikipedia thường giữ cấu trúc tuyến tính rõ:

$$

$$

w_{Paris} - w_{France} + w_{Germany} \approx w_{Berlin}

$$

$$

Twitter có thể nhiễu hơn do:
	•	Từ viết tắt
	•	Thiếu chuẩn hóa

Sai số:

$$
\epsilon =
$$

\| ($w_a$ - $w_b$ + $w_c$) - $w_d$ \|_2

Thường:

\epsilon_{twitter} > \epsilon_{wiki}

⸻

7. Độ tổng quát hóa (Generalization)

Giả sử huấn luyện classifier:

$$
f(E(w)) = y
$$

Huấn luyện trên wiki, test trên twitter:

Sai số:

$\mathcal${L}_{cross-domain}

Tăng theo khoảng cách phân bố:

D_{KL}$P_{wiki} \\mid  P_{twitter}$

Với:

$$

$$

D_{KL}P\\mid Q = \sum_i P(i)\log\frac{P(i)}{Qi}

$$

$$

⸻

8. Phân tích SVD và Cấu trúc Phổ

Ma trận đồng xuất hiện:

$$

$$

X = U\Sigma V^\top

$$

$$

So sánh phổ trị riêng:

\lambda_r^{wiki} \neq \lambda_r^{twitter}

Wikipedia thường có:
	•	Phổ giảm chậm
	•	Cấu trúc ngữ nghĩa ổn định

Twitter:
	•	Nhiễu cao
	•	Phổ phẳng hơn

⸻

9. Ảnh hưởng đến Transformer

Embedding đầu vào cho mô hình như BERT hoặc GPT chịu ảnh hưởng miền dữ liệu.

Self-attention:

$$
\text{Attention}(Q,K,V)=
$$

$$
\text{softmax}\left\frac{QK^\top}{\sqrt{d}}\rightV
$$

$$
Nếu embedding nhiễu: \|QK^\top\| \text{ giảm ổn định} → attention phân tán hơn. ⸻ 10. Phân tích Định lượng Ta định nghĩa: 10.1 Độ tương đồng trung bình
$$

\bar{s} =

$$
\frac{1}{|P|} \sum_{(i,j)\in P} \text{sim}(i,j) 10.2 Độ lệch miền trung bình
$$

\bar{\Delta} =