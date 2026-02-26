<!-- Aero-Navigation-Start -->
[🏠 Home](../index.md) > [02 words to tokens to numbers](index.md)

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

# Chuẩn bị văn bản cho Tokenization trong mô hình Transformer: Cơ sở lý thuyết và phân tích toán học

---

## Tóm tắt

Bài viết trình bày quy trình **chuẩn bị văn bản trước khi tokenization** trong các mô hình ngôn ngữ lớn dựa trên kiến trúc Transformer. Nội dung bao gồm: chuẩn hóa văn bản, làm sạch dữ liệu, Byte Pair Encoding (BPE), và ánh xạ sang không gian embedding. Các công thức toán học mô tả xác suất chuỗi, ánh xạ rời rạc–liên tục và cấu trúc đại số của quá trình mã hóa được trình bày chặt chẽ.

---

# 1. Giới thiệu

Trong các mô hình Transformer hiện đại, văn bản đầu vào trải qua chuỗi biến đổi:

$$
\text{Raw Text} \rightarrow \text{Normalization} \rightarrow \text{Tokenization} \rightarrow \text{Embedding}
$$

Gọi chuỗi ký tự ban đầu là:

$$
X = (c_1, c_2, \dots, c_n)
$$

Tokenizer định nghĩa ánh xạ:

$$
\tau : \Sigma^{\ast} \rightarrow \mathbb{Z}^m
$$

Trong đó:

- $\Sigma$ là bảng chữ cái
- $\mathbb{Z}^m$ là chuỗi ID token
- $m \le n$

---

# 2. Chuẩn hóa văn bản (Text Normalization)

## 2.1 Lowercasing

$$
f_{\text{lower}}(c) = \text{lowercase}(c)
$$

Ví dụ:

$$
\text{"ChatGPT"} \rightarrow \text{"chatgpt"}
$$

---

## 2.2 Unicode Normalization

Một ký tự có thể có nhiều biểu diễn Unicode.

Ví dụ:

$$
\text{é} = e + \text{´}
$$

Chuẩn hóa NFC đảm bảo:

$$
\text{NFC}(x_1) = \text{NFC}(x_2)
$$

nếu hai chuỗi tương đương ngữ nghĩa.

---

# 3. Tokenization và mô hình xác suất

Mô hình ngôn ngữ tối ưu:

$$
P(X) = \prod_{t=1}^{T} P(x_t \mid x_{<t})
$$

Nếu làm việc ở mức ký tự:

$$
T = n
$$

Giải pháp: chia thành token:

$$
X = (w_1, w_2, \dots, w_m), \quad m < n
$$

Giảm độ dài chuỗi và tăng khả năng biểu diễn.

---

# 4. Byte Pair Encoding (BPE)

## 4.1 Thuật toán

Khởi tạo:

$$
V_0 = \{ \text{tập ký tự đơn} \}
$$

Lặp:

1. Tìm cặp ký tự xuất hiện nhiều nhất
2. Gộp thành token mới
3. Cập nhật từ vựng

Tần suất cặp:

$$
f(a,b) = \sum_i \mathbf{1}[(a,b) \in X_i]
$$

Chọn:

$$
(a^{\ast}, b^{\ast}) = \arg\max_{a,b} f(a,b)
$$

Cập nhật:

$$
V_{k+1} = V_k \cup \{ab\}
$$

---

# 5. Ánh xạ sang embedding

Sau tokenization:

$$
w_i \rightarrow \text{id}_i \in \{1, \dots, |V|\}
$$

Ma trận embedding:

$$
E \in \mathbb{R}^{|V| \times d}
$$

Ánh xạ:

$$
e_i = E[\text{id}_i]
$$

Chuỗi embedding:

$$
X \rightarrow (e_1, e_2, \dots, e_m)
$$

---

# 6. Phân tích độ phức tạp

Chi phí xây dựng BPE:

$$
\mathcal{O}(N \log V)
$$

Chi phí suy luận tokenization:

$$
\mathcal{O}(m)
$$

---

# 7. Out-of-Vocabulary (OOV)

BPE đảm bảo:

$$
\forall x \in \Sigma^{\ast}, \quad \exists \text{ decomposition into subwords}
$$

Do đó:

$$
P(x) > 0
$$

cho mọi chuỗi hợp lệ.

---

# 8. So sánh các phương pháp tokenization

| Phương pháp        | Ưu điểm            | Hạn chế              |
|-------------------|-------------------|----------------------|
| Word-level        | Ngắn, dễ hiểu     | OOV cao              |
| Character-level   | Không OOV         | Chuỗi rất dài        |
| BPE               | Cân bằng tốt      | Phụ thuộc dữ liệu    |
| Unigram LM        | Tối ưu xác suất   | Tính toán cao        |

Unigram Language Model tối ưu:

$$
\max_{\theta} \prod_i \sum_{z \in \mathcal{Z}(x_i)} P(z \mid \theta)
$$

---

# 9. Tác động đến Attention

Self-attention có độ phức tạp:

$$
\mathcal{O}(T^2)
$$

Tokenization kém → $T$ lớn → chi phí tăng mạnh.

Tokenizer tốt giúp:

- Giảm memory footprint  
- Tăng tốc inference  
- Cải thiện biểu diễn ngữ nghĩa  

---

# 10. Liên hệ với GPT

Các mô hình GPT sử dụng biến thể của BPE hoặc byte-level BPE.

Xác suất sinh token:

$$
P(w_t = i \mid w_{<t}) =
\frac{\exp((z_t W_{\text{out}})_i)}
{\sum_j \exp((z_t W_{\text{out}})_j)}
$$

Chất lượng tokenization ảnh hưởng trực tiếp đến phân phối logits.

---

# 11. Góc nhìn thông tin học

Theo Shannon:

$$
H(X) = - \sum_x P(x) \log P(x)
$$

Tokenizer tốt giúp độ dài chuỗi token xấp xỉ:

$$
\text{Length}(X_{\text{tokens}})
\approx
\frac{H(X)}{\log |V|}
$$

---

# 12. Kết luận

Quy trình chuẩn bị văn bản bao gồm:

1. Chuẩn hóa Unicode  
2. Làm sạch dữ liệu  
3. Áp dụng BPE hoặc Unigram LM  
4. Ánh xạ sang embedding  

Về mặt toán học, tokenization là ánh xạ:

$$
\Sigma^{\ast} \rightarrow V^{\ast}
$$

đóng vai trò cầu nối giữa không gian ký tự rời rạc và không gian vector liên tục trong Transformer.

📚 Tài liệu tham khảo (bổ sung)

1. Transformer & GPT
	1.	Vaswani, A., et al. (2017).
Attention Is All You Need. NeurIPS.
→ Bài báo nền tảng giới thiệu kiến trúc Transformer.
	2.	Radford, A., et al. (2019).
Language Models are Unsupervised Multitask Learners. OpenAI.
→ GPT-2 và cơ chế autoregressive modeling.
	3.	Brown, T., et al. (2020).
Language Models are Few-Shot Learners. NeurIPS.
→ GPT-3 và scaling law.
	4.	OpenAI (2023).
GPT-4 Technical Report.
→ Tổng quan kỹ thuật về GPT-4.

⸻

2. Tokenization & Subword Methods
	5.	Sennrich, R., Haddow, B., Birch, A. (2016).
Neural Machine Translation of Rare Words with Subword Units. ACL.
→ BPE trong NLP.
	6.	Kudo, T. (2018).
Subword Regularization: Improving Neural Network Translation Models with Multiple Subword Candidates. ACL.
→ Unigram Language Model (SentencePiece).
	7.	Kudo, T., Richardson, J. (2018).
SentencePiece: A simple and language independent subword tokenizer. EMNLP.
	8.	Gage, P. (1994).
A New Algorithm for Data Compression.
→ BPE gốc trong nén dữ liệu.

⸻

3. Information Theory
	9.	Shannon, C. E. (1948).
A Mathematical Theory of Communication. Bell System Technical Journal.
	10.	Cover, T., Thomas, J. (2006).
Elements of Information Theory. Wiley.

⸻

4. Representation & Embedding
	11.	Mikolov, T., et al. (2013).
Efficient Estimation of Word Representations in Vector Space. arXiv.
	12.	Pennington, J., Socher, R., Manning, C. (2014).
GloVe: Global Vectors for Word Representation. EMNLP.
	13.	Jurafsky, D., Martin, J. (2023 draft).
Speech and Language Processing (3rd ed.).

⸻

5. Complexity & Scaling Laws
	14.	Kaplan, J., et al. (2020).
Scaling Laws for Neural Language Models. arXiv.
	15.	Hoffmann, J., et al. (2022).
Training Compute-Optimal Large Language Models. (Chinchilla paper)
