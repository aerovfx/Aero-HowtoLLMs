
<!-- Aero-Navigation-Start -->
[🏠 Home](../../index.md) > [02 Words to tokens to numbers](../index.md)

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
Chuyển đổi giữa các Tokenizer trong Mô hình Ngôn ngữ Lớn:

Phân tích Lý thuyết, Biểu diễn Hình thức và Hệ quả Tính toán

⸻

Tóm tắt

Tokenization là bước tiền xử lý cốt lõi trong các mô hình ngôn ngữ lớn (LLMs). Tuy nhiên, sự khác biệt giữa các thuật toán token hóa như WordPiece, BPE và Unigram LM tạo ra những thách thức khi cần chuyển đổi giữa các tokenizer khác nhau. Bài viết này, dựa trên tài liệu đính kèm về Translating between tokenizers, trình bày một khung lý thuyết hình thức cho bài toán ánh xạ giữa hai hệ tokenizer, phân tích điều kiện tồn tại ánh xạ song ánh, ước lượng sai số thông tin, và đánh giá tác động đến độ phức tạp tính toán trong Transformer. Các ví dụ được minh họa với tokenizer của BERT, GPT-2, và thư viện SentencePiece.

⸻

1. Giới thiệu

Mỗi mô hình ngôn ngữ định nghĩa một hàm token hóa:

\mathcal{T}: \Sigma^* \rightarrow V^*

Trong đó:
	•	\Sigma^*: tập tất cả chuỗi ký tự
	•	V: từ vựng token
	•	V^*: chuỗi token

Hai tokenizer khác nhau \mathcal{T}_A và \mathcal{T}_B sẽ tạo ra hai biểu diễn khác nhau cho cùng một chuỗi đầu vào x:

\mathcal{T}_A(x) \neq \mathcal{T}_B(x)

Vấn đề đặt ra:
Làm thế nào để ánh xạ chuỗi token từ không gian V_A^* sang V_B^* mà không mất thông tin?

⸻

2. Các Hệ Tokenizer Phổ biến

2.1 WordPiece

Được sử dụng trong BERT do Google phát triển.

Thuật toán tối đa hóa xác suất:

\arg\max_{s_1,\dots,s_k} \prod_{i=1}^{k} P(s_i)

⸻

2.2 Byte Pair Encoding (BPE)

Được sử dụng trong GPT-2 bởi OpenAI.

Quá trình lặp:

(\alpha, \beta) = \arg\max_{(u,v)} \text{freq}(uv)

Sau đó thay thế cặp phổ biến nhất.

⸻

2.3 Unigram Language Model

Áp dụng trong SentencePiece.

Tối ưu hóa:

\max_{V} \sum_{x \in D} \log \sum_{s \in \mathcal{S}(x)} \prod_{i} P(s_i)

⸻

3. Mô hình Toán học của Bài toán Chuyển đổi

Giả sử:

\mathcal{T}_A: \Sigma^* \rightarrow V_A^*

\mathcal{T}_B: \Sigma^* \rightarrow V_B^*

Ta cần xây dựng:

\Phi: V_A^* \rightarrow V_B^*

3.1 Điều kiện tồn tại ánh xạ chính xác

Nếu tồn tại hàm giải mã:

\mathcal{D}_A: V_A^* \rightarrow \Sigma^*

thì:

\Phi = \mathcal{T}_B \circ \mathcal{D}_A

Khi đó:

\Phi(\mathcal{T}_A(x)) = \mathcal{T}_B(x)

⸻

4. Phân tích Sai số Thông tin

Nếu tokenizer không khả nghịch hoàn toàn, ta có sai số:

\epsilon = d(\mathcal{D}_A(\mathcal{T}_A(x)), x)

Trong đó d là khoảng cách Levenshtein.

Entropy trước và sau:

H_A = - \sum p(t_i)\log p(t_i)

H_B = - \sum p(u_j)\log p(u_j)

Độ chênh entropy:

\Delta H = |H_A - H_B|

Nếu \Delta H lớn → thay đổi cấu trúc phân bố token đáng kể.

⸻

5. Ảnh hưởng đến Độ dài Chuỗi và Self-Attention

Giả sử văn bản có n ký tự.

Số token:

m_A = \frac{n}{\mathbb{E}[L_A]}

m_B = \frac{n}{\mathbb{E}[L_B]}

Self-attention có độ phức tạp:

O(m^2)

Tỷ lệ chi phí:

\frac{C_A}{C_B} = \left(\frac{m_A}{m_B}\right)^2

Nếu tokenizer B tạo token dài hơn:

\mathbb{E}[L_B] > \mathbb{E}[L_A]
\Rightarrow C_B < C_A

⸻

6. Bài toán Căn chỉnh Token (Token Alignment)

Giả sử:

\mathcal{T}_A(x) = (a_1, a_2, \dots, a_m)

\mathcal{T}_B(x) = (b_1, b_2, \dots, b_k)

Ta cần tìm ánh xạ căn chỉnh:

\pi: \{1,\dots,m\} \rightarrow \{1,\dots,k\}

Tối ưu hóa:

\min_{\pi} \sum_{i=1}^{m} d(\text{span}(a_i), \text{span}(b_{\pi(i)}))

Đây tương đương bài toán căn chỉnh chuỗi động (dynamic programming).

⸻

7. Biểu diễn Ma trận Ánh xạ

Ta có thể định nghĩa ma trận chuyển đổi:

M \in \mathbb{R}^{|V_A| \times |V_B|}

Trong đó:

M_{ij} = P(b_j \mid a_i)

Nếu ánh xạ xác định:

M_{ij} \in \{0,1\}

Nếu ánh xạ xác suất:

\sum_j M_{ij} = 1

⸻

8. Ứng dụng Thực tiễn
	1.	Chuyển embedding giữa hai mô hình
	2.	Fine-tune chéo tokenizer
	3.	Distillation giữa hai LLM
	4.	Interoperability giữa hệ sinh thái NLP

⸻

9. Thảo luận

Sự khác biệt giữa tokenizer không chỉ ảnh hưởng đến:
	•	Độ dài chuỗi
	•	Chi phí attention
	•	Entropy hệ biểu diễn

Mà còn ảnh hưởng đến:
	•	Phân bố gradient
	•	Ổn định huấn luyện
	•	Tính chuyển giao embedding

Bài toán chuyển đổi tokenizer thực chất là bài toán ánh xạ giữa hai hệ mã hóa rời rạc có cấu trúc phân cấp.

⸻

10. Kết luận

Việc chuyển đổi giữa hai tokenizer có thể được mô hình hóa hình thức bằng:

\Phi = \mathcal{T}_B \circ \mathcal{D}_A

Sai số thông tin được đo bằng:

\epsilon = d(\mathcal{D}_A(\mathcal{T}_A(x)), x)

Độ phức tạp tính toán phụ thuộc vào:

O\left(\left(\frac{n}{\mathbb{E}[L]}\right)^2\right)

Thiết kế tokenizer không chỉ là vấn đề tiền xử lý mà là một thành phần cấu trúc của toàn bộ kiến trúc Transformer.

⸻

Tài liệu tham khảo
	1.	Devlin et al. (2018). BERT: Pre-training of Deep Bidirectional Transformers.
	2.	Radford et al. (2019). Language Models are Unsupervised Multitask Learners.
	3.	Kudo & Richardson (2018). SentencePiece: A simple and language independent subword tokenizer.
	4.	Sennrich et al. (2016). Neural Machine Translation of Rare Words with Subword Units.
	5.	Shannon, C. (1948). A Mathematical Theory of Communication.
	6.	Vaswani et al. (2017). Attention Is All You Need.
<!-- Aero-Footer-Start -->

## 📄 Tài liệu cùng chuyên mục
| Bài học | Liên kết |
| :--- | :--- |
| [Tại sao văn bản cần được đánh số?](aero_LLM_01_Why text needs to be numbered.md) | [Xem bài viết →](aero_LLM_01_Why text needs to be numbered.md) |
| [Phân tích và chuyển đổi văn bản thành chuỗi token được đánh số: Cơ sở toán học và ứng dụng trong mô hình ngôn ngữ lớn](aero_LLM_02_Parsing text to numbered tokens.md) | [Xem bài viết →](aero_LLM_02_Parsing text to numbered tokens.md) |
| [Tạo và trực quan hóa Token trong mô hình ngôn ngữ lớn: Cơ sở toán học và phân tích biểu diễn](aero_LLM_03_CodeChallenge Create and visualize tokens (part 1).md) | [Xem bài viết →](aero_LLM_03_CodeChallenge Create and visualize tokens (part 1).md) |
| [Tạo và trực quan hóa Token (Phần 2): Phân tích hình học không gian embedding và Attention Map trong mô hình Transformer](aero_LLM_04_CodeChallenge Create and visualize tokens (part 2).md) | [Xem bài viết →](aero_LLM_04_CodeChallenge Create and visualize tokens (part 2).md) |
| [Chuẩn bị văn bản cho Tokenization trong mô hình Transformer: Cơ sở lý thuyết và phân tích toán học](aero_LLM_05_Preparing text for tokenization.md) | [Xem bài viết →](aero_LLM_05_Preparing text for tokenization.md) |
| [Phân tích quy trình Tokenization qua ví dụ *The Time Machine*: Cơ sở thuật toán và mô hình hóa toán học](aero_LLM_06_CodeChallenge Tokenizing The Time Machine.md) | [Xem bài viết →](aero_LLM_06_CodeChallenge Tokenizing The Time Machine.md) |
| [So sánh Tokenization mức ký tự, từ và subword: Phân tích lý thuyết và mô hình toán học](aero_LLM_07_Tokenizing characters vs. subwords vs. words.md) | [Xem bài viết →](aero_LLM_07_Tokenizing characters vs. subwords vs. words.md) |
| [aero_LLM_08_Byte-pair encoding algorithm.md](aero_LLM_08_Byte-pair encoding algorithm.md) | [Xem bài viết →](aero_LLM_08_Byte-pair encoding algorithm.md) |
| [Thuật toán Byte Pair Encoding (BPE) và Bài toán Tối ưu Kích thước Từ vựng trong Mô hình Ngôn ngữ](aero_LLM_09_CodeChallenge Byte-pair encoding to a desired vocab size.md) | [Xem bài viết →](aero_LLM_09_CodeChallenge Byte-pair encoding to a desired vocab size.md) |
| [aero_LLM_10_Exploring ChatGPT4's tokenizer.md](aero_LLM_10_Exploring ChatGPT4's tokenizer.md) | [Xem bài viết →](aero_LLM_10_Exploring ChatGPT4's tokenizer.md) |
| [aero_LLM_11_CodeChallenge Token count by subword length (part 1).md](aero_LLM_11_CodeChallenge Token count by subword length (part 1).md) | [Xem bài viết →](aero_LLM_11_CodeChallenge Token count by subword length (part 1).md) |
| [aero_LLM_12_CodeChallenge Token count by subword length (part 2).md](aero_LLM_12_CodeChallenge Token count by subword length (part 2).md) | [Xem bài viết →](aero_LLM_12_CodeChallenge Token count by subword length (part 2).md) |
| [aero_LLM_13_How many rs in strawberry.md](aero_LLM_13_How many rs in strawberry.md) | [Xem bài viết →](aero_LLM_13_How many rs in strawberry.md) |
| [aero_LLM_14_CodeChallenge Create your algorithmic rapper name ).md](aero_LLM_14_CodeChallenge Create your algorithmic rapper name ).md) | [Xem bài viết →](aero_LLM_14_CodeChallenge Create your algorithmic rapper name ).md) |
| [aero_LLM_15_Tokenization in BERT.md](aero_LLM_15_Tokenization in BERT.md) | [Xem bài viết →](aero_LLM_15_Tokenization in BERT.md) |
| [aero_LLM_16_CodeChallenge Character counts in BERT tokens.md](aero_LLM_16_CodeChallenge Character counts in BERT tokens.md) | [Xem bài viết →](aero_LLM_16_CodeChallenge Character counts in BERT tokens.md) |
| 📌 **[aero_LLM_17_Translating between tokenizers.md](aero_LLM_17_Translating between tokenizers.md)** | [Xem bài viết →](aero_LLM_17_Translating between tokenizers.md) |
| [aero_LLM_18_CodeChallenge More on token translation.md](aero_LLM_18_CodeChallenge More on token translation.md) | [Xem bài viết →](aero_LLM_18_CodeChallenge More on token translation.md) |
| [aero_LLM_19_CodeChallenge Tokenization compression ratios.md](aero_LLM_19_CodeChallenge Tokenization compression ratios.md) | [Xem bài viết →](aero_LLM_19_CodeChallenge Tokenization compression ratios.md) |
| [aero_LLM_20_Tokenization in different languages.md](aero_LLM_20_Tokenization in different languages.md) | [Xem bài viết →](aero_LLM_20_Tokenization in different languages.md) |
| [aero_LLM_21_CodeChallenge Zipf's law in characters and tokens.md](aero_LLM_21_CodeChallenge Zipf's law in characters and tokens.md) | [Xem bài viết →](aero_LLM_21_CodeChallenge Zipf's law in characters and tokens.md) |
| [aero_LLM_22_Word variations in Claude tokenizer.md](aero_LLM_22_Word variations in Claude tokenizer.md) | [Xem bài viết →](aero_LLM_22_Word variations in Claude tokenizer.md) |

---
## 🤝 Liên hệ & Đóng góp
Dự án được phát triển bởi **Pixibox**. Mọi đóng góp về nội dung và mã nguồn đều được chào đón.

> *"Kiến thức là để chia sẻ. Hãy cùng nhau xây dựng cộng đồng AI vững mạnh!"* 🚀

*Cập nhật tự động bởi Aero-Indexer - 2026*
<!-- Aero-Footer-End -->
