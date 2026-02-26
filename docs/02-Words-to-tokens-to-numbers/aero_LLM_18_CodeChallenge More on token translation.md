
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
Mở rộng Bài toán Chuyển đổi Token:

Phân tích Hình thức, Định lượng Sai số và Ảnh hưởng đến Biểu diễn Ngữ nghĩa

⸻

Tóm tắt

Dựa trên tài liệu đính kèm “More on Token Translation”, bài viết này mở rộng phân tích bài toán chuyển đổi giữa các hệ tokenizer trong mô hình ngôn ngữ lớn (LLMs). Chúng tôi xây dựng một khung toán học cho ánh xạ giữa hai không gian token rời rạc, phân tích sai số tích lũy khi chuyển đổi nhiều bước, đề xuất mô hình ma trận xác suất chuyển đổi, và đánh giá ảnh hưởng đến embedding và attention trong kiến trúc Transformer. Các ví dụ được minh họa với tokenizer của BERT và GPT-2.

⸻

1. Giới thiệu

Tokenization định nghĩa một phép mã hóa:

\mathcal{T}: \Sigma^* \rightarrow V^*

với:
	•	\Sigma: bảng chữ cái ký tự
	•	V: tập token
	•	V^*: chuỗi token

Khi tồn tại hai tokenizer \mathcal{T}_A và \mathcal{T}_B, bài toán đặt ra là xây dựng ánh xạ:

\Phi_{A \to B}: V_A^* \rightarrow V_B^*

sao cho bảo toàn nội dung ngữ nghĩa và hạn chế sai số thông tin.

⸻

2. Phân rã Hai Bước: Decode và Re-tokenize

Cách tự nhiên nhất:

\Phi_{A \to B} = \mathcal{T}_B \circ \mathcal{D}_A

Trong đó:
	•	\mathcal{D}_A: V_A^* \rightarrow \Sigma^* là hàm giải mã

Khi tokenizer khả nghịch:

\mathcal{D}_A(\mathcal{T}_A(x)) = x

Tuy nhiên, trong thực tế có thể xuất hiện chuẩn hóa Unicode hoặc xử lý khoảng trắng gây sai số.

⸻

3. Sai số Tích lũy khi Chuyển đổi Nhiều Lần

Giả sử thực hiện chuỗi chuyển đổi:

A \to B \to C

Sai số tổng:

\epsilon_{A \to C} \le \epsilon_{A \to B} + \epsilon_{B \to C}

Đây là hệ quả của bất đẳng thức tam giác đối với khoảng cách Levenshtein:

d(x,z) \le d(x,y) + d(y,z)

Nếu mỗi bước có sai số nhỏ nhưng lặp nhiều lần, sai số tích lũy có thể tăng tuyến tính theo số bước:

\epsilon_k \le k \epsilon

⸻

4. Mô hình Xác suất cho Chuyển đổi Token

Thay vì ánh xạ xác định, ta định nghĩa phân bố xác suất:

P(b_j \mid a_i)

Tạo thành ma trận:

M \in \mathbb{R}^{|V_A| \times |V_B|}

với:

\sum_{j} M_{ij} = 1

Khi đó embedding có thể chuyển đổi tuyến tính:

E_B = M^\top E_A

Trong đó:
	•	E_A \in \mathbb{R}^{|V_A| \times d}
	•	E_B \in \mathbb{R}^{|V_B| \times d}

⸻

5. Phân tích Sai số Ngữ nghĩa

Giả sử embedding của token:

e(a_i), \quad e(b_j)

Sai số chuyển đổi:

\delta_i = \| e(a_i) - \sum_j M_{ij} e(b_j) \|_2

Sai số trung bình:

\mathbb{E}[\delta] = \frac{1}{|V_A|} \sum_i \delta_i

Nếu embedding hai mô hình nằm trong cùng không gian ngữ nghĩa, ta có thể tối ưu:

\min_M \sum_i \delta_i^2

⸻

6. Ảnh hưởng đến Self-Attention

Cho văn bản độ dài n ký tự:

m_A = \frac{n}{\mathbb{E}[L_A]}

m_B = \frac{n}{\mathbb{E}[L_B]}

Chi phí attention:

C_A = O(m_A^2)

C_B = O(m_B^2)

Tỷ lệ:

\frac{C_A}{C_B} = \left(\frac{\mathbb{E}[L_B]}{\mathbb{E}[L_A]}\right)^2

Tokenizer tạo token dài hơn giúp giảm chi phí tính toán.

⸻

7. Căn chỉnh Span Ký tự

Mỗi token tương ứng một đoạn ký tự:

a_i \leftrightarrow [s_i, e_i)

b_j \leftrightarrow [u_j, v_j)

Bài toán căn chỉnh trở thành:

\text{match}(a_i, b_j) \iff [s_i, e_i) \cap [u_j, v_j) \neq \emptyset

Có thể xây dựng ánh xạ nhiều-nhiều.

⸻

8. Độ phức tạp Thuật toán

Nếu:
	•	Chuỗi có m token ở A
	•	k token ở B

Thuật toán căn chỉnh span có thể thực hiện trong:

O(m + k)

vì chỉ cần quét hai con trỏ.

Tuy nhiên nếu so khớp embedding:

O(mk)

⸻

9. Liên hệ đến Lý thuyết Thông tin

Entropy của phân bố token:

H(V) = - \sum_{t \in V} p(t)\log p(t)

Chuyển tokenizer làm thay đổi phân bố:

\Delta H = |H(V_A) - H(V_B)|

Theo Claude Shannon (1948), entropy đo lượng thông tin trung bình trên mỗi token.

⸻

10. Thảo luận

Mở rộng từ tài liệu đính kèm, có thể thấy:
	1.	Token translation không chỉ là thao tác chuỗi
	2.	Là bài toán ánh xạ giữa hai hệ mã hóa rời rạc
	3.	Có thể xem như biến đổi tuyến tính trong không gian embedding
	4.	Sai số có thể tích lũy nếu chuyển đổi nhiều bước

Trong thực tế, các hệ như OpenAI hay Google thiết kế tokenizer gắn chặt với kiến trúc mô hình, do đó việc chuyển đổi đòi hỏi phân tích cẩn trọng.

⸻

11. Kết luận

Bài toán chuyển đổi tokenizer có thể được mô hình hóa:

\Phi_{A \to B} = \mathcal{T}_B \circ \mathcal{D}_A

Sai số tích lũy:

\epsilon_k \le k \epsilon

Embedding có thể chuyển đổi bằng:

E_B = M^\top E_A

Đây là một bài toán kết hợp giữa:
	•	Lý thuyết mã hóa
	•	Lý thuyết thông tin
	•	Tối ưu hóa tuyến tính
	•	Kiến trúc Transformer

⸻

Tài liệu tham khảo
	1.	Devlin et al. (2018). BERT: Pre-training of Deep Bidirectional Transformers.
	2.	Radford et al. (2019). GPT-2: Language Models are Unsupervised Multitask Learners.
	3.	Vaswani et al. (2017). Attention Is All You Need.
	4.	Shannon, C. (1948). A Mathematical Theory of Communication.
	5.	Kudo & Richardson (2018). SentencePiece: A simple and language independent subword tokenizer.
	6.	Sennrich et al. (2016). Neural Machine Translation of Rare Words with Subword Units.
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
| [aero_LLM_17_Translating between tokenizers.md](aero_LLM_17_Translating between tokenizers.md) | [Xem bài viết →](aero_LLM_17_Translating between tokenizers.md) |
| 📌 **[aero_LLM_18_CodeChallenge More on token translation.md](aero_LLM_18_CodeChallenge More on token translation.md)** | [Xem bài viết →](aero_LLM_18_CodeChallenge More on token translation.md) |
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
