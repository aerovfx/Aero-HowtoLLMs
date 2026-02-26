
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
Phân tích Số lượng Ký tự trong Token của BERT:

Mô hình Thống kê, Entropy và Ảnh hưởng đến Độ phức tạp Transformer

⸻

Tóm tắt

Bài viết này phân tích số lượng ký tự cấu thành mỗi token trong bộ tokenizer của BERT do Google phát triển, dựa trên dữ liệu từ tài liệu đính kèm. Chúng tôi xây dựng mô hình thống kê cho phân bố độ dài token, ước lượng entropy hệ subword, và phân tích tác động của độ dài ký tự đến độ phức tạp tính toán trong kiến trúc Transformer. Kết quả cho thấy phân bố độ dài token có xu hướng lệch phải (right-skewed), gần với phân bố hình học hoặc log-linear, phản ánh sự cân bằng giữa kích thước từ vựng và độ dài chuỗi đầu vào.

⸻

1. Giới thiệu

Trong BERT, văn bản đầu vào được token hóa bằng thuật toán WordPiece thành các subword token:

$$
S = (w_1, w_2, ..., w_n)
$$

$$
T = (t_1, t_2, ..., t_m)
$$

Với:

$$
m $\ge$ n
$$

Mỗi token t_i có độ dài ký tự:

$\ell$(t_i)

Mục tiêu nghiên cứu:
	1.	Phân bố xác suất của $\ell$(t)
	2.	Độ dài trung bình token
	3.	Ảnh hưởng đến chi phí self-attention

⸻

2. Mô hình Thống kê Phân bố Độ dài Token

2.1 Định nghĩa

Gọi:
	•	V: tập từ vựng BERT

$$
•	|V| $\approx$ 30{,}000
$$

	•	N_k: số token có độ dài ký tự bằng k

Xác suất:

$P(L = k)$ = \frac{N_k}{|V|}

Chuẩn hóa:

$$
$\sum$_{k=1}^{K_{\max}} $P(L=k)$ = 1
$$

⸻

2.2 Mô hình Hình học (Geometric Approximation)

Quan sát thực nghiệm cho thấy:

$$
N_k $\approx$ Ae^{-\lambda k}
$$

Suy ra:

$P(L=k)$ = (1-q)q^{k-1}

Trong đó:

$$
q = e^{-\lambda}
$$

Đây là phân bố hình học rời rạc.

⸻

2.3 Kỳ vọng và Phương sai

Kỳ vọng:

$$
$\mathbb${E}[L] = \frac{1}{1-q}
$$

Phương sai:

\mathrm{Var}$L$ = \frac{q}{(1-q)^2}

Nếu q \to 1, phân bố có đuôi dài hơn (nhiều token dài).

⸻

3. Ảnh hưởng đến Độ dài Chuỗi Văn bản

Giả sử văn bản có tổng số ký tự n.

Số token trung bình:

$$
m = \frac{n}{$\mathbb${E}[L]}
$$

Self-attention trong Transformer encoder:

$O(m^2)$

Thay vào:

$$
O$\le$ft($\le$ft(\frac{n}{$\mathbb${E}[L]}\right)^2\right)
$$

Khi $\mathbb${E}[L] \uparrow, chi phí giảm.

⸻

4. Entropy của Hệ Token

Entropy theo phân bố độ dài:

$$
H_L = - $\sum$_{k} $P(L=k)$\log $P(L=k)$
$$

Thay phân bố hình học:

$$
H_L = - $\sum$_{k=1}^{$\infty$} (1-q)q^{k-1} $\log$[(1-q)q^{k-1}]
$$

Rút gọn:

$$
H_L = -$\log$(1-q) - \frac{q}{1-q}$\log$ q
$$

Entropy càng lớn → độ đa dạng độ dài càng cao.

⸻

5. Quan hệ với Luật Zipf

Tần suất token thường tuân theo:

f$r$ $\propto$ \frac{1}{r^\alpha}

Trong đó:
	•	r: thứ hạng token

$$
•	\alpha $\approx$ 1
$$

Token ngắn thường:
	•	Có tần suất cao
	•	Ở thứ hạng thấp

Do đó tồn tại tương quan nghịch:

$\ell$(t) $\propto$ $\log$ r

⸻

6. Ảnh hưởng đến Embedding Matrix

Embedding:

E: V \rightarrow $\mathbb${R}^d

Ma trận embedding:

W \in $\mathbb${R}^{|V| \times d}

Bài toán tối ưu:

$$
\min_{V} $\le$ft( $\mathbb${E}[m] + \lambda |V| \right)
$$

Trong đó:
	•	$\mathbb${E}[m]: số token trung bình
	•	|V|: kích thước từ vựng
	•	\lambda: hệ số cân bằng

⸻

7. So sánh với Character-level Modeling

Mô hình	Độ dài trung bình	OOV	Chi phí
Character-level	1	Không	Rất cao
Word-level	Lớn	Cao	Trung bình
WordPiece	Trung bình	Thấp	Tối ưu

Nếu xử lý ở mức ký tự:

$$
m = n
$$

Chi phí:

$O(n^2)$

WordPiece giảm:

$$
m = \frac{n}{$\mathbb${E}[L]}
$$

⸻

8. Thảo luận

Dữ liệu thực nghiệm cho thấy:
	•	Phần lớn token có độ dài nhỏ (1–5 ký tự)
	•	Token dài tồn tại nhưng ít
	•	Phân bố có đuôi nhẹ (mild heavy-tail)

Điều này phản ánh:
	•	Sự cân bằng giữa khả năng tổng quát hóa và độ nén
	•	Tối ưu hóa thực nghiệm hơn là lý thuyết thuần túy

⸻

9. Kết luận

Phân bố độ dài ký tự của token trong BERT có thể mô hình hóa gần đúng bằng phân bố hình học:

$P(L=k)$ \sim q^{k-1}

Tác động trực tiếp đến:

$$
m = \frac{n}{$\mathbb${E}[L]}
$$

\text{Attention Cost} \sim $O(m^2)$

$$
H_L = - $\sum$ $P(L)$\log $P(L)$
$$

Thiết kế tokenizer là bài toán tối ưu đa mục tiêu giữa:
	•	Kích thước từ vựng
	•	Độ dài chuỗi
	•	Entropy thông tin
	•	Chi phí tính toán

⸻

Tài liệu tham khảo
	1.	BERT – Devlin et al. (2018). BERT: Pre-training of Deep Bidirectional Transformers.
	2.	Sennrich et al. (2016). Neural Machine Translation of Rare Words with Subword Units.
	3.	Vaswani et al. (2017). Attention Is All You Need.
	4.	Shannon, C. (1948). A Mathematical Theory of Communication.
	5.	Jurafsky & Martin. Speech and Language Processing.
<!-- Aero-Footer-Start -->

## 📄 Tài liệu cùng chuyên mục
| Bài học | Liên kết |
| :--- | :--- |
| [Tại sao văn bản cần được đánh số?](aero_llm_01_why_text_needs_to_be_numbered.md) | [Xem bài viết →](aero_llm_01_why_text_needs_to_be_numbered.md) |
| [Phân tích và chuyển đổi văn bản thành chuỗi token được đánh số: Cơ sở toán học và ứng dụng trong mô hình ngôn ngữ lớn](aero_llm_02_parsing_text_to_numbered_tokens.md) | [Xem bài viết →](aero_llm_02_parsing_text_to_numbered_tokens.md) |
| [Tạo và trực quan hóa Token trong mô hình ngôn ngữ lớn: Cơ sở toán học và phân tích biểu diễn](aero_llm_03_codechallenge_create_and_visualize_tokens_part_1_.md) | [Xem bài viết →](aero_llm_03_codechallenge_create_and_visualize_tokens_part_1_.md) |
| [Tạo và trực quan hóa Token (Phần 2): Phân tích hình học không gian embedding và Attention Map trong mô hình Transformer](aero_llm_04_codechallenge_create_and_visualize_tokens_part_2_.md) | [Xem bài viết →](aero_llm_04_codechallenge_create_and_visualize_tokens_part_2_.md) |
| [Chuẩn bị văn bản cho Tokenization trong mô hình Transformer: Cơ sở lý thuyết và phân tích toán học](aero_llm_05_preparing_text_for_tokenization.md) | [Xem bài viết →](aero_llm_05_preparing_text_for_tokenization.md) |
| [Phân tích quy trình Tokenization qua ví dụ *The Time Machine*: Cơ sở thuật toán và mô hình hóa toán học](aero_llm_06_codechallenge_tokenizing_the_time_machine.md) | [Xem bài viết →](aero_llm_06_codechallenge_tokenizing_the_time_machine.md) |
| [So sánh Tokenization mức ký tự, từ và subword: Phân tích lý thuyết và mô hình toán học](aero_llm_07_tokenizing_characters_vs_subwords_vs_words.md) | [Xem bài viết →](aero_llm_07_tokenizing_characters_vs_subwords_vs_words.md) |
| [aero llm 08 byte pair encoding algorithm](aero_llm_08_byte_pair_encoding_algorithm.md) | [Xem bài viết →](aero_llm_08_byte_pair_encoding_algorithm.md) |
| [Thuật toán Byte Pair Encoding (BPE) và Bài toán Tối ưu Kích thước Từ vựng trong Mô hình Ngôn ngữ](aero_llm_09_codechallenge_byte_pair_encoding_to_a_desired_vocab_size.md) | [Xem bài viết →](aero_llm_09_codechallenge_byte_pair_encoding_to_a_desired_vocab_size.md) |
| [aero llm 10 exploring chatgpt4 s tokenizer](aero_llm_10_exploring_chatgpt4_s_tokenizer.md) | [Xem bài viết →](aero_llm_10_exploring_chatgpt4_s_tokenizer.md) |
| [aero llm 11 codechallenge token count by subword length part 1](aero_llm_11_codechallenge_token_count_by_subword_length_part_1_.md) | [Xem bài viết →](aero_llm_11_codechallenge_token_count_by_subword_length_part_1_.md) |
| [aero llm 12 codechallenge token count by subword length part 2](aero_llm_12_codechallenge_token_count_by_subword_length_part_2_.md) | [Xem bài viết →](aero_llm_12_codechallenge_token_count_by_subword_length_part_2_.md) |
| [aero llm 13 how many rs in strawberry](aero_llm_13_how_many_rs_in_strawberry.md) | [Xem bài viết →](aero_llm_13_how_many_rs_in_strawberry.md) |
| [aero llm 14 codechallenge create your algorithmic rapper name](aero_llm_14_codechallenge_create_your_algorithmic_rapper_name_.md) | [Xem bài viết →](aero_llm_14_codechallenge_create_your_algorithmic_rapper_name_.md) |
| [aero llm 15 tokenization in bert](aero_llm_15_tokenization_in_bert.md) | [Xem bài viết →](aero_llm_15_tokenization_in_bert.md) |
| 📌 **[aero llm 16 codechallenge character counts in bert tokens](aero_llm_16_codechallenge_character_counts_in_bert_tokens.md)** | [Xem bài viết →](aero_llm_16_codechallenge_character_counts_in_bert_tokens.md) |
| [aero llm 17 translating between tokenizers](aero_llm_17_translating_between_tokenizers.md) | [Xem bài viết →](aero_llm_17_translating_between_tokenizers.md) |
| [aero llm 18 codechallenge more on token translation](aero_llm_18_codechallenge_more_on_token_translation.md) | [Xem bài viết →](aero_llm_18_codechallenge_more_on_token_translation.md) |
| [aero llm 19 codechallenge tokenization compression ratios](aero_llm_19_codechallenge_tokenization_compression_ratios.md) | [Xem bài viết →](aero_llm_19_codechallenge_tokenization_compression_ratios.md) |
| [aero llm 20 tokenization in different languages](aero_llm_20_tokenization_in_different_languages.md) | [Xem bài viết →](aero_llm_20_tokenization_in_different_languages.md) |
| [aero llm 21 codechallenge zipf s law in characters and tokens](aero_llm_21_codechallenge_zipf_s_law_in_characters_and_tokens.md) | [Xem bài viết →](aero_llm_21_codechallenge_zipf_s_law_in_characters_and_tokens.md) |
| [aero llm 22 word variations in claude tokenizer](aero_llm_22_word_variations_in_claude_tokenizer.md) | [Xem bài viết →](aero_llm_22_word_variations_in_claude_tokenizer.md) |

---
## 🤝 Liên hệ & Đóng góp
Dự án được phát triển bởi **Pixibox**. Mọi đóng góp về nội dung và mã nguồn đều được chào đón.

> *"Kiến thức là để chia sẻ. Hãy cùng nhau xây dựng cộng đồng AI vững mạnh!"* 🚀

*Cập nhật tự động bởi Aero-Indexer - 2026*
<!-- Aero-Footer-End -->
