
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
Tokenization trong BERT: Phân tích Cơ chế WordPiece và Mô hình Toán học

⸻

Tóm tắt

Bài viết này phân tích cơ chế tokenization trong BERT dựa trên tài liệu đính kèm, tập trung vào thuật toán WordPiece, nền tảng của quá trình phân tách subword trong mô hình Google phát triển – BERT. Chúng tôi trình bày cơ sở toán học của WordPiece, so sánh với Byte Pair Encoding (BPE), đồng thời phân tích ảnh hưởng của tokenization đến embedding, attention và hàm mất mát trong huấn luyện.

⸻

1. Giới thiệu

BERT (Bidirectional Encoder Representations from Transformers) là mô hình Transformer encoder hai chiều được giới thiệu năm 2018.

Chuỗi đầu vào:

S = (w_1, w_2, ..., w_n)

Được ánh xạ thành chuỗi token:

T = (t_1, t_2, ..., t_m)

Với:

m \ge n

Do một từ có thể bị tách thành nhiều subword.

⸻

2. Thuật toán WordPiece

2.1 Nguyên lý cơ bản

WordPiece bắt đầu từ tập ký tự cơ sở và lặp lại quá trình:
	•	Chọn cặp subword có xác suất cao nhất
	•	Gộp lại thành một token mới

Khác với BPE (chọn theo tần suất), WordPiece tối ưu theo xác suất tối đa hóa likelihood.

⸻

2.2 Hàm Mục tiêu

Giả sử tập dữ liệu huấn luyện D.

WordPiece tối đa hóa:

\mathcal{L} = \sum_{w \in D} \log P$w$

Trong đó một từ w được phân rã thành:

w = (t_1, t_2, ..., t_k)

Xác suất:

P$w$ = \prod_{i=1}^{k} P$t_i$

Thuật toán chọn phép gộp làm tăng likelihood nhiều nhất.

⸻

2.3 Quy tắc Tiền tố “##”

Ví dụ:

playing → play + ##ing

Ký hiệu “##” cho biết token không ở đầu từ.

Điều này giúp mô hình phân biệt:
	•	“play” (từ độc lập)
	•	“##play” (không hợp lệ)

⸻

3. Mô hình Toán học của Tokenization

3.1 Phân bố Subword

Gọi:
	•	V: tập từ vựng
	•	|V|: kích thước (≈ 30k với BERT-base)

Phân bố:

P$t$ = \frac{\text{count}$t$}{\sum_{t' \in V} \text{count}(t')}

Entropy:

H = - \sum_{t \in V} P$t$\log P$t$

⸻

3.2 Độ dài Trung bình Chuỗi Token

Nếu văn bản có n từ và trung bình mỗi từ tách thành \alpha subword:

m = \alpha n

Self-attention trong Transformer encoder:

O$m^2$

Do đó:

O$(\alpha n$^2)

Tokenization ảnh hưởng trực tiếp đến chi phí tính toán.

⸻

4. So sánh WordPiece và BPE

Đặc điểm	WordPiece	BPE
Tiêu chí gộp	Tối đa hóa likelihood	Tần suất
Mô hình xác suất	Có	Không trực tiếp
Ứng dụng	BERT	GPT
Tối ưu	Theo corpus	Theo tần suất thuần

⸻

5. Biểu diễn Embedding

Mỗi token được ánh xạ:

E: V \rightarrow \mathbb{R}^d

Chuỗi token tạo thành ma trận:

X \in \mathbb{R}^{m \times d}

BERT cộng thêm:
	•	Positional embedding
	•	Segment embedding

Tổng embedding:

E_{\text{total}} = E_{\text{token}} + E_{\text{position}} + E_{\text{segment}}

⸻

6. Masked Language Modeling (MLM)

BERT huấn luyện bằng cách che một số token:

P$t_i \mid  T_{\setminus i}$

Loss:

\mathcal{L}_{MLM} = - \sum_{i \in M} \log P$t_i \mid  T_{\setminus i}$

Trong đó M là tập token bị mask.

Tokenization ảnh hưởng trực tiếp đến:
	•	Số token bị mask
	•	Độ khó của nhiệm vụ dự đoán

⸻

7. Phân tích Lý thuyết Thông tin

Tokenization tối ưu hóa sự cân bằng giữa:
	•	Vocabulary size |V|
	•	Độ dài chuỗi m

Bài toán tối ưu:

\min_{V} \left( \mathbb{E}[m] + \lambda |V| \right)

Với:
	•	\lambda: hệ số điều chỉnh
	•	\mathbb{E}[m]: số token trung bình

⸻

8. Tính Khái quát hóa (Generalization)

WordPiece cho phép xử lý từ hiếm:

Ví dụ:

unbelievable → un + ##believ + ##able

Do đó:

\forall w \notin V_{word}, \exists \text{decomposition in } V_{subword}

Giảm vấn đề OOV (Out-of-Vocabulary).

⸻

9. Hạn chế
	1.	Phụ thuộc corpus huấn luyện
	2.	Có thể tách không tự nhiên về mặt ngôn ngữ
	3.	Tăng độ dài chuỗi trong ngôn ngữ có cấu trúc phức tạp

⸻

10. Kết luận

Tokenization trong BERT dựa trên WordPiece có thể được mô hình hóa:

\max \sum_{w \in D} \log \prod_{i=1}^{k} P$t_i$

Ảnh hưởng trực tiếp đến:

m = \alpha n

\text{Attention Cost} = O$m^2$

H = - \sum P$t$\log P$t$

Do đó, thiết kế tokenizer là bài toán tối ưu đa mục tiêu giữa:
	•	Khả năng tổng quát hóa
	•	Hiệu suất tính toán
	•	Độ nén thông tin

⸻

Tài liệu tham khảo
	1.	BERT: Devlin, J. et al. (2018). BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding.
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
| 📌 **[aero llm 15 tokenization in bert](aero_llm_15_tokenization_in_bert.md)** | [Xem bài viết →](aero_llm_15_tokenization_in_bert.md) |
| [aero llm 16 codechallenge character counts in bert tokens](aero_llm_16_codechallenge_character_counts_in_bert_tokens.md) | [Xem bài viết →](aero_llm_16_codechallenge_character_counts_in_bert_tokens.md) |
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
