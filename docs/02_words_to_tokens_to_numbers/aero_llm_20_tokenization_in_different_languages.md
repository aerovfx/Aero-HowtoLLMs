
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
Tokenization trong Các Ngôn ngữ Khác nhau:

Phân tích Toán học về Tỷ lệ Nén, Hình thái học và Ảnh hưởng đến Transformer

⸻

Tóm tắt

Dựa trên tài liệu đính kèm “Tokenization in Different Languages”, bài viết này phân tích sự khác biệt trong hành vi tokenization giữa các ngôn ngữ có đặc điểm hình thái và hệ chữ viết khác nhau. Chúng tôi xây dựng mô hình toán học cho tỷ lệ nén, entropy và độ dài chuỗi token, đồng thời phân tích tác động đến độ phức tạp tính toán trong kiến trúc Transformer. Các ví dụ minh họa được trình bày với tokenizer của BERT, mBERT và thư viện SentencePiece.

⸻

1. Giới thiệu

Tokenization ánh xạ chuỗi ký tự:

x \in \Sigma^*

thành chuỗi token:

\mathcal{T}(x) = (t_1, t_2, \dots, t_m)

Tuy nhiên, đặc điểm của ngôn ngữ (morphology, hệ chữ viết, khoảng trắng) ảnh hưởng mạnh đến:
	•	Độ dài trung bình của token
	•	Tỷ lệ nén
	•	Kích thước từ vựng
	•	Chi phí attention

⸻

2. Phân loại Ngôn ngữ theo Đặc điểm Tokenization

2.1 Ngôn ngữ phân tích (Analytic languages)

Ví dụ: tiếng Anh.
Từ thường tách bằng khoảng trắng.

Tokenizer như WordPiece (trong BERT) hoạt động hiệu quả.

⸻

2.2 Ngôn ngữ chắp dính (Agglutinative languages)

Ví dụ: tiếng Thổ Nhĩ Kỳ, tiếng Phần Lan.
Một từ có thể chứa nhiều hậu tố.

Nếu một từ có cấu trúc:

w = r + s_1 + s_2 + \dots + s_k

Độ dài ký tự tăng tuyến tính theo k.

Tokenizer phải chia nhỏ hơn:

m \uparrow

⸻

2.3 Ngôn ngữ không phân tách bằng khoảng trắng

Ví dụ: tiếng Trung.

Chuỗi ký tự:

x = c_1 c_2 \dots c_n

Mỗi ký tự có thể là một đơn vị nghĩa.

Trong trường hợp này:

R \approx 1

(trừ khi tokenizer gộp nhiều ký tự thành một token).

⸻

3. Mô hình Tỷ lệ Nén

Giả sử:
	•	n: số ký tự
	•	m: số token

3.1 Compression Ratio

R = \frac{n}{m}

Tương đương:

R = \mathbb{E}[L]

trong đó L là độ dài token.

⸻

3.2 So sánh giữa Ngôn ngữ

Giả sử:

R_{\text{EN}} = 4

R_{\text{ZH}} = 1.5

Chi phí attention:

C = O(m^2) = O\left(\left(\frac{n}{R}\right)^2\right)

Tỷ lệ chi phí:

\frac{C_{\text{ZH}}}{C_{\text{EN}}}
=
\left(\frac{R_{\text{EN}}}{R_{\text{ZH}}}\right)^2

Nếu R_{\text{EN}} = 4, R_{\text{ZH}} = 2:

= \left(\frac{4}{2}\right)^2 = 4

Tiếng Trung tốn gấp 4 lần chi phí attention cho cùng số ký tự.

⸻

4. Entropy theo Ngôn ngữ

Theo lý thuyết của Claude Shannon:

Entropy ký tự:

H_c = -\sum p(c)\log p(c)

Entropy token:

H_t = -\sum p(t)\log p(t)

Bảo toàn thông tin:

n H_c \approx m H_t

Suy ra:

R \approx \frac{H_t}{H_c}

Ngôn ngữ có bảng chữ cái lớn (như tiếng Trung) có:

H_c \uparrow
\Rightarrow R \downarrow

⸻

5. Tác động đến Mô hình Đa ngôn ngữ

5.1 mBERT

mBERT dùng chung từ vựng ~110k token cho nhiều ngôn ngữ.

Phân bố token không đồng đều:

p_{\text{lang}}(t) \neq \text{uniform}

Ngôn ngữ có ít dữ liệu → ít token chuyên biệt.

⸻

5.2 Tối ưu hóa Từ vựng

Bài toán:

\min_{V} \sum_{\ell} \alpha_\ell \left(\frac{n_\ell}{R_\ell}\right)^2 + \lambda |V|

Trong đó:
	•	\ell: ngôn ngữ
	•	\alpha_\ell: trọng số dữ liệu
	•	R_\ell: compression ratio của ngôn ngữ đó

⸻

6. Phân bố Độ dài Token

Gọi:

P_\ell(L=k)

Kỳ vọng:

\mathbb{E}_\ell[L] = \sum_k k P_\ell(L=k)

Ngôn ngữ chắp dính có:

\text{Var}(L) \uparrow

vì từ dài bị chia thành nhiều subword không đều.

⸻

7. Ảnh hưởng đến Độ phức tạp Huấn luyện

Transformer:

\text{Cost} = O(m^2 d)

Thay m = \frac{n}{R}:

\text{Cost} = O\left(\frac{n^2}{R^2} d\right)

Ngôn ngữ có R nhỏ làm tăng:
	•	Bộ nhớ GPU
	•	Thời gian huấn luyện
	•	Độ trễ suy luận

⸻

8. Phân tích Hình thái học

Nếu số hậu tố trung bình mỗi từ là k:

|w| \sim O(k)

Tokenizer tối ưu sẽ cố gắng học các đơn vị có xác suất cao:

\arg\max_{s} P(s)

Trong ngôn ngữ chắp dính, xác suất hậu tố phân tán → khó đạt nén cao.

⸻

9. Thảo luận

Khác biệt giữa các ngôn ngữ dẫn đến:
	1.	Compression ratio khác nhau
	2.	Chi phí attention khác nhau
	3.	Phân bố gradient khác nhau
	4.	Hiệu năng mô hình không đồng đều

Các hệ như Google và OpenAI phải cân bằng giữa:
	•	Bao phủ đa ngôn ngữ
	•	Kích thước từ vựng
	•	Chi phí tính toán

⸻

10. Kết luận

Tokenization phụ thuộc mạnh vào cấu trúc ngôn ngữ.

Các hệ thức quan trọng:

R = \frac{n}{m}

n H_c \approx m H_t

\text{Cost} = O\left(\frac{n^2}{R^2}\right)

Ngôn ngữ có compression ratio thấp sẽ chịu chi phí tính toán cao hơn trong Transformer.

Do đó, thiết kế tokenizer đa ngôn ngữ là bài toán tối ưu đa mục tiêu giữa:
	•	Entropy
	•	Kích thước từ vựng
	•	Phân bố dữ liệu
	•	Chi phí attention

⸻

Tài liệu tham khảo
	1.	Devlin et al. (2018). BERT: Pre-training of Deep Bidirectional Transformers.
	2.	Vaswani et al. (2017). Attention Is All You Need.
	3.	Shannon, C. (1948). A Mathematical Theory of Communication.
	4.	Kudo & Richardson (2018). SentencePiece.
	5.	Sennrich et al. (2016). Neural Machine Translation of Rare Words with Subword Units.
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
| [aero llm 16 codechallenge character counts in bert tokens](aero_llm_16_codechallenge_character_counts_in_bert_tokens.md) | [Xem bài viết →](aero_llm_16_codechallenge_character_counts_in_bert_tokens.md) |
| [aero llm 17 translating between tokenizers](aero_llm_17_translating_between_tokenizers.md) | [Xem bài viết →](aero_llm_17_translating_between_tokenizers.md) |
| [aero llm 18 codechallenge more on token translation](aero_llm_18_codechallenge_more_on_token_translation.md) | [Xem bài viết →](aero_llm_18_codechallenge_more_on_token_translation.md) |
| [aero llm 19 codechallenge tokenization compression ratios](aero_llm_19_codechallenge_tokenization_compression_ratios.md) | [Xem bài viết →](aero_llm_19_codechallenge_tokenization_compression_ratios.md) |
| 📌 **[aero llm 20 tokenization in different languages](aero_llm_20_tokenization_in_different_languages.md)** | [Xem bài viết →](aero_llm_20_tokenization_in_different_languages.md) |
| [aero llm 21 codechallenge zipf s law in characters and tokens](aero_llm_21_codechallenge_zipf_s_law_in_characters_and_tokens.md) | [Xem bài viết →](aero_llm_21_codechallenge_zipf_s_law_in_characters_and_tokens.md) |
| [aero llm 22 word variations in claude tokenizer](aero_llm_22_word_variations_in_claude_tokenizer.md) | [Xem bài viết →](aero_llm_22_word_variations_in_claude_tokenizer.md) |

---
## 🤝 Liên hệ & Đóng góp
Dự án được phát triển bởi **Pixibox**. Mọi đóng góp về nội dung và mã nguồn đều được chào đón.

> *"Kiến thức là để chia sẻ. Hãy cùng nhau xây dựng cộng đồng AI vững mạnh!"* 🚀

*Cập nhật tự động bởi Aero-Indexer - 2026*
<!-- Aero-Footer-End -->
